# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from admm_solver import ADMMSolver                                  # noqa: E402
from chen2024_baseline import (ACADMMChenSolver, _quiet_solve,      # noqa: E402
                               _quiet_solve_wrapper)
from centralized_solver import CentralizedSolver                    # noqa: E402
from load_real_data import load_experiment_microgrids               # noqa: E402
from microgrid import DistributionNetwork, Microgrid                 # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_data_real" / "acadmm_fullrepair_20260915.json"
DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
SIZES = (3, 5, 10)
MAX_ITER = 500
OUR_ARMS = [("ours_fixed_0.01", 0.01, False),
            ("ours_fixed_0.1", 0.1, False),
            ("ours_adaptive_0.1", 0.1, True)]
ALL_ARMS_DAYS = {15}


def arms_for(day):
    return OUR_ARMS if day in ALL_ARMS_DAYS else OUR_ARMS[:1]


def repaired_run(solver):
    """"""
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        solver.solve()
    wall = time.perf_counter() - t0
    import numpy as np
    imb = float(np.abs(np.asarray(solver.P_local) -
                       np.asarray(solver.P_global)).sum(axis=1).max())
    pre = float(solver.history["objective"][-1])
    _, rep = solver.repair_finalize()
    settled = float(rep["repair_cost"]) if rep else pre
    return (len(solver.history["objective"]), wall, settled,
            round(settled - pre, 2), round(imb, 4))


def main():
    out = (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
           {"config": {"days": DAYS, "sizes": list(SIZES), "max_iter": MAX_ITER,
                       "protocol": "every arm: solve + identical terminal repair + "
                                   "full-cost repricing; day-15 AC-ADMM re-run in-session",
                       "our_arms": [a[0] for a in OUR_ARMS]},
            "runs": []})
    done = {(r["day"], r["N"], r["method"]) for r in out["runs"]}

    for day in DAYS:
        raw_all = load_experiment_microgrids(day=day)
        for n in SIZES:
            mgs = [Microgrid(it["mg_id"], None, real_data=it["data"],
                             soc_bounds=it["soc_bounds"]) for it in raw_all[:n]]
            net = DistributionNetwork(n)
            cen = float(_quiet_solve_wrapper(CentralizedSolver(mgs, network=net))["optimal_cost"])

            for name, rho, adaptive in arms_for(day):
                if (day, n, name) in done:
                    continue
                s = ADMMSolver(mgs, rho_init=rho, max_iter=MAX_ITER, tol=1e-3,
                               adaptive=adaptive, network=net)
                iters, wall, settled, delta, imb = repaired_run(s)
                out["runs"].append({
                    "day": day, "N": n, "method": name, "centralized": cen,
                    "converged": bool(s.converged), "stop": s.stop_reason or "max_iter",
                    "iterations": iters, "time_s": round(wall, 1),
                    "settled_cost": settled, "repair_delta": delta,
                    "pre_repair_imbalance_kw": imb,
                    "gap_pct": round(100.0 * (settled - cen) / cen, 3)})
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                print(f"  day{day:>2} N={n:>2} {name:<18} it={iters:>4} "
                      f"stop={str(s.stop_reason):<8} settled={settled:9.2f} "
                      f"rep={delta:+6.2f} gap={100.0*(settled-cen)/cen:5.2f}% "
                      f"wall={wall:6.1f}s", flush=True)

            if day == 15 and (day, n, "ac_admm") not in done:
                ac = ACADMMChenSolver(mgs, rho_init=0.1, max_iter=MAX_ITER,
                                      tol=1e-3, network=net)
                iters, wall, settled, delta, imb = repaired_run(ac)
                stop = ("boyd" if ac.converged else "") + \
                       ("+chen" if ac.chen_converged else "") or "max_iter"
                out["runs"].append({
                    "day": day, "N": n, "method": "ac_admm", "centralized": cen,
                    "converged": bool(ac.converged or ac.chen_converged),
                    "stop": stop, "iterations": iters, "time_s": round(wall, 1),
                    "settled_cost": settled, "repair_delta": delta,
                    "pre_repair_imbalance_kw": imb,
                    "gap_pct": round(100.0 * (settled - cen) / cen, 3)})
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                print(f"  day{day:>2} N={n:>2} ac_admm            it={iters:>4} "
                      f"stop={stop:<8} settled={settled:9.2f} gap="
                      f"{100.0*(settled-cen)/cen:5.2f}% wall={wall:6.1f}s", flush=True)
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
