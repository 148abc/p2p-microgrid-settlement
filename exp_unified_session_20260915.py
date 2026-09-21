# -*- coding: utf-8 -*-
""""""
import io
import json
import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from admm_solver import ADMMSolver                       # noqa: E402
from load_real_data import load_experiment_microgrids    # noqa: E402
from microgrid import Microgrid, build_network           # noqa: E402

OUT = HERE / "experiment_data_real" / "unified_session_20260915.json"
DAY, N = 15, 10
CENTRALIZED = 2809.16
C_ITER_PAPER = 0.3959
FIXED_RHOS = [0.002, 0.005, 0.01, 0.05, 0.1, 0.2]
TAUS = [0.1, 0.3, 1.0, 3.0]
WINDOW, K_MIN, GAMMA_BAR = 5, 15, 0.9


def make_mgs(day, n):
    raw = load_experiment_microgrids(day=day)[:n]
    return [Microgrid(it["mg_id"], None, real_data=it["data"],
                      soc_bounds=it["soc_bounds"]) for it in raw]


def run(mgs, **kw):
    solver = ADMMSolver(mgs, network=build_network(mgs), max_iter=kw.pop("max_iter", 1000),
                        tol=1e-3, **kw)
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        solver.solve()
    return solver, time.perf_counter() - t0


def settle(solver):
    """"""
    d = np.asarray(solver.P_local) - np.asarray(solver.P_global)
    imb = float(np.abs(d).sum(axis=1).max())
    _, stats = solver.repair_finalize()
    return (round(float(stats["repair_cost"]), 2) if stats else None), round(imb, 4)


def c_solve_mean(solver):
    s = solver.history.get("solve_time_s") or []
    return round(float(np.mean(s)), 4) if s else None


def main():
    data = (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
            {"config": {"day": DAY, "N": N, "centralized": CENTRALIZED,
                        "c_iter_paper": C_ITER_PAPER, "fixed_rhos": FIXED_RHOS,
                        "taus": TAUS, "window": WINDOW, "k_min": K_MIN,
                        "gamma_bar": GAMMA_BAR, "max_iter": 1000,
                        "protocol": "single session; identical terminal repair for every row",
                        "host": platform.node(), "python": sys.version.split()[0]},
             "rows": []})
    done = {(r["kind"], r["param"]) for r in data["rows"]}

    for rho in FIXED_RHOS:
        if ("fixed", rho) in done:
            continue
        s, wall = run(make_mgs(DAY, N), rho_init=rho, adaptive=False)
        pre = round(float(s.history["objective"][-1]), 2)
        settled, imb = settle(s)
        rec = {"kind": "fixed", "param": rho, "iters": len(s.history["objective"]),
               "stop": s.stop_reason, "converged": bool(s.converged),
               "cost_at_stop": pre, "settled_cost": settled,
               "repair_delta": (round(settled - pre, 2) if settled else None),
               "pre_repair_imbalance_kw": imb,
               "gap_pct": (round(100 * (settled - CENTRALIZED) / CENTRALIZED, 3) if settled else None),
               "wall_s": round(wall, 1), "c_solve_measured": c_solve_mean(s)}
        data["rows"].append(rec)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  fixed rho={rho:<6} it={rec['iters']:<4} stop={rec['stop']:<8} "
              f"settled={rec['settled_cost']} wall={rec['wall_s']}s", flush=True)

    for tau in TAUS:
        if ("sa", tau) in done:
            continue
        s, wall = run(make_mgs(DAY, N), rho_init=0.01, adaptive=True,
                      economic_stop={"delta_tol": tau * C_ITER_PAPER,
                                     "window": WINDOW, "k_min": K_MIN,
                                     "gamma_bar": GAMMA_BAR})
        pre = round(float(s.history["objective"][-1]), 2)
        settled = round(float(s.final_system_cost), 2)
        rec = {"kind": "sa", "param": tau, "iters": len(s.history["objective"]),
               "stop": s.stop_reason, "converged": bool(s.converged),
               "gamma_hat": s.economic_certified,
               "cost_at_stop": pre, "settled_cost": settled,
               "repair_delta": round(settled - pre, 2),
               "gap_pct": round(100 * (settled - CENTRALIZED) / CENTRALIZED, 3),
               "wall_s": round(wall, 1), "c_solve_measured": c_solve_mean(s)}
        data["rows"].append(rec)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  SA tau_V={tau:<4} it={rec['iters']:<4} stop={rec['stop']:<8} "
              f"gamma={rec['gamma_hat']} settled={settled} wall={rec['wall_s']}s", flush=True)

    print("\n===== 单-session 全表 =====")
    print(f"{'rule':<16}{'iters':>6}{'stop':>10}{'cost@stop':>11}{'settled':>10}"
          f"{'gap%':>8}{'wall_s':>8}{'c_solve':>9}")
    for r in data["rows"]:
        tag = f"fixed rho={r['param']}" if r["kind"] == "fixed" else f"SA tau={r['param']}"
        print(f"{tag:<16}{r['iters']:>6}{str(r['stop']):>10}{r['cost_at_stop']:>11}"
              f"{r['settled_cost']:>10}{r['gap_pct']:>8}{r['wall_s']:>8}"
              f"{r['c_solve_measured']:>9}")
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
