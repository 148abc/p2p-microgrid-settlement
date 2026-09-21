# -*- coding: utf-8 -*-
""""""
import io
import json
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

OUT = HERE / "experiment_data_real" / "stop_multiday_20260915.json"
DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
SIZES = (3, 10)
C_SOLVE = {3: 0.0527, 10: 0.3959}
TAUS = [1.0, 3.0]
WINDOW, K_MIN, GAMMA_BAR = 5, 15, 0.9


def make_mgs(day, n):
    raw = load_experiment_microgrids(day=day)[:n]
    return [Microgrid(it["mg_id"], None, real_data=it["data"],
                      soc_bounds=it["soc_bounds"]) for it in raw]


def run(mgs, **kw):
    s = ADMMSolver(mgs, network=build_network(mgs), max_iter=1000, tol=1e-3, **kw)
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        s.solve()
    return s, time.perf_counter() - t0


def settle(s):
    d = np.asarray(s.P_local) - np.asarray(s.P_global)
    imb = float(np.abs(d).sum(axis=1).max())
    _, stats = s.repair_finalize()
    return (round(float(stats["repair_cost"]), 2) if stats else None), round(imb, 4)


def main():
    data = (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
            {"config": {"days": DAYS, "sizes": list(SIZES), "taus": TAUS,
                        "c_solve": C_SOLVE, "window": WINDOW, "k_min": K_MIN,
                        "gamma_bar": GAMMA_BAR, "max_iter": 1000,
                        "protocol": "Boyd-or-economic stop; identical terminal repair"},
             "rows": []})
    done = {(r["day"], r["N"], r["arm"]) for r in data["rows"]}

    for n in SIZES:
        print(f"===== N={n} =====", flush=True)
        for day in DAYS:
            arms = [("fixed_0.01", dict(rho_init=0.01, adaptive=False))]
            for tau in TAUS:
                arms.append((f"sa_tau{tau:g}",
                             dict(rho_init=0.01, adaptive=True,
                                  economic_stop={"delta_tol": tau * C_SOLVE[n],
                                                 "window": WINDOW, "k_min": K_MIN,
                                                 "gamma_bar": GAMMA_BAR})))
            for arm, kw in arms:
                if (day, n, arm) in done:
                    continue
                s, wall = run(make_mgs(day, n), **kw)
                pre = round(float(s.history["objective"][-1]), 2)
                settled, imb = settle(s)
                rec = {"day": day, "N": n, "arm": arm,
                       "iters": len(s.history["objective"]), "stop": s.stop_reason,
                       "gamma_hat": s.economic_certified,
                       "cost_at_stop": pre, "settled_cost": settled,
                       "repair_delta": (round(settled - pre, 2) if settled else None),
                       "pre_repair_imbalance_kw": imb,
                       "wall_s": round(wall, 1)}
                data["rows"].append(rec)
                OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                print(f"  day{day:2d} N={n:2d} {arm:<12} it={rec['iters']:<4} "
                      f"stop={rec['stop']:<8} settled={settled} wall={rec['wall_s']}s",
                      flush=True)
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
