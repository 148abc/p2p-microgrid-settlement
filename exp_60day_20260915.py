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

OUT = HERE / "experiment_data_real" / "exp60day_20260915.json"
DAYS = list(range(60))
N = 10
SIGMA_REF = 0.4189
CENTRALIZED = {0: 1146.93, 7: 1749.69, 15: 2809.16, 23: 4042.79,
               31: 1772.88, 39: 3453.50, 47: 1255.78, 55: 2063.24}
ARMS = [("fixed_0.005", 0.005, False), ("fixed_0.01", 0.01, False)]


def make_mgs(day, n):
    raw = load_experiment_microgrids(day=day)[:n]
    return [Microgrid(it["mg_id"], None, real_data=it["data"],
                      soc_bounds=it["soc_bounds"]) for it in raw]


def volatility(mgs):
    sig = [float(np.std((mg.total_generation - mg.load) / np.maximum(mg.load, 1e-9)))
           for mg in mgs]
    return float(np.mean(sig))


def prior_seed(sig):
    return float(np.clip(0.01 * SIGMA_REF / max(sig, 1e-9), 0.002, 0.05))


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
            {"config": {"days": DAYS, "N": N, "max_iter": 1000, "sigma_ref": SIGMA_REF,
                        "arms": [a[0] for a in ARMS] + ["prior_seed"],
                        "protocol": "Boyd stop; identical terminal repair"},
             "rows": []})
    done = {(r["day"], r["arm"]) for r in data["rows"]}

    for day in DAYS:
        mgs = None
        sig = None
        for arm, rho0, adapt in ARMS + [("prior_seed", None, True)]:
            if (day, arm) in done:
                continue
            if mgs is None:
                mgs = make_mgs(day, N)
                sig = volatility(mgs)
            r = prior_seed(sig) if arm == "prior_seed" else rho0
            s, wall = run(mgs, rho_init=r, adaptive=adapt)
            pre = round(float(s.history["objective"][-1]), 2)
            settled, imb = settle(s)
            cref = CENTRALIZED.get(day)
            rec = {"day": day, "arm": arm, "rho0": round(float(r), 4),
                   "sigma": round(sig, 4), "iters": len(s.history["objective"]),
                   "stop": s.stop_reason, "converged": bool(s.converged),
                   "cost_at_stop": pre, "settled_cost": settled,
                   "pre_repair_imbalance_kw": imb,
                   "gap_pct": (round(100 * (settled - cref) / cref, 3)
                               if (settled and cref) else None),
                   "wall_s": round(wall, 1)}
            data["rows"].append(rec)
            OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            print(f"  day{day:2d} {arm:<12} rho0={r:.4f} it={rec['iters']:<4} "
                  f"stop={rec['stop']:<8} settled={settled} wall={rec['wall_s']}s",
                  flush=True)
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
