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

OUT = HERE / "experiment_data_real" / "rho_band_20260915.json"
DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
RHOS = {3: [0.002, 0.005, 0.01, 0.02, 0.05],
        10: [0.002, 0.005, 0.01, 0.02, 0.05],
        20: [0.005, 0.01, 0.02]}
CENTRALIZED = {3: {0: 47.80, 7: 2192.39, 15: 2156.92, 23: 2951.88,
                   31: 1525.92, 39: 2169.98, 47: 2176.35, 55: 1829.81},
               10: {0: 1146.93, 7: 1749.69, 15: 2809.16, 23: 4042.79,
                    31: 1772.88, 39: 3453.50, 47: 1255.78, 55: 2063.24},
               20: {}}


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
    todo = [int(a) for a in sys.argv[1:]] or [3, 20]
    data = (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
            {"config": {"days": DAYS, "rhos": RHOS, "max_iter": 1000,
                        "protocol": "Boyd stop, adaptive=False, identical terminal repair"},
             "rows": []})
    done = {(r["day"], r["N"], r["rho"]) for r in data["rows"]}

    for n in todo:
        print(f"===== N={n} =====", flush=True)
        for day in DAYS:
            for rho in RHOS[n]:
                if (day, n, rho) in done:
                    continue
                s, wall = run(make_mgs(day, n), rho_init=rho, adaptive=False)
                pre = round(float(s.history["objective"][-1]), 2)
                settled, imb = settle(s)
                cref = CENTRALIZED[n].get(day)
                rec = {"day": day, "N": n, "rho": rho, "iters": len(s.history["objective"]),
                       "stop": s.stop_reason, "converged": bool(s.converged),
                       "settled_cost": settled,
                       "repair_delta": (round(settled - pre, 2) if settled else None),
                       "pre_repair_imbalance_kw": imb,
                       "gap_pct_centralized": (round(100 * (settled - cref) / cref, 3)
                                               if (settled and cref) else None),
                       "wall_s": round(wall, 1)}
                data["rows"].append(rec)
                OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                               encoding="utf-8")
                print(f"  day{day:2d} N={n:2d} rho={rho:<6} it={rec['iters']:<4} "
                      f"stop={rec['stop']:<8} settled={settled} "
                      f"gap={rec['gap_pct_centralized']} wall={rec['wall_s']}s", flush=True)

    print("\n===== 区间汇总 (收敛率 / 迭代中位) =====")
    rows = data["rows"]
    for n in sorted({r["N"] for r in rows}):
        for rho in RHOS[n]:
            sub = [r for r in rows if r["N"] == n and r["rho"] == rho]
            if not sub:
                continue
            conv = [r for r in sub if r["converged"]]
            med = sorted(r["iters"] for r in conv)[len(conv) // 2] if conv else None
            print(f"N={n:2d} rho={rho:<6} 收敛 {len(conv)}/{len(sub)}  迭代中位 {med}")
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
