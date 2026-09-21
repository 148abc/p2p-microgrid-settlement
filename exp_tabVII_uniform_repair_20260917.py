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
from admm_solver import ADMMSolver                        # noqa: E402
from load_real_data import load_experiment_microgrids     # noqa: E402
from microgrid import Microgrid, build_network            # noqa: E402

OUT = HERE / "experiment_data_real" / "tabVII_uniform_repair_20260917.json"
DAY, N = 15, 10
C_ITER_PAPER = 0.3959
FIXED_RHOS = [0.002, 0.005, 0.01, 0.05, 0.1, 0.2]
TAUS = [0.1, 0.3, 1.0, 3.0]
WINDOW, K_MIN, GAMMA_BAR = 5, 15, 0.9


def make_mgs():
    raw = load_experiment_microgrids(day=DAY)[:N]
    return [Microgrid(it["mg_id"], None, real_data=it["data"],
                      soc_bounds=it["soc_bounds"]) for it in raw]


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return fn(*a, **kw)


def run_row(kind, param):
    mgs = make_mgs()
    kw = {"rho_init": param, "adaptive": False} if kind == "fixed" else {
        "rho_init": 0.01, "adaptive": True,
        "economic_stop": {"delta_tol": param * C_ITER_PAPER, "window": WINDOW,
                          "k_min": K_MIN, "gamma_bar": GAMMA_BAR}}
    solver = ADMMSolver(mgs, network=build_network(mgs), max_iter=1000, tol=1e-3, **kw)
    t0 = time.perf_counter()
    quiet(solver.solve)
    wall = time.perf_counter() - t0
    pre = float(solver.final_system_cost)
    imb = float(np.abs(np.asarray(solver.P_local) - np.asarray(solver.P_global)).sum(axis=1).max())
    results, stats = quiet(solver.repair_finalize)
    repaired = float(stats["repair_cost"]) if stats else None
    return {"kind": kind, "param": param, "iters": int(solver.final_iteration + 1),
            "stop": solver.stop_reason, "converged": bool(solver.converged),
            "gamma_hat": solver.economic_certified,
            "cost_at_stop": round(pre, 2),
            "repaired_cost": round(repaired, 2) if repaired else None,
            "repair_delta": round(repaired - pre, 2) if repaired else None,
            "pre_repair_imbalance_kw": round(imb, 4),
            "repair_time_s": round(stats["repair_time_s"], 3) if stats else None,
            "wall_s": round(wall, 1)}


def main():
    force = "--force" in sys.argv
    state = ({"config": {"day": DAY, "N": N, "max_iter": 1000, "tol": 1e-3,
                         "protocol": "every row: solve, then the identical terminal repair round",
                         "c_iter_paper": C_ITER_PAPER}, "rows": []}
             if force else
             (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
              {"config": {"day": DAY, "N": N, "max_iter": 1000, "tol": 1e-3,
                          "protocol": "every row: solve, then the identical terminal repair round",
                          "c_iter_paper": C_ITER_PAPER}, "rows": []}))
    done = set() if force else {(r["kind"], r["param"]) for r in state["rows"]}
    jobs = [("fixed", p) for p in FIXED_RHOS] + [("sa", p) for p in TAUS]
    for k, (kind, param) in enumerate(jobs, 1):
        if (kind, param) in done:
            print(f"[{k}/{len(jobs)}] {kind} {param}: cached")
            continue
        r = run_row(kind, param)
        state["rows"].append(r)
        OUT.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"[{k}/{len(jobs)}] {kind:5s} {param:<6} iters={r['iters']:<4} "
              f"stop={r['stop']:<8} stop_cost={r['cost_at_stop']:<8} "
              f"repaired={r['repaired_cost']:<8} delta={r['repair_delta']}", flush=True)
    print("saved ->", OUT.name)


if __name__ == "__main__":
    main()
