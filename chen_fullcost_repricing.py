# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chen2024_baseline import ACADMMChenSolver
from load_real_data import load_experiment_microgrids
from microgrid import Microgrid, DistributionNetwork

DAY = 15
MAX_ITER = 300
out = {"day": DAY, "max_iter": MAX_ITER, "note": "full-cost repricing of AC-ADMM cleared state", "rows": []}

for n in (3, 5, 10):
    raw = load_experiment_microgrids(day=DAY)[:n]
    mgs = [Microgrid(it["mg_id"], None, real_data=it["data"],
                     soc_bounds=it["soc_bounds"]) for it in raw]
    net = DistributionNetwork(n)
    solver = ACADMMChenSolver(mgs, rho_init=0.1, max_iter=MAX_ITER, tol=1e-3,
                              network=net)
    old = sys.stdout
    sys.stdout = io.StringIO()
    solver.solve()
    _, rep = solver.repair_finalize()
    sys.stdout = old

    full_cost = rep["repair_cost"] if rep else float("nan")
    iters = solver.final_iteration + 1
    stop = ("boyd" if solver.converged else "") + \
           ("+chen" if solver.chen_converged else "") or "max_iter"
    rows = {"N": n, "iterations": iters, "stop_reason": stop,
            "grid_only_cost": float(solver.final_system_cost),
            "full_cost_settled": float(full_cost),
            "repair_time_s": rep["repair_time_s"] if rep else None}
    out["rows"].append(rows)
    print(f"N={n}: iters={iters} stop={stop} grid-only={rows['grid_only_cost']:.2f} "
          f"full-cost settled={full_cost:.2f}")

Path("experiment_data_real/chen_acadmm_fullcost.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
print("saved -> experiment_data_real/chen_acadmm_fullcost.json")
