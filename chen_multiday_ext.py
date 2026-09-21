# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chen2024_baseline import (ADMMSolver, ACADMMChenSolver,
                               _quiet_solve, _quiet_solve_wrapper)
from centralized_solver import CentralizedSolver
from load_real_data import load_experiment_microgrids
from microgrid import Microgrid, DistributionNetwork

DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
SIZES = (3, 5, 10)
MAX_ITER = 500

out = {"days": DAYS, "sizes": list(SIZES), "max_iter": MAX_ITER,
       "protocol": "identical repair + full-cost repricing; AC rho0=0.1, ours rho0=0.1",
       "runs": []}

for day in DAYS:
    raw_all = load_experiment_microgrids(day=day)
    for n in SIZES:
        mgs = [Microgrid(it["mg_id"], None, real_data=it["data"],
                         soc_bounds=it["soc_bounds"]) for it in raw_all[:n]]
        net = DistributionNetwork(n)

        cs = CentralizedSolver(mgs, network=net)
        cinfo = _quiet_solve_wrapper(cs)
        cen = float(cinfo["optimal_cost"])

        row = {"day": day, "N": n, "centralized": cen, "methods": {}}
        for name, adaptive in (("ours_adaptive", True), ("ours_fixed", False)):
            solver = ADMMSolver(mgs, rho_init=0.1, max_iter=MAX_ITER, tol=1e-3,
                                adaptive=adaptive, network=net)
            old = sys.stdout
            sys.stdout = io.StringIO()
            t0 = time.perf_counter()
            _quiet_solve(solver)
            dt = time.perf_counter() - t0
            sys.stdout = old
            row["methods"][name] = {
                "converged": bool(solver.converged),
                "iterations": int(solver.final_iteration + 1),
                "time_s": dt,
                "settled_cost": float(solver.final_system_cost),
                "gap_pct": 100.0 * (float(solver.final_system_cost) - cen) / cen,
            }

        solver = ACADMMChenSolver(mgs, rho_init=0.1, max_iter=MAX_ITER, tol=1e-3,
                                  network=net)
        old = sys.stdout
        sys.stdout = io.StringIO()
        t0 = time.perf_counter()
        solver.solve()
        dt = time.perf_counter() - t0
        _, rep = solver.repair_finalize()
        sys.stdout = old
        settled = float(rep["repair_cost"]) if rep else float(solver.final_system_cost)
        stop = ("boyd" if solver.converged else "") + \
               ("+chen" if solver.chen_converged else "") or "max_iter"
        row["methods"]["ac_admm"] = {
            "converged": bool(solver.converged or solver.chen_converged),
            "stop_reason": stop,
            "iterations": int(solver.final_iteration + 1),
            "time_s": dt,
            "settled_cost": settled,
            "gap_pct": 100.0 * (settled - cen) / cen,
        }
        out["runs"].append(row)
        m = row["methods"]
        print(f"day={day:>2} N={n:>2} cen={cen:9.2f} | "
              f"ours {m['ours_adaptive']['iterations']:>3}it {m['ours_adaptive']['time_s']:7.1f}s "
              f"gap={m['ours_adaptive']['gap_pct']:6.2f}% | "
              f"ac {m['ac_admm']['iterations']:>3}it {m['ac_admm']['time_s']:7.1f}s "
              f"gap={m['ac_admm']['gap_pct']:6.2f}% ({m['ac_admm']['stop_reason']})",
              flush=True)

        Path("experiment_data_real/chen_multiday_ext.json").write_text(
            json.dumps(out, indent=1), encoding="utf-8")

print("[OK] saved -> experiment_data_real/chen_multiday_ext.json", flush=True)
