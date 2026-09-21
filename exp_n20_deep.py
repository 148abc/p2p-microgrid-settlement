# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admm_solver import ADMMSolver
from load_real_data import load_all_microgrids
from microgrid import Microgrid, build_network
from standalone_baseline import standalone_total

DAY = 15
CENT = json.load(open("experiment_data_real/sa_frontier_data.json"))["20"]["cent"]

raw = load_all_microgrids(day=DAY)
by_id = {it["mg_id"]: it for it in raw}
mgs = [Microgrid(i, None, real_data=by_id[i]["data"], soc_bounds=by_id[i]["soc_bounds"])
       for i in range(1, 21)]

sa = standalone_total(mgs)
print(f"standalone total (post-curtail-fix): {sa:.2f}")
print(f"centralized reference: {CENT:.2f}")

solver = ADMMSolver(mgs, rho_init=0.01, max_iter=1000, tol=1e-5,
                    adaptive=True, network=build_network(mgs))
t0 = time.time()
buf = io.StringIO()
with redirect_stdout(buf):
    solver.solve()
wall = time.time() - t0

hist = {k: (len(v) if hasattr(v, "__len__") else v) for k, v in solver.history.items()}
print("history keys:", hist)
costs = solver.history.get("objective")
if costs is not None:
    costs = np.asarray(costs, dtype=float)
    marks = [10, 25, 50, 100, 200, 400, 600, 800, len(costs) - 1]
    for m in marks:
        if m < len(costs):
            print(f"  iter {m+1:4d}: cost {costs[m]:9.2f}  gap {(costs[m]-CENT)/CENT*100:6.2f}%")
print(f"iters={len(costs) if costs is not None else 'n/a'} converged={solver.converged} wall={wall:.0f}s")
final = float(solver.final_system_cost)
print(f"FINAL settled={final:.2f}  gap={(final-CENT)/CENT*100:.2f}%  "
      f"savings={(sa-final)/sa*100:.2f}%")
out_path = Path(__file__).resolve().parent / "experiment_data_real" / "n20_deep.json"
out_path.write_text(json.dumps({"standalone": float(sa), "settled": final, "cent": CENT,
                                "iters": int(len(costs)) if costs is not None else None,
                                "gap_pct": (final - CENT) / CENT * 100,
                                "saving_pct": (sa - final) / sa * 100},
                               indent=1), encoding="utf-8")
