# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from admm_solver import ADMMSolver
from load_real_data import load_experiment_microgrids
from microgrid import Microgrid, build_network
from standalone_baseline import standalone_total

DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
SIZES = [3, 10, 20]
SWEEP_DAY = 15
SWEEP_SIZES = [3, 5, 10, 15, 20]

RHO = 0.01
MAX_ITER = 1000
TOL = 1e-3

OUT = Path(__file__).resolve().parent / "experiment_data_real" / "deployed_market_20260917.json"


def quiet(fn):
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


def build_mgs(day, n):
    raw = load_experiment_microgrids(day=day)[:n]
    return [
        Microgrid(item["mg_id"], None, real_data=item["data"],
                  soc_bounds=item["soc_bounds"])
        for item in raw
    ]


def run_one(day, n):
    mgs = build_mgs(day, n)

    t0 = time.perf_counter()
    sa_cost = standalone_total(mgs)
    t_sa = time.perf_counter() - t0

    solver = ADMMSolver(
        mgs,
        rho_init=RHO,
        max_iter=MAX_ITER,
        tol=TOL,
        adaptive=False,
        network=build_network(mgs),
    )
    t0 = time.perf_counter()
    quiet(solver.solve)
    t_coop = time.perf_counter() - t0

    coop = float(solver.final_system_cost)
    return {
        "day": day,
        "N": n,
        "arm": "deployed_fixed_rho0.01",
        "rho_init": RHO,
        "adaptive": False,
        "standalone_cost": sa_cost,
        "coop_cost": coop,
        "saving_pct": (sa_cost - coop) / sa_cost * 100.0,
        "converged": bool(solver.converged),
        "iterations": int(solver.final_iteration + 1),
        "time_standalone_s": t_sa,
        "time_admm_s": t_coop,
    }


def main():
    state = {"config": {"rho_init": RHO, "adaptive": False, "max_iter": MAX_ITER,
                        "tol": TOL, "arm": "deployed_fixed",
                        "days": DAYS, "sizes": SIZES,
                        "sweep_day": SWEEP_DAY, "sweep_sizes": SWEEP_SIZES},
             "multiday": [], "sweep": [], "failures": []}

    def flush():
        OUT.write_text(json.dumps(state, indent=1), encoding="utf-8")

    flush()
    jobs = [("multiday", d, n) for d in DAYS for n in SIZES] + \
           [("sweep", SWEEP_DAY, n) for n in SWEEP_SIZES]
    for k, (phase, day, n) in enumerate(jobs, 1):
        try:
            r = run_one(day, n)
            state[phase].append(r)
            print(f"[{k}/{len(jobs)}] {phase} day={day:>2} N={n:>2}: "
                  f"standalone={r['standalone_cost']:.2f} coop={r['coop_cost']:.2f} "
                  f"saving={r['saving_pct']:.3f}% iters={r['iterations']} "
                  f"conv={r['converged']}", flush=True)
        except Exception:
            state["failures"].append({"phase": phase, "day": day, "N": n,
                                      "error": traceback.format_exc()})
            print(f"[{k}/{len(jobs)}] {phase} day={day} N={n}: FAILED", flush=True)
        flush()

    print("saved ->", OUT.name)


if __name__ == "__main__":
    main()
