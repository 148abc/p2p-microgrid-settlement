""""""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_real_data import load_experiment_microgrids
from microgrid import Microgrid, build_network
from admm_solver import ADMMSolver
from witness_generator_v2 import build_witness_v2

HERE = Path(__file__).resolve().parent
ZKP = HERE.parent / "zkp_circuit"
DAY, N = 15, 10
GO = "go"


def quiet(fn, *a, **kw):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


def main():
    t_all = time.perf_counter()
    results = {"day": DAY, "N": N, "mg_results": [], "tamper": {}}

    raw = load_experiment_microgrids(day=DAY)[:N]
    mgs = [Microgrid(it["mg_id"], None, real_data=it["data"],
                     soc_bounds=it["soc_bounds"]) for it in raw]
    solver = ADMMSolver(mgs, rho_init=0.01, max_iter=1000, tol=1e-3,
                        adaptive=True, network=build_network(mgs))
    t0 = time.perf_counter()
    quiet(solver.solve)
    _, rep = quiet(solver.repair_finalize)
    wall = time.perf_counter() - t0
    print(f"ADMM reason={solver.stop_reason} iters={solver.final_iteration+1} "
          f"repair={rep['repair_cost']:.2f}$ wall={wall:.0f}s")
    results["admm"] = {"stop_reason": solver.stop_reason,
                       "iterations": solver.final_iteration + 1,
                       "settled_cost": rep["repair_cost"],
                       "wall_s": round(wall, 1)}

    env_base = dict(os.environ)
    t_w = time.perf_counter()
    paths = []
    for i in range(N):
        w = build_witness_v2(solver, mg_idx=i, session_id=f"day{DAY}-N{N}-settle")
        wpath = HERE / f"witness_v2_mg{i+1}.json"
        wpath.write_text(json.dumps(w), encoding="utf-8")
        paths.append((wpath, w))
    wit_wall = time.perf_counter() - t_w
    t_p = time.perf_counter()
    for i, (wpath, w) in enumerate(paths):
        env = dict(env_base, WITNESS_V2=str(wpath))
        t0 = time.perf_counter()
        r = subprocess.run(
            [GO, "test", "-run", "TestProveV2FromJSON", "-count=1", "-v",
             "-timeout", "600s", "."],
            cwd=ZKP, env=env, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = "PASS" in r.stdout and "v2 e2e OK" in r.stdout
        cons = None
        for line in r.stdout.splitlines():
            if "constraints=" in line:
                cons = int(line.split("constraints=")[1].split()[0])
        results["mg_results"].append({
            "mg": i + 1, "proved": ok, "constraints": cons,
            "wall_s": round(dt, 2), "cost_claim": w["public"]["cost_claim"]})
        print(f"MG{i+1}: proved={ok} constraints={cons} wall={dt:.1f}s")
    prove_wall = time.perf_counter() - t_p

    w = build_witness_v2(solver, mg_idx=0, session_id=f"day{DAY}-N{N}-settle")
    w["public"]["cost_claim"] += 1_000_000
    wpath = HERE / "witness_v2_tampered.json"
    wpath.write_text(json.dumps(w), encoding="utf-8")
    env = dict(env_base, WITNESS_V2=str(wpath))
    r = subprocess.run(
        [GO, "test", "-run", "TestProveV2FromJSON", "-count=1", "-timeout",
         "600s", "."], cwd=ZKP, env=env, capture_output=True, text=True)
    rejected = "FAIL" in r.stdout
    results["tamper"] = {"kind": "cost_claim+1$", "rejected": rejected}
    print(f"tamper rejected={rejected}")

    results["timings"] = {"admm_s": round(wall, 1),
                          "witness_build_s": round(wit_wall, 1),
                          "prove_verify_s": round(prove_wall, 1),
                          "total_s": round(time.perf_counter() - t_all, 1)}
    print("timings:", results["timings"])
    (HERE / "e2e_v2_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print("saved -> e2e_v2_results.json")


if __name__ == "__main__":
    main()
