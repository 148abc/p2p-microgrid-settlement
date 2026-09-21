# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gurobipy
import admm_solver
from admm_solver import ADMMSolver
from load_real_data import load_experiment_microgrids
from microgrid import Microgrid, build_network
from centralized_solver import CentralizedSolver

OUT_DIR = Path(__file__).resolve().parent / "experiment_data_real"
OUT_JSON = OUT_DIR / "seed_stab_sweep.json"

DAY = 15
N = 10
MAX_ITER = 1000
TOL = 1e-3
TAU_GRID = [0.1, 0.3, 1.0, 3.0]
WINDOW = 5
K_MIN = 15
GAMMA_BAR = 0.9
SIGMA_REF = 0.4189
SEEDS = [0, 1, 2, 3, 4]
W_STAB = 10

# ----------------------------------------------------------------
# ----------------------------------------------------------------
_REAL_MODEL = gurobipy.Model
_CURRENT_SEED = [0]


def _seeded_model(name=""):
    m = _REAL_MODEL(name)
    m.setParam("Seed", _CURRENT_SEED[0])
    return m


class SeedScope:
    """"""
    def __enter__(self):
        admm_solver.gp.Model = _seeded_model
        _CURRENT_SEED[0] = self.seed
        return self

    def __init__(self, seed):
        self.seed = seed

    def __exit__(self, *a):
        admm_solver.gp.Model = _REAL_MODEL
        _CURRENT_SEED[0] = 0


# ----------------------------------------------------------------
# ----------------------------------------------------------------
class InstrumentedADMMSolver(ADMMSolver):
    """"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cur = {}
        self.mode_hist = []

    def solve_local_problem(self, mg_idx, update_state=True, fix_trades=False):
        res = super().solve_local_problem(mg_idx, update_state, fix_trades)
        if update_state:
            self._cur[mg_idx] = ((res["P_ch"] > 1e-6).astype(int)
                                 - (res["P_dis"] > 1e-6).astype(int))
            if len(self._cur) == self.N:
                self.mode_hist.append(np.stack([self._cur[i] for i in range(self.N)]))
                self._cur = {}
        return res


def quiet(fn, *args, **kwargs):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = old


def make_mgs(day, n):
    raw = load_experiment_microgrids(day=day)[:n]
    return [
        Microgrid(it["mg_id"], None, real_data=it["data"], soc_bounds=it["soc_bounds"])
        for it in raw
    ]


def renewable_volatility(mgs):
    sigmas = []
    for mg in mgs:
        res = (mg.total_generation - mg.load) / np.maximum(mg.load, 1e-9)
        sigmas.append(float(np.std(res)))
    return float(np.mean(sigmas))


def stab_stats(mode_hist, w=W_STAB):
    """"""
    modes = np.stack(mode_hist)              # [K, N, T]
    flip = (np.diff(modes, axis=0) != 0).any(axis=(1, 2))  # [K-1] bool
    last_flip = int(np.nonzero(flip)[0].max()) + 1 if flip.any() else 0
    trigger = None
    cnt = 0
    for k in range(len(flip)):
        cnt = 0 if flip[k] else cnt + 1
        if cnt >= w and trigger is None:
            trigger = k + 1
            break
    return {
        "iters": int(len(mode_hist)),
        "last_flip_iter": last_flip,
        "stab_trigger_iter": trigger,
        "stab_margin_iters": (None if trigger is None
                              else int(len(mode_hist) - trigger)),
        "flips_total": int(flip.sum()),
    }


def run_one(method, rho_init, economic_stop, seed):
    mgs = make_mgs(DAY, N)
    with SeedScope(seed):
        solver = InstrumentedADMMSolver(
            mgs,
            rho_init=rho_init,
            max_iter=MAX_ITER,
            tol=TOL,
            adaptive=method.startswith(("adaptive", "sa")),
            network=build_network(mgs),
            economic_stop=economic_stop,
        )
        t0 = time.perf_counter()
        quiet(solver.solve)
        wall_s = time.perf_counter() - t0
    rec = {
        "method": method, "seed": seed,
        "stop_reason": solver.stop_reason,
        "iterations": len(solver.history["objective"]),
        "settled_cost": float(solver.final_system_cost),
        "wall_s": wall_s,
        "c_solve_hist": [float(x) for x in solver.history["solve_time_s"]],
        "stab": stab_stats(solver.mode_hist),
    }
    return rec


def save(results):
    OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")


def main():
    mgs = make_mgs(DAY, N)
    sigma = renewable_volatility(mgs)
    rho_prior = float(np.clip(0.01 * SIGMA_REF / max(sigma, 1e-9), 0.002, 0.05))

    results = {"config": {
        "day": DAY, "N": N, "max_iter": MAX_ITER, "tol": TOL,
        "tau_grid": TAU_GRID, "window": WINDOW, "k_min": K_MIN,
        "gamma_bar": GAMMA_BAR, "sigma": sigma, "rho_prior": rho_prior,
        "seeds": SEEDS, "w_stab": W_STAB,
        "note": "Seed=0 = Gurobi 默认 (论文 tab:sa 口径); "
                "settled_cost 为修复后全成本; stab=在线二值稳定检测器",
    }, "runs": []}

    if OUT_JSON.exists():
        try:
            old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            results["runs"] = old.get("runs", [])
            print(f"[resume] 已有 {len(results['runs'])} 条运行", flush=True)
        except Exception:
            pass

    done = {(r["method"], r["seed"]) for r in results["runs"]}

    cent = CentralizedSolver(mgs, network=build_network(mgs))
    cent_cost = float(quiet(cent.solve, verbose=False)["optimal_cost"])
    results["config"]["centralized_cost"] = cent_cost
    print(f"sigma={sigma:.4f} rho_prior={rho_prior:.4f} cent={cent_cost:.2f}",
          flush=True)

    plan = [("fixed_0.05", 0.05, None),
            ("fixed_0.1", 0.1, None),
            ("fixed_0.2", 0.2, None),
            ("adaptive_boyd", 0.01, None)]
    for tau in TAU_GRID:
        plan.append((f"sa_{tau}", rho_prior,
                     {"delta_tol": None, "window": WINDOW, "k_min": K_MIN,
                      "gamma_bar": GAMMA_BAR, "tau": tau}))

    for method, rho0, estop in plan:
        for seed in SEEDS:
            if (method, seed) in done:
                continue
            if estop is not None:
                boyd0 = next((r for r in results["runs"]
                              if r["method"] == "adaptive_boyd" and r["seed"] == 0), None)
                if boyd0 is None:
                    raise RuntimeError("需要先跑 adaptive_boyd seed=0")
                c_iter = float(np.mean(boyd0["c_solve_hist"]))
                estop = dict(estop, delta_tol=estop["tau"] * c_iter)
            t0 = time.time()
            r = run_one(method, rho0, estop, seed)
            r.update({"N": N, "day": DAY, "sigma": sigma,
                      "centralized_cost": cent_cost})
            results["runs"].append(r)
            save(results)
            st = r["stab"]
            print(f"[{method:<14} seed={seed}] iters={r['iterations']:<4} "
                  f"cost={r['settled_cost']:.2f} reason={r['stop_reason'] or '-':<8} "
                  f"stab_trigger={st['stab_trigger_iter']} "
                  f"last_flip={st['last_flip_iter']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print("saved ->", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
