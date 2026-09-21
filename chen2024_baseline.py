""""""
import io
import json
import sys
import time

from pathlib import Path

import numpy as np

from admm_solver import ADMMSolver
from microgrid import DistributionNetwork, Microgrid


class ACADMMChenSolver(ADMMSolver):
    """"""
    def __init__(
        self,
        microgrids,
        rho_init=0.1,
        max_iter=300,
        tol=1e-3,
        reltol=1e-3,
        network=None,
        freeze_rho_iter=20,
        mu=10.0,
        tau=2.0,
        rho_bounds=(1e-4, 1e4),
        chen_eps=1e-3,
    ):
        super().__init__(
            microgrids,
            rho_init=rho_init,
            max_iter=max_iter,
            tol=tol,
            reltol=reltol,
            adaptive=False,
            network=network,
        )
        self.rho_vec = np.full(self.N, float(rho_init))
        self.freeze_rho_iter = freeze_rho_iter
        self.mu = float(mu)
        self.tau = float(tau)
        self.rho_bounds = rho_bounds
        self.chen_eps = float(chen_eps)

        self.r_agent = np.zeros(self.N)
        self.s_agent = np.zeros(self.N)
        self.chen_converged = False
        self.rho_frozen_value = None
        self.rho_frozen_at = None

        self.history["rho_vec"] = []
        self.history["r_agent"] = []
        self.history["s_agent"] = []

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_global_variables(self):
        self.P_global_old = self.P_global.copy()
        for i in range(self.N):
            for j in range(i + 1, self.N):
                avg = 0.5 * (
                    self.P_local[i, j, :]
                    - self.P_local[j, i, :]
                    + self.lambda_dual[i, j, :] / self.rho_vec[i]
                    - self.lambda_dual[j, i, :] / self.rho_vec[j]
                )
                self.P_global[i, j, :] = avg
                self.P_global[j, i, :] = -avg

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_dual_variables(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.lambda_dual[i, j, :] += self.rho_vec[i] * (
                        self.P_local[i, j, :] - self.P_global[i, j, :]
                    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def compute_residuals(self):
        for i in range(self.N):
            self.r_agent[i] = np.linalg.norm(
                self.P_local[i, :, :] - self.P_global[i, :, :]
            )
        dz = self.P_global - self.P_global_old
        for i in range(self.N):
            self.s_agent[i] = self.rho_vec[i] * np.linalg.norm(dz[i, :, :])

        # primal = ||P_local - P_global||_F;
        primal_res = np.linalg.norm(self.P_local - self.P_global)
        dual_res = np.linalg.norm(self.rho_vec[:, None, None] * dz)
        return primal_res, dual_res

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def adapt_rho_chen(self, k):
        lo, hi = self.rho_bounds
        if self.freeze_rho_iter is not None and k >= self.freeze_rho_iter:
            if self.rho_frozen_value is None:
                self.rho_frozen_value = float(np.mean(self.rho_vec))
                self.rho_frozen_at = k
                self.rho_vec[:] = self.rho_frozen_value
            return
        for i in range(self.N):
            if self.r_agent[i] > self.mu * self.s_agent[i]:
                self.rho_vec[i] = min(self.rho_vec[i] * self.tau, hi)
            elif self.s_agent[i] > self.mu * self.r_agent[i]:
                self.rho_vec[i] = max(self.rho_vec[i] / self.tau, lo)

    def chen_stopping_test(self):
        """"""
        return (
            np.max(self.r_agent) <= self.chen_eps
            and np.max(self.s_agent) <= self.chen_eps
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def solve(self):
        print("开始 AC-ADMM (Chen 2024) 迭代...")
        print(
            f"{'迭代':<6} {'||r||':<14} {'||s||':<14} {'rho(均值)':<10} "
            f"{'系统成本':<12} {'总网损(kWh)':<15}"
        )
        print("-" * 85)

        converged = False
        chen_converged = False
        primal_res = float("inf")
        dual_res = float("inf")
        total_system_cost = 0.0
        k = 0
        wall_start = time.perf_counter()

        for k in range(self.max_iter):
            results = []
            for i in range(self.N):
                self.rho = self.rho_vec[i]
                results.append(self.solve_local_problem(i, update_state=True))
            self._cached_local_results = {i: results[i] for i in range(self.N)}
            cached = {i: results[i] for i in range(self.N)}

            total_grid_cost = 0.0
            for i, result in enumerate(results):
                mg = self.microgrids[i]
                total_grid_cost += np.sum(mg.grid_price * result["P_grid"])

            self.update_global_variables()

            self.update_dual_variables()

            primal_res, dual_res = self.compute_residuals()

            self._compute_line_losses(cached_results=cached)
            total_loss = np.sum(self.line_loss)
            loss_cost = 0.0
            for i in range(self.N):
                mg = self.microgrids[i]
                loss_cost += np.sum(mg.grid_price * self.loss_allocation[i, :])
            total_system_cost = total_grid_cost + loss_cost

            self.history["primal_residual"].append(primal_res)
            self.history["dual_residual"].append(dual_res)
            self.history["rho"].append(float(np.mean(self.rho_vec)))
            self.history["rho_vec"].append(self.rho_vec.copy())
            self.history["r_agent"].append(self.r_agent.copy())
            self.history["s_agent"].append(self.s_agent.copy())
            self.history["objective"].append(total_system_cost)
            self.history["total_loss"].append(total_loss)
            self.history["lambda_dual"].append(np.mean(np.abs(self.lambda_dual)))

            if k % 10 == 0 or k < 5:
                print(
                    f"{k:<6} {primal_res:<14.6f} {dual_res:<14.6f} "
                    f"{np.mean(self.rho_vec):<10.4f} {total_system_cost:<12.2f} "
                    f"{total_loss:<15.4f}"
                )

            if self.check_convergence(primal_res, dual_res):
                converged = True
            if self.chen_stopping_test():
                chen_converged = True
            if converged or chen_converged:
                print(
                    f"\n停机于第 {k} 轮: Boyd 判据={converged}, "
                    f"Chen(B.7) 判据={chen_converged}"
                )
                print(f"  最终系统成本: {total_system_cost:.2f} 元")
                break

            self.adapt_rho_chen(k)

        if not converged and not chen_converged:
            print(f"\n达到最大迭代次数 {self.max_iter} 未停机")
            print(f"  最终 ||r||: {primal_res:.6f}, ||s||: {dual_res:.6f}")
            print(f"  最终系统成本: {total_system_cost:.2f} 元")
        self.rho = float(np.mean(self.rho_vec))
        self.final_results = self.get_results()
        self.converged = converged
        self.chen_converged = chen_converged
        self.final_iteration = k
        self.final_primal_residual = primal_res
        self.final_dual_residual = dual_res
        self.final_system_cost = total_system_cost
        self.solve_time = time.perf_counter() - wall_start

        print(f"  rho 终值: {self.rho_vec.tolist()} "
              f"(冻结于第 {self.rho_frozen_at} 轮, 值 {self.rho_frozen_value})")
        print(f"  总耗时: {self.solve_time:.2f} s")
        return self.final_results


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _quiet_solve(solver):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = solver.solve()
    finally:
        sys.stdout = old
    return result


def _quiet_solve_wrapper(cs):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        info = cs.solve(verbose=False)
    finally:
        sys.stdout = old
    return info


def run_ac_admm(microgrids, network, freeze_rho_iter=20, rho_init=0.1,
                max_iter=300, tol=1e-3):
    solver = ACADMMChenSolver(
        microgrids,
        rho_init=rho_init,
        max_iter=max_iter,
        tol=tol,
        network=network,
        freeze_rho_iter=freeze_rho_iter,
    )
    _quiet_solve(solver)
    return {
        "name": "AC-ADMM (Chen 2024)",
        "converged": bool(solver.converged),
        "chen_converged": bool(solver.chen_converged),
        "iterations": solver.final_iteration + 1,
        "time_s": float(solver.solve_time),
        "cost": float(solver.final_system_cost),
        "primal_res": float(solver.final_primal_residual),
        "dual_res": float(solver.final_dual_residual),
        "rho_final": solver.rho_vec.tolist(),
        "rho_frozen_at": solver.rho_frozen_at,
        "rho_frozen_value": solver.rho_frozen_value,
        "_solver": solver,
    }


def run_problem(label, microgrids, network, freeze_rho_iter=20):
    """"""
    from centralized_solver import CentralizedSolver

    print(f"\n{'=' * 70}\n问题: {label}  (N={len(microgrids)})\n{'=' * 70}")
    out = {"problem": label, "N": len(microgrids)}

    print("-- 集中式基准 --")
    cs = CentralizedSolver(microgrids, network=network)
    cinfo = _quiet_solve_wrapper(cs)
    out["centralized"] = {
        "cost": float(cinfo["optimal_cost"]),
        "time_s": float(cinfo["solve_time"]),
        "mip_gap": float(cinfo.get("gap", 0.0) or 0.0),
    }
    print(f"  cost={cinfo['optimal_cost']:.2f}, time={cinfo['solve_time']:.2f}s")

    for key, adaptive in (("ours_adaptive", True), ("ours_fixed", False)):
        print(f"-- {key} --")
        solver = ADMMSolver(
            microgrids,
            rho_init=0.1,
            max_iter=300,
            tol=1e-3,
            adaptive=adaptive,
            network=network,
        )
        t0 = time.perf_counter()
        _quiet_solve(solver)
        dt = time.perf_counter() - t0
        out[key] = {
            "converged": bool(solver.converged),
            "iterations": int(solver.final_iteration + 1),
            "time_s": dt,
            "cost": float(solver.final_system_cost),
            "primal_res": float(solver.final_primal_residual),
            "dual_res": float(solver.final_dual_residual),
            "rho_final": float(solver.rho),
        }
        print(
            f"  conv={solver.converged}, iters={out[key]['iterations']}, "
            f"time={dt:.2f}s, cost={out[key]['cost']:.2f}"
        )

    print("-- AC-ADMM (Chen 2024) --")
    out["ac_admm"] = {
        k: v for k, v in run_ac_admm(
            microgrids, network, freeze_rho_iter=freeze_rho_iter
        ).items()
        if k != "_solver"
    }
    print(
        f"  conv={out['ac_admm']['converged']}, "
        f"chen_conv={out['ac_admm']['chen_converged']}, "
        f"iters={out['ac_admm']['iterations']}, "
        f"time={out['ac_admm']['time_s']:.2f}s, "
        f"cost={out['ac_admm']['cost']:.2f}"
    )

    # gap to centralized
    c_cost = out["centralized"]["cost"]
    for key in ("ours_adaptive", "ours_fixed", "ac_admm"):
        out[key]["gap_pct"] = 100.0 * (out[key]["cost"] - c_cost) / c_cost

    return out


def main():
    results = {}

    mgs_a = [
        Microgrid(1, "industrial"),
        Microgrid(2, "commercial"),
        Microgrid(3, "residential"),
    ]
    net_a = DistributionNetwork(len(mgs_a))
    results["synthetic_3mg"] = run_problem(
        "合成 3 微网 (industrial/commercial/residential)", mgs_a, net_a
    )

    from load_real_data import load_experiment_microgrids

    raw = load_experiment_microgrids()
    mgs_b = [
        Microgrid(item["mg_id"], None, real_data=item["data"],
                  soc_bounds=item["soc_bounds"])
        for item in raw[:3]
    ]
    net_b = DistributionNetwork(len(mgs_b))
    results["real_N3"] = run_problem(
        "真实数据规范排序 N=3 (MG1/MG2/MG19, day=15)", mgs_b, net_b
    )

    out_path = Path(__file__).resolve().parent / "chen2024_baseline_results.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] JSON 结果已写入 {out_path}")
    return results


if __name__ == "__main__":
    main()
