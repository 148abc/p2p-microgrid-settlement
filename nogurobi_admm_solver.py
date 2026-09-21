"""Gurobi-free ADMM backend.

This backend keeps the consensus, dual-update, line-loss, settlement, and
history logic from :class:`ADMMSolver`, but solves each local subproblem with
CVXPY/OSQP (or CLARABEL as a fallback).  The local problem is a convex QP, so
the binary charge/discharge mutex used by the Gurobi MILP is relaxed to
continuous charge and discharge variables.  A tiny throughput penalty makes
simultaneous charge/discharge unattractive; the resulting violation is
reported in each local result for validation.
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - depends on the active environment
    cp = None

from admm_solver import ADMMSolver


class NoGurobiADMMSolver(ADMMSolver):
    """ADMM solver using CVXPY instead of Gurobi for local QPs."""

    backend_name = "cvxpy-relaxed"

    def __init__(self, *args, cvxpy_solver="OSQP", **kwargs):
        if cp is None:
            raise ImportError(
                "NoGurobiADMMSolver requires cvxpy. Install cvxpy with OSQP "
                "or use the project's power_market environment."
            )
        super().__init__(*args, **kwargs)
        self.cvxpy_solver = cvxpy_solver
        self.max_mutex_violation = 0.0

    def _solve_problem(self, problem):
        """Solve a convex QP with a non-Gurobi solver and a safe fallback."""
        if self.cvxpy_solver == "OSQP":
            try:
                problem.solve(
                    solver=cp.OSQP,
                    verbose=False,
                    warm_start=False,
                    max_iter=200_000,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    polish=True,
                )
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    return
            except cp.error.SolverError:
                pass

        problem.solve(solver=cp.CLARABEL, verbose=False)

    def solve_local_problem(self, mg_idx, update_state=True, fix_trades=False):
        """"""
        mg = self.microgrids[mg_idx]
        i = mg_idx
        T = self.T

        p_grid = cp.Variable(T, nonneg=True, name=f"p_grid_{i}")
        p_export = cp.Variable(T, nonneg=True, name=f"p_export_{i}")
        p_ch = cp.Variable(T, nonneg=True, name=f"p_ch_{i}")
        p_dis = cp.Variable(T, nonneg=True, name=f"p_dis_{i}")
        p_curtail = cp.Variable(T, nonneg=True, name=f"p_curtail_{i}")
        soc = cp.Variable(T + 1, name=f"soc_{i}")

        use_dg = mg.dg_cost is not None and mg.diesel_capacity > 0
        p_dg = (
            cp.Variable(T, nonneg=True, name=f"p_dg_{i}") if use_dg else None
        )

        trade_limit = max(mg.max_load, mg.pv_capacity + mg.wind_capacity)
        if fix_trades:
            p_trade = {
                j: cp.Constant(np.asarray(self.P_global[i, j, :], dtype=float))
                for j in range(self.N)
                if j != i
            }
        else:
            p_trade = {
                j: cp.Variable(T, name=f"p_{i}_{j}")
                for j in range(self.N)
                if j != i
            }

        objective = cp.sum(cp.multiply(mg.grid_price, p_grid))
        objective -= cp.sum(cp.multiply(mg.fit_price, p_export))
        for j, trade in p_trade.items():
            consensus = self.P_global[i, j, :]
            objective += cp.sum(cp.multiply(self.p2p_price[i, j, :], trade))
            if fix_trades:
                continue
            objective += cp.sum(cp.multiply(self.lambda_dual[i, j, :], trade - consensus))
            objective += (self.rho / 2.0) * cp.sum_squares(trade - consensus)

        # Small regularizer to avoid an arbitrary simultaneous charge/discharge
        # solution in the continuous relaxation.
        objective += 1e-7 * cp.sum(p_ch + p_dis)

        if use_dg:
            objective += mg.dg_cost * cp.sum(p_dg)
        objective += mg.battery_cost * cp.sum(p_dis)

        soc_target = 0.5 * (mg.soc_lb[0] + mg.soc_ub[0])
        constraints = [
            p_grid <= mg.p_grid_max,
            p_export <= mg.p_grid_max,
            p_ch <= mg.battery_power,
            p_dis <= mg.battery_power,
            p_curtail <= mg.total_generation,
            soc >= mg.soc_lb,
            soc <= mg.soc_ub,
            soc[0] == soc_target,
            soc[T] == soc_target,
        ]
        if use_dg:
            constraints.append(p_dg <= mg.diesel_capacity)

        for t in range(T):
            net_trade = sum(
                (trade[t] for j, trade in p_trade.items()), cp.Constant(0.0)
            )
            dg_gen = p_dg[t] if use_dg else 0.0
            constraints.append(
                mg.load[t] + p_curtail[t] + self.loss_allocation[i, t] + p_export[t]
                == mg.total_generation[t]
                + p_dis[t]
                - p_ch[t]
                + p_grid[t]
                + net_trade
                + dg_gen
            )
            constraints.append(
                soc[t + 1]
                == soc[t]
                + p_ch[t] * mg.battery_efficiency
                - p_dis[t] / mg.battery_efficiency
            )

        for j, trade in p_trade.items():
            if not fix_trades:
                constraints.extend([trade <= trade_limit, trade >= -trade_limit])
            cap = self._get_trade_capacity(i, j)
            if not np.isfinite(cap):
                continue
            if fix_trades:
                frozen = np.asarray(self.P_global[i, j, :], dtype=float)
                if float(np.max(np.abs(frozen))) > cap:
                    raise RuntimeError(
                        f"CVXPY 修复轮交易越界: MG{i}-MG{j} "
                        f"|P|max={np.max(np.abs(frozen)):.6g} > cap={cap:.6g}"
                    )
                continue
            constraints.extend([trade <= cap, trade >= -cap])

        problem = cp.Problem(cp.Minimize(objective), constraints)
        self._solve_problem(problem)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(
                f"CVXPY local optimization failed for microgrid {i}: "
                f"status={problem.status}"
            )

        p_grid_value = np.asarray(p_grid.value, dtype=float).reshape(T)
        p_ch_value = np.asarray(p_ch.value, dtype=float).reshape(T)
        p_dis_value = np.asarray(p_dis.value, dtype=float).reshape(T)
        soc_value = np.asarray(soc.value, dtype=float).reshape(T + 1)[:-1]
        mutex_violation = float(np.max(np.minimum(p_ch_value, p_dis_value)))
        self.max_mutex_violation = max(self.max_mutex_violation, mutex_violation)

        trade_arr = np.zeros((self.N, T), dtype=float)
        for j, trade in p_trade.items():
            trade_arr[j, :] = np.asarray(trade.value, dtype=float).reshape(T)

        result = {
            "P_grid": p_grid_value,
            "P_export": np.asarray(p_export.value, dtype=float).reshape(T),
            "P_ch": p_ch_value,
            "P_dis": p_dis_value,
            "SOC": soc_value,
            "P_trade_local": trade_arr,
            "P_curtail": np.asarray(p_curtail.value, dtype=float).reshape(T),
            "P_dg": (
                np.asarray(p_dg.value, dtype=float).reshape(T)
                if use_dg
                else np.zeros(T)
            ),
            "cost": float(problem.value),
            "simultaneous_charge_discharge_max": mutex_violation,
        }

        if update_state:
            for j in range(self.N):
                if j != i:
                    self.P_local[i, j, :] = trade_arr[j, :]

        return result


if __name__ == "__main__":
    from microgrid import DistributionNetwork, Microgrid

    microgrids = [
        Microgrid(1, "industrial"),
        Microgrid(2, "commercial"),
        Microgrid(3, "residential"),
    ]
    solver = NoGurobiADMMSolver(
        microgrids,
        rho_init=0.1,
        max_iter=300,
        tol=1e-3,
        adaptive=True,
        network=DistributionNetwork(3),
    )
    solver.solve()
    print(
        f"No-Gurobi backend={solver.backend_name}, converged={solver.converged}, "
        f"iterations={len(solver.history['primal_residual'])}, "
        f"cost={solver.get_system_grid_cost():.4f}, "
        f"max_mutex_violation={solver.max_mutex_violation:.6g}"
    )
