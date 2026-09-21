""""""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from microgrid import DistributionNetwork


class CentralizedSolver:
    """"""
    def __init__(self, microgrids, network=None):
        self.microgrids = microgrids
        self.N = len(microgrids)
        self.T = microgrids[0].time_slots
        self.network = network if network is not None else DistributionNetwork(self.N)

        self.model = None
        self.solve_time = 0.0
        self.optimal_cost = None
        self.status = None

        self._results = None  # {mg_idx: dict}
        self._P_trade = None  # [N, N, T]
        self._line_loss = None  # [num_lines, T]
        self._loss_allocation = None  # [N, T]

    def solve(self, verbose=True):
        """"""
        N = self.N
        T = self.T
        mgs = self.microgrids
        network = self.network

        model = gp.Model("Centralized_P2P")
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("TimeLimit", 600)

        # ==================================================================
        # ==================================================================
        P_grid = {}
        P_export = {}
        P_ch = {}
        P_dis = {}
        P_curtail = {}
        SOC = {}
        z_bat = {}
        P_trade = {}
        P_dg = {}

        for i in range(N):
            mg = mgs[i]
            use_dg = mg.dg_cost is not None and mg.diesel_capacity > 0
            for t in range(T):
                P_grid[i, t] = model.addVar(lb=0, ub=mg.p_grid_max, name=f"P_grid_{i}_{t}")
                P_export[i, t] = model.addVar(lb=0, ub=mg.p_grid_max, name=f"P_export_{i}_{t}")
                P_ch[i, t] = model.addVar(
                    lb=0, ub=mg.battery_power, name=f"P_ch_{i}_{t}"
                )
                P_dis[i, t] = model.addVar(
                    lb=0, ub=mg.battery_power, name=f"P_dis_{i}_{t}"
                )
                P_curtail[i, t] = model.addVar(lb=0, name=f"P_curtail_{i}_{t}")
                z_bat[i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_bat_{i}_{t}")
                if use_dg:
                    P_dg[i, t] = model.addVar(
                        lb=0, ub=mg.diesel_capacity, name=f"P_dg_{i}_{t}"
                    )

            for t in range(T + 1):
                SOC[i, t] = model.addVar(
                    lb=mg.soc_lb[t] if hasattr(mg, "soc_lb") else 0.1 * mg.battery_capacity,
                    ub=mg.soc_ub[t] if hasattr(mg, "soc_ub") else 0.9 * mg.battery_capacity,
                    name=f"SOC_{i}_{t}",
                )

        for i in range(N):
            for j in range(N):
                if i != j:
                    mg = mgs[i]
                    trade_limit = max(mg.max_load, mg.pv_capacity + mg.wind_capacity)
                    for t in range(T):
                        P_trade[i, j, t] = model.addVar(
                            lb=-trade_limit,
                            ub=trade_limit,
                            name=f"P_trade_{i}_{j}_{t}",
                        )

        P_line_flow = {}
        P_line_abs = {}

        for line_idx, (from_node, to_node) in enumerate(network.lines):
            for t in range(T):
                P_line_flow[line_idx, t] = model.addVar(
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    name=f"P_flow_{line_idx}_{t}",
                )
                P_line_abs[line_idx, t] = model.addVar(
                    lb=0,
                    ub=GRB.INFINITY,
                    name=f"P_abs_{line_idx}_{t}",
                )
                # loss = (|P|/V)^2 * R / 1000

        model.update()

        # ==================================================================
        # ==================================================================

        for i in range(N):
            mg = mgs[i]
            for t in range(T):
                net_trade = gp.quicksum(P_trade[i, j, t] for j in range(N) if j != i)
                dg_gen = P_dg[i, t] if (mg.dg_cost is not None and mg.diesel_capacity > 0) else 0.0
                model.addConstr(
                    mg.load[t] + P_curtail[i, t] + P_export[i, t]
                    == mg.total_generation[t]
                    + P_dis[i, t]
                    - P_ch[i, t]
                    + P_grid[i, t]
                    + net_trade
                    + dg_gen,
                    name=f"power_balance_{i}_{t}",
                )

        for i in range(N):
            mg = mgs[i]
            for t in range(T):
                model.addConstr(
                    SOC[i, t + 1]
                    == SOC[i, t]
                    + P_ch[i, t] * mg.battery_efficiency
                    - P_dis[i, t] / mg.battery_efficiency,
                    name=f"soc_dynamics_{i}_{t}",
                )

        for i in range(N):
            mg = mgs[i]
            soc_target = 0.5 * mg.battery_capacity
            model.addConstr(SOC[i, 0] == soc_target, name=f"soc_init_{i}")
            model.addConstr(SOC[i, T] == soc_target, name=f"soc_final_{i}")

        for i in range(N):
            mg = mgs[i]
            for t in range(T):
                model.addConstr(
                    P_ch[i, t] <= z_bat[i, t] * mg.battery_power,
                    name=f"ch_mutex_{i}_{t}",
                )
                model.addConstr(
                    P_dis[i, t] <= (1 - z_bat[i, t]) * mg.battery_power,
                    name=f"dis_mutex_{i}_{t}",
                )

        for i in range(N):
            for j in range(i + 1, N):
                for t in range(T):
                    model.addConstr(
                        P_trade[i, j, t] + P_trade[j, i, t] == 0,
                        name=f"trade_balance_{i}_{j}_{t}",
                    )

        for i in range(N):
            for j in range(N):
                if i != j:
                    node_i = i + 1
                    node_j = j + 1
                    lo = min(node_i, node_j)
                    hi = max(node_i, node_j)
                    min_cap = float("inf")
                    for k in range(lo, hi):
                        cap = network.get_line_capacity(k, k + 1)
                        min_cap = min(min_cap, cap)
                    if min_cap < float("inf"):
                        for t in range(T):
                            model.addConstr(
                                P_trade[i, j, t] <= min_cap,
                                name=f"line_cap_pos_{i}_{j}_{t}",
                            )
                            model.addConstr(
                                P_trade[i, j, t] >= -min_cap,
                                name=f"line_cap_neg_{i}_{j}_{t}",
                            )

        for line_idx, (from_node, to_node) in enumerate(network.lines):
            for t in range(T):
                downstream_grid = gp.quicksum(
                    P_grid[mg_idx, t] for mg_idx in range(N) if (mg_idx + 1) > from_node
                )
                p2p_cross = gp.quicksum(
                    P_trade[i, j, t]
                    for i in range(N)
                    for j in range(N)
                    if i != j and (i + 1) <= from_node and (j + 1) > from_node
                )
                model.addConstr(
                    P_line_flow[line_idx, t] == downstream_grid + p2p_cross,
                    name=f"line_flow_def_{line_idx}_{t}",
                )
                model.addConstr(
                    P_line_abs[line_idx, t] >= P_line_flow[line_idx, t],
                    name=f"abs_pos_{line_idx}_{t}",
                )
                model.addConstr(
                    P_line_abs[line_idx, t] >= -P_line_flow[line_idx, t],
                    name=f"abs_neg_{line_idx}_{t}",
                )

        # ==================================================================
        # ==================================================================
        grid_cost_expr = gp.quicksum(
            mgs[i].grid_price[t] * P_grid[i, t] for i in range(N) for t in range(T)
        )
        fit_revenue_expr = gp.quicksum(
            mgs[i].fit_price[t] * P_export[i, t] for i in range(N) for t in range(T)
        )
        dg_cost_expr = gp.quicksum(
            mgs[i].dg_cost * P_dg[i, t]
            for i in range(N)
            for t in range(T)
            if mgs[i].dg_cost is not None and mgs[i].diesel_capacity > 0
        )
        wear_cost_expr = gp.quicksum(
            mgs[i].battery_cost * P_dis[i, t] for i in range(N) for t in range(T)
        )

        V = 10.0  # kV
        loss_cost_expr = gp.QuadExpr()
        for line_idx, (from_node, to_node) in enumerate(network.lines):
            R, _ = network.line_impedance[(from_node, to_node)]
            coeff = R / (V**2 * 1000.0)
            for t in range(T):
                price_t = mgs[0].grid_price[t]
                # loss_cost += price * coeff * P_abs^2
                loss_cost_expr += (
                    price_t * coeff * P_line_abs[line_idx, t] * P_line_abs[line_idx, t]
                )

        model.setObjective(
            grid_cost_expr - fit_revenue_expr + dg_cost_expr + wear_cost_expr + loss_cost_expr,
            GRB.MINIMIZE,
        )

        # ==================================================================
        # ==================================================================
        self.model = model

        t_start = time.time()
        model.optimize()
        self.solve_time = time.time() - t_start

        self.status = model.status

        if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            self.optimal_cost = model.objVal
            self._extract_results(
                P_grid, P_ch, P_dis, SOC, P_trade, P_line_flow, P_line_abs
            )

            if verbose:
                print(f"\n集中式优化完成:")
                print(f"  状态: {'最优' if model.status == GRB.OPTIMAL else '次优'}")
                print(f"  最优成本: {self.optimal_cost:.2f} 元")
                print(f"  求解时间: {self.solve_time:.2f} 秒")
                if hasattr(model, "MIPGap"):
                    print(f"  MIP Gap: {model.MIPGap:.6f}")

            return {
                "optimal_cost": self.optimal_cost,
                "solve_time": self.solve_time,
                "gap": model.MIPGap if hasattr(model, "MIPGap") else 0.0,
                "status": model.status,
            }
        else:
            print(f"集中式优化失败，状态码: {model.status}")
            if model.status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write("centralized_infeasible.ilp")
                print("  IIS 已保存至 centralized_infeasible.ilp")
            return {
                "optimal_cost": float("inf"),
                "solve_time": self.solve_time,
                "gap": float("inf"),
                "status": model.status,
            }

    def _extract_results(
        self, P_grid, P_ch, P_dis, SOC, P_trade, P_line_flow, P_line_abs
    ):
        """"""
        N = self.N
        T = self.T
        network = self.network
        V = 10.0

        self._results = {}
        for i in range(N):
            self._results[i] = {
                "P_grid": np.array([P_grid[i, t].X for t in range(T)]),
                "P_ch": np.array([P_ch[i, t].X for t in range(T)]),
                "P_dis": np.array([P_dis[i, t].X for t in range(T)]),
                "SOC": np.array([SOC[i, t].X for t in range(T)]),
            }

        self._P_trade = np.zeros((N, N, T))
        for i in range(N):
            for j in range(N):
                if i != j:
                    for t in range(T):
                        self._P_trade[i, j, t] = P_trade[i, j, t].X

        num_lines = len(network.lines)
        self._line_loss = np.zeros((num_lines, T))
        for line_idx, (from_node, to_node) in enumerate(network.lines):
            R, _ = network.line_impedance[(from_node, to_node)]
            for t in range(T):
                p_abs = P_line_abs[line_idx, t].X
                loss = (p_abs / V) ** 2 * R / 1000.0
                self._line_loss[line_idx, t] = loss

        self._loss_allocation = np.zeros((N, T))
        for line_idx, (from_node, to_node) in enumerate(network.lines):
            for t in range(T):
                loss = self._line_loss[line_idx, t]
                if loss < 1e-9:
                    continue

                mg_contributions = np.zeros(N)
                for mg_idx in range(N):
                    mg_node = mg_idx + 1
                    contrib = 0.0
                    if mg_node > from_node:
                        contrib += self._results[mg_idx]["P_grid"][t]
                    for j in range(N):
                        if j == mg_idx:
                            continue
                        node_j = j + 1
                        if mg_node <= from_node and node_j > from_node:
                            contrib += abs(self._P_trade[mg_idx, j, t])
                        elif mg_node > from_node and node_j <= from_node:
                            contrib += abs(self._P_trade[mg_idx, j, t])
                    mg_contributions[mg_idx] = abs(contrib)

                total_contrib = np.sum(mg_contributions)
                if total_contrib > 1e-6:
                    for mg_idx in range(N):
                        share = mg_contributions[mg_idx] / total_contrib
                        self._loss_allocation[mg_idx, t] += loss * share

    def get_local_result(self, mg_idx):
        """"""
        if self._results is None or self._P_trade is None:
            raise RuntimeError("请先调用 solve()")
        return self._results[mg_idx]

    def get_system_grid_cost(self):
        """"""
        if self._results is None or self._loss_allocation is None:
            raise RuntimeError("请先调用 solve()")

        total = 0.0
        for i in range(self.N):
            mg = self.microgrids[i]
            total += np.sum(mg.grid_price * self._results[i]["P_grid"])
        for i in range(self.N):
            mg = self.microgrids[i]
            total += np.sum(mg.grid_price * self._loss_allocation[i, :])
        return total

    def get_mg_cost_breakdown(self, mg_idx):
        """"""
        if (
            self._results is None
            or self._P_trade is None
            or self._loss_allocation is None
        ):
            raise RuntimeError("请先调用 solve()")

        mg = self.microgrids[mg_idx]
        result = self._results[mg_idx]

        grid_cost = np.sum(mg.grid_price * result["P_grid"])
        loss_cost = np.sum(mg.grid_price * self._loss_allocation[mg_idx, :])

        from admm_solver import ADMMSolver

        temp_solver = ADMMSolver.__new__(ADMMSolver)
        temp_solver.N = self.N
        temp_solver.T = self.T
        temp_solver.microgrids = self.microgrids

        type_factor = {}
        for idx in range(self.N):
            mg_type = self.microgrids[idx].mg_type
            if mg_type == "industrial":
                type_factor[idx] = 0.55
            elif mg_type == "commercial":
                type_factor[idx] = 0.65
            elif mg_type == "residential":
                type_factor[idx] = 0.70
            else:
                type_factor[idx] = 0.65

        p2p_buy_cost = 0.0
        p2p_sell_revenue = 0.0
        for j in range(self.N):
            if j != mg_idx:
                for t in range(self.T):
                    avg_factor = (type_factor[mg_idx] + type_factor[j]) / 2.0
                    price = mg.grid_price[t] * avg_factor
                    trade = self._P_trade[mg_idx, j, t]
                    cost = trade * price
                    if cost > 0:
                        p2p_buy_cost += cost
                    else:
                        p2p_sell_revenue += abs(cost)

        p2p_net_cost = p2p_buy_cost - p2p_sell_revenue
        total_cost = grid_cost + loss_cost + p2p_net_cost

        return {
            "grid_cost": grid_cost,
            "loss_cost": loss_cost,
            "p2p_buy_cost": p2p_buy_cost,
            "p2p_sell_revenue": p2p_sell_revenue,
            "p2p_net_cost": p2p_net_cost,
            "total_cost": total_cost,
        }

    def get_total_line_loss(self):
        """"""
        if self._line_loss is None:
            raise RuntimeError("请先调用 solve()")
        return float(np.sum(self._line_loss))

    def _get_total_line_loss_impl(self):
        """"""
        if self._line_loss is None:
            raise RuntimeError("请先调用 solve()")
        return np.sum(self._line_loss)

    def get_total_trade_volume(self):
        """"""
        if self._P_trade is None:
            raise RuntimeError("请先调用 solve()")
        P_trade = self._P_trade
        total = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                total += float(np.sum(np.abs(P_trade[i, j, :])))
        return total


def compare_centralized_vs_admm(microgrids, network=None, verbose=True):
    """"""
    from admm_solver import ADMMSolver

    N = len(microgrids)
    if network is None:
        network = DistributionNetwork(N)

    if verbose:
        print("=" * 70)
        print("  集中式全局优化 (Centralized)")
        print("=" * 70)
    cent = CentralizedSolver(microgrids, network)
    cent.solve(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("  ADMM 分布式优化 (Distributed)")
        print("=" * 70)

    admm = ADMMSolver(
        microgrids,
        rho_init=0.1,
        max_iter=500,
        tol=1e-3,
        adaptive=True,
        network=network,
    )
    t0 = time.time()
    admm.solve()
    admm_time = time.time() - t0

    admm_cost = admm.get_system_grid_cost()
    cent_cost = cent.get_system_grid_cost()

    if cent_cost > 1e-6:
        optimality_gap = (admm_cost - cent_cost) / cent_cost * 100.0
    else:
        optimality_gap = 0.0

    cost_comparison = {}
    for i in range(N):
        mg = microgrids[i]
        cent_bd = cent.get_mg_cost_breakdown(i)
        admm_bd = admm.get_mg_cost_breakdown(i)
        cost_comparison[i] = {
            "mg_type": mg.mg_type,
            "centralized_grid_cost": cent_bd["grid_cost"],
            "admm_grid_cost": admm_bd["grid_cost"],
            "centralized_total": cent_bd["total_cost"],
            "admm_total": admm_bd["total_cost"],
        }

    if verbose:
        print("\n" + "=" * 70)
        print("  集中式 vs ADMM 对比结果")
        print("=" * 70)
        print(
            f"  集中式系统成本:  {cent_cost:>10.2f} 元  (求解时间: {cent.solve_time:.2f}s)"
        )
        print(f"  ADMM 系统成本:   {admm_cost:>10.2f} 元  (求解时间: {admm_time:.2f}s)")
        print(f"  最优间隙:        {optimality_gap:>10.4f} %")
        print(f"  时间比:          {admm_time / max(cent.solve_time, 1e-6):>10.2f} x")
        print(f"  集中式总网损:    {cent.get_total_line_loss():>10.4f} kWh")
        print(f"  ADMM 总网损:     {np.sum(admm.line_loss):>10.4f} kWh")
        print()
        print(
            f"  {'微网':<15} {'集中式(¥)':>12} {'ADMM(¥)':>12} {'差异(¥)':>12} {'差异(%)':>10}"
        )
        print("  " + "-" * 63)
        for i in range(N):
            c = cost_comparison[i]
            diff = c["admm_grid_cost"] - c["centralized_grid_cost"]
            pct = diff / max(c["centralized_grid_cost"], 1e-6) * 100
            print(
                f"  MG{i + 1} ({c['mg_type']:<10}) "
                f"{c['centralized_grid_cost']:>12.2f} "
                f"{c['admm_grid_cost']:>12.2f} "
                f"{diff:>12.2f} "
                f"{pct:>9.2f}%"
            )

    return {
        "centralized": cent,
        "admm": admm,
        "centralized_cost": cent_cost,
        "admm_cost": admm_cost,
        "optimality_gap": optimality_gap,
        "centralized_time": cent.solve_time,
        "admm_time": admm_time,
        "time_ratio": admm_time / max(cent.solve_time, 1e-6),
        "cost_comparison": cost_comparison,
    }


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    from microgrid import Microgrid

    mg1 = Microgrid(1, "industrial")
    mg2 = Microgrid(2, "commercial")
    mg3 = Microgrid(3, "residential")
    microgrids = [mg1, mg2, mg3]

    results = compare_centralized_vs_admm(microgrids)

    print("\n最终结论:")
    gap = results["optimality_gap"]
    if gap < 1.0:
        print(f"  ADMM 与集中式最优间隙仅 {gap:.4f}%，几乎无损。")
    elif gap < 5.0:
        print(f"  ADMM 与集中式最优间隙 {gap:.2f}%，在可接受范围内。")
    else:
        print(f"  ADMM 与集中式最优间隙 {gap:.2f}%，需要调参优化。")
