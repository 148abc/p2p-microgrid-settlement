import json
import time

from pathlib import Path

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised by the no-Gurobi backend
    gp = None
    GRB = None

from microgrid import DistributionNetwork


class ADMMSolver:
    """"""
    def __init__(
        self,
        microgrids,
        rho_init=10.0,
        max_iter=300,
        tol=1e-3,
        adaptive=True,
        network=None,
        reltol=1e-3,
        economic_stop=None,
        rho_floor=0.01,
        rho_ceil=50.0,
    ):
        self.microgrids = microgrids
        self.N = len(microgrids)
        self.T = microgrids[0].time_slots
        self.rho = rho_init
        self.max_iter = max_iter
        self.tol = tol
        self.reltol = reltol
        self.rho_floor = rho_floor
        self.rho_ceil = rho_ceil
        self.adaptive = adaptive
        self.rho_update_iter = 0

        self.economic_stop = dict(economic_stop) if economic_stop else None
        self.stop_reason = None  # None | "boyd" | "economic" | "economic_repair_failed"
        self.repair_stats = None  # {"repair_cost", "repair_time_s"}
        self.economic_certified = None

        self.network = network if network is not None else DistributionNetwork(self.N)

        self.P_local = np.zeros((self.N, self.N, self.T))
        self.P_global = np.zeros((self.N, self.N, self.T))
        self.P_global_old = np.zeros((self.N, self.N, self.T))
        self.lambda_dual = np.zeros((self.N, self.N, self.T))

        self.p2p_price = self._initialize_p2p_prices()

        self.line_loss = np.zeros((len(self.network.lines), self.T))

        self.loss_allocation = np.zeros((self.N, self.T))

        self.history = {
            "primal_residual": [],
            "dual_residual": [],
            "rho": [],
            "objective": [],
            "lambda_dual": [],
            "lambda_dual_full": [],
            "total_loss": [],
            "solve_time_s": [],
        }

        self.final_results = None
        self._cached_local_results = None  # {mg_idx: result_dict}

        self.converged = False
        self.final_iteration = 0
        self.final_primal_residual = float("inf")
        self.final_dual_residual = float("inf")
        self.final_system_cost = 0.0

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _initialize_p2p_prices(self):
        """"""
        p2p_price = np.zeros((self.N, self.N, self.T))

        grid_price = self.microgrids[0].grid_price

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

        for t in range(self.T):
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        avg_factor = (type_factor[i] + type_factor[j]) / 2.0
                        p2p_price[i, j, t] = grid_price[t] * avg_factor

        return p2p_price

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def solve_local_problem(self, mg_idx, update_state=True, fix_trades=False):
        """"""
        mg = self.microgrids[mg_idx]
        i = mg_idx

        model = gp.Model(f"MG_{i}")
        model.setParam("OutputFlag", 0)

        P_grid = model.addVars(self.T, lb=0, ub=mg.p_grid_max, name="P_grid")
        P_export = model.addVars(self.T, lb=0, ub=mg.p_grid_max, name="P_export")
        P_ch = model.addVars(self.T, lb=0, ub=mg.battery_power, name="P_ch")
        P_dis = model.addVars(self.T, lb=0, ub=mg.battery_power, name="P_dis")
        P_curtail = model.addVars(
            self.T, lb=0, ub=mg.total_generation.tolist(),
            name="P_curtail",
        )
        SOC = model.addVars(
            self.T + 1,
            lb=mg.soc_lb.tolist(),
            ub=mg.soc_ub.tolist(),
            name="SOC",
        )

        z_bat = model.addVars(self.T, vtype=GRB.BINARY, name="z_bat")

        use_dg = mg.dg_cost is not None and mg.diesel_capacity > 0
        if use_dg:
            P_dg = model.addVars(
                self.T, lb=0, ub=mg.diesel_capacity, name="P_dg"
            )

        trade_limit = max(mg.max_load, mg.pv_capacity + mg.wind_capacity)
        P_trade = {}
        for j in range(self.N):
            if j != i:
                if fix_trades:
                    fixed = self.P_global[i, j, :]
                    P_trade[j] = model.addVars(
                        self.T, lb=fixed.tolist(), ub=fixed.tolist(),
                        name=f"P_{i}{j}",
                    )
                else:
                    P_trade[j] = model.addVars(
                        self.T, lb=-trade_limit, ub=trade_limit, name=f"P_{i}{j}"
                    )

        obj = gp.quicksum(mg.grid_price[t] * P_grid[t] for t in range(self.T))
        obj -= gp.quicksum(mg.fit_price[t] * P_export[t] for t in range(self.T))

        if use_dg:
            obj += gp.quicksum(mg.dg_cost * P_dg[t] for t in range(self.T))
        obj += gp.quicksum(mg.battery_cost * P_dis[t] for t in range(self.T))


        for j in range(self.N):
            if j != i:
                for t in range(self.T):
                    obj += self.p2p_price[i, j, t] * P_trade[j][t]
                    if fix_trades:
                        continue
                    obj += self.lambda_dual[i, j, t] * (
                        P_trade[j][t] - self.P_global[i, j, t]
                    )
                    obj += (
                        (self.rho / 2.0)
                        * (P_trade[j][t] - self.P_global[i, j, t])
                        * (P_trade[j][t] - self.P_global[i, j, t])
                    )

        model.setObjective(obj, GRB.MINIMIZE)

        for t in range(self.T):
            net_trade = gp.quicksum(P_trade[j][t] for j in range(self.N) if j != i)
            dg_gen = P_dg[t] if use_dg else 0.0
            model.addConstr(
                mg.load[t] + P_curtail[t] + self.loss_allocation[i, t] + P_export[t]
                == mg.total_generation[t]
                + P_dis[t]
                - P_ch[t]
                + P_grid[t]
                + net_trade
                + dg_gen,
                name=f"power_balance_{t}",
            )

        for t in range(self.T):
            model.addConstr(
                SOC[t + 1]
                == SOC[t]
                + P_ch[t] * mg.battery_efficiency
                - P_dis[t] / mg.battery_efficiency,
                name=f"soc_dynamics_{t}",
            )

        soc_target = 0.5 * (mg.soc_lb[0] + mg.soc_ub[0])
        model.addConstr(SOC[0] == soc_target, name="soc_initial")
        model.addConstr(SOC[self.T] == soc_target, name="soc_final")

        for t in range(self.T):
            model.addConstr(
                P_ch[t] <= z_bat[t] * mg.battery_power, name=f"ch_mutex_{t}"
            )
            model.addConstr(
                P_dis[t] <= (1 - z_bat[t]) * mg.battery_power, name=f"dis_mutex_{t}"
            )

        for j in range(self.N):
            if j != i:
                cap = self._get_trade_capacity(i, j)
                if cap < float("inf"):
                    for t in range(self.T):
                        model.addConstr(
                            P_trade[j][t] <= cap, name=f"line_cap_pos_{i}_{j}_{t}"
                        )
                        model.addConstr(
                            P_trade[j][t] >= -cap, name=f"line_cap_neg_{i}_{j}_{t}"
                        )

        model.optimize()

        if model.status == GRB.OPTIMAL:
            result = {
                "P_grid": np.array([P_grid[t].X for t in range(self.T)]),
                "P_export": np.array([P_export[t].X for t in range(self.T)]),
                "P_ch": np.array([P_ch[t].X for t in range(self.T)]),
                "P_dis": np.array([P_dis[t].X for t in range(self.T)]),
                "SOC": np.array([SOC[t].X for t in range(self.T)]),
                "P_dg": (
                    np.array([P_dg[t].X for t in range(self.T)])
                    if use_dg
                    else np.zeros(self.T)
                ),
                "cost": model.objVal,
            }

            trade_arr = np.zeros((self.N, self.T))
            for j in range(self.N):
                if j != i:
                    trade_arr[j, :] = [P_trade[j][t].X for t in range(self.T)]

            if update_state:
                for j in range(self.N):
                    if j != i:
                        self.P_local[i, j, :] = trade_arr[j, :]

            result["P_trade_local"] = trade_arr
            return result

        elif model.status == GRB.INFEASIBLE:
            print(f"警告: 微网 {i} 的优化问题不可行")
            model.computeIIS()
            model.write(f"infeasible_mg{i}.ilp")
            raise RuntimeError(f"微网 {i} 优化不可行，IIS已保存")
        else:
            raise RuntimeError(f"微网 {i} 优化失败，状态: {model.status}")

    def _get_trade_capacity(self, i, j):
        """"""
        node_i = i + 1
        node_j = j + 1
        lo = min(node_i, node_j)
        hi = max(node_i, node_j)

        min_cap = float("inf")
        for k in range(lo, hi):
            cap = self.network.get_line_capacity(k, k + 1)
            min_cap = min(min_cap, cap)
        return min_cap

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_global_variables(self):
        """"""
        self.P_global_old = self.P_global.copy()

        for i in range(self.N):
            for j in range(i + 1, self.N):
                avg = (self.P_local[i, j, :] - self.P_local[j, i, :]) / 2.0
                self.P_global[i, j, :] = avg
                self.P_global[j, i, :] = -avg

    def update_dual_variables(self):
        """"""
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    residual = self.P_local[i, j, :] - self.P_global[i, j, :]
                    self.lambda_dual[i, j, :] += self.rho * residual

    def compute_residuals(self):
        """"""
        primal_res = np.linalg.norm(self.P_local - self.P_global)
        dual_res = self.rho * np.linalg.norm(self.P_global - self.P_global_old)
        return primal_res, dual_res

    def _system_cost_from_results(self, results):
        """"""
        total_grid_cost = 0.0
        total_dg_cost = 0.0
        total_wear_cost = 0.0
        total_fit_revenue = 0.0
        for i, result in enumerate(results):
            mg = self.microgrids[i]
            total_grid_cost += np.sum(mg.grid_price * result["P_grid"])
            total_dg_cost += (
                np.sum(mg.dg_cost * result["P_dg"]) if mg.dg_cost else 0.0
            )
            total_wear_cost += np.sum(mg.battery_cost * result["P_dis"])
            total_fit_revenue += np.sum(mg.fit_price * result["P_export"])
        loss_cost = 0.0
        for i in range(self.N):
            mg = self.microgrids[i]
            loss_cost += np.sum(mg.grid_price * self.loss_allocation[i, :])
        return {
            "total": float(
                total_grid_cost
                + loss_cost
                + total_dg_cost
                + total_wear_cost
                - total_fit_revenue
            ),
            "grid": float(total_grid_cost),
            "loss": float(loss_cost),
            "dg": float(total_dg_cost),
            "wear": float(total_wear_cost),
            "fit": float(total_fit_revenue),
        }

    def repair_finalize(self):
        """"""
        self.P_local = self.P_global.copy()
        results = []
        t0 = time.perf_counter()
        for i in range(self.N):
            try:
                results.append(
                    self.solve_local_problem(i, update_state=False, fix_trades=True)
                )
            except RuntimeError:
                return None, None
        stats = {
            "repair_cost": self._system_cost_from_results(results)["total"],
            "repair_time_s": time.perf_counter() - t0,
        }
        self._cached_local_results = {i: results[i] for i in range(self.N)}
        return results, stats

    def _economic_stop_due(self, k):
        """"""
        es = self.economic_stop
        window = int(es.get("window", 5))
        k_min = int(es.get("k_min", window + 5))
        gamma_bar = float(es.get("gamma_bar", 0.9))
        if k < max(k_min, window + 1):
            return False
        F = self.history["objective"]
        if "tau_rel" in es:
            delta_tol = float(es["tau_rel"]) * abs(F[k])
        else:
            delta_tol = float(es["delta_tol"])
        dF = [abs(F[j] - F[j - 1]) for j in range(k - window + 1, k + 1)]
        if max(dF) > delta_tol:
            return False
        ratios = [dF[i] / max(dF[i - 1], 1e-12) for i in range(1, len(dF))]
        gamma_hat = float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-12)))))
        if gamma_hat > gamma_bar:
            return False
        self.economic_certified = gamma_hat
        return True

    def check_convergence(self, primal_res, dual_res):
        """"""
        num_pairs = self.N * (self.N - 1)
        p = num_pairs * self.T
        sqrt_p = np.sqrt(p)

        eps_pri = sqrt_p * self.tol + self.reltol * max(
            np.linalg.norm(self.P_local), np.linalg.norm(self.P_global)
        )
        eps_dual = sqrt_p * self.tol + self.reltol * np.linalg.norm(self.lambda_dual)

        return primal_res <= eps_pri and dual_res <= eps_dual

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def adapt_rho(self, primal_res, dual_res, k, mu=10, tau=2):
        """"""
        old_rho = self.rho

        if k - self.rho_update_iter < 10:
            return

        if primal_res > mu * dual_res:
            self.rho = min(self.rho * tau, self.rho_ceil)
        elif dual_res > mu * primal_res:
            self.rho = max(self.rho / tau, self.rho_floor)

        if self.rho != old_rho:
            scale = self.rho / old_rho
            self.lambda_dual *= scale
            self.rho_update_iter = k

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _compute_line_losses(self, cached_results=None):
        """"""
        self.line_loss[:] = 0.0
        self.loss_allocation[:] = 0.0

        p_grids = []
        for i in range(self.N):
            if cached_results and i in cached_results:
                p_grids.append(cached_results[i]["P_grid"])
            elif self._cached_local_results and i in self._cached_local_results:
                p_grids.append(self._cached_local_results[i]["P_grid"])
            else:
                result = self.solve_local_problem(i, update_state=False)
                p_grids.append(result["P_grid"])

        for line_idx, (from_node, to_node) in enumerate(self.network.lines):
            for t in range(self.T):

                mg_contributions = np.zeros(self.N)

                for mg_idx in range(self.N):
                    mg_node = mg_idx + 1
                    contrib = 0.0

                    if mg_node > from_node:
                        contrib += p_grids[mg_idx][t]

                    for j in range(self.N):
                        if j == mg_idx:
                            continue
                        node_j = j + 1
                        if mg_node <= from_node and node_j > from_node:
                            contrib += abs(self.P_global[mg_idx, j, t])
                        elif mg_node > from_node and node_j <= from_node:
                            contrib += abs(self.P_global[mg_idx, j, t])

                    mg_contributions[mg_idx] = abs(contrib)

                downstream_grid = 0.0
                for mg_idx in range(self.N):
                    mg_node = mg_idx + 1
                    if mg_node > from_node:
                        downstream_grid += p_grids[mg_idx][t]

                p2p_cross = 0.0
                for i in range(self.N):
                    for j in range(self.N):
                        if i == j:
                            continue
                        node_i = i + 1
                        node_j = j + 1
                        if node_i <= from_node and node_j > from_node:
                            p2p_cross += self.P_global[i, j, t]

                net_flow = downstream_grid + p2p_cross

                loss = self.network.calculate_line_loss(
                    abs(net_flow), from_node, to_node
                )
                self.line_loss[line_idx, t] = loss

                total_contrib = np.sum(mg_contributions)
                if total_contrib > 1e-6 and loss > 1e-9:
                    for mg_idx in range(self.N):
                        share = mg_contributions[mg_idx] / total_contrib
                        self.loss_allocation[mg_idx, t] += loss * share

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _print_blockchain_submission(self, k, results, p_global_prev, lambda_prev, primal_res, dual_res):
        """"""
        print("\n" + "=" * 80)
        print(f"  区块链提交数据 — 迭代 {k}")
        print("=" * 80)

        for i in range(self.N):
            mg = self.microgrids[i]
            result = results[i]

            p_trade = []
            for j in range(self.N):
                if j != i:
                    p_trade.append(self.P_local[i, j, :].tolist())

            cost = result["cost"]

            pglobal_flat = p_global_prev[i, :, :].flatten().tolist()
            lambda_flat = lambda_prev[i, :, :].flatten().tolist()
            p2p_flat = self.p2p_price[i, :, :].flatten().tolist()
            grid_price_list = mg.grid_price.tolist()
            loss_alloc_flat = self.loss_allocation[i, :].tolist()

            line_caps = []
            for j in range(self.N):
                if j != i:
                    line_caps.append(self._get_trade_capacity(i, j))

            p_trade_flat = [v for row in p_trade for v in row]

            n_elements = (
                1  # iteration
                + 1  # rho
                + len(pglobal_flat)
                + len(lambda_flat)
                + len(p2p_flat)
                + len(grid_price_list)
                + len(loss_alloc_flat)
                + len(line_caps)
                + len(p_trade_flat)
                + 1  # costSubmit
            )

            print(f"\n--- 微网 MG{i+1} ({mg.mg_type}) 提交 ---")
            print(f"  [1] pTrade [{len(p_trade)}x{self.T}]:")
            trade_labels = [f"MG{j+1}" for j in range(self.N) if j != i]
            for idx, row in enumerate(p_trade):
                vals = "[" + ", ".join(f"{v:8.2f}" for v in row) + "]"
                print(f"      与{trade_labels[idx]}: {vals}")

            print(f"  [2] cost = {cost:.4f} $")

            print(f"  [3] publicInputs ({n_elements} 个域元素):")
            print(f"      iteration     = {k}")
            print(f"      rho           = {self.rho:.4f}")
            print(f"      pglobalPrev   = [{len(pglobal_flat)}个] "
                  f"前5: {[round(x, 4) for x in pglobal_flat[:5]]}")
            print(f"      lambdaPrev    = [{len(lambda_flat)}个] "
                  f"前5: {[round(x, 4) for x in lambda_flat[:5]]}")
            print(f"      p2pPrice      = [{len(p2p_flat)}个] "
                  f"前5: {[round(x, 4) for x in p2p_flat[:5]]}")
            print(f"      gridPrice     = {grid_price_list}")
            print(f"      lossAllocation= [{len(loss_alloc_flat)}个] "
                  f"前5: {[round(x, 4) for x in loss_alloc_flat[:5]]}")
            print(f"      lineCapacity  = {line_caps}")
            print(f"      pTradeSubmit  = [{len(p_trade_flat)}个] "
                  f"前5: {[round(x, 4) for x in p_trade_flat[:5]]}")
            print(f"      costSubmit    = {cost:.4f}")

            print(f"  [4] proof (Groth16, 待生成):")
            print(f"      ar = <G1点, 32 bytes>")
            print(f"      bs = <G2点, 64 bytes>")
            print(f"      cr = <G1点, 32 bytes>")
            print(f"  [5] vk (验证密钥, 一次性生成):")
            print(f"      alpha, beta, gamma, delta, ic[{n_elements + 1}个G1点]")

        print("\n" + "=" * 80)
        print("  链上操作 (智能合约执行):")
        print(f"    1. 验证 {self.N} 个 ZKP 证明 → 通过/拒绝")
        print(f"    2. 聚合 P_local → 更新 P_global")
        print(f"    3. 更新对偶变量 λ += ρ * (P_local - P_global)")
        print(f"    4. 计算残差: primal={primal_res:.6f}, dual={dual_res:.6f}")
        print(f"    5. 自适应调整 ρ (当前 ρ={self.rho:.4f})")
        print("=" * 80 + "\n")

    def solve(self):
        """"""
        print("开始ADMM迭代...")
        print(
            f"{'迭代':<6} {'原始残差':<14} {'对偶残差':<14} {'ρ':<10} "
            f"{'系统成本':<12} {'总网损(kWh)':<15}"
        )
        print("-" * 85)

        converged = False
        primal_res = float("inf")
        dual_res = float("inf")
        total_system_cost = 0.0

        for k in range(self.max_iter):
            p_global_prev = self.P_global.copy()
            lambda_prev = self.lambda_dual.copy()

            t_solve0 = time.perf_counter()
            results = []
            for i in range(self.N):
                result = self.solve_local_problem(i, update_state=True)
                results.append(result)
            solve_dt = time.perf_counter() - t_solve0

            self._cached_local_results = {i: results[i] for i in range(self.N)}
            cached = {i: results[i] for i in range(self.N)}

            breakdown = self._system_cost_from_results(results)
            total_grid_cost = breakdown["grid"]
            total_dg_cost = breakdown["dg"]
            total_wear_cost = breakdown["wear"]
            total_fit_revenue = breakdown["fit"]

            self.update_global_variables()

            self.update_dual_variables()

            primal_res, dual_res = self.compute_residuals()

            self._compute_line_losses(cached_results=cached)
            total_loss = np.sum(self.line_loss)

            loss_cost = breakdown["loss"]

            total_system_cost = breakdown["total"]

            self.history["primal_residual"].append(primal_res)
            self.history["dual_residual"].append(dual_res)
            self.history["rho"].append(self.rho)
            self.history["objective"].append(total_system_cost)
            self.history["total_loss"].append(total_loss)
            self.history["solve_time_s"].append(solve_dt)
            self.history["lambda_dual"].append(np.mean(np.abs(self.lambda_dual)))
            self.history["lambda_dual_full"].append(self.lambda_dual.copy())

            if k % 5 == 0 or k < 10:
                print(
                    f"{k:<6} {primal_res:<14.6f} {dual_res:<14.6f} "
                    f"{self.rho:<10.4f} {total_system_cost:<12.2f} {total_loss:<15.4f}"
                )

            if self.check_convergence(primal_res, dual_res):
                converged = True
                self.stop_reason = "boyd"
                num_pairs = self.N * (self.N - 1)
                p = num_pairs * self.T
                sqrt_p = np.sqrt(p)
                norm_P_local = np.linalg.norm(self.P_local)
                norm_P_global = np.linalg.norm(self.P_global)
                norm_lambda = np.linalg.norm(self.lambda_dual)
                eps_pri = sqrt_p * self.tol + self.reltol * max(norm_P_local, norm_P_global)
                eps_dual = sqrt_p * self.tol + self.reltol * norm_lambda

                print(f"\n收敛于第 {k} 次迭代 (Boyd 2011 标准)")
                print(f"  最终原始残差: {primal_res:.6f}")
                print(f"  最终对偶残差: {dual_res:.6f}")
                print(f"  最终系统成本: {total_system_cost:.2f} $")
                print(f"    其中主网购电: {total_grid_cost:.2f} $")
                print(f"    其中网损成本: {loss_cost:.2f} $")
                print(f"    其中DG发电:   {total_dg_cost:.2f} $")
                print(f"    其中VES磨损:  {total_wear_cost:.2f} $")
                print(f"    其中FiT收益:  -{total_fit_revenue:.2f} $")
                print(f"  最终总网损: {total_loss:.4f} kWh")
                print("  --- Convergence thresholds and norms (debug) ---")
                print(f"  sqrt(p): {sqrt_p:.4f}, sqrt(p)*tol: {sqrt_p * self.tol:.6f}")
                print(f"  ||P_local||: {norm_P_local:.6f}, ||P_global||: {norm_P_global:.6f}")
                print(f"  ||lambda||: {norm_lambda:.6f}")
                print(f"  eps_pri: {eps_pri:.6f}, eps_dual: {eps_dual:.6f}")

                self._print_blockchain_submission(
                    k, results, p_global_prev, lambda_prev, primal_res, dual_res
                )
                break

            if self.economic_stop is not None and self._economic_stop_due(k):
                self.stop_reason = "economic"
                self.final_iteration = k
                self.final_primal_residual = primal_res
                self.final_dual_residual = dual_res
                self.final_system_cost = total_system_cost
                if "tau_rel" in self.economic_stop:
                    stop_desc = (
                        f"τ_V^rel={self.economic_stop['tau_rel']:g} "
                        f"(阈值 {self.economic_stop['tau_rel'] * abs(self.history['objective'][-1]):.4f} $)"
                    )
                else:
                    stop_desc = f"{self.economic_stop['delta_tol']:.4f} $"
                print(f"\n经济停机于第 {k} 轮 (窗口内目标改善均低于 {stop_desc})")
                break

            if self.adaptive:
                self.adapt_rho(primal_res, dual_res, k)

        if not converged and self.stop_reason is None:
            print(f"\n达到最大迭代次数 {self.max_iter}")
            print(f"  最终原始残差: {primal_res:.6f}")
            print(f"  最终对偶残差: {dual_res:.6f}")
            print(f"  最终系统成本: {total_system_cost:.2f} $")
            self._print_blockchain_submission(
                self.max_iter - 1, results, p_global_prev, lambda_prev, primal_res, dual_res
            )

        if self.stop_reason == "economic":
            _, rep_stats = self.repair_finalize()
            if rep_stats is not None:
                self.repair_stats = rep_stats
                print(
                    f"  修复轮完成: 结算成本 {rep_stats['repair_cost']:.2f} $ "
                    f"(停机轮 {total_system_cost:.2f} $, "
                    f"修复开销 {rep_stats['repair_cost'] - total_system_cost:+.4f} $, "
                    f"耗时 {rep_stats['repair_time_s']:.3f} s)"
                )
            else:
                self.stop_reason = "economic_repair_failed"

        self.final_results = self.get_results()

        self.converged = converged
        if self.stop_reason == "economic" and self.repair_stats is not None:
            self.final_system_cost = self.repair_stats["repair_cost"]
        else:
            self.final_iteration = k if converged else self.max_iter - 1
            self.final_primal_residual = primal_res
            self.final_dual_residual = dual_res
            self.final_system_cost = total_system_cost

        return self.final_results

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def get_results(self):
        """"""
        return {
            "P_trade": self.P_global.copy(),
            "lambda": self.lambda_dual.copy(),
            "p2p_price": self.p2p_price.copy(),
            "history": self.history,
        }

    def get_local_result(self, mg_idx):
        """"""
        if self._cached_local_results and mg_idx in self._cached_local_results:
            return self._cached_local_results[mg_idx]
        return self.solve_local_problem(mg_idx, update_state=False)

    def get_mg_cost_breakdown(self, mg_idx):
        """"""
        mg = self.microgrids[mg_idx]
        result = self.get_local_result(mg_idx)

        grid_cost = np.sum(mg.grid_price * result["P_grid"])

        loss_cost = np.sum(mg.grid_price * self.loss_allocation[mg_idx, :])

        p2p_buy_cost = 0.0
        p2p_sell_revenue = 0.0
        for j in range(self.N):
            if j != mg_idx:
                for t in range(self.T):
                    trade = self.P_global[mg_idx, j, t]
                    price = self.p2p_price[mg_idx, j, t]
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

    def get_system_grid_cost(self):
        """"""
        total = 0.0
        for i in range(self.N):
            result = self.get_local_result(i)
            mg = self.microgrids[i]
            total += np.sum(mg.grid_price * result["P_grid"])
            if mg.dg_cost is not None:
                total += np.sum(mg.dg_cost * result["P_dg"])
            total += np.sum(mg.battery_cost * result["P_dis"])
            total -= np.sum(mg.fit_price * result["P_export"])
        for i in range(self.N):
            mg = self.microgrids[i]
            total += np.sum(mg.grid_price * self.loss_allocation[i, :])
        return total

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def export_settlement_json(self, filepath, session_id="admm-session", mg_id_prefix="MG"):
        """"""
        if self.final_results is None:
            raise RuntimeError("请先调用 solve() 完成 ADMM 求解")

        microgrid_ids = [f"{mg_id_prefix}{mg.mg_id}" for mg in self.microgrids]

        p_global = self.P_global.tolist()

        p2p_price = self.p2p_price.tolist()

        grid_price = [mg.grid_price.tolist() for mg in self.microgrids]

        settlement = {}
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                net_energy = float(np.sum(self.P_global[i, j, :]))  # kWh
                net_amount = float(np.sum(self.P_global[i, j, :] * self.p2p_price[i, j, :]))  # $
                if abs(net_energy) < 1e-9:
                    continue
                key = f"{microgrid_ids[i]}-{microgrid_ids[j]}"
                settlement[key] = {
                    "energy_kwh": round(net_energy, 6),
                    "amount_yuan": round(net_amount, 6),
                }

        output = {
            "session_id": session_id,
            "n": self.N,
            "t": self.T,
            "converged": bool(self.converged),
            "iterations": int(self.final_iteration),
            "rho_final": float(self.rho),
            "primal_residual": float(self.final_primal_residual),
            "dual_residual": float(self.final_dual_residual),
            "system_cost_yuan": float(self.final_system_cost),
            "microgrid_ids": microgrid_ids,
            "p_global": p_global,
            "p2p_price": p2p_price,
            "grid_price": grid_price,
            "settlement": settlement,
            "proof": None,
            "public_inputs": None,
            "vk": None,
        }

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[OK] 终态 JSON 已导出: {filepath}")
        print(f"   会话: {session_id}, 收敛: {self.converged}, 迭代: {self.final_iteration}")
        print(f"   微网: {microgrid_ids}, 系统成本: {self.final_system_cost:.2f} $")
        return output
