""""""
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from admm_solver import ADMMSolver
from microgrid import DistributionNetwork, Microgrid, get_grid_price
from standalone_baseline import single_mg_cost
from experiment_data import export_figure_data, export_result_bundle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.alpha"] = 0.3

COLORS_SENS = {
    "cost": "#2E4057",
    "saving": "#F4A259",
    "trade": "#8B5A8C",
    "mg1": "#BC4B51",
    "mg2": "#5FA8D3",
    "mg3": "#52A675",
    "loss": "#E07A5F",
    "iters": "#BC4B51",
    "baseline": "#888888",
}


# ======================================================================
# ======================================================================


class ModifiedMicrogrid(Microgrid):
    """"""
    def __init__(self, mg_id, mg_type, time_slots=24):
        super().__init__(mg_id, mg_type, time_slots)

    def set_grid_price(self, new_price):
        """"""
        self.grid_price = new_price.copy()

    def set_battery_params(self, capacity_factor=1.0, power_factor=1.0):
        """"""
        if not hasattr(self, "_orig_battery_capacity"):
            self._orig_battery_capacity = self.battery_capacity
            self._orig_battery_power = self.battery_power

        self.battery_capacity = self._orig_battery_capacity * capacity_factor
        self.battery_power = self._orig_battery_power * power_factor

        self.battery_soc, self.battery_power_profile = self._simulate_battery()


def create_modified_microgrids(
    price_func=None,
    battery_capacity_factor=1.0,
    battery_power_factor=1.0,
):
    """"""
    mg_configs = [
        (1, "industrial"),
        (2, "commercial"),
        (3, "residential"),
    ]
    microgrids = []
    for mg_id, mg_type in mg_configs:
        mg = ModifiedMicrogrid(mg_id, mg_type)

        if price_func is not None:
            mg.set_grid_price(price_func())

        if battery_capacity_factor != 1.0 or battery_power_factor != 1.0:
            mg.set_battery_params(battery_capacity_factor, battery_power_factor)

        microgrids.append(mg)
    return microgrids


def run_admm_quiet(microgrids, max_iter=500, tol=1e-3):
    """"""
    N = len(microgrids)
    network = DistributionNetwork(N)

    individual_costs = []
    for i in range(N):
        cost = single_mg_cost(microgrids[i])
        individual_costs.append(cost)
    total_individual = sum(individual_costs)

    # ADMM
    solver = ADMMSolver(
        microgrids,
        rho_init=0.1,
        max_iter=max_iter,
        tol=tol,
        adaptive=True,
        network=network,
    )

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        solver.solve()
    finally:
        sys.stdout = old_stdout

    admm_cost = solver.get_system_grid_cost()
    admm_iters = len(solver.history["primal_residual"])
    total_loss = np.sum(solver.line_loss)

    total_trade = 0.0
    P_trade = solver.P_global
    for i in range(N):
        for j in range(i + 1, N):
            total_trade += np.sum(np.abs(P_trade[i, j, :]))

    mg_grid_costs = []
    mg_total_costs = []
    for i in range(N):
        bd = solver.get_mg_cost_breakdown(i)
        mg_grid_costs.append(bd["grid_cost"])
        mg_total_costs.append(bd["total_cost"])

    saving = total_individual - admm_cost
    saving_pct = saving / total_individual * 100 if total_individual > 1e-6 else 0.0

    return {
        "admm_cost": admm_cost,
        "individual_costs": individual_costs,
        "total_individual": total_individual,
        "saving": saving,
        "saving_pct": saving_pct,
        "total_trade": total_trade,
        "total_loss": total_loss,
        "admm_iters": admm_iters,
        "mg_grid_costs": mg_grid_costs,
        "mg_total_costs": mg_total_costs,
    }


# ======================================================================
# ======================================================================


def generate_price_with_spread(spread_factor=1.0, time_slots=24):
    """"""
    base_price = get_grid_price(time_slots)

    mean_price = np.mean(base_price)

    new_price = mean_price + spread_factor * (base_price - mean_price)

    new_price = np.maximum(new_price, 0.05)

    return new_price


def run_price_sensitivity(spread_factors=None):
    """"""
    if spread_factors is None:
        spread_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print("\n" + "=" * 70)
    print("  实验3A: 电价结构敏感性分析")
    print(f"  峰谷价差倍率: {spread_factors}")
    print("=" * 70)

    results = []
    for sf in spread_factors:
        print(f"\n  价差倍率 = {sf:.2f} ...")

        def price_func(sf=sf):
            return generate_price_with_spread(sf)

        microgrids = create_modified_microgrids(price_func=price_func)

        price = microgrids[0].grid_price
        peak = np.max(price)
        valley = np.min(price)
        print(
            f"    峰: {peak:.3f}, 谷: {valley:.3f}, 差: {peak - valley:.3f}, 均: {np.mean(price):.3f}"
        )

        r = run_admm_quiet(microgrids)
        r["spread_factor"] = sf
        r["peak_price"] = peak
        r["valley_price"] = valley
        r["spread"] = peak - valley

        print(
            f"    系统成本: {r['admm_cost']:.2f}, 节省: {r['saving_pct']:.1f}%, 交易量: {r['total_trade']:.1f}"
        )
        results.append(r)

    return results


# ======================================================================
# ======================================================================


def run_battery_sensitivity(capacity_factors=None):
    """"""
    if capacity_factors is None:
        capacity_factors = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    print("\n" + "=" * 70)
    print("  实验3B: 储能容量敏感性分析")
    print(f"  容量倍率: {capacity_factors}")
    print("=" * 70)

    results = []
    for cf in capacity_factors:
        print(f"\n  容量倍率 = {cf:.2f} ...")

        microgrids = create_modified_microgrids(
            battery_capacity_factor=cf,
            battery_power_factor=max(cf, 0.01),
        )

        for mg in microgrids:
            cap = mg.battery_capacity
            pwr = mg.battery_power
            if mg == microgrids[0]:
                print(f"    MG1: {cap:.0f} kWh / {pwr:.0f} kW")

        r = run_admm_quiet(microgrids)
        r["capacity_factor"] = cf

        print(
            f"    系统成本: {r['admm_cost']:.2f}, 节省: {r['saving_pct']:.1f}%, 交易量: {r['total_trade']:.1f}"
        )
        results.append(r)

    return results


# ======================================================================
# ======================================================================


class ModifiedNetwork(DistributionNetwork):
    """"""
    def __init__(self, num_microgrids=3, resistance_factor=1.0, capacity_factor=1.0):
        """"""
        super().__init__(num_microgrids)

        for line in self.lines:
            if line in self.line_params:
                self.line_params[line]["R"] *= resistance_factor
                self.line_params[line]["X"] *= resistance_factor
                original_cap = self.line_params[line]["capacity"]
                self.line_params[line]["capacity"] = max(
                    original_cap * capacity_factor, 10.0
                )

        self.line_impedance = {}
        for line, params in self.line_params.items():
            r_total = params["R"] * params["length"]
            x_total = params["X"] * params["length"]
            self.line_impedance[line] = (r_total, x_total)


def run_line_resistance_sensitivity(resistance_factors=None):
    """"""
    if resistance_factors is None:
        resistance_factors = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    print("\n" + "=" * 70)
    print("  实验3C-1: 线路电阻敏感性分析（网损影响）")
    print(f"  电阻倍率: {resistance_factors}")
    print("=" * 70)

    results = []
    for rf in resistance_factors:
        print(f"\n  电阻倍率 = {rf:.2f} ...")

        mg1 = Microgrid(1, "industrial")
        mg2 = Microgrid(2, "commercial")
        mg3 = Microgrid(3, "residential")
        microgrids = [mg1, mg2, mg3]

        network = ModifiedNetwork(3, resistance_factor=rf, capacity_factor=1.0)

        individual_costs = []
        for i in range(3):
            cost = single_mg_cost(microgrids[i])
            individual_costs.append(cost)
        total_individual = sum(individual_costs)

        solver = ADMMSolver(
            microgrids,
            rho_init=0.1,
            max_iter=500,
            tol=1e-3,
            adaptive=True,
            network=network,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            solver.solve()
        finally:
            sys.stdout = old_stdout

        admm_cost = solver.get_system_grid_cost()
        total_loss = np.sum(solver.line_loss)
        admm_iters = len(solver.history["primal_residual"])

        total_trade = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                total_trade += np.sum(np.abs(solver.P_global[i, j, :]))

        saving = total_individual - admm_cost
        saving_pct = saving / total_individual * 100 if total_individual > 1e-6 else 0

        print(
            f"    系统成本: {admm_cost:.2f}, 网损: {total_loss:.2f} kWh, "
            f"节省: {saving_pct:.1f}%, 交易量: {total_trade:.1f}"
        )

        results.append(
            {
                "resistance_factor": rf,
                "admm_cost": admm_cost,
                "total_individual": total_individual,
                "saving": saving,
                "saving_pct": saving_pct,
                "total_trade": total_trade,
                "total_loss": total_loss,
                "admm_iters": admm_iters,
            }
        )

    return results


def run_line_capacity_sensitivity(capacity_factors=None):
    """"""
    if capacity_factors is None:
        capacity_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]

    print("\n" + "=" * 70)
    print("  实验3C-2: 线路容量敏感性分析（交易约束）")
    print(f"  容量倍率: {capacity_factors}")
    print("=" * 70)

    results = []
    for cf in capacity_factors:
        print(f"\n  容量倍率 = {cf:.2f} ...")

        mg1 = Microgrid(1, "industrial")
        mg2 = Microgrid(2, "commercial")
        mg3 = Microgrid(3, "residential")
        microgrids = [mg1, mg2, mg3]

        network = ModifiedNetwork(3, resistance_factor=1.0, capacity_factor=cf)

        individual_costs = []
        for i in range(3):
            cost = single_mg_cost(microgrids[i])
            individual_costs.append(cost)
        total_individual = sum(individual_costs)

        solver = ADMMSolver(
            microgrids,
            rho_init=0.1,
            max_iter=500,
            tol=1e-3,
            adaptive=True,
            network=network,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            solver.solve()
        finally:
            sys.stdout = old_stdout

        admm_cost = solver.get_system_grid_cost()
        total_loss = np.sum(solver.line_loss)
        admm_iters = len(solver.history["primal_residual"])

        total_trade = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                total_trade += np.sum(np.abs(solver.P_global[i, j, :]))

        saving = total_individual - admm_cost
        saving_pct = saving / total_individual * 100 if total_individual > 1e-6 else 0

        cap_01 = network.get_line_capacity(0, 1)
        print(
            f"    线路容量(0→1): {cap_01:.0f} kW, "
            f"系统成本: {admm_cost:.2f}, 节省: {saving_pct:.1f}%, "
            f"交易量: {total_trade:.1f}"
        )

        results.append(
            {
                "capacity_factor": cf,
                "line_capacity_01": cap_01,
                "admm_cost": admm_cost,
                "total_individual": total_individual,
                "saving": saving,
                "saving_pct": saving_pct,
                "total_trade": total_trade,
                "total_loss": total_loss,
                "admm_iters": admm_iters,
            }
        )

    return results


# ======================================================================
# ======================================================================


def plot_price_sensitivity(results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 17: 电价敏感性分析...")

    spreads = [r["spread_factor"] for r in results]
    costs = [r["admm_cost"] for r in results]
    savings = [r["saving_pct"] for r in results]
    trades = [r["total_trade"] for r in results]
    mg1_costs = [r["mg_grid_costs"][0] for r in results]
    mg2_costs = [r["mg_grid_costs"][1] for r in results]
    mg3_costs = [r["mg_grid_costs"][2] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    base_idx = None
    for i, sf in enumerate(spreads):
        if abs(sf - 1.0) < 1e-6:
            base_idx = i
            break

    ax = axes[0]
    ax.plot(
        spreads,
        costs,
        color=COLORS_SENS["cost"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="System Cost",
    )
    if base_idx is not None:
        ax.scatter(
            [spreads[base_idx]],
            [costs[base_idx]],
            color="red",
            s=120,
            zorder=10,
            marker="*",
            label="Baseline",
        )

    ax.set_xlabel("Peak-Valley Spread Factor")
    ax.set_ylabel("System Cost (¥)", color=COLORS_SENS["cost"])
    ax.tick_params(axis="y", labelcolor=COLORS_SENS["cost"])

    ax_twin = ax.twinx()
    ax_twin.plot(
        spreads,
        savings,
        color=COLORS_SENS["saving"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="Saving Rate",
    )
    ax_twin.set_ylabel("Cooperation Saving (%)", color=COLORS_SENS["saving"])
    ax_twin.tick_params(axis="y", labelcolor=COLORS_SENS["saving"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9, fontsize=9
    )
    ax.set_title("(a) System Cost & Saving Rate", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    ax = axes[1]
    ax.bar(
        range(len(spreads)),
        trades,
        color=COLORS_SENS["trade"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )
    if base_idx is not None:
        ax.patches[base_idx].set_edgecolor("red")
        ax.patches[base_idx].set_linewidth(2.5)

    for i, (sf, tv) in enumerate(zip(spreads, trades)):
        ax.text(
            i,
            tv + max(trades) * 0.02,
            f"{tv:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(spreads)))
    ax.set_xticklabels([f"{sf:.2f}" for sf in spreads])
    ax.set_xlabel("Peak-Valley Spread Factor")
    ax.set_ylabel("Total P2P Trade Volume (kWh)")
    ax.set_title("(b) P2P Trade Volume", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    ax = axes[2]
    ax.plot(
        spreads,
        mg1_costs,
        color=COLORS_SENS["mg1"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="MG1 (Industrial)",
    )
    ax.plot(
        spreads,
        mg2_costs,
        color=COLORS_SENS["mg2"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="MG2 (Commercial)",
    )
    ax.plot(
        spreads,
        mg3_costs,
        color=COLORS_SENS["mg3"],
        marker="D",
        markersize=6,
        linewidth=2,
        label="MG3 (Residential)",
    )

    ax.set_xlabel("Peak-Valley Spread Factor")
    ax.set_ylabel("Grid Purchase Cost (¥)")
    ax.set_title("(c) Per-Microgrid Cost", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle(
        "Figure 17: Price Structure Sensitivity Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.90, bottom=0.12, wspace=0.38)

    if save:
        export_figure_data(fig, "figure17_price_sensitivity")
        plt.savefig("figure17_price_sensitivity.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure17_price_sensitivity.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure17_price_sensitivity.pdf/png")
    plt.close()


def plot_battery_sensitivity(results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 18: 储能容量敏感性分析...")

    factors = [r["capacity_factor"] for r in results]
    costs = [r["admm_cost"] for r in results]
    savings = [r["saving_pct"] for r in results]
    trades = [r["total_trade"] for r in results]
    ind_costs = [r["total_individual"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    base_idx = None
    for i, cf in enumerate(factors):
        if abs(cf - 1.0) < 1e-6:
            base_idx = i
            break

    ax = axes[0]
    ax.plot(
        factors,
        ind_costs,
        color=COLORS_SENS["baseline"],
        marker="^",
        markersize=7,
        linewidth=2,
        linestyle="--",
        label="Individual (No P2P)",
        alpha=0.7,
    )
    ax.plot(
        factors,
        costs,
        color=COLORS_SENS["cost"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="Alliance (ADMM)",
    )
    if base_idx is not None:
        ax.scatter(
            [factors[base_idx]],
            [costs[base_idx]],
            color="red",
            s=120,
            zorder=10,
            marker="*",
            label="Baseline",
        )

    ax.fill_between(
        factors,
        costs,
        ind_costs,
        color=COLORS_SENS["saving"],
        alpha=0.12,
        label="Cooperation Saving",
    )

    ax.set_xlabel("Battery Capacity Factor")
    ax.set_ylabel("System Cost (¥)")
    ax.set_title("(a) System Cost: Individual vs Alliance", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax = axes[1]
    ax.plot(
        factors,
        savings,
        color=COLORS_SENS["saving"],
        marker="s",
        markersize=8,
        linewidth=2.5,
    )
    if base_idx is not None:
        ax.scatter(
            [factors[base_idx]],
            [savings[base_idx]],
            color="red",
            s=120,
            zorder=10,
            marker="*",
        )

    ax.set_xlabel("Battery Capacity Factor")
    ax.set_ylabel("Cooperation Saving (%)")
    ax.set_title("(b) Cooperation Saving Rate", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    ax = axes[2]
    ax.bar(
        range(len(factors)),
        trades,
        color=COLORS_SENS["trade"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )
    if base_idx is not None:
        ax.patches[base_idx].set_edgecolor("red")
        ax.patches[base_idx].set_linewidth(2.5)

    for i, tv in enumerate(trades):
        ax.text(
            i,
            tv + max(trades) * 0.02,
            f"{tv:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f"{cf:.2f}x" for cf in factors], fontsize=8)
    ax.set_xlabel("Battery Capacity Factor")
    ax.set_ylabel("Total P2P Trade Volume (kWh)")
    ax.set_title("(c) P2P Trade Volume", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.suptitle(
        "Figure 18: Battery Capacity Sensitivity Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.90, bottom=0.12, wspace=0.35)

    if save:
        export_figure_data(fig, "figure18_battery_sensitivity")
        plt.savefig("figure18_battery_sensitivity.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure18_battery_sensitivity.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure18_battery_sensitivity.pdf/png")
    plt.close()


def plot_line_sensitivity(resistance_results, capacity_results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 19: 线路参数敏感性分析...")

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ================================================================
    # ================================================================
    ax = fig.add_subplot(gs[0, 0])
    rfs = [r["resistance_factor"] for r in resistance_results]
    r_costs = [r["admm_cost"] for r in resistance_results]
    r_losses = [r["total_loss"] for r in resistance_results]

    ax.plot(
        rfs,
        r_costs,
        color=COLORS_SENS["cost"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="System Cost",
    )
    ax.set_xlabel("Line Resistance Factor")
    ax.set_ylabel("System Cost (¥)", color=COLORS_SENS["cost"])
    ax.tick_params(axis="y", labelcolor=COLORS_SENS["cost"])

    ax_twin = ax.twinx()
    ax_twin.plot(
        rfs,
        r_losses,
        color=COLORS_SENS["loss"],
        marker="D",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="Total Line Loss",
    )
    ax_twin.set_ylabel("Line Loss (kWh)", color=COLORS_SENS["loss"])
    ax_twin.tick_params(axis="y", labelcolor=COLORS_SENS["loss"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=9
    )
    ax.set_title("(a) Resistance → Cost & Line Loss", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax = fig.add_subplot(gs[0, 1])
    r_savings = [r["saving_pct"] for r in resistance_results]
    r_trades = [r["total_trade"] for r in resistance_results]

    ax.plot(
        rfs,
        r_savings,
        color=COLORS_SENS["saving"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        label="Saving Rate",
    )
    ax.set_xlabel("Line Resistance Factor")
    ax.set_ylabel("Cooperation Saving (%)", color=COLORS_SENS["saving"])
    ax.tick_params(axis="y", labelcolor=COLORS_SENS["saving"])

    ax_twin = ax.twinx()
    ax_twin.plot(
        rfs,
        r_trades,
        color=COLORS_SENS["trade"],
        marker="^",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="Trade Volume",
    )
    ax_twin.set_ylabel("P2P Trade Volume (kWh)", color=COLORS_SENS["trade"])
    ax_twin.tick_params(axis="y", labelcolor=COLORS_SENS["trade"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9, fontsize=9
    )
    ax.set_title("(b) Resistance → Saving & Trade", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax = fig.add_subplot(gs[1, 0])
    cfs = [r["capacity_factor"] for r in capacity_results]
    c_costs = [r["admm_cost"] for r in capacity_results]
    c_savings = [r["saving_pct"] for r in capacity_results]

    ax.plot(
        cfs,
        c_costs,
        color=COLORS_SENS["cost"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="System Cost",
    )
    ax.set_xlabel("Line Capacity Factor")
    ax.set_ylabel("System Cost (¥)", color=COLORS_SENS["cost"])
    ax.tick_params(axis="y", labelcolor=COLORS_SENS["cost"])

    ax_twin = ax.twinx()
    ax_twin.plot(
        cfs,
        c_savings,
        color=COLORS_SENS["saving"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        label="Saving Rate",
    )
    ax_twin.set_ylabel("Cooperation Saving (%)", color=COLORS_SENS["saving"])
    ax_twin.tick_params(axis="y", labelcolor=COLORS_SENS["saving"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9, fontsize=9
    )
    ax.set_title("(c) Line Capacity → Cost & Saving", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax = fig.add_subplot(gs[1, 1])
    c_trades = [r["total_trade"] for r in capacity_results]

    ax.bar(
        range(len(cfs)),
        c_trades,
        color=COLORS_SENS["trade"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )

    base_idx = None
    for i, cf in enumerate(cfs):
        if abs(cf - 1.0) < 1e-6:
            base_idx = i
            break
    if base_idx is not None:
        ax.patches[base_idx].set_edgecolor("red")
        ax.patches[base_idx].set_linewidth(2.5)

    for i, tv in enumerate(c_trades):
        ax.text(
            i,
            tv + max(c_trades) * 0.02,
            f"{tv:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(cfs)))
    ax.set_xticklabels([f"{cf:.2f}x" for cf in cfs], fontsize=8)
    ax.set_xlabel("Line Capacity Factor")
    ax.set_ylabel("Total P2P Trade Volume (kWh)")
    ax.set_title("(d) Line Capacity → Trade Volume", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.suptitle(
        "Figure 19: Line Parameter Sensitivity Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.93, bottom=0.07)

    if save:
        export_figure_data(fig, "figure19_line_sensitivity")
        plt.savefig("figure19_line_sensitivity.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure19_line_sensitivity.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure19_line_sensitivity.pdf/png")
    plt.close()


def plot_spider_chart(
    price_results,
    battery_results,
    resistance_results,
    capacity_results,
    save=True,
):
    """"""
    print("\n[绘图] 生成 Figure 20: 综合敏感性蛛网图...")

    param_names = [
        "Price\nSpread",
        "Battery\nCapacity",
        "Line\nResistance",
        "Line\nCapacity",
    ]

    metric_names = ["System Cost", "Saving Rate", "Trade Volume", "Line Loss"]
    metric_colors = [
        COLORS_SENS["cost"],
        COLORS_SENS["saving"],
        COLORS_SENS["trade"],
        COLORS_SENS["loss"],
    ]

    def find_result_by_factor(results, factor_key, target):
        """"""
        best = None
        best_diff = float("inf")
        for r in results:
            diff = abs(r[factor_key] - target)
            if diff < best_diff:
                best_diff = diff
                best = r
        return best

    base_price = find_result_by_factor(price_results, "spread_factor", 1.0)
    base_batt = find_result_by_factor(battery_results, "capacity_factor", 1.0)
    base_res = find_result_by_factor(resistance_results, "resistance_factor", 1.0)
    base_cap = find_result_by_factor(capacity_results, "capacity_factor", 1.0)

    if any(b is None for b in [base_price, base_batt, base_res, base_cap]):
        print("  警告: 基准结果缺失，无法绘制蛛网图")
        return

    bases = [base_price, base_batt, base_res, base_cap]
    factor_keys = [
        "spread_factor",
        "capacity_factor",
        "resistance_factor",
        "capacity_factor",
    ]
    result_sets = [price_results, battery_results, resistance_results, capacity_results]

    up_factor = 1.5
    down_factor = 0.5

    def compute_pct_change(base_r, perturbed_r, metric_key):
        base_val = base_r.get(metric_key, 0)
        pert_val = perturbed_r.get(metric_key, 0)
        if abs(base_val) < 1e-9:
            return 0.0
        return (pert_val - base_val) / abs(base_val) * 100.0

    metrics_keys = ["admm_cost", "saving_pct", "total_trade", "total_loss"]

    up_changes = np.zeros((len(param_names), len(metric_names)))
    down_changes = np.zeros((len(param_names), len(metric_names)))

    for p_idx in range(len(param_names)):
        base_r = bases[p_idx]
        fk = factor_keys[p_idx]
        rs = result_sets[p_idx]

        up_r = find_result_by_factor(rs, fk, up_factor)
        down_r = find_result_by_factor(rs, fk, down_factor)

        for m_idx, mk in enumerate(metrics_keys):
            if up_r is not None:
                up_changes[p_idx, m_idx] = compute_pct_change(base_r, up_r, mk)
            if down_r is not None:
                down_changes[p_idx, m_idx] = compute_pct_change(base_r, down_r, mk)

    fig, (ax_up, ax_down) = plt.subplots(1, 2, figsize=(16, 6.5))

    x = np.arange(len(param_names))
    n_metrics = len(metric_names)
    w = 0.18

    for m_idx in range(n_metrics):
        offset = (m_idx - n_metrics / 2 + 0.5) * w
        vals_up = up_changes[:, m_idx]
        vals_down = down_changes[:, m_idx]

        ax_up.bar(
            x + offset,
            vals_up,
            w,
            color=metric_colors[m_idx],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            label=metric_names[m_idx] if True else None,
        )
        ax_down.bar(
            x + offset,
            vals_down,
            w,
            color=metric_colors[m_idx],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            label=metric_names[m_idx],
        )

    for ax, title_suffix in [(ax_up, "+50%"), (ax_down, "−50%")]:
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, fontsize=10)
        ax.set_ylabel("Change from Baseline (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_title(f"Parameter {title_suffix}", fontweight="bold", fontsize=12)

    ax_up.legend(
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
        ncol=2,
        edgecolor="black",
    )

    plt.suptitle(
        "Figure 20: Comprehensive Sensitivity Analysis (Tornado Chart)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.90, bottom=0.10, wspace=0.25)

    if save:
        export_figure_data(fig, "figure20_sensitivity_spider")
        plt.savefig("figure20_sensitivity_spider.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure20_sensitivity_spider.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure20_sensitivity_spider.pdf/png")
    plt.close()


# ======================================================================
# ======================================================================


def print_sensitivity_summary(
    price_results, battery_results, resistance_results, capacity_results
):
    """"""
    def _find_base(results, key):
        for r in results:
            if abs(r[key] - 1.0) < 1e-6:
                return r
        return None

    print("\n" + "=" * 80)
    print("  参数敏感性分析汇总")
    print("=" * 80)

    print("\n  [A] 电价峰谷价差:")
    print(f"    {'倍率':>6} {'系统成本':>12} {'节省率':>8} {'交易量':>10} {'迭代':>6}")
    print("    " + "-" * 50)
    for r in price_results:
        sf = r["spread_factor"]
        flag = " ← baseline" if abs(sf - 1.0) < 1e-6 else ""
        print(
            f"    {sf:>6.2f} {r['admm_cost']:>12.2f} {r['saving_pct']:>7.1f}% "
            f"{r['total_trade']:>10.1f} {r['admm_iters']:>6}{flag}"
        )

    print("\n  [B] 储能容量:")
    print(f"    {'倍率':>6} {'系统成本':>12} {'节省率':>8} {'交易量':>10} {'迭代':>6}")
    print("    " + "-" * 50)
    for r in battery_results:
        cf = r["capacity_factor"]
        flag = " ← baseline" if abs(cf - 1.0) < 1e-6 else ""
        print(
            f"    {cf:>6.2f} {r['admm_cost']:>12.2f} {r['saving_pct']:>7.1f}% "
            f"{r['total_trade']:>10.1f} {r['admm_iters']:>6}{flag}"
        )

    print("\n  [C-1] 线路电阻:")
    print(
        f"    {'倍率':>6} {'系统成本':>12} {'节省率':>8} {'网损(kWh)':>10} {'交易量':>10}"
    )
    print("    " + "-" * 56)
    for r in resistance_results:
        rf = r["resistance_factor"]
        flag = " ← baseline" if abs(rf - 1.0) < 1e-6 else ""
        print(
            f"    {rf:>6.2f} {r['admm_cost']:>12.2f} {r['saving_pct']:>7.1f}% "
            f"{r['total_loss']:>10.2f} {r['total_trade']:>10.1f}{flag}"
        )

    print("\n  [C-2] 线路容量:")
    print(f"    {'倍率':>6} {'系统成本':>12} {'节省率':>8} {'交易量':>10} {'迭代':>6}")
    print("    " + "-" * 50)
    for r in capacity_results:
        cf = r["capacity_factor"]
        flag = " ← baseline" if abs(cf - 1.0) < 1e-6 else ""
        print(
            f"    {cf:>6.2f} {r['admm_cost']:>12.2f} {r['saving_pct']:>7.1f}% "
            f"{r['total_trade']:>10.1f} {r['admm_iters']:>6}{flag}"
        )

    print("\n" + "=" * 80)


# ======================================================================
# ======================================================================


def run_and_plot():
    """"""
    print("=" * 70)
    print("  实验3: 参数敏感性分析 (Sensitivity Analysis)")
    print("=" * 70)

    price_results = run_price_sensitivity()

    battery_results = run_battery_sensitivity()

    resistance_results = run_line_resistance_sensitivity()
    capacity_results = run_line_capacity_sensitivity()
    export_result_bundle(
        "sensitivity_results",
        {
            "price": price_results,
            "battery": battery_results,
            "resistance": resistance_results,
            "capacity": capacity_results,
        },
    )

    print_sensitivity_summary(
        price_results, battery_results, resistance_results, capacity_results
    )

    plot_price_sensitivity(price_results)
    plot_battery_sensitivity(battery_results)
    plot_line_sensitivity(resistance_results, capacity_results)
    plot_spider_chart(
        price_results, battery_results, resistance_results, capacity_results
    )

    return {
        "price": price_results,
        "battery": battery_results,
        "resistance": resistance_results,
        "capacity": capacity_results,
    }


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    all_results = run_and_plot()
