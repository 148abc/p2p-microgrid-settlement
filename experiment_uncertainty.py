""""""
import sys
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from admm_solver import ADMMSolver
from microgrid import DistributionNetwork, Microgrid
from shapley_allocation import calculate_shapley_values
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

COLORS_UNC = {
    "mg1": "#BC4B51",
    "mg2": "#5FA8D3",
    "mg3": "#52A675",
    "system": "#2E4057",
    "saving": "#F4A259",
    "trade": "#8B5A8C",
    "shapley1": "#BC4B51",
    "shapley2": "#5FA8D3",
    "shapley3": "#52A675",
}


class UncertainMicrogrid(Microgrid):
    """"""
    def __init__(
        self,
        mg_id,
        mg_type,
        sigma_pv=0.0,
        sigma_wind=0.0,
        sigma_load=0.0,
        seed=None,
        time_slots=24,
    ):
        super().__init__(mg_id, mg_type, time_slots)

        self._base_pv = self.pv_generation.copy()
        self._base_wind = self.wind_generation.copy()
        self._base_load = self.load.copy()

        self.sigma_pv = sigma_pv
        self.sigma_wind = sigma_wind
        self.sigma_load = sigma_load

        if seed is not None or (sigma_pv > 0 or sigma_wind > 0 or sigma_load > 0):
            self.apply_perturbation(seed)

    def apply_perturbation(self, seed=None):
        """"""
        rng = np.random.RandomState(seed)
        T = self.time_slots

        if self.sigma_pv > 0:
            pv_noise = 1.0 + self.sigma_pv * rng.randn(T)
            pv_noise = np.clip(pv_noise, 0.0, 2.0)
            self.pv_generation = self._base_pv * pv_noise
            self.pv_generation[self._base_pv < 1e-6] = 0.0
        else:
            self.pv_generation = self._base_pv.copy()

        if self.sigma_wind > 0 and np.sum(self._base_wind) > 1e-6:
            wind_noise = 1.0 + self.sigma_wind * rng.randn(T)
            wind_noise = np.clip(wind_noise, 0.0, 2.0)
            self.wind_generation = self._base_wind * wind_noise
        else:
            self.wind_generation = self._base_wind.copy()

        if self.sigma_load > 0:
            load_noise = 1.0 + self.sigma_load * rng.randn(T)
            load_noise = np.clip(load_noise, 0.5, 1.5)
            self.load = self._base_load * load_noise
        else:
            self.load = self._base_load.copy()

        self.total_generation = self.pv_generation + self.wind_generation
        self.net_load = self.load - self.total_generation

        self.battery_soc, self.battery_power_profile = self._simulate_battery()


def create_uncertain_microgrids(sigma_pv, sigma_wind, sigma_load, seed=None):
    """"""
    mg_configs = [
        (1, "industrial"),
        (2, "commercial"),
        (3, "residential"),
    ]
    microgrids = []
    for mg_id, mg_type in mg_configs:
        mg_seed = seed * 100 + mg_id if seed is not None else None
        mg = UncertainMicrogrid(
            mg_id,
            mg_type,
            sigma_pv=sigma_pv,
            sigma_wind=sigma_wind,
            sigma_load=sigma_load,
            seed=mg_seed,
        )
        microgrids.append(mg)
    return microgrids


def run_single_scenario(
    microgrids, run_shapley=False, max_iter=500, tol=1e-3, verbose=False
):
    """"""
    N = len(microgrids)
    network = DistributionNetwork(N)

    individual_costs = []
    for i in range(N):
        cost = single_mg_cost(microgrids[i])
        individual_costs.append(cost)
    total_individual = sum(individual_costs)

    solver = ADMMSolver(
        microgrids,
        rho_init=0.1,
        max_iter=max_iter,
        tol=tol,
        adaptive=True,
        network=network,
    )

    old_stdout = sys.stdout
    if not verbose:
        sys.stdout = StringIO()
    try:
        solver.solve()
    finally:
        if not verbose:
            sys.stdout = old_stdout

    admm_cost = solver.get_system_grid_cost()
    admm_iters = len(solver.history["primal_residual"])
    total_loss = np.sum(solver.line_loss)

    mg_costs = []
    mg_grid_costs = []
    for i in range(N):
        bd = solver.get_mg_cost_breakdown(i)
        mg_costs.append(bd["total_cost"])
        mg_grid_costs.append(bd["grid_cost"])

    total_trade = 0.0
    P_trade = solver.P_global
    for i in range(N):
        for j in range(i + 1, N):
            total_trade += np.sum(np.abs(P_trade[i, j, :]))

    saving = total_individual - admm_cost
    saving_pct = saving / total_individual * 100 if total_individual > 1e-6 else 0.0

    result = {
        "admm_cost": admm_cost,
        "individual_costs": individual_costs,
        "total_individual": total_individual,
        "mg_costs": mg_costs,
        "mg_grid_costs": mg_grid_costs,
        "total_trade": total_trade,
        "total_loss": total_loss,
        "saving": saving,
        "saving_pct": saving_pct,
        "admm_iters": admm_iters,
        "converged": bool(solver.converged),
        "shapley_values": None,
    }

    if run_shapley:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            shapley_vals, coalition_costs = calculate_shapley_values(microgrids)
        finally:
            sys.stdout = old_stdout
        result["shapley_values"] = shapley_vals
        result["coalition_costs"] = coalition_costs

    return result


def run_monte_carlo(
    n_scenarios=50,
    sigma_pv=0.15,
    sigma_wind=0.20,
    sigma_load=0.10,
    run_shapley=False,
    verbose=False,
):
    """"""
    print(f"\n{'=' * 70}")
    print(f"  蒙特卡洛模拟: {n_scenarios} 个场景")
    print(f"  σ_pv={sigma_pv:.0%}, σ_wind={sigma_wind:.0%}, σ_load={sigma_load:.0%}")
    print(f"  Shapley: {'开启' if run_shapley else '关闭'}")
    print(f"{'=' * 70}")

    all_results = []
    t_start = time.time()

    for s in range(n_scenarios):
        if (s + 1) % 10 == 0 or s == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (s + 1) * (n_scenarios - s - 1) if s > 0 else 0
            print(
                f"  场景 {s + 1}/{n_scenarios} "
                f"(已用 {elapsed:.1f}s, 预计剩余 {eta:.0f}s)..."
            )

        microgrids = create_uncertain_microgrids(
            sigma_pv, sigma_wind, sigma_load, seed=s + 1
        )

        result = run_single_scenario(
            microgrids, run_shapley=run_shapley, verbose=verbose
        )
        result["scenario_id"] = s
        all_results.append(result)

    total_time = time.time() - t_start
    print(
        f"\n  蒙特卡洛模拟完成: {total_time:.1f}s ({total_time / n_scenarios:.2f}s/场景)"
    )

    costs = [r["admm_cost"] for r in all_results]
    savings = [r["saving_pct"] for r in all_results]
    trades = [r["total_trade"] for r in all_results]

    print(f"\n  系统成本统计:")
    print(f"    均值: {np.mean(costs):.2f} ± {np.std(costs):.2f} 元")
    print(f"    范围: [{np.min(costs):.2f}, {np.max(costs):.2f}] 元")
    print(f"  合作节省率:")
    print(f"    均值: {np.mean(savings):.2f} ± {np.std(savings):.2f} %")
    print(f"  P2P总交易量:")
    print(f"    均值: {np.mean(trades):.2f} ± {np.std(trades):.2f} kWh")

    return all_results


def run_uncertainty_levels(
    sigma_levels=None,
    n_scenarios=30,
    run_shapley=False,
):
    """"""
    if sigma_levels is None:
        sigma_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    print("=" * 70)
    print("  实验2: 不确定性分析 (Uncertainty Analysis)")
    print(f"  不确定性水平: {sigma_levels}")
    print(f"  每水平场景数: {n_scenarios}")
    print("=" * 70)

    all_level_results = {}

    for sigma in sigma_levels:
        print(f"\n{'─' * 60}")
        print(f"  不确定性水平: σ = {sigma:.0%}")
        print(f"{'─' * 60}")

        if sigma == 0.0:
            microgrids = create_uncertain_microgrids(0, 0, 0, seed=0)
            result = run_single_scenario(
                microgrids, run_shapley=run_shapley, verbose=False
            )
            result["scenario_id"] = 0
            all_level_results[sigma] = [result]
        else:
            results = run_monte_carlo(
                n_scenarios=n_scenarios,
                sigma_pv=sigma,
                sigma_wind=min(sigma * 1.3, 0.5),
                sigma_load=min(sigma * 0.6, 0.25),
                run_shapley=run_shapley,
                verbose=False,
            )
            all_level_results[sigma] = results

    print(f"\n{'=' * 80}")
    print("  不确定性水平实验汇总")
    print(f"{'=' * 80}")
    header = (
        f"{'σ':>6} {'#Scen':>6} {'Cost Mean':>12} {'Cost Std':>10} "
        f"{'Saving(%)':>10} {'Trade Mean':>12} {'Iters Mean':>11}"
    )
    print(header)
    print("-" * 80)

    for sigma in sigma_levels:
        results = all_level_results[sigma]
        costs = [r["admm_cost"] for r in results]
        savings = [r["saving_pct"] for r in results]
        trades = [r["total_trade"] for r in results]
        iters_list = [r["admm_iters"] for r in results]
        n_scen = len(results)

        print(
            f"{sigma:>6.0%} {n_scen:>6} "
            f"{np.mean(costs):>12.2f} {np.std(costs):>10.2f} "
            f"{np.mean(savings):>9.2f}% "
            f"{np.mean(trades):>12.2f} {np.mean(iters_list):>11.1f}"
        )
    print("=" * 80)

    return all_level_results


# ======================================================================
# ======================================================================


def plot_uncertainty_distributions(mc_results, sigma_label="", save=True):
    """"""
    print(f"\n[绘图] 生成 Figure 14: 不确定性分布分析 ({sigma_label})...")

    n_scen = len(mc_results)
    if n_scen < 2:
        print("  警告: 场景数不足（< 2），跳过分布图")
        return

    costs = [r["admm_cost"] for r in mc_results]
    savings = [r["saving_pct"] for r in mc_results]
    trades = [r["total_trade"] for r in mc_results]
    mg1_costs = [r["mg_grid_costs"][0] for r in mc_results]
    mg2_costs = [r["mg_grid_costs"][1] for r in mc_results]
    mg3_costs = [r["mg_grid_costs"][2] for r in mc_results]

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ================================================================
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        costs,
        bins=max(10, n_scen // 5),
        color=COLORS_UNC["system"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
        density=True,
        label="Distribution",
    )
    mean_cost = float(np.mean(costs))
    ax1.axvline(
        mean_cost,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: ¥{mean_cost:.0f}",
    )
    p5, p95 = np.percentile(costs, [5, 95])
    ax1.axvspan(
        p5, p95, color="red", alpha=0.08, label=f"90% CI: [{p5:.0f}, {p95:.0f}]"
    )

    ax1.set_xlabel("System Cost (¥)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) System Cost Distribution", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    # ================================================================
    # ================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    violin_data = [mg1_costs, mg2_costs, mg3_costs]
    violin_colors = [COLORS_UNC["mg1"], COLORS_UNC["mg2"], COLORS_UNC["mg3"]]
    mg_labels = ["MG1\n(Industrial)", "MG2\n(Commercial)", "MG3\n(Residential)"]

    parts = ax2.violinplot(
        violin_data, positions=[1, 2, 3], showmeans=True, showextrema=True
    )

    bodies: list = parts.get("bodies", [])  # type: ignore[assignment]
    for idx, pc in enumerate(bodies):
        pc.set_facecolor(violin_colors[idx])
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")

    for partname in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1.5)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(mg_labels)
    ax2.set_ylabel("Grid Purchase Cost (¥)")
    ax2.set_title("(b) Per-Microgrid Cost Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    # ================================================================
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(
        trades,
        bins=max(10, n_scen // 5),
        color=COLORS_UNC["trade"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
        density=True,
    )
    mean_trade = float(np.mean(trades))
    ax3.axvline(
        mean_trade,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_trade:.0f} kWh",
    )
    ax3.set_xlabel("Total P2P Trade Volume (kWh)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) P2P Trade Volume Distribution", fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y", linestyle="--")

    # ================================================================
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(
        savings,
        bins=max(10, n_scen // 5),
        color=COLORS_UNC["saving"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
        density=True,
    )
    mean_saving = float(np.mean(savings))
    ax4.axvline(
        mean_saving,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_saving:.1f}%",
    )
    p5s, p95s = np.percentile(savings, [5, 95])
    ax4.axvspan(
        p5s, p95s, color="red", alpha=0.08, label=f"90% CI: [{p5s:.1f}%, {p95s:.1f}%]"
    )

    ax4.set_xlabel("Cooperation Saving (%)")
    ax4.set_ylabel("Density")
    ax4.set_title("(d) Cooperation Saving Distribution", fontweight="bold")
    ax4.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.suptitle(
        f"Figure 14: Uncertainty Impact Analysis ({sigma_label}, {n_scen} scenarios)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.93, bottom=0.07)

    if save:
        export_figure_data(fig, "figure14_uncertainty_dist")
        plt.savefig("figure14_uncertainty_dist.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure14_uncertainty_dist.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure14_uncertainty_dist.pdf/png")

    plt.close()


def plot_uncertainty_levels(all_level_results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 15: 不确定性水平对比分析...")

    sigma_levels = sorted(all_level_results.keys())
    n_levels = len(sigma_levels)

    cost_means, cost_stds = [], []
    saving_means, saving_stds = [], []
    trade_means, trade_stds = [], []
    iter_means, iter_stds = [], []

    for sigma in sigma_levels:
        results = all_level_results[sigma]
        costs = [r["admm_cost"] for r in results]
        savings = [r["saving_pct"] for r in results]
        trades = [r["total_trade"] for r in results]
        iters_list = [r["admm_iters"] for r in results]

        cost_means.append(np.mean(costs))
        cost_stds.append(np.std(costs) if len(costs) > 1 else 0)
        saving_means.append(np.mean(savings))
        saving_stds.append(np.std(savings) if len(savings) > 1 else 0)
        trade_means.append(np.mean(trades))
        trade_stds.append(np.std(trades) if len(trades) > 1 else 0)
        iter_means.append(np.mean(iters_list))
        iter_stds.append(np.std(iters_list) if len(iters_list) > 1 else 0)

    cost_means = np.array(cost_means)
    cost_stds = np.array(cost_stds)
    saving_means = np.array(saving_means)
    saving_stds = np.array(saving_stds)
    trade_means = np.array(trade_means)
    trade_stds = np.array(trade_stds)
    iter_means = np.array(iter_means)
    iter_stds = np.array(iter_stds)

    sigma_pct = [s * 100 for s in sigma_levels]

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ================================================================
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        sigma_pct,
        cost_means,
        color=COLORS_UNC["system"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="Mean System Cost",
        zorder=5,
    )
    ax1.fill_between(
        sigma_pct,
        cost_means - cost_stds,
        cost_means + cost_stds,
        color=COLORS_UNC["system"],
        alpha=0.15,
        label="±1 Std Dev",
    )
    ax1.fill_between(
        sigma_pct,
        cost_means - 2 * cost_stds,
        cost_means + 2 * cost_stds,
        color=COLORS_UNC["system"],
        alpha=0.06,
        label="±2 Std Dev",
    )

    ax1.set_xlabel("Uncertainty Level σ (%)")
    ax1.set_ylabel("System Cost (¥)")
    ax1.set_title("(a) System Cost vs Uncertainty Level", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        sigma_pct,
        saving_means,
        color=COLORS_UNC["saving"],
        marker="s",
        markersize=8,
        linewidth=2.5,
        label="Mean Saving Rate",
        zorder=5,
    )
    ax2.fill_between(
        sigma_pct,
        saving_means - saving_stds,
        saving_means + saving_stds,
        color=COLORS_UNC["saving"],
        alpha=0.15,
        label="±1 Std Dev",
    )
    ax2.fill_between(
        sigma_pct,
        saving_means - 2 * saving_stds,
        saving_means + 2 * saving_stds,
        color=COLORS_UNC["saving"],
        alpha=0.06,
        label="±2 Std Dev",
    )

    ax2.set_xlabel("Uncertainty Level σ (%)")
    ax2.set_ylabel("Cooperation Saving (%)")
    ax2.set_title("(b) Cooperation Saving vs Uncertainty Level", fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        sigma_pct,
        trade_means,
        color=COLORS_UNC["trade"],
        marker="D",
        markersize=8,
        linewidth=2.5,
        label="Mean Trade Volume",
        zorder=5,
    )
    ax3.fill_between(
        sigma_pct,
        trade_means - trade_stds,
        trade_means + trade_stds,
        color=COLORS_UNC["trade"],
        alpha=0.15,
        label="±1 Std Dev",
    )

    ax3.set_xlabel("Uncertainty Level σ (%)")
    ax3.set_ylabel("Total P2P Trade Volume (kWh)")
    ax3.set_title("(c) P2P Trade Volume vs Uncertainty Level", fontweight="bold")
    ax3.legend(loc="best", framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle="--")

    # ================================================================
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.bar(
        range(n_levels),
        iter_means,
        color=COLORS_UNC["mg1"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
        yerr=iter_stds,
        capsize=4,
        error_kw={"linewidth": 1.5},
    )

    for idx, (im, ist) in enumerate(zip(iter_means, iter_stds)):
        ax4.text(
            idx,
            im + ist + 2,
            f"{im:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax4.set_xticks(range(n_levels))
    ax4.set_xticklabels([f"{s:.0%}" for s in sigma_levels])
    ax4.set_xlabel("Uncertainty Level σ")
    ax4.set_ylabel("ADMM Iterations")
    ax4.set_title("(d) Convergence Difficulty vs Uncertainty Level", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.suptitle(
        "Figure 15: Impact of Uncertainty Level on P2P Trading Performance",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.93, bottom=0.07)

    if save:
        export_figure_data(fig, "figure15_uncertainty_levels")
        plt.savefig("figure15_uncertainty_levels.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure15_uncertainty_levels.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure15_uncertainty_levels.pdf/png")

    plt.close()


def plot_shapley_uncertainty(all_level_results, save=True):
    """"""
    has_shapley = False
    for sigma, results in all_level_results.items():
        for r in results:
            if r.get("shapley_values") is not None:
                has_shapley = True
                break
        if has_shapley:
            break

    if not has_shapley:
        print("  [跳过] Figure 16: 无 Shapley 数据（需设置 run_shapley=True）")
        return

    print("\n[绘图] 生成 Figure 16: Shapley值稳定性分析...")

    sigma_levels = sorted(all_level_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    mg_labels = ["MG1 (Industrial)", "MG2 (Commercial)", "MG3 (Residential)"]
    mg_colors = [COLORS_UNC["shapley1"], COLORS_UNC["shapley2"], COLORS_UNC["shapley3"]]

    for mg_idx in range(3):
        ax = axes[mg_idx]
        data_by_level = []
        tick_labels = []

        for sigma in sigma_levels:
            results = all_level_results[sigma]
            vals = [
                r["shapley_values"][mg_idx]
                for r in results
                if r.get("shapley_values") is not None
            ]
            if vals:
                data_by_level.append(vals)
                tick_labels.append(f"{sigma:.0%}")

        if not data_by_level:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center"
            )
            continue

        bp = ax.boxplot(
            data_by_level,
            patch_artist=True,
            labels=tick_labels,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(mg_colors[mg_idx])
            patch.set_alpha(0.7)

        ax.set_xlabel("Uncertainty Level σ")
        ax.set_ylabel("Shapley Value (¥)")
        ax.set_title(mg_labels[mg_idx], fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.suptitle(
        "Figure 16: Shapley Value Stability Under Uncertainty",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.30)

    if save:
        export_figure_data(fig, "figure16_shapley_uncertainty")
        plt.savefig("figure16_shapley_uncertainty.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure16_shapley_uncertainty.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure16_shapley_uncertainty.pdf/png")

    plt.close()


# ======================================================================
# ======================================================================


def run_and_plot(
    n_scenarios=50,
    sigma_levels=None,
    run_shapley=False,
):
    """"""
    if sigma_levels is None:
        sigma_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    all_level_results = run_uncertainty_levels(
        sigma_levels=sigma_levels,
        n_scenarios=n_scenarios,
        run_shapley=run_shapley,
    )
    export_result_bundle("uncertainty_results", all_level_results)
    audit_summary = {}
    for sigma, scenario_results in all_level_results.items():
        iters = [int(r["admm_iters"]) for r in scenario_results]
        audit_summary[str(sigma)] = {
            "sigma": float(sigma),
            "scenario_count": len(scenario_results),
            "iterations_min": min(iters) if iters else None,
            "iterations_mean": float(np.mean(iters)) if iters else None,
            "iterations_max": max(iters) if iters else None,
            "diverged_count": sum(not bool(r.get("converged", False)) for r in scenario_results),
        }
    export_result_bundle("uncertainty_audit_summary", audit_summary)

    typical_sigma = 0.15
    if (
        typical_sigma in all_level_results
        and len(all_level_results[typical_sigma]) >= 2
    ):
        plot_uncertainty_distributions(
            all_level_results[typical_sigma],
            sigma_label=f"σ={typical_sigma:.0%}",
        )
    else:
        nonzero = [
            s for s in sigma_levels if s > 0 and len(all_level_results.get(s, [])) >= 2
        ]
        if nonzero:
            s_use = nonzero[-1]
            plot_uncertainty_distributions(
                all_level_results[s_use],
                sigma_label=f"σ={s_use:.0%}",
            )

    plot_uncertainty_levels(all_level_results)

    plot_shapley_uncertainty(all_level_results)

    return all_level_results


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    results = run_and_plot(n_scenarios=50, run_shapley=False)
