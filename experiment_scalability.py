""""""
import sys
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from admm_solver import ADMMSolver
from centralized_solver import CentralizedSolver
from microgrid import DistributionNetwork, Microgrid, build_network
from standalone_baseline import single_mg_cost, standalone_total
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

COLORS_SCALE = {
    "iters": "#BC4B51",
    "time": "#2E4057",
    "cost_admm": "#5FA8D3",
    "cost_cent": "#52A675",
    "saving": "#F4A259",
    "gap": "#8B5A8C",
}

MG_TYPE_CYCLE = ["industrial", "commercial", "residential"]


def create_microgrids(n):
    """"""
    microgrids = []
    for i in range(n):
        mg_type = MG_TYPE_CYCLE[i % len(MG_TYPE_CYCLE)]
        mg = Microgrid(i + 1, mg_type)
        microgrids.append(mg)
    return microgrids


def run_single_scale(n, max_iter=500, tol=1e-3, run_centralized=True, verbose=False):
    """"""
    print(f"\n{'=' * 60}")
    print(f"  N = {n} microgrids")
    print(f"{'=' * 60}")

    microgrids = create_microgrids(n)
    network = DistributionNetwork(n)

    print(f"  计算 {n} 个微网的单独运行成本...")
    individual_costs = []
    for i in range(n):
        cost = single_mg_cost(microgrids[i])
        individual_costs.append(cost)
    total_individual = sum(individual_costs)
    print(f"  单独运行总成本: {total_individual:.2f} $")

    print(f"  运行 ADMM 分布式优化...")
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

    t_admm_start = time.time()
    try:
        solver.solve()
    finally:
        if not verbose:
            sys.stdout = old_stdout
    t_admm = time.time() - t_admm_start

    admm_cost = solver.get_system_grid_cost()
    admm_iters = len(solver.history["primal_residual"])
    admm_primal_hist = solver.history["primal_residual"]
    admm_dual_hist = solver.history["dual_residual"]
    admm_total_loss = np.sum(solver.line_loss)

    saving = total_individual - admm_cost
    saving_pct = saving / total_individual * 100 if total_individual > 1e-6 else 0.0

    print(f"  ADMM 系统成本: {admm_cost:.2f} $ ({admm_iters} iters, {t_admm:.2f}s)")
    print(f"  合作节省: {saving:.2f} $ ({saving_pct:.1f}%)")

    result = {
        "n": n,
        "individual_costs": individual_costs,
        "total_individual": total_individual,
        "admm_cost": admm_cost,
        "admm_time": t_admm,
        "admm_iters": admm_iters,
        "admm_primal_hist": admm_primal_hist,
        "admm_dual_hist": admm_dual_hist,
        "admm_total_loss": admm_total_loss,
        "saving": saving,
        "saving_pct": saving_pct,
        "centralized_cost": None,
        "centralized_time": None,
        "optimality_gap": None,
    }

    if run_centralized:
        print(f"  运行集中式优化 (Benchmark)...")
        cent = CentralizedSolver(microgrids, network)

        old_stdout = sys.stdout
        if not verbose:
            sys.stdout = StringIO()
        try:
            cent.solve(verbose=verbose)
        finally:
            if not verbose:
                sys.stdout = old_stdout

        cent_cost = cent.get_system_grid_cost()
        cent_time = cent.solve_time

        if cent_cost > 1e-6:
            gap = (admm_cost - cent_cost) / cent_cost * 100.0
        else:
            gap = 0.0

        result["centralized_cost"] = cent_cost
        result["centralized_time"] = cent_time
        result["optimality_gap"] = gap

        print(f"  集中式成本: {cent_cost:.2f} $ ({cent_time:.2f}s)")
        print(f"  最优间隙: {gap:.4f}%")

    return result


def run_scalability_experiment(
    n_values=None,
    max_iter=500,
    tol=1e-3,
    run_centralized_threshold=12,
):
    """"""
    if n_values is None:
        n_values = [3, 5, 8, 10]

    print("=" * 70)
    print("  实验1: 可扩展性分析 (Scalability Analysis)")
    print(f"  测试规模: N = {n_values}")
    print("=" * 70)

    all_results = []
    for n in n_values:
        run_cent = n <= run_centralized_threshold
        result = run_single_scale(
            n,
            max_iter=max_iter,
            tol=tol,
            run_centralized=run_cent,
            verbose=False,
        )
        all_results.append(result)

    print("\n" + "=" * 90)
    print("  可扩展性实验汇总")
    print("=" * 90)
    header = (
        f"{'N':>4} {'ADMM Cost':>12} {'Cent Cost':>12} {'Gap(%)':>10} "
        f"{'Saving(%)':>10} {'ADMM Iters':>11} {'ADMM Time':>10} {'Cent Time':>10}"
    )
    print(header)
    print("-" * 90)
    for r in all_results:
        cent_str = (
            f"{r['centralized_cost']:.2f}"
            if r["centralized_cost"] is not None
            else "N/A"
        )
        gap_str = (
            f"{r['optimality_gap']:.4f}" if r["optimality_gap"] is not None else "N/A"
        )
        cent_t_str = (
            f"{r['centralized_time']:.2f}s"
            if r["centralized_time"] is not None
            else "N/A"
        )
        print(
            f"{r['n']:>4} {r['admm_cost']:>12.2f} {cent_str:>12} {gap_str:>10} "
            f"{r['saving_pct']:>9.1f}% {r['admm_iters']:>11} "
            f"{r['admm_time']:>9.2f}s {cent_t_str:>10}"
        )
    print("=" * 90)

    return all_results


def plot_scalability_results(all_results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 12: 可扩展性分析...")

    n_values = [r["n"] for r in all_results]
    admm_iters = [r["admm_iters"] for r in all_results]
    admm_times = [r["admm_time"] for r in all_results]
    admm_costs = [r["admm_cost"] for r in all_results]
    saving_pcts = [r["saving_pct"] for r in all_results]

    has_cent = [r["centralized_cost"] is not None for r in all_results]
    cent_costs = [
        r["centralized_cost"] if r["centralized_cost"] is not None else np.nan
        for r in all_results
    ]
    cent_times = [
        r["centralized_time"] if r["centralized_time"] is not None else np.nan
        for r in all_results
    ]
    gaps = [
        r["optimality_gap"] if r["optimality_gap"] is not None else np.nan
        for r in all_results
    ]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ================================================================
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(n_values))
    w = 0.35

    bars1 = ax1.bar(
        x - w / 2,
        admm_iters,
        w,
        label="ADMM Iterations",
        color=COLORS_SCALE["iters"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_xlabel("Number of Microgrids (N)")
    ax1.set_ylabel("Iterations", color=COLORS_SCALE["iters"])
    ax1.tick_params(axis="y", labelcolor=COLORS_SCALE["iters"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_values])
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    for bar, val in zip(bars1, admm_iters):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS_SCALE["iters"],
        )

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        x,
        admm_times,
        color=COLORS_SCALE["time"],
        marker="s",
        markersize=8,
        linewidth=2.5,
        label="ADMM Time",
        zorder=10,
    )
    if any(has_cent):
        cent_t_valid = [(xi, ct) for xi, ct, hc in zip(x, cent_times, has_cent) if hc]
        if cent_t_valid:
            cx, ct = zip(*cent_t_valid)
            ax1_twin.plot(
                list(cx),
                list(ct),
                color=COLORS_SCALE["cost_cent"],
                marker="D",
                markersize=8,
                linewidth=2.5,
                linestyle="--",
                label="Centralized Time",
                zorder=10,
            )

    ax1_twin.set_ylabel("Computation Time (s)", color=COLORS_SCALE["time"])
    ax1_twin.tick_params(axis="y", labelcolor=COLORS_SCALE["time"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=9
    )

    ax1.set_title("(a) Convergence Iterations & Computation Time", fontweight="bold")

    # ================================================================
    # ================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.bar(
        x - w / 2,
        admm_costs,
        w,
        label="ADMM System Cost",
        color=COLORS_SCALE["cost_admm"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    if any(has_cent):
        cent_c_vals = [cc if hc else 0 for cc, hc in zip(cent_costs, has_cent)]
        ax2.bar(
            x + w / 2,
            cent_c_vals,
            w,
            label="Centralized Cost",
            color=COLORS_SCALE["cost_cent"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

    ax2.set_xlabel("Number of Microgrids (N)")
    ax2.set_ylabel("System Cost ($)", color=COLORS_SCALE["cost_admm"])
    ax2.tick_params(axis="y", labelcolor=COLORS_SCALE["cost_admm"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in n_values])
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        x,
        saving_pcts,
        color=COLORS_SCALE["saving"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="Saving (%)",
        zorder=10,
    )
    ax2_twin.set_ylabel("Cost Saving (%)", color=COLORS_SCALE["saving"])
    ax2_twin.tick_params(axis="y", labelcolor=COLORS_SCALE["saving"])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9, fontsize=9
    )

    ax2.set_title("(b) System Cost & Cooperation Saving", fontweight="bold")

    # ================================================================
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    valid_gaps = [
        (n, g) for n, g, hc in zip(n_values, gaps, has_cent) if hc and not np.isnan(g)
    ]

    if valid_gaps:
        gap_ns, gap_vals = zip(*valid_gaps)

        ax3.bar(
            range(len(gap_ns)),
            gap_vals,
            color=COLORS_SCALE["gap"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
            width=0.5,
        )

        for idx, (gn, gv) in enumerate(zip(gap_ns, gap_vals)):
            ax3.text(
                idx,
                gv + 0.02,
                f"{gv:.3f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=COLORS_SCALE["gap"],
            )

        ax3.set_xticks(range(len(gap_ns)))
        ax3.set_xticklabels([f"N={n}" for n in gap_ns])

        ax3.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax3.text(
            len(gap_ns) - 0.5,
            1.05,
            "1% threshold",
            fontsize=9,
            color="red",
            ha="right",
            alpha=0.7,
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "No centralized\nresults available",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    ax3.set_xlabel("Number of Microgrids (N)")
    ax3.set_ylabel("Optimality Gap (%)")
    ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax3.set_title("(c) Optimality Gap (ADMM vs Centralized)", fontweight="bold")

    y_low, y_high = ax3.get_ylim()
    ax3.set_ylim(0, max(y_high, 1.5))

    # ================================================================
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    cmap = plt.cm.get_cmap("tab10")
    for idx, r in enumerate(all_results):
        primal = np.array(r["admm_primal_hist"])
        iters = np.arange(1, len(primal) + 1)
        color = cmap(idx / max(len(all_results) - 1, 1))
        ax4.plot(
            iters,
            primal,
            color=color,
            linewidth=2,
            label=f"N={r['n']} ({r['admm_iters']} iters)",
            alpha=0.9,
        )

    ax4.axhline(
        y=1e-3,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    ax4.text(5, 1.5e-3, "Tolerance", fontsize=8, color="gray")

    ax4.set_yscale("log")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Primal Residual (log scale)")
    ax4.grid(True, which="both", alpha=0.3, linestyle="--")
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax4.set_title("(d) Primal Residual Convergence by Scale", fontweight="bold")

    # ================================================================
    # ================================================================
    plt.suptitle(
        "Figure 12: Scalability Analysis of ADMM-based P2P Trading",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.subplots_adjust(top=0.93, bottom=0.07)

    if save:
        export_figure_data(fig, "figure12_scalability")
        plt.savefig("figure12_scalability.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure12_scalability.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure12_scalability.pdf/png")

    plt.close()


def plot_per_mg_cost_breakdown(all_results, save=True):
    """"""
    print("\n[绘图] 生成 Figure 13: 各类型微网平均节省 vs 规模...")

    fig, ax = plt.subplots(figsize=(10, 6))

    type_savings = {t: [] for t in MG_TYPE_CYCLE}
    n_labels = []

    for r in all_results:
        n = r["n"]
        n_labels.append(f"N={n}")
        type_sums = {t: {"individual": 0.0, "count": 0} for t in MG_TYPE_CYCLE}
        for i in range(n):
            mg_type = MG_TYPE_CYCLE[i % len(MG_TYPE_CYCLE)]
            type_sums[mg_type]["individual"] += r["individual_costs"][i]
            type_sums[mg_type]["count"] += 1

        total_ind = r["total_individual"]
        admm_cost = r["admm_cost"]

        for t in MG_TYPE_CYCLE:
            if type_sums[t]["count"] > 0:
                ind = type_sums[t]["individual"]
                share = ind / total_ind * admm_cost if total_ind > 1e-6 else 0
                avg_saving_pct = (ind - share) / ind * 100 if ind > 1e-6 else 0
                type_savings[t].append(avg_saving_pct)
            else:
                type_savings[t].append(0.0)

    x = np.arange(len(n_labels))
    w = 0.25

    type_colors = {
        "industrial": "#BC4B51",
        "commercial": "#5FA8D3",
        "residential": "#52A675",
    }

    for idx, mg_type in enumerate(MG_TYPE_CYCLE):
        offset = (idx - 1) * w
        vals = type_savings[mg_type]
        bars = ax.bar(
            x + offset,
            vals,
            w,
            label=mg_type.capitalize(),
            color=type_colors[mg_type],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Number of Microgrids (N)", fontsize=12)
    ax.set_ylabel("Average Cost Saving (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    plt.suptitle(
        "Figure 13: Average Cost Saving by Microgrid Type at Different Scales",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.subplots_adjust(top=0.92, bottom=0.10)

    if save:
        export_figure_data(fig, "figure13_scalability_by_type")
        plt.savefig("figure13_scalability_by_type.pdf", dpi=300, bbox_inches="tight")
        plt.savefig("figure13_scalability_by_type.png", dpi=300, bbox_inches="tight")
        print("  已保存: figure13_scalability_by_type.pdf/png")

    plt.close()


def run_and_plot(n_values=None):
    """"""
    if n_values is None:
        n_values = [3, 5, 8, 10]

    all_results = run_scalability_experiment(n_values=n_values)
    export_result_bundle("scalability_results", all_results)
    plot_scalability_results(all_results)
    plot_per_mg_cost_breakdown(all_results)

    return all_results




def run_real_data_scalability(sizes=(3, 5, 10, 15, 20), day=15, verbose=False):
    """"""
    from load_real_data import load_experiment_microgrids

    print("\n" + "=" * 80)
    print(f"真实数据扩展性实验 (day={day}, N={list(sizes)}, 自然顺序)")
    print("=" * 80)

    raw = load_experiment_microgrids(day=day)
    results = {}

    for n in sizes:
        microgrids = [
            Microgrid(
                item["mg_id"],
                None,
                real_data=item["data"],
                soc_bounds=item["soc_bounds"],
            )
            for item in raw[:n]
        ]
        network = build_network(microgrids)
        solver = ADMMSolver(
            microgrids,
            rho_init=0.01,
            max_iter=1000,
            tol=1e-3,
            adaptive=True,
            network=network,
        )

        old_stdout = sys.stdout
        if not verbose:
            sys.stdout = StringIO()
        t0 = time.time()
        try:
            solver.solve()
        finally:
            if not verbose:
                sys.stdout = old_stdout
        elapsed = time.time() - t0

        r = {
            "iterations": len(solver.history["primal_residual"]),
            "time": elapsed,
            "system_cost": solver.get_system_grid_cost(),
            "converged": solver.converged,
            "total_loss": float(np.sum(solver.line_loss)),
            "final_primal_residual": solver.final_primal_residual,
        }

        individual_total = standalone_total(microgrids)
        r["individual_cost"] = individual_total
        r["saving"] = individual_total - r["system_cost"]
        r["saving_pct"] = (
            r["saving"] / individual_total * 100 if individual_total > 1e-6 else 0.0
        )
        results[n] = r

        print(
            f"  N={n:>3}: iters={r['iterations']:>4}  time={r['time']:7.2f}s  "
            f"cost={r['system_cost']:>12.2f}  独立={individual_total:>12.2f}  "
            f"节省={r['saving_pct']:>5.1f}%  converged={r['converged']}"
        )

    return results


# ======================================================================
# ======================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        run_real_data_scalability()
    else:
        results = run_and_plot()
