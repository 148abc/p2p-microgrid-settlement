""""""
import numpy as np

from admm_solver import ADMMSolver
from microgrid import DistributionNetwork, Microgrid
from shapley_allocation import calculate_coalition_cost


def _compute_line_losses_reference(solver):
    """"""
    network = solver.network
    N = solver.N
    T = solver.T
    P_global = solver.P_global

    p_grids = []
    for i in range(N):
        result = solver.get_local_result(i)
        p_grids.append(result["P_grid"])

    total_loss = 0.0
    per_line_loss = []

    for line_idx, (from_node, to_node) in enumerate(network.lines):
        R, _ = network.line_impedance[(from_node, to_node)]
        V = 10.0  # kV
        line_loss_total = 0.0

        for t in range(T):
            downstream_grid = 0.0
            for mg_idx in range(N):
                mg_node = mg_idx + 1
                if mg_node > from_node:
                    downstream_grid += p_grids[mg_idx][t]

            p2p_cross = 0.0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    node_i = i + 1
                    node_j = j + 1
                    if node_i <= from_node and node_j > from_node:
                        p2p_cross += P_global[i, j, t]

            net_flow = downstream_grid + p2p_cross
            loss = (abs(net_flow) / V) ** 2 * R / 1000  # kW
            line_loss_total += loss

        per_line_loss.append(line_loss_total)
        total_loss += line_loss_total

    return total_loss, per_line_loss


def main():
    print("=" * 70)
    print("  VALIDATION SCRIPT - Checking all fixes")
    print("=" * 70)

    mg1 = Microgrid(1, "industrial")
    mg2 = Microgrid(2, "commercial")
    mg3 = Microgrid(3, "residential")
    microgrids = [mg1, mg2, mg3]

    all_pass = True

    # ==================================================================
    # TEST 1: DistributionNetwork dynamic topology
    # ==================================================================
    print("\n[TEST 1] DistributionNetwork dynamic topology")
    print("-" * 70)
    for n_mg in [1, 2, 3, 4]:
        net = DistributionNetwork(n_mg)
        expected_lines = n_mg
        actual_lines = len(net.lines)
        ok = actual_lines == expected_lines
        status = "PASS" if ok else "FAIL"
        print(
            f"  num_microgrids={n_mg}: lines={actual_lines} (expected {expected_lines}) [{status}]"
        )
        if not ok:
            all_pass = False
        # Check no line references nodes beyond num_nodes
        for f, t in net.lines:
            if f >= net.num_nodes or t >= net.num_nodes:
                print(f"    FAIL: line ({f},{t}) references node >= {net.num_nodes}")
                all_pass = False

    # ==================================================================
    # TEST 2: Wind generation differs per mg_id
    # ==================================================================
    print("\n[TEST 2] Wind generation uses different seeds per mg_id")
    print("-" * 70)
    mg_a = Microgrid(1, "industrial")
    mg_b = Microgrid(2, "industrial")  # same type, different id
    wind_same = np.allclose(mg_a.wind_generation, mg_b.wind_generation)
    status = "PASS" if not wind_same else "FAIL"
    print(f"  MG1 vs MG2 (same type, diff id) wind identical: {wind_same} [{status}]")
    if wind_same:
        all_pass = False

    # ==================================================================
    # TEST 3: ADMM solve + convergence
    # ==================================================================
    print("\n[TEST 3] ADMM solve and convergence (Boyd 2011)")
    print("-" * 70)
    network = DistributionNetwork(len(microgrids))
    solver = ADMMSolver(
        microgrids, rho_init=0.1, max_iter=500, tol=1e-3, adaptive=True, network=network
    )
    _results = solver.solve()

    n_iters = len(solver.history["primal_residual"])
    final_primal = solver.history["primal_residual"][-1]
    final_dual = solver.history["dual_residual"][-1]
    grid_cost = solver.get_system_grid_cost()

    print(f"  Iterations: {n_iters}")
    print(f"  Final primal residual: {final_primal:.6f}")
    print(f"  Final dual residual:   {final_dual:.6f}")
    print(f"  System grid cost:      {grid_cost:.2f}")

    converged = n_iters < 500
    status = "PASS" if converged else "FAIL"
    print(f"  Converged before max_iter: {converged} [{status}]")
    if not converged:
        all_pass = False

    # ==================================================================
    # TEST 4: P2P net cost is ~0 (symmetric pricing)
    # ==================================================================
    print("\n[TEST 4] P2P net cost balance (symmetric pricing)")
    print("-" * 70)
    total_p2p = 0.0
    total_loss_cost = 0.0
    for i in range(3):
        bd = solver.get_mg_cost_breakdown(i)
        total_p2p += bd["p2p_net_cost"]
        total_loss_cost += bd["loss_cost"]
        print(
            f"  MG{i + 1}: grid={bd['grid_cost']:>9.2f}, "
            f"loss={bd['loss_cost']:>7.2f}, "
            f"p2p_buy={bd['p2p_buy_cost']:>8.2f}, "
            f"p2p_sell={bd['p2p_sell_revenue']:>8.2f}, "
            f"p2p_net={bd['p2p_net_cost']:>9.2f}, "
            f"total={bd['total_cost']:>9.2f}"
        )

    p2p_ok = abs(total_p2p) < 0.01
    status = "PASS" if p2p_ok else "FAIL"
    print(f"\n  System P2P net cost: {total_p2p:.6f} [{status}]")
    print(f"  System loss cost:   {total_loss_cost:.2f}")
    if not p2p_ok:
        all_pass = False

    # ==================================================================
    # TEST 5: get_local_result does NOT modify P_local (no side effect)
    # ==================================================================
    print("\n[TEST 5] get_local_result has no side effects")
    print("-" * 70)
    p_local_before = solver.P_local.copy()
    for i in range(3):
        _ = solver.get_local_result(i)
    p_local_after = solver.P_local.copy()
    side_effect_free = np.allclose(p_local_before, p_local_after)
    status = "PASS" if side_effect_free else "FAIL"
    print(f"  P_local unchanged after get_local_result: {side_effect_free} [{status}]")
    if not side_effect_free:
        all_pass = False

    # ==================================================================
    # TEST 6: Charge/discharge mutual exclusion
    # ==================================================================
    print("\n[TEST 6] Charge/discharge mutual exclusion")
    print("-" * 70)
    _mutex_ok = True
    for i in range(3):
        r = solver.get_local_result(i)
        simultaneous = np.sum((r["P_ch"] > 1e-3) & (r["P_dis"] > 1e-3))
        if simultaneous > 0:
            print(
                f"  MG{i + 1}: {simultaneous} time slots with simultaneous charge+discharge [FAIL]"
            )
            _mutex_ok = False
            all_pass = False
        else:
            print(f"  MG{i + 1}: No simultaneous charge+discharge [PASS]")

    # ==================================================================
    # ==================================================================
    # TEST 7: Grand coalition achieves cost reduction
    # ==================================================================
    print("\n[TEST 7] Grand coalition achieves cost reduction")
    print("-" * 70)
    from standalone_baseline import standalone_total

    gc_cost = calculate_coalition_cost(microgrids, (0, 1, 2))
    ind_total = standalone_total(microgrids)
    reduction = ind_total - gc_cost
    pct = reduction / ind_total * 100
    ok = reduction > 0
    status = "PASS" if ok else "FAIL"
    print(f"  Individual total: {ind_total:.2f}")
    print(f"  Grand coalition:  {gc_cost:.2f}")
    print(f"  Reduction:        {reduction:.2f} ({pct:.1f}%) [{status}]")
    if not ok:
        all_pass = False

    # ==================================================================
    # TEST 11: Line loss calculation correctness
    # ==================================================================
    print("\n[TEST 11] Line loss calculation correctness")
    print("-" * 70)

    total_loss_solver = np.sum(solver.line_loss)
    loss_positive = total_loss_solver > 0.1
    status = "PASS" if loss_positive else "FAIL"
    print(f"  Total line loss: {total_loss_solver:.4f} kWh (>0.1?) [{status}]")
    if not loss_positive:
        all_pass = False

    line01_loss = np.sum(solver.line_loss[0, :])
    line01_ok = line01_loss > 0.01
    status = "PASS" if line01_ok else "FAIL"
    print(f"  Line (0,1) loss: {line01_loss:.4f} kWh (>0?) [{status}]")
    if not line01_ok:
        all_pass = False

    ref_total, ref_per_line = _compute_line_losses_reference(solver)
    match = abs(total_loss_solver - ref_total) < 0.01
    status = "PASS" if match else "FAIL"
    print(
        f"  Solver loss: {total_loss_solver:.4f}, Reference: {ref_total:.4f}, "
        f"diff={abs(total_loss_solver - ref_total):.6f} [{status}]"
    )
    if not match:
        all_pass = False

    for line_idx, (f, t) in enumerate(solver.network.lines):
        solver_line = np.sum(solver.line_loss[line_idx, :])
        ref_line = ref_per_line[line_idx]
        line_match = abs(solver_line - ref_line) < 0.01
        status_l = "OK" if line_match else "MISMATCH"
        print(
            f"    Line ({f},{t}): solver={solver_line:.4f}, "
            f"ref={ref_line:.4f} [{status_l}]"
        )
        if not line_match:
            all_pass = False

    total_alloc = np.sum(solver.loss_allocation)
    alloc_match = abs(total_alloc - total_loss_solver) < 0.01
    status = "PASS" if alloc_match else "FAIL"
    print(
        f"  Loss allocation sum: {total_alloc:.4f}, "
        f"total loss: {total_loss_solver:.4f} [{status}]"
    )
    if not alloc_match:
        all_pass = False

    sys_cost = solver.get_system_grid_cost()
    grid_only = sum(
        np.sum(solver.microgrids[i].grid_price * solver.get_local_result(i)["P_grid"])
        for i in range(3)
    )
    loss_cost_in_system = sys_cost - grid_only
    loss_cost_positive = loss_cost_in_system > 0.1
    status = "PASS" if loss_cost_positive else "FAIL"
    print(
        f"  System cost: {sys_cost:.2f}, grid only: {grid_only:.2f}, "
        f"loss cost: {loss_cost_in_system:.2f} [{status}]"
    )
    if not loss_cost_positive:
        all_pass = False

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED - please check above")
    print("=" * 70)


if __name__ == "__main__":
    main()
