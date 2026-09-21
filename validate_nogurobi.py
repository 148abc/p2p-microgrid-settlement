"""Small smoke/regression check for the Gurobi-free ADMM backend."""

import numpy as np

from microgrid import DistributionNetwork, Microgrid
from nogurobi_admm_solver import NoGurobiADMMSolver


def main():
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

    assert solver.final_results is not None
    assert np.allclose(solver.P_global + np.swapaxes(solver.P_global, 0, 1), 0.0, atol=1e-4)
    assert solver.get_system_grid_cost() > 0.0

    for i in range(solver.N):
        result = solver.get_local_result(i)
        assert np.all(result["P_grid"] >= -1e-4)
        assert np.max(np.minimum(result["P_ch"], result["P_dis"])) < 1e-2

    print("NO-GUROBI VALIDATION PASSED")
    print(f"backend={solver.backend_name}")
    print(f"converged={solver.converged}")
    print(f"iterations={len(solver.history['primal_residual'])}")
    print(f"system_cost={solver.get_system_grid_cost():.6f}")
    print(f"total_loss={float(np.sum(solver.line_loss)):.6f}")
    print(f"max_mutex_violation={solver.max_mutex_violation:.9f}")


if __name__ == "__main__":
    main()
