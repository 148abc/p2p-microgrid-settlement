# P2P Microgrid Settlement Experiments

Experiment code and archived results for the paper *"Provable Settlement of
Peer-to-Peer Energy Trading in Networked Microgrids with Anchored
Zero-Knowledge Dispatch Proofs"*.

The repository contains two parts:

- `./` (root) — the distributed market clearing stack: a settlement-aware
  ADMM (SA-ADMM) solver for peer-to-peer energy trading among networked
  microgrids, a centralized full-cost benchmark, a standalone (no-trade)
  baseline, the real-profile data loader, and the experiment drivers used for
  every table and figure of the paper.
- `zkp_circuit/` — the fixed-point Groth16 circuit family (Go, gnark, BN254)
  that proves each microgrid's dispatch against the market equations, plus
  the measured circuit benchmarks reported in the paper.

## Contents

```
admm_solver.py              SA-ADMM consensus clearing solver
centralized_solver.py       centralized full-cost benchmark (eq. 12 of the paper)
standalone_baseline.py      no-trade baseline used as the savings denominator
nogurobi_admm_solver.py     Gurobi-free cvxpy/Clarabel path (mutex-inactive regimes)
microgrid.py                microgrid and radial-network model
load_real_data.py           loader for the public 20-microgrid dataset
experiment_data.py          synthetic three-type fleet generator
validate.py                 synthetic-fleet validation entry point
experiment_*.py             synthetic-fleet studies (scalability/sensitivity/uncertainty/seeds)
exp_*.py                    experiment drivers for the real-profile results
chen2024_baseline.py        re-implementation of the A-ADMM baseline (Chen et al., 2024)
e2e_zkp_v2.py               end-to-end clear -> repair -> prove -> settle driver
gen_witness_from_settlement.py  witness extraction from a settled state
witness_generator_v2.py     per-microgrid witness generator (integer pre-check)
bench_zkp.py                Go benchmark launcher
experiment_data_real/       archived JSON results cited in the paper (incl. profiles.json)
zkp_circuit/                Go circuit sources, tests, and benchmark JSONs
```

## Requirements

- Python 3.10+, `numpy`, `matplotlib`, `cvxpy` (and `gurobipy` for the full
  MILP path; a Gurobi license is required by Gurobi, not by us — the
  `nogurobi_*` scripts run without it via cvxpy/Clarabel).
- Go 1.21+ with [gnark](https://github.com/Consensys/gnark) for the ZKP part
  (pulled automatically by `go.mod`).

```bash
pip install -r requirements.txt
python validate.py                 # synthetic three-type fleet
python exp_deployed_market_20260917.py   # real-profile deployed configuration
cd zkp_circuit && go test ./...    # circuit tests and benchmarks
```

## Data

`experiment_data_real/profiles.json` carries the per-microgrid profiles used
by the real-profile study. The underlying source data is the public
20-microgrid dataset released under CC BY 4.0 with the study of Ning et al.
(2026); the underlying load/PV/wind series originate from AEMO NEMWEB market
data, scaled to each microgrid's rated capacity.

## Note

Tariff values, feeder-model conventions, and sizing rules are declared
assumptions of the market design and are documented in the paper. The
timings in the archived JSONs are wall-clock measurements of the original
testbed sessions and are session-dependent; re-running on other hardware
will give different absolute times.
