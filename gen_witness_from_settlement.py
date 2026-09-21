""""""
import json
import os
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from microgrid import DistributionNetwork, Microgrid, get_grid_price

SCALE = 1_000_000
SLOTS = [0, 6, 12, 18]
SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else "session-wp2-test"


def to_field(v):
    """"""
    return round(v * SCALE)


def path_capacity(network, i, j):
    """"""
    node_i = i + 1
    node_j = j + 1
    lo = min(node_i, node_j)
    hi = max(node_i, node_j)
    min_cap = float("inf")
    for k in range(lo, hi):
        min_cap = min(min_cap, network.get_line_capacity(k, k + 1))
    return min_cap


def compute_line_losses(P_global, p_grids, network):
    """"""
    N = len(p_grids)
    T = P_global.shape[2]
    loss_allocation = np.zeros((N, T))
    for line_idx, (from_node, to_node) in enumerate(network.lines):
        for t in range(T):
            mg_contributions = np.zeros(N)
            for mg_idx in range(N):
                mg_node = mg_idx + 1
                contrib = 0.0
                if mg_node > from_node:
                    contrib += p_grids[mg_idx][t]
                for j in range(N):
                    if j == mg_idx:
                        continue
                    node_j = j + 1
                    if mg_node <= from_node and node_j > from_node:
                        contrib += abs(P_global[mg_idx, j, t])
                    elif mg_node > from_node and node_j <= from_node:
                        contrib += abs(P_global[mg_idx, j, t])
                mg_contributions[mg_idx] = abs(contrib)

            downstream_grid = sum(
                p_grids[mg_idx][t]
                for mg_idx in range(N)
                if mg_idx + 1 > from_node
            )
            p2p_cross = 0.0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if (i + 1) <= from_node and (j + 1) > from_node:
                        p2p_cross += P_global[i, j, t]
            net_flow = downstream_grid + p2p_cross
            loss = network.calculate_line_loss(abs(net_flow), from_node, to_node)

            total_contrib = np.sum(mg_contributions)
            if total_contrib > 1e-6 and loss > 1e-9:
                for mg_idx in range(N):
                    share = mg_contributions[mg_idx] / total_contrib
                    loss_allocation[mg_idx, t] += loss * share
    return loss_allocation


def main():
    base = os.path.dirname(__file__)
    settle = json.load(open(os.path.join(base, "settlement.json")))
    N, T24 = settle["n"], settle["t"]
    assert N == 3 and T24 == 24

    mgs = [Microgrid(1, "industrial"), Microgrid(2, "commercial"), Microgrid(3, "residential")]
    network = DistributionNetwork(N)
    T4 = len(SLOTS)

    pg24 = np.array(settle["p_global"])  # [N][N][24] float kW
    p2p24 = np.array(settle["p2p_price"])
    grid_price24 = np.array([get_grid_price(24) for _ in range(N)])
    load24 = np.array([mg.load for mg in mgs])
    pv24 = np.array([mg.pv_generation for mg in mgs])
    wind24 = np.array([mg.wind_generation for mg in mgs])

    def take3d(arr):
        return [arr[i, j, t] for i in range(N) for j in range(N) for t in SLOTS]

    def take2d(arr):
        return [arr[i, t] for i in range(N) for t in SLOTS]

    pg = np.array(take3d(pg24)).reshape(N, N, T4)
    p2p = np.array(take3d(p2p24)).reshape(N, N, T4)
    gp = np.array(take2d(grid_price24)).reshape(N, T4)
    load = np.array(take2d(load24)).reshape(N, T4)
    pv = np.array(take2d(pv24)).reshape(N, T4)
    wind = np.array(take2d(wind24)).reshape(N, T4)

    assert np.allclose(pg, -np.transpose(pg, (1, 0, 2)), atol=1e-6), "p_global 非反对称"

    line_capacity = [path_capacity(network, i, j) for i in range(N) for j in range(N) if i != j]

    gen = pv + wind
    net_trade = pg.sum(axis=1)  # [N][T]
    p_grid = load - gen - net_trade
    for _ in range(2):
        loss_alloc = compute_line_losses(pg, p_grid, network)
        p_grid = load + loss_alloc - gen - net_trade

    def field3(arr):
        return [to_field(v) for v in arr.reshape(-1)]

    def field2(arr):
        return [to_field(v) for v in arr.reshape(-1)]

    f_pg = np.array(field3(pg)).reshape(N, N, T4)
    f_p2p = np.array(field3(p2p)).reshape(N, N, T4)
    f_gp = np.array(field2(gp)).reshape(N, T4)
    f_loss = np.array(field2(loss_alloc)).reshape(N, T4)
    f_load = np.array(field2(load)).reshape(N, T4)
    f_pv = np.array(field2(pv)).reshape(N, T4)
    f_wind = np.array(field2(wind)).reshape(N, T4)
    f_curtail = np.zeros((N, T4), dtype=np.int64)
    f_pch = np.zeros((N, T4), dtype=np.int64)
    f_pdis = np.zeros((N, T4), dtype=np.int64)
    f_net = f_pg.sum(axis=1)
    f_pgrid = f_loss + f_load + f_curtail - (f_pv + f_wind) - (f_pdis - f_pch) - f_net

    lhs = f_load + f_curtail + f_loss
    rhs = (f_pv + f_wind) + (f_pdis - f_pch) + f_pgrid + f_net
    assert np.array_equal(lhs, rhs), "整数域功率平衡不成立"

    witness = {
        "session_id": SESSION_ID,
        "n": N,
        "t": T4,
        "microgrid_ids": ["MG1", "MG2", "MG3"],
        "public": {
            "p_global": f_pg.reshape(-1).tolist(),
            "p2p_price": f_p2p.reshape(-1).tolist(),
            "grid_price": f_gp.reshape(-1).tolist(),
            "loss_alloc": f_loss.reshape(-1).tolist(),
            "line_capacity": [to_field(c) for c in line_capacity],
        },
        "private": {
            "p_grid": f_pgrid.reshape(-1).tolist(),
            "p_ch": f_pch.reshape(-1).tolist(),
            "p_dis": f_pdis.reshape(-1).tolist(),
            "load": f_load.reshape(-1).tolist(),
            "pv": f_pv.reshape(-1).tolist(),
            "wind": f_wind.reshape(-1).tolist(),
            "curtail": f_curtail.reshape(-1).tolist(),
        },
    }
    wpath = Path(base).resolve().parent / "zkp_circuit" / "witness.json"
    wpath.parent.mkdir(parents=True, exist_ok=True)
    wpath.write_text(json.dumps(witness, indent=2), encoding="utf-8")
    print(f"witness 已保存: {wpath}")

    def field_to_float(x):
        return x / SCALE

    import hashlib

    h = hashlib.sha256(SESSION_ID.encode()).digest()[:8]
    session_hash_f = float(int.from_bytes(h, "big") & ((1 << 43) - 1))

    public_input = {
        "sessionHash": session_hash_f,
        "costClaim": 0.0,
        "residualClaim": 0.0,
        "pGlobal": [field_to_float(v) for v in witness["public"]["p_global"]],
        "p2pPrice": [field_to_float(v) for v in witness["public"]["p2p_price"]],
        "gridPrice": [field_to_float(v) for v in witness["public"]["grid_price"]],
        "lossAlloc": [field_to_float(v) for v in witness["public"]["loss_alloc"]],
        "lineCapacity": [field_to_float(v) for v in witness["public"]["line_capacity"]],
        "commitLoad": 0.0,
        "commitPV": 0.0,
        "commitWind": 0.0,
    }

    payload = {
        "sessionID": SESSION_ID,
        "microgridIDs": ["MG1", "MG2", "MG3"],
        "pGlobal": [[[field_to_float(f_pg[i, j, t]) for t in range(T4)] for j in range(N)] for i in range(N)],
        "p2pPrice": [[[field_to_float(f_p2p[i, j, t]) for t in range(T4)] for j in range(N)] for i in range(N)],
        "publicInput": public_input,
    }
    ppath = Path(base).resolve().parent / "zkp_circuit" / "settle_payload.json"
    ppath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"settle payload 已保存: {ppath}")

    net_e = np.zeros((N, N))
    net_a = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                net_e[i, j] = f_pg[i, j].sum() / SCALE
                net_a[i, j] = (f_pg[i, j] * f_p2p[i, j]).sum() / (SCALE * SCALE)
    for i in range(N):
        for j in range(i + 1, N):
            print(f"  MG{i+1}<->MG{j+1}: 净能量 {net_e[i,j]:+.2f} kWh, 净金额 {net_a[i,j]:+.2f} 元")
    return 0


if __name__ == "__main__":
    sys.exit(main())
