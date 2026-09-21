""""""
import hashlib
import json
import os

from pathlib import Path

import numpy as np

SCALE = 1_000_000
ETA_NUM, ETA_DEN = 19, 20
CIRCUIT_VERSION = 2
TRADE_SLOTS = 19
SOC_BAND_WIDEN_MAX_F = 2000


def session_hash_field(session_id: str) -> int:
    """"""
    h = hashlib.sha256(session_id.encode()).digest()
    v = int.from_bytes(h[:8], "big") & ((1 << 43) - 1)
    return int(float(v) * SCALE)


def _f(x) -> int:
    return int(round(float(x) * SCALE))


def _f_floor(x) -> int:
    """"""
    return int(np.floor(float(x) * SCALE))


def _f_ceil(x) -> int:
    """"""
    return int(np.ceil(float(x) * SCALE))


def build_witness_v2(solver, mg_idx: int, session_id: str) -> dict:
    """"""
    N, T = solver.N, solver.T
    i = mg_idx
    mg = solver.microgrids[i]
    res = solver.get_local_result(i)

    load_f = [_f(x) for x in mg.load]
    pv_f = [_f(x) for x in mg.pv_generation]
    wind_f = [_f(x) for x in mg.wind_generation]
    p_export_f = [_f(x) for x in res["P_export"]]
    p_dg_f = [_f(x) for x in res["P_dg"]]
    p_grid_src = [_f(x) for x in res["P_grid"]]

    p_ch_f, p_dis_f, z_f = [], [], []
    for t in range(T):
        ch, dis = _f(res["P_ch"][t]), _f(res["P_dis"][t])
        if ch > 0 and dis > 0:
            raise ValueError(
                f"MG{i} t{t}: 同槽同时充放 (ch={ch}, dis={dis}), 互斥位无法一致")
        if ch > 0:
            ch -= ch % 20
            if ch == 0:
                ch, z = 0, 0
            else:
                z = 1
        elif dis > 0:
            dis -= dis % 19
            if dis == 0:
                dis, z = 0, 0
            else:
                z = 0
        else:
            z = 0
        p_ch_f.append(ch)
        p_dis_f.append(dis)
        z_f.append(z)

    if N > TRADE_SLOTS + 1:
        raise ValueError(
            f"MG{i}: 联盟规模 N={N} 超过语句定长市场规模 {TRADE_SLOTS + 1}")
    trade_rows = [[0] * T for _ in range(TRADE_SLOTS)]
    for j in range(N):
        if j == i:
            continue
        slot = j if j < i else j - 1
        for t in range(T):
            trade_rows[slot][t] = _f(solver.P_global[i, j, t])
    net_trade_f = [sum(trade_rows[s][t] for s in range(TRADE_SLOTS))
                   for t in range(T)]

    loss_f = [_f(x) for x in solver.loss_allocation[i, :]]
    grid_price_f = [_f(x) for x in mg.grid_price]
    fit_price_f = [_f(x) for x in mg.fit_price]

    curtail_f = [0] * T
    grid_cap_f = _f(mg.p_grid_max)
    ess_power_f = _f(mg.battery_power)
    wind_base = list(wind_f)

    def resolve_grid(first):
        curtail_f[:] = [0] * T
        wind_f[:] = wind_base
        """"""
        out = []
        for t in range(T):
            gen = pv_f[t] + wind_f[t]
            v = (load_f[t] + loss_f[t] + p_export_f[t]
                 - gen - (p_dis_f[t] - p_ch_f[t]) - net_trade_f[t] - p_dg_f[t])
            if first and abs(v - p_grid_src[t]) > 50_000:
                raise ValueError(
                    f"MG{i} t{t}: 反推 p_grid 与调度偏差 {v - p_grid_src[t]} 域单元 "
                    f"(>{50_000 / SCALE:g} kW), 市场模型与电路语句不一致")
            if v < 0:
                curtail_f[t] += -v
                v = 0
            if v > grid_cap_f:
                wind_f[t] += v - grid_cap_f
                v = grid_cap_f
            out.append(v)
        return out

    p_grid_f = resolve_grid(first=True)

    soc0_f = (_f(mg.soc_lb[0]) + _f(mg.soc_ub[0])) // 2
    soc_f = [soc0_f]
    for t in range(T):
        val = (int(ETA_NUM) * int(ETA_DEN) * soc_f[t]
               + int(ETA_NUM) ** 2 * p_ch_f[t]
               - int(ETA_DEN) ** 2 * p_dis_f[t])
        if val % (int(ETA_NUM) * int(ETA_DEN)) != 0:
            raise ValueError(f"MG{i} t{t}: SOC 递归不可整除 (ch/dis 量化失效)")
        soc_f.append(val // (int(ETA_NUM) * int(ETA_DEN)))
    delta = soc_f[T] - soc0_f
    if delta != 0:
        m_sol = n_sol = None
        for m in range(0, 200):
            num = 19 * m + delta
            if num >= 0 and num % 20 == 0:
                m_sol, n_sol = m, num // 20
                break
        if m_sol is None:
            raise ValueError(f"MG{i}: SOC 回零残差 {delta} 无格点解")
        if m_sol > 0:
            ch_slot = next((t for t in range(T) if p_dis_f[t] == 0), None)
            if ch_slot is None:
                raise ValueError(f"MG{i}: 无充电侧可修正槽位")
            p_ch_f[ch_slot] += 20 * m_sol
            z_f[ch_slot] = 1
        if n_sol > 0:
            dis_slot = next((t for t in range(T) if p_ch_f[t] == 0), None)
            if dis_slot is None:
                raise ValueError(f"MG{i}: 无放电侧可修正槽位")
            p_dis_f[dis_slot] += 19 * n_sol
            z_f[dis_slot] = 0
        soc_f = [soc0_f]
        for t in range(T):
            val = (int(ETA_NUM) * int(ETA_DEN) * soc_f[t]
                   + int(ETA_NUM) ** 2 * p_ch_f[t]
                   - int(ETA_DEN) ** 2 * p_dis_f[t])
            soc_f.append(val // (int(ETA_NUM) * int(ETA_DEN)))
        if soc_f[T] != soc0_f:
            raise ValueError(f"MG{i}: 格点修正后仍不回零 (残差 {soc_f[T]-soc0_f})")
        p_grid_f = resolve_grid(first=True)


    band_lb = [_f_floor(x) for x in mg.soc_lb]
    band_ub = [_f_ceil(x) for x in mg.soc_ub]
    s_min_f = [max(band_lb[t], 0) for t in range(T + 1)]
    s_max_f = [band_ub[t] for t in range(T + 1)]
    widen_f = 0
    for t in range(T + 1):
        if soc_f[t] < s_min_f[t]:
            widen_f = max(widen_f, s_min_f[t] - soc_f[t])
            s_min_f[t] = soc_f[t]
        elif soc_f[t] > s_max_f[t]:
            widen_f = max(widen_f, soc_f[t] - s_max_f[t])
            s_max_f[t] = soc_f[t]
    if widen_f > SOC_BAND_WIDEN_MAX_F:
        raise ValueError(
            f"MG{i}: 量化 SOC 越出市场带 {widen_f / SCALE:.6f} kWh, 超过最小外扩"
            f"上限 {SOC_BAND_WIDEN_MAX_F / SCALE:.3f} kWh: 拒绝放宽市场带 "
            f"(需修调度或量化对账, 不得改用自生成包络)")
    s_min_f[0] = 2 * soc_f[0] - s_max_f[0]
    if s_min_f[0] < 0:
        raise ValueError(f"MG{i}: 首槽下带 {s_min_f[0]} 为负 (市场带中点约定被破坏)")
    if widen_f:
        print(f"[info] MG{i}: 市场 SOC 带按量化最小外扩 {widen_f / SCALE:.6f} kWh "
              f"(上限 {SOC_BAND_WIDEN_MAX_F / SCALE:.3f} kWh)")

    dg_cost_f = _f(mg.dg_cost) if mg.dg_cost else 0
    wear_cost_f = _f(mg.battery_cost) if mg.battery_cost else 0
    cost_claim = 0
    for t in range(T):
        cost_claim += grid_price_f[t] * (p_grid_f[t] + loss_f[t])
        cost_claim += dg_cost_f * p_dg_f[t]
        cost_claim += wear_cost_f * p_dis_f[t]
        cost_claim -= fit_price_f[t] * p_export_f[t]

    witness = {
        "version": CIRCUIT_VERSION,
        "session_id": session_id,
        "mg_id": f"MG{mg.mg_id}",
        "t": T,
        "public": {
            "session_hash": session_hash_field(session_id),
            "version_tag": CIRCUIT_VERSION,
            "cost_claim": cost_claim,
            "trade": trade_rows,
            "grid_price": grid_price_f,
            "fit_price": fit_price_f,
            "loss_alloc": loss_f,
            "grid_cap": grid_cap_f,
            "dg_cap": _f(mg.diesel_capacity),
            "ess_power": ess_power_f,
            "eta_num": ETA_NUM,
            "eta_den": ETA_DEN,
            "s_min": s_min_f,
            "s_max": s_max_f,
            "dg_cost": dg_cost_f,
            "wear_cost": wear_cost_f,
        },
        "private": {
            "load": load_f,
            "pv": pv_f,
            "wind": wind_f,
            "curtail": curtail_f,
            "p_grid": p_grid_f,
            "p_export": p_export_f,
            "p_dg": p_dg_f,
            "p_ch": p_ch_f,
            "p_dis": p_dis_f,
            "soc": soc_f,
            "z": z_f,
        },
    }
    _self_check(witness)
    return witness


def _self_check(w: dict) -> None:
    """"""
    T = w["t"]
    pub, priv = w["public"], w["private"]
    a, b = pub["eta_num"], pub["eta_den"]
    rows = pub["trade"]
    assert len(rows) == TRADE_SLOTS, f"交易行槽位数 {len(rows)} != {TRADE_SLOTS}"
    for t in range(T):
        net = sum(rows[s][t] for s in range(TRADE_SLOTS))
        lhs = priv["load"][t] + priv["curtail"][t] + pub["loss_alloc"][t] + priv["p_export"][t]
        rhs = (priv["pv"][t] + priv["wind"][t] + priv["p_dis"][t] - priv["p_ch"][t]
               + priv["p_grid"][t] + net + priv["p_dg"][t])
        assert lhs == rhs, f"balance t{t}: {lhs} != {rhs}"
        assert 0 <= priv["p_grid"][t] <= pub["grid_cap"], f"p_grid RC t{t}"
        assert 0 <= priv["p_export"][t] <= pub["grid_cap"], f"p_export RC t{t}"
        assert 0 <= priv["p_dg"][t] <= pub["dg_cap"], f"p_dg RC t{t}"
        assert 0 <= priv["p_ch"][t] <= pub["ess_power"], f"p_ch RC t{t}"
        assert 0 <= priv["p_dis"][t] <= pub["ess_power"], f"p_dis RC t{t}"
        assert priv["p_ch"][t] % b == 0 and priv["p_dis"][t] % a == 0, f"量化 t{t}"
        z = priv["z"][t]
        if not (z in (0, 1) and priv["p_ch"][t] * (1 - z) == 0
                and priv["p_dis"][t] * z == 0):
            raise AssertionError(
                f"mutex t{t}: z={z} ch={priv['p_ch'][t]} dis={priv['p_dis'][t]}")
    for t in range(T + 1):
        assert pub["s_min"][t] <= priv["soc"][t] <= pub["s_max"][t], f"SOC band t{t}"
    assert priv["soc"][0] == priv["soc"][T]


if __name__ == "__main__":
    import io
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_real_data import load_experiment_microgrids
    from microgrid import Microgrid, build_network
    from admm_solver import ADMMSolver

    day, n, mg_idx = 15, 10, 0
    raw = load_experiment_microgrids(day=day)[:n]
    mgs = [Microgrid(it["mg_id"], None, real_data=it["data"],
                     soc_bounds=it["soc_bounds"]) for it in raw]
    solver = ADMMSolver(mgs, rho_init=0.01, max_iter=1000, tol=1e-3,
                        adaptive=True, network=build_network(mgs))
    old = sys.stdout
    sys.stdout = io.StringIO()
    solver.solve()
    _, rep = solver.repair_finalize()
    sys.stdout = old
    print(f"ADMM 停机 reason={solver.stop_reason}, 修复轮: "
          f"结算成本 {rep['repair_cost']:.2f} $") if rep else print("无修复")
    w = build_witness_v2(solver, mg_idx=mg_idx,
                         session_id=f"day{day}-N{n}-boyd")
    mg_label = str(w["mg_id"]).lower()
    if not (mg_label.startswith("mg") and mg_label[2:].isdigit()):
        raise ValueError(f"unexpected mg_id: {w['mg_id']!r}")
    out = Path(__file__).resolve().parent / f"witness_v2_{mg_label}.json"
    out.write_text(json.dumps(w), encoding="utf-8")
    print(f"OK: {out} (t={w['t']}, cost_claim={w['public']['cost_claim']})")
