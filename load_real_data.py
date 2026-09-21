""""""
import os

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "shujv",
    "Real-Time Peer-to-Peer Energy Trading for Multi-Mi",
)

_NUM_MG = 20
_DAYS = 60
_POINTS_PER_DAY = 288
_TIME_SLOTS = 24

_FILE_MAP = {
    "load": "60-day load power data for 20 microgrids.xlsx",
    "pv": "60-day photovoltaic power data for 20 microgrids.xlsx",
    "wind": "60-day wind power data for 20 microgrids.xlsx",
}

_SOC_FILES = {
    "lower": "Lower SOC Boundary for VES.xlsx",
    "upper": "Upper SOC Boundary for VES.xlsx",
}

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".realdata_cache.npz"
)
_RAW_MEMO = None


def _xlsx_latest_mtime():
    latest = 0.0
    for fname in list(_FILE_MAP.values()) + list(_SOC_FILES.values()):
        latest = max(latest, os.path.getmtime(os.path.join(_DATA_DIR, fname)))
    return latest


def _read_soc_sheet(fname):
    df = pd.read_excel(os.path.join(_DATA_DIR, fname), header=None)
    values = df.values.astype(float).ravel()
    if values.size != _POINTS_PER_DAY:
        raise ValueError(f"{fname} 有 {values.size} 个点, 期望 {_POINTS_PER_DAY}")
    return values


def _load_raw():
    """"""
    global _RAW_MEMO
    if _RAW_MEMO is not None:
        return _RAW_MEMO

    if os.path.exists(_CACHE_PATH) and (
        os.path.getmtime(_CACHE_PATH) >= _xlsx_latest_mtime()
    ):
        with np.load(_CACHE_PATH) as z:
            _RAW_MEMO = {k: z[k] for k in z.files}
        return _RAW_MEMO

    raw = {}
    for kind in _FILE_MAP:
        arr = np.empty((_NUM_MG, _DAYS, _POINTS_PER_DAY))
        for mg_id in range(1, _NUM_MG + 1):
            arr[mg_id - 1] = _read_sheet(mg_id, kind)
        raw[kind] = arr
    for kind, fname in _SOC_FILES.items():
        raw[f"soc_{kind}"] = _read_soc_sheet(fname)

    np.savez_compressed(_CACHE_PATH, **raw)
    _RAW_MEMO = raw
    return _RAW_MEMO

# (wind_rated_kW, pv_rated_kW, load_cap_kW, dg_rated_kW, dg_cost_usd_per_kwh,
#  ves_kwh, ves_hours, ves_cost_usd_per_kwh, topology)
_MG_CONFIG = {
    1:  (500, 300, 500, 150, 0.127, 400, 2, 0.014, "IEEE 33-bus"),
    2:  (800, 200, 700, 250, 0.172, 500, 2, 0.013, "IEEE 12-bus"),
    3:  (600, 400, 700, 100, 0.143, 600, 3, 0.016, "IEEE 12-bus"),
    4:  (800, 300, 500, 100, 0.125, 600, 3, 0.017, "IEEE 33-bus"),
    5:  (900, 200, 300, 150, 0.184, 500, 2, 0.015, "IEEE 33-bus"),
    6:  (600, 300, 400, 100, 0.153, 400, 2, 0.014, "IEEE 15-bus"),
    7:  (700, 300, 400, 250, 0.132, 600, 3, 0.014, "IEEE 12-bus"),
    8:  (600, 300, 600, 150, 0.168, 600, 3, 0.013, "IEEE 12-bus"),
    9:  (400, 400, 700, 100, 0.161, 600, 3, 0.015, "IEEE 15-bus"),
    10: (700, 300, 200, 200, 0.185, 300, 3, 0.015, "IEEE 12-bus"),
    11: (800, 200, 500, 250, 0.158, 500, 2, 0.015, "IEEE 15-bus"),
    12: (800, 400, 500, 200, 0.155, 300, 3, 0.016, "IEEE 33-bus"),
    13: (400, 300, 400, 200, 0.137, 300, 3, 0.017, "IEEE 33-bus"),
    14: (400, 300, 600, 100, 0.142, 600, 3, 0.016, "IEEE 15-bus"),
    15: (500, 300, 500, 250, 0.170, 300, 3, 0.014, "IEEE 12-bus"),
    16: (900, 400, 400, 150, 0.164, 500, 2, 0.018, "IEEE 15-bus"),
    17: (400, 400, 300, 200, 0.152, 600, 3, 0.013, "IEEE 33-bus"),
    18: (600, 300, 700, 150, 0.162, 400, 2, 0.016, "IEEE 33-bus"),
    19: (900, 400, 400, 200, 0.141, 300, 3, 0.018, "IEEE 12-bus"),
    20: (500, 400, 300, 150, 0.125, 300, 3, 0.018, "IEEE 15-bus"),
}


_RES_RATIO_THRESHOLDS = (1.5, 2.5)


def _read_sheet(mg_id, data_type):
    """"""
    path = os.path.join(_DATA_DIR, _FILE_MAP[data_type])
    df = pd.read_excel(path, sheet_name=f"MG_{mg_id}", header=None)
    values = df.values.astype(float)
    if values.shape != (_DAYS, _POINTS_PER_DAY):
        raise ValueError(
            f"{_FILE_MAP[data_type]} sheet MG_{mg_id} 形状 {values.shape} "
            f"!= ({_DAYS}, {_POINTS_PER_DAY})"
        )
    return values


def _aggregate_hourly(points_288):
    """"""
    return points_288.reshape(_TIME_SLOTS, _POINTS_PER_DAY // _TIME_SLOTS).mean(axis=1)


def load_microgrid_data(mg_id, day=15):
    """"""
    if not 1 <= mg_id <= _NUM_MG:
        raise ValueError(f"mg_id 必须在 1~{_NUM_MG}, 收到 {mg_id}")
    if not 0 <= day < _DAYS:
        raise ValueError(f"day 必须在 0~{_DAYS - 1}, 收到 {day}")

    raw = _load_raw()
    return {
        kind: _aggregate_hourly(raw[kind][mg_id - 1][day])
        for kind in ("load", "pv", "wind")
    }


def load_soc_bounds():
    """"""
    raw = _load_raw()
    bounds = {}
    for kind in ("lower", "upper"):
        values = raw[f"soc_{kind}"]
        blocks = values.reshape(_TIME_SLOTS, _POINTS_PER_DAY // _TIME_SLOTS)
        bounds[kind] = blocks.max(axis=1) if kind == "lower" else blocks.min(axis=1)
    return bounds

CANONICAL_ORDER = tuple(range(1, _NUM_MG + 1))


def load_experiment_microgrids(day=15):
    """"""
    by_id = {it["mg_id"]: it for it in load_all_microgrids(day=day)}
    return [by_id[i] for i in CANONICAL_ORDER]


def build_soc_arrays(soc_bounds_frac, battery_capacity, time_slots=_TIME_SLOTS):
    """"""
    lower = np.asarray(soc_bounds_frac["lower"], dtype=float) * battery_capacity
    upper = np.asarray(soc_bounds_frac["upper"], dtype=float) * battery_capacity
    if lower.size != time_slots:
        raise ValueError(f"SOC 边界长度 {lower.size} != {time_slots}")

    lb = np.empty(time_slots + 1)
    ub = np.empty(time_slots + 1)
    lb[0], ub[0] = lower[0], upper[0]
    lb[time_slots], ub[time_slots] = lower[-1], upper[-1]
    for t in range(1, time_slots):
        lb[t] = max(lower[t - 1], lower[t])
        ub[t] = min(upper[t - 1], upper[t])
    return lb, ub


def classify_microgrid(load_capacity, res_capacity):
    """"""
    ratio = res_capacity / load_capacity
    lo, hi = _RES_RATIO_THRESHOLDS
    if ratio < lo:
        return "industrial"
    if ratio <= hi:
        return "commercial"
    return "residential"


def _scaled_params(mg_id, load, pv, wind):
    """"""
    (wind_rated, pv_rated, load_cap, dg_rated, dg_cost_usd,
     ves_kwh, ves_hours, ves_cost_usd, _) = _MG_CONFIG[mg_id]
    mg_type = classify_microgrid(load_cap, wind_rated + pv_rated)
    return {
        "mg_type": mg_type,
        "battery_capacity": float(ves_kwh),
        "battery_power": float(ves_kwh / ves_hours),
        "battery_efficiency": 0.95,
        "battery_cost": float(ves_cost_usd),
        "has_diesel": True,
        "diesel_capacity": float(dg_rated),
        "dg_cost": float(dg_cost_usd),
        "max_load": float(load.max()),
        "pv_capacity": float(pv.max()),
        "wind_capacity": float(wind.max()),
        "load_capacity": float(load_cap),
        "wind_rated": float(wind_rated),
        "pv_rated": float(pv_rated),
        "topology": _MG_CONFIG[mg_id][8],
    }


def load_all_microgrids(day=15):
    """"""
    soc_frac = load_soc_bounds()
    microgrids = []
    for mg_id in range(1, _NUM_MG + 1):
        data = load_microgrid_data(mg_id, day=day)
        params = _scaled_params(mg_id, data["load"], data["pv"], data["wind"])
        lb, ub = build_soc_arrays(soc_frac, params["battery_capacity"])
        data["mg_type"] = params["mg_type"]
        data["params"] = params
        microgrids.append(
            {
                "mg_id": mg_id,
                "mg_type": params["mg_type"],
                "data": data,
                "params": params,
                "soc_bounds": {"lb": lb, "ub": ub},
            }
        )
    return microgrids


if __name__ == "__main__":
    items = load_all_microgrids(day=15)
    print(f"{'MG':>4} {'类型':<12} {'avg_load':>9} {'peak_load':>10} "
          f"{'avg_pv':>8} {'avg_wind':>9} {'ves_kwh':>8} {'ves_kw':>7} "
          f"{'soc_lb[0]':>10} {'soc_ub[0]':>10}")
    for it in items:
        d, p = it["data"], it["params"]
        print(f"MG{it['mg_id']:<3} {it['mg_type']:<12} {d['load'].mean():>9.1f} "
              f"{d['load'].max():>10.1f} {d['pv'].mean():>8.1f} {d['wind'].mean():>9.1f} "
              f"{p['battery_capacity']:>8.0f} {p['battery_power']:>7.0f} "
              f"{it['soc_bounds']['lb'][0]:>10.1f} {it['soc_bounds']['ub'][0]:>10.1f}")
    types = [it["mg_type"] for it in items]
    print(f"\n分类统计: industrial={types.count('industrial')}, "
          f"commercial={types.count('commercial')}, residential={types.count('residential')}")
