""""""
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def single_mg_cost(mg, return_components=False):
    """"""
    model = gp.Model("single_mg")
    model.setParam("OutputFlag", 0)
    T = mg.time_slots

    P_grid = model.addVars(T, lb=0, ub=mg.p_grid_max, name="P_grid")
    P_export = model.addVars(T, lb=0, ub=mg.p_grid_max, name="P_export")
    P_ch = model.addVars(T, lb=0, ub=mg.battery_power, name="P_ch")
    P_dis = model.addVars(T, lb=0, ub=mg.battery_power, name="P_dis")
    P_curtail = model.addVars(T, lb=0, name="P_curtail")
    SOC = model.addVars(
        T + 1, lb=mg.soc_lb.tolist(), ub=mg.soc_ub.tolist(), name="SOC"
    )
    z_bat = model.addVars(T, vtype=GRB.BINARY, name="z_bat")

    use_dg = getattr(mg, "dg_cost", None) is not None and mg.diesel_capacity > 0
    P_dg = None
    if use_dg:
        P_dg = model.addVars(T, lb=0, ub=mg.diesel_capacity, name="P_dg")

    obj = gp.quicksum(mg.grid_price[t] * P_grid[t] for t in range(T))
    obj -= gp.quicksum(mg.fit_price[t] * P_export[t] for t in range(T))
    if use_dg:
        obj += gp.quicksum(mg.dg_cost * P_dg[t] for t in range(T))
    obj += gp.quicksum(mg.battery_cost * P_dis[t] for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)

    soc_target = 0.5 * (mg.soc_lb[0] + mg.soc_ub[0])
    model.addConstr(SOC[0] == soc_target, name="soc_initial")
    model.addConstr(SOC[T] == soc_target, name="soc_final")

    for t in range(T):
        dg_gen = P_dg[t] if use_dg else 0.0
        model.addConstr(
            mg.load[t] + P_curtail[t] + P_export[t]
            == mg.total_generation[t] + P_dis[t] - P_ch[t] + P_grid[t] + dg_gen,
            name=f"power_balance_{t}",
        )
        model.addConstr(
            SOC[t + 1]
            == SOC[t]
            + P_ch[t] * mg.battery_efficiency
            - P_dis[t] / mg.battery_efficiency,
            name=f"soc_dynamics_{t}",
        )
        model.addConstr(P_ch[t] <= z_bat[t] * mg.battery_power, name=f"ch_mutex_{t}")
        model.addConstr(P_dis[t] <= (1 - z_bat[t]) * mg.battery_power, name=f"dis_mutex_{t}")

    model.optimize()
    if model.status == GRB.OPTIMAL:
        if not return_components:
            return model.objVal
        grid_c = float(np.sum(mg.grid_price * np.array([P_grid[t].X for t in range(T)])))
        dg_c = (
            float(np.sum(mg.dg_cost * np.array([P_dg[t].X for t in range(T)])))
            if use_dg
            else 0.0
        )
        wear_c = float(np.sum(mg.battery_cost * np.array([P_dis[t].X for t in range(T)])))
        fit_c = float(np.sum(mg.fit_price * np.array([P_export[t].X for t in range(T)])))
        return {
            "cost": model.objVal,
            "grid": grid_c,
            "dg": dg_c,
            "wear": wear_c,
            "fit": fit_c,
        }
    return float("inf") if not return_components else {"cost": float("inf")}


def standalone_total(microgrids):
    """"""
    return float(sum(single_mg_cost(mg) for mg in microgrids))


if __name__ == "__main__":
    from microgrid import Microgrid

    mgs = [Microgrid(1, "industrial"), Microgrid(2, "commercial"), Microgrid(3, "residential")]
    for i, mg in enumerate(mgs):
        print(f"MG{i+1} standalone cost = {single_mg_cost(mg):.4f} $")
    print(f"total = {standalone_total(mgs):.4f} $")
