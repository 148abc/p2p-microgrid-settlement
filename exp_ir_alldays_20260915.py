# -*- coding: utf-8 -*-
""""""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from admm_solver import ADMMSolver                       # noqa: E402
from load_real_data import load_all_microgrids           # noqa: E402
from microgrid import Microgrid, build_network           # noqa: E402
from standalone_baseline import single_mg_cost           # noqa: E402

OUT = HERE / "experiment_data_real" / "ir_alldays_20260915.json"
DAYS = [0, 7, 15, 23, 31, 39, 47, 55]
IDS = list(range(1, 21))


def build(day):
    raw = load_all_microgrids(day=day)
    by_id = {it["mg_id"]: it for it in raw}
    return [Microgrid(it["mg_id"], None, real_data=it["data"],
                      soc_bounds=it["soc_bounds"]) for it in (by_id[i] for i in IDS)]


def run_coop(mgs):
    """"""
    s = ADMMSolver(mgs, rho_init=0.01, max_iter=1000, tol=1e-3, adaptive=True,
                   network=build_network(mgs))
    buf = io.StringIO()
    with redirect_stdout(buf):
        s.solve()
    return s


def mg_bill_coop(solver, i, mg):
    """"""
    r = solver.get_local_result(i)
    bill = (float(np.sum(mg.grid_price * r["P_grid"]))
            + float(np.sum(mg.grid_price * solver.loss_allocation[i, :]))
            + (float(np.sum(mg.dg_cost * r["P_dg"])) if mg.dg_cost is not None else 0.0)
            + float(np.sum(mg.battery_cost * r["P_dis"]))
            - float(np.sum(mg.fit_price * r["P_export"])))
    p2p_pay = 0.0
    for j in range(solver.N):
        if j != i:
            p2p_pay += float(np.sum(solver.p2p_price[i, j, :] * r["P_trade_local"][j, :]))
    return bill + p2p_pay


def main():
    data = (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else
            {"config": {"days": DAYS, "N": 20,
                        "protocol": "identical to per_mg_benefits.json (day 15); "
                                    "standalone vs coop bill per microgrid"},
             "days": []})
    done = {d["day"] for d in data["days"]}

    for day in DAYS:
        if day in done:
            continue
        mgs = build(day)
        sa = [float(single_mg_cost(mg)) for mg in mgs]
        solver = run_coop(mgs)
        bills = [mg_bill_coop(solver, i, mg) for i, mg in enumerate(mgs)]
        rows = [{"mg_id": mg.mg_id, "benefit": s - b}
                for mg, s, b in zip(mgs, sa, bills)]
        deficits = [r for r in rows if r["benefit"] < 0]
        pos = [r["benefit"] for r in rows if r["benefit"] > 0]
        tot_pos, tot_def = float(sum(pos)), float(-sum(r["benefit"] for r in deficits))
        rate = (tot_def / tot_pos) if tot_pos > 0 else None
        rec = {"day": day, "iters": len(solver.history["objective"]),
               "stop": solver.stop_reason, "converged": bool(solver.converged),
               "n_deficit": len(deficits),
               "deficit_total": round(tot_def, 2),
               "worst_deficit": round(min((r["benefit"] for r in rows), default=0.0), 2),
               "surplus": round(tot_pos, 2),
               "levy_rate_pct": (round(100 * rate, 3) if rate is not None else None),
               "max_beneficiary_cost": (round(rate * max(pos), 2)
                                        if (rate is not None and pos) else None),
               "deficit_mgs": [{"mg_id": r["mg_id"], "benefit": round(r["benefit"], 2)}
                               for r in sorted(deficits, key=lambda x: x["benefit"])]}
        data["days"].append(rec)
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  day{day:>2}: 赤字成员 {len(deficits)} 个 总额 ${tot_def:8.2f}  "
              f"盈余 ${tot_pos:9.2f}  征收率 {100*rate if rate else 0:5.3f}%  "
              f"最大受益者承担 ${rec['max_beneficiary_cost']}"
              + (f"  {rec['deficit_mgs']}" if deficits else ""), flush=True)

    print("\n=== 汇总 ===")
    ds = data["days"]
    print(f"  评估日 {len(ds)} 天, 出现赤字成员的 {sum(1 for d in ds if d['n_deficit'])} 天")
    print(f"  征收率范围 {min(d['levy_rate_pct'] for d in ds if d['levy_rate_pct'] is not None):.3f}%"
          f" ~ {max(d['levy_rate_pct'] for d in ds if d['levy_rate_pct'] is not None):.3f}%")
    worst = max((d for d in ds if d["max_beneficiary_cost"] is not None),
                key=lambda d: d["max_beneficiary_cost"])
    print(f"  最大受益者的最大承担: ${worst['max_beneficiary_cost']} (day {worst['day']})")
    print("\nALL DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
