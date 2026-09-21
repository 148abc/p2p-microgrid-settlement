# 真实数据实验数据快照 (experiment_data_real/)

口径: ToU($0.20/$0.10/$0.045) + FiT($0.036/0) 美元全成本, 自然顺序, rho=0.01, max_iter=1000, Gurobi 13.0。生成: `python fig_paper.py`

| 文件 | 内容 | 关键字段 |
|---|---|---|
| profiles.json | 20 微网汇总 + 3 代表微网 24h 曲线 (day=15) | summary[], representative_curves |
| dispatch_trio.json | MG1-3 典型日 ADMM 调度 | microgrids{load,pv,wind,P_grid,P_export,P_dg,P_ch,P_dis,SOC,P_trade_net} |
| scal_results.json | 扩展性 N=3..20 | {standalone,coop,saving_pct,iters,converged} |
| multiday_results.json | 多天 8 日 x N∈{3,20} 快照 | results[{day,N,standalone_cost,coop_cost,saving_pct,converged,iterations}] |
| cost_decomposition.json | 独立/合作成本分量 (N=3/20) | {standalone,coop}{cost,grid,dg,wear,fit[,loss]} |
| sensitivity_results.json | DG 成本/P2P 折扣扫描 (N=10, day=15) | dg_scales/p2p_scales + saving |
| per_mg_benefits.json | 逐微网受益 (standalone_i − 合作账单_i, N=20) | rows[{mg_id,mg_type,standalone,coop_bill,benefit}] |
| penetration_correlation.json | 节省% vs 日可再生渗透率 (multiday 挖掘) | rows[{day,N,saving_pct,res_penetration}] |

重绘: python fig_paper.py (缓存命中时秒级出图, 无缓存时重算约 25 分钟)。
