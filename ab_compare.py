"""A/B 对照：工时化容量口径下，不同护理员编制的逐年结果。

A  = 现状编制 2/5/10（config.yaml 原值）
B1 = 1:100 老人  3/10/30
B2 = 1:50  老人  6/20/60
B3 = 1:30  老人  10/33/100
B4 = 1:20  老人  15/50/150

用法：PYTHONPATH=. .venv/bin/python ab_compare.py
"""
import os
import numpy as np
import pandas as pd
from src.config_loader import load_config
from src.demand_calc import DemandCalculator
from src.site_optimize import SiteOptimizer

SCENARIOS = {
    "A_现状编制_2_5_10": {"小型": 2, "中型": 5, "大型": 10},
    "B1_1比100": {"小型": 3, "中型": 10, "大型": 30},
    "B2_1比50": {"小型": 6, "中型": 20, "大型": 60},
    "B3_1比30": {"小型": 10, "中型": 33, "大型": 100},
    "B4_1比20": {"小型": 15, "中型": 50, "大型": 150},
}
T = 0  # 只对比首年方案，避免逐年演化干扰归因


def year0_metrics(cfg):
    labels = list(cfg["communities"]["labels"])
    pop0 = np.array([cfg["communities"]["initial_population"][c] for c in labels], float)
    income = np.array([cfg["communities"]["income"][c] for c in labels], float)
    will = np.array([cfg["communities"]["willingness"][c] for c in labels], float)
    D = np.array(cfg["distance_matrix"], float)
    actual, M = DemandCalculator(cfg).calc(pop0, income, will)
    res = SiteOptimizer(cfg).optimize(actual, D, history_sites=None)
    return actual.sum(), M, res


rows = []
for name, nurses in SCENARIOS.items():
    cfg = load_config("config.yaml", overrides={"facility": {"min_nurse_count": nurses}})
    total, M, res = year0_metrics(cfg)
    m = res["metrics"]
    rows.append({
        "情景": name,
        "护理员编制(小/中/大)": f"{nurses['小型']}/{nurses['中型']}/{nurses['大型']}",
        "月总需求(次)": round(total),
        "需求释放率M": round(M, 4),
        "最优站数": len(res["selected_sites"]),
        "站点": "+".join(f"{s['cid']}·{s['scale']}" for s in res["selected_sites"]),
        "覆盖率": round(m["coverage_rate"], 4),
        "满足率": round(m["demand_satisfaction_rate"], 4),
        "利用率": round(m["avg_utilization"], 4),
        "瓶颈": "+".join(sorted({s["binding_constraint"] for s in res["selected_sites"]})),
        "月工时预算(分)": round(m["labor_minutes_available"]),
        "已用工时(分)": round(m["total_labor_minutes_used"]),
        "净投资(元)": round(m["net_invest"]),
        "成本效率(次/万元)": round(m["cost_per_demand_wan"], 2),
    })

df = pd.DataFrame(rows)
os.makedirs("simulation_output", exist_ok=True)
out = "simulation_output/scenario_AB_compare.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print(f"\n已写入 {out}")
