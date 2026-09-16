"""
sim_main.py — 主循环（接口式）
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from .config_loader import load_config
from .pop_evolve import PopulationModel
from .demand_calc import DemandCalculator
from .site_optimize import SiteOptimizer
from .price_subsidy import SiteFinance


def run(config: dict, output_dir: str = "simulation_output") -> None:
    """主入口：跑完整仿真流程。"""
    os.makedirs(output_dir, exist_ok=True)

    # 初始化各模块
    pop_model = PopulationModel(config)
    dem_calc = DemandCalculator(config)
    optimizer = SiteOptimizer(config)
    finance = SiteFinance(config)

    # 从 config 读取基础数据
    labels = list(config["communities"]["labels"])
    n = len(labels)
    pop_0_array = np.array(
        [config["communities"]["initial_population"][c] for c in labels],
        dtype=float,
    )
    income = np.array([config["communities"]["income"][c] for c in labels], dtype=float)
    willingness = np.array(
        [config["communities"]["willingness"][c] for c in labels], dtype=float
    )
    distance_matrix = np.array(config["distance_matrix"], dtype=float)

    # 当前人口（list[dict] 供 population 使用）
    current_pop = [
        {"name": labels[i], "Z": pop_0_array[i, 0],
         "B": pop_0_array[i, 1], "S": pop_0_array[i, 2]}
        for i in range(n)
    ]

    years = config["simulation"]["years"]
    budget = config["site"]["annual_build_budget"]

    for t in range(years + 1):
        print(f"\n========== 仿真年份 t = {t} ==========")

        # 当前人口转 numpy 供需求计算
        pop_array = np.array([[d["Z"], d["B"], d["S"]] for d in current_pop], dtype=float)

        # 1. 需求计算
        actual_demand, match_degree = dem_calc.calc(pop_array, income, willingness)
        print(f"匹配度 M = {match_degree:.4f}")
        print(f"月总需求 = {actual_demand.sum():.2f}")

        # 2. 选址（枚举）
        site_result = optimizer.optimize(actual_demand, distance_matrix)
        selected_sites = site_result["selected_sites"]
        print(f"选址完成：{len(selected_sites)} 个站点")
        print(f"覆盖率 = {site_result['metrics']['coverage_rate']:.2%}")
        print(f"满足率 = {site_result['metrics']['demand_satisfaction_rate']:.2%}")

        # 3. 财务测算
        for site in selected_sites:
            fin_result = finance.calc(
                scale=site["scale"],
                monthly_demand=site["monthly_demand_by_service"],
                price_ratio=config["demand"]["price_ratio"],
            )
            site.update(fin_result)

        # 控制台打印
        for s in selected_sites:
            print(f"  {s['cid']} {s['scale']} 月利润={s.get('monthly_profit', 0):.2f}")

        # 4. 保存快照
        _save_snapshot(t, pop_array, actual_demand, match_degree, selected_sites, labels, output_dir)

        # 5. 人口演化
        trans_p = pop_model.sample_transition()
        city_float = pop_model.evolve_one_year(current_pop, trans_p)
        current_pop = pop_model.round_total(city_float)

    print("\n===== 仿真全部完成 =====")


def _save_snapshot(t, pop_array, demand, match_deg, sites, labels, output_dir):
    # 人口+需求快照
    rows = []
    for i, cid in enumerate(labels):
        rows.append({
            "t": t, "name": cid,
            "Z": int(pop_array[i, 0]),
            "B": int(pop_array[i, 1]),
            "S": int(pop_array[i, 2]),
            "match_degree": round(match_deg, 4),
            "助餐": round(demand[i, 0]),
            "日间照料": round(demand[i, 1]),
            "上门护理": round(demand[i, 2]),
            "康复理疗": round(demand[i, 3]),
            "助浴": round(demand[i, 4]),
            "紧急救助": round(demand[i, 5]),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"snap_year_{t}_block.csv"),
        index=False, encoding="utf-8-sig",
    )

    # 站点快照
    site_rows = []
    for s in sites:
        demand_json = json.dumps(s.get("monthly_demand_by_service", {}), ensure_ascii=False)
        site_rows.append({
            "cid": s["cid"],
            "scale": s["scale"],
            "monthly_demand": demand_json,
            "monthly_total_demand": round(s.get("monthly_total_demand", 0), 2),
            "monthly_max_capacity": round(s.get("monthly_max_capacity", 0), 2),
            "monthly_revenue": round(s.get("monthly_revenue", 0), 2),
            "monthly_var_cost": round(s.get("monthly_var_cost", 0), 2),
            "monthly_fixed_cost": round(s.get("monthly_fixed_cost", 0), 2),
            "monthly_op_profit": round(s.get("monthly_profit", 0), 2),
        })
    pd.DataFrame(site_rows).to_csv(
        os.path.join(output_dir, f"snap_year_{t}_site.csv"),
        index=False, encoding="utf-8-sig",
    )
    print(f"✅ 已保存 t={t} 快照")