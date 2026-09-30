"""
sim_main.py — 主循环（带站点生命周期）
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from .pop_evolve import PopulationModel
from .demand_calc import DemandCalculator
from .site_optimize import SiteOptimizer
from .price_subsidy import SiteFinance


def run(config: dict, output_dir: str = "simulation_output") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模块
    pop_model = PopulationModel(config)
    dem_calc = DemandCalculator(config)
    optimizer = SiteOptimizer(config)
    finance = SiteFinance(config)

    # 基础数据
    labels = list(config["communities"]["labels"])
    n = len(labels)
    pop_0 = np.array(
        [config["communities"]["initial_population"][c] for c in labels],
        dtype=float,
    )
    income = np.array([config["communities"]["income"][c] for c in labels], dtype=float)
    willingness = np.array(
        [config["communities"]["willingness"][c] for c in labels], dtype=float
    )
    distance_matrix = np.array(config["distance_matrix"], dtype=float)

    current_pop = [
        {"name": labels[i], "Z": pop_0[i, 0], "B": pop_0[i, 1], "S": pop_0[i, 2]}
        for i in range(n)
    ]

    # 站点池（关键：跨年份维护）
    history_sites: list = []

    years = config["simulation"]["years"]

    for t in range(years + 1):
        print(f"\n========== 仿真年份 t = {t} ==========")

        pop_array = np.array([[d["Z"], d["B"], d["S"]] for d in current_pop], dtype=float)

        # 1. 需求
        actual_demand, match_degree = dem_calc.calc(pop_array, income, willingness)
        print(f"匹配度 M = {match_degree:.4f} | 月总需求 = {actual_demand.sum():.2f}")

        # 2. 选址（带生命周期）
        site_result = optimizer.optimize(
            actual_demand,
            distance_matrix,
            history_sites=history_sites,
        )
        selected_sites = site_result["selected_sites"]
        m = site_result["metrics"]
        print(f"覆盖率 = {m['coverage_rate']:.2%} | 满足率 = {m['demand_satisfaction_rate']:.2%}")

        # 3. 财务（容量校验按护理员工时口径，与选址模块的分配口径一致）
        for site in selected_sites:
            fin = finance.calc(
                scale=site["scale"],
                monthly_demand=site["monthly_demand_by_service"],
                price_ratio=config["demand"]["price_ratio"],
                cid=site["cid"],
                max_capacity=site["effective_monthly_capacity"],
                labor_minutes_limit=site["labor_minutes_budget"],
            )
            # 展示用的有效容量（含小区系数与新站磨合）
            site["effective_labor_minutes"] = site["effective_monthly_capacity"] * (
                finance.avg_service_duration
            )
            site.update(fin)

        for s in selected_sites:
            tag = "新" if s["is_new"] else "旧"
            print(f"  [{tag}] {s['cid']} {s['scale']} 月利润={s.get('monthly_profit', 0):.2f}")

        # 4. 更新站点池（给下一年用）
        #    站龄：保留的站点年龄 +1；当年新建的站点记 0。
        #    它决定下一年的"磨合期容量系数"是否继续生效（只有建成当年打 0.8 折）。
        history_sites = [
            {
                "cid": s["cid"],
                "scale": s["scale"],
                "build_cost": s["build_cost"],
                "build_year": s.get("build_year", t),
                "age": 0 if s.get("is_new", True) else s.get("site_age", 1) + 1,
            }
            for s in selected_sites
        ]

        # 5. 保存快照
        _save_snapshot(t, pop_array, actual_demand, match_degree, selected_sites, labels, output_dir)

        # 6. 人口演化
        trans_p = pop_model.sample_transition()
        city_float = pop_model.evolve_one_year(current_pop, trans_p)
        current_pop = pop_model.round_total(city_float)

    print("\n===== 仿真全部完成 =====")


def _save_snapshot(t, pop_array, demand, match_deg, sites, labels, output_dir):
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

    site_rows = []
    for s in sites:
        demand_json = json.dumps(s.get("monthly_demand_by_service", {}), ensure_ascii=False)
        site_rows.append({
            "cid": s["cid"],
            "scale": s["scale"],
            "is_new": s.get("is_new", False),
            "monthly_demand": demand_json,
            "monthly_total_demand": round(s.get("monthly_total_demand", 0), 2),
            "monthly_max_capacity": round(s.get("monthly_max_capacity", 0), 2),
            # 容量工时化（P0-1）：真实约束是护理员分钟，次数上限仅为等权口径展示
            "labor_minutes_budget": round(s.get("labor_minutes_budget", 0), 2),
            "labor_minutes_used": round(s.get("labor_minutes_used", 0), 2),
            "labor_utilization": round(
                s.get("labor_minutes_used", 0) / s["labor_minutes_budget"], 4
            ) if s.get("labor_minutes_budget") else 0.0,
            "binding_constraint": s.get("binding_constraint", ""),
            "monthly_revenue": round(s.get("monthly_revenue", 0), 2),
            "monthly_var_cost": round(s.get("monthly_var_cost", 0), 2),
            "monthly_fixed_cost": round(s.get("monthly_fixed_cost", 0), 2),
            "monthly_op_profit": round(s.get("monthly_profit", 0), 2),
        })
    pd.DataFrame(site_rows).to_csv(
        os.path.join(output_dir, f"snap_year_{t}_site.csv"),
        index=False, encoding="utf-8-sig",
    )