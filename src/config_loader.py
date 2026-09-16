"""
config_loader.py — 配置加载与校验

全项目唯一读取 yaml 的入口。
其他模块一律接收 dict，不直接读文件。

对外接口：
    config = load_config("config.yaml")                    # 读默认配置
    config = load_config("config.yaml", overrides={...})   # 带覆盖
"""
from __future__ import annotations

import os
import yaml
from copy import deepcopy
from typing import Dict, Any, Optional


# ===================== 主入口 =====================
def load_config(
    path: str = "config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    加载 yaml 配置，可选覆盖部分参数，返回校验后的配置字典。

    :param path: 配置文件路径
    :param overrides: 要覆盖的参数字典，例如 {"simulation": {"years": 10}}
    :return: 完整的配置字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空: {path}")

    # 应用覆盖参数
    if overrides:
        config = _deep_merge(config, overrides)

    # 校验
    validate_config(config)

    return config


# ===================== 深合并 =====================
def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个 dict，override 中的值覆盖 base。"""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


# ===================== 校验 =====================
def validate_config(config: Dict[str, Any]) -> None:
    """校验配置合法性，非法参数抛 ValueError。"""

    # ---------- 顶层字段 ----------
    required_top = [
        "simulation", "population", "demand", "site",
        "finance", "facility", "services", "communities",
        "distance_matrix", "demand_matrix",
    ]
    for key in required_top:
        if key not in config:
            raise ValueError(f"配置缺少顶层字段: {key}")

    # ---------- simulation ----------
    sim = config["simulation"]
    if not isinstance(sim["years"], int) or sim["years"] <= 0:
        raise ValueError("simulation.years 必须为正整数")
    if not isinstance(sim["replan_interval"], int) or sim["replan_interval"] <= 0:
        raise ValueError("simulation.replan_interval 必须为正整数")

    # ---------- population ----------
    pop = config["population"]
    if not (0 <= pop["lambda_rate"] <= 1):
        raise ValueError("population.lambda_rate 应在 [0, 1]")
    if pop.get("float_ratio", 0.10) < 0:
        raise ValueError("population.float_ratio 应 ≥ 0")

    for k, v in pop["transitions"].items():
        if not (0 <= v <= 1):
            raise ValueError(f"population.transitions.{k} 应在 [0, 1]，当前 {v}")

    for k, v in pop["death"].items():
        if not (0 <= v <= 1):
            raise ValueError(f"population.death.{k} 应在 [0, 1]，当前 {v}")

    if not (0 <= pop["migrate_out"] <= 1):
        raise ValueError("population.migrate_out 应在 [0, 1]")

    # ---------- demand ----------
    dem = config["demand"]
    for k, v in dem["consumption_limits"].items():
        if not (0 < v <= 1):
            raise ValueError(f"demand.consumption_limits.{k} 应在 (0, 1]")
    for k, v in dem["price_elasticity"].items():
        if v > 0:
            raise ValueError(f"demand.price_elasticity.{k} 应为负数")
    if dem["price_ratio"] <= 0:
        raise ValueError("demand.price_ratio 必须 > 0")

    # ---------- site ----------
    site = config["site"]
    if site["annual_build_budget"] <= 0:
        raise ValueError("site.annual_build_budget 必须 > 0")
    if site["service_radius"] <= 0:
        raise ValueError("site.service_radius 必须 > 0")
    if not (0 <= site["adapt_threshold"] <= 1):
        raise ValueError("site.adapt_threshold 应在 [0, 1]")
    if not (0 <= site["salvage_ratio"] <= 1):
        raise ValueError("site.salvage_ratio 应在 [0, 1]")

    # ---------- finance ----------
    fin = config["finance"]
    if fin["max_profit_ratio"] <= 0:
        raise ValueError("finance.max_profit_ratio 必须 > 0")
    if fin["depreciation_years"] <= 0:
        raise ValueError("finance.depreciation_years 必须 > 0")
    if fin["nurse_monthly_salary"] <= 0:
        raise ValueError("finance.nurse_monthly_salary 必须 > 0")

    # ---------- facility ----------
    fac = config["facility"]
    for key in ["build_cost", "daily_fix_cost", "daily_max_capacity", "min_nurse_count"]:
        if key not in fac:
            raise ValueError(f"facility 缺少字段: {key}")
        for scale in ["小型", "中型", "大型"]:
            if scale not in fac[key]:
                raise ValueError(f"facility.{key} 缺少规模: {scale}")

    # ---------- services ----------
    svc = config["services"]
    for key in ["names", "revenue", "variable_cost"]:
        if key not in svc:
            raise ValueError(f"services 缺少字段: {key}")
    for name in svc["names"]:
        if name not in svc["revenue"]:
            raise ValueError(f"services.revenue 缺少服务: {name}")
        if name not in svc["variable_cost"]:
            raise ValueError(f"services.variable_cost 缺少服务: {name}")

    # ---------- communities ----------
    comm = config["communities"]
    labels = comm["labels"]
    if len(labels) == 0:
        raise ValueError("communities.labels 不能为空")

    for label in labels:
        if label not in comm["initial_population"]:
            raise ValueError(f"communities.initial_population 缺少小区: {label}")
        if len(comm["initial_population"][label]) != 3:
            raise ValueError(f"communities.initial_population[{label}] 应为 [Z, B, S] 三个数")
        if label not in comm["income"]:
            raise ValueError(f"communities.income 缺少小区: {label}")
        if label not in comm["adapt"]:
            raise ValueError(f"communities.adapt 缺少小区: {label}")
        if not (0 <= comm["adapt"][label] <= 1):
            raise ValueError(f"communities.adapt[{label}] 应在 [0, 1]")
        if label not in comm["willingness"]:
            raise ValueError(f"communities.willingness 缺少小区: {label}")
        if len(comm["willingness"][label]) != 3:
            raise ValueError(f"communities.willingness[{label}] 应为 3 个数")

    # ---------- distance_matrix ----------
    dist = config["distance_matrix"]
    n = len(labels)
    if len(dist) != n:
        raise ValueError(f"distance_matrix 应有 {n} 行，当前 {len(dist)} 行")
    for i, row in enumerate(dist):
        if len(row) != n:
            raise ValueError(f"distance_matrix 第 {i} 行应有 {n} 列，当前 {len(row)} 列")
        if row[i] != 0:
            raise ValueError(f"distance_matrix 对角线应为 0，第 {i} 行第 {i} 列为 {row[i]}")

    # ---------- demand_matrix ----------
    dm = config["demand_matrix"]
    n_services = len(svc["names"])
    for state in ["Z", "B", "S"]:
        if state not in dm:
            raise ValueError(f"demand_matrix 缺少状态: {state}")
        if len(dm[state]) != n_services:
            raise ValueError(
                f"demand_matrix.{state} 应有 {n_services} 项，当前 {len(dm[state])}"
            )


# ===================== 自测 =====================
if __name__ == "__main__":
    try:
        cfg = load_config("config.yaml")
        print("✅ 配置加载成功")
        print(f"   顶层字段: {list(cfg.keys())}")
        print(f"   仿真年数: {cfg['simulation']['years']}")
        print(f"   小区数量: {len(cfg['communities']['labels'])}")
        print(f"   服务数量: {len(cfg['services']['names'])}")
        print(f"   距离矩阵: {len(cfg['distance_matrix'])} × {len(cfg['distance_matrix'][0])}")

        # 测试覆盖功能
        cfg2 = load_config("config.yaml", overrides={"simulation": {"years": 10}})
        print(f"\n✅ 覆盖测试: years 从 {cfg['simulation']['years']} → {cfg2['simulation']['years']}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   请确认 config.yaml 在项目根目录")
    except ValueError as e:
        print(f"❌ 配置校验失败: {e}")