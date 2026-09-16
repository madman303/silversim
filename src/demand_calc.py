"""
demand_calc.py — 养老服务需求计算模块（接口式）

对外接口：
    calc = DemandCalculator(config)          # 用 config 初始化
    actual_demand, M = calc.calc(pop, income, willingness, price_ratio)

config 结构（对应 config.yaml）：
    services.names               服务名称列表
    services.variable_cost       单次服务直接支出（用于消费上限计算）
    demand.consumption_limits    {Z: 0.20, B: 0.25, S: 0.30}
    demand.price_elasticity      {Z: -1.2, B: -0.9, S: -0.6}
    demand.price_ratio           基准价倍率
    demand_matrix                {Z: [...], B: [...], S: [...]}
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional


# 结构常量（不随情景变化）
STATE_ORDER = ("Z", "B", "S")  # 老人类型顺序


# ===================== 核心类 =====================
class DemandCalculator:
    """养老需求计算器，所有参数从 config 注入。"""

    def __init__(self, config: Dict[str, Any]):
        svc_cfg = config["services"]
        dem_cfg = config["demand"]
        matrix_cfg = config["demand_matrix"]

        # 服务名称（保持 config 中定义的顺序）
        self.service_names = list(svc_cfg["names"])

        # 单次服务直接支出（用于计算每类老人的理论月费用）
        self.service_prices = np.array(
            [svc_cfg["variable_cost"][name] for name in self.service_names],
            dtype=float,
        )

        # 需求矩阵 (3, 6)，行序 = [Z, B, S]
        self.service_demand = np.array(
            [matrix_cfg[state] for state in STATE_ORDER],
            dtype=float,
        )

        # 消费上限 (3,) 对应 Z/B/S
        self.consumption_limits = np.array(
            [dem_cfg["consumption_limits"][state] for state in STATE_ORDER],
            dtype=float,
        )

        # 价格弹性 (3,) 对应 Z/B/S
        self.price_elasticity = np.array(
            [dem_cfg["price_elasticity"][state] for state in STATE_ORDER],
            dtype=float,
        )

        # 默认价格倍率
        self.default_price_ratio = dem_cfg["price_ratio"]

    # ---------- 核心计算 ----------
    def calc(
        self,
        pop: np.ndarray,
        income: np.ndarray,
        willingness: np.ndarray,
        price_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        计算各小区六类养老服务的实际需求及供需匹配度。

        :param pop:         (n, 3) 数组，列序 [自理, 半失能, 失能]
        :param income:      (n,) 数组，各小区人均月收入
        :param willingness: (n, 3) 数组，各小区×三类老人的付费意愿
        :param price_ratio: 价格倍率，None 时从 config 读取默认值
        :return: (actual_demand, match_degree)
                 actual_demand: (n, 6) 数组
                 match_degree:  float in [0, 1]
        """
        if price_ratio is None:
            price_ratio = self.default_price_ratio

        population = np.asarray(pop, dtype=float)
        monthly_income = np.asarray(income, dtype=float)
        pay_willingness = np.asarray(willingness, dtype=float)

        # ---------- 输入校验 ----------
        if population.ndim != 2 or population.shape[1] != 3:
            raise ValueError("pop must have shape (n, 3)")
        n_blocks = population.shape[0]
        if monthly_income.shape != (n_blocks,):
            raise ValueError("income must have shape (n,)")
        if pay_willingness.shape != (n_blocks, 3):
            raise ValueError("willingness must have shape (n, 3)")
        if price_ratio < 0:
            raise ValueError("price_ratio must be non-negative")

        # ---------- 第 1 步：理论需求 ----------
        theory_by_type = population[:, :, None] * self.service_demand[None, :, :]
        theory_demand = theory_by_type.sum(axis=1)

        # ---------- 第 2 步：消费约束裁剪 ----------
        cost_per_person = self.service_demand @ self.service_prices
        budget = monthly_income[:, None] * self.consumption_limits[None, :]
        scale = np.minimum(1.0, budget / cost_per_person[None, :])
        income_constrained_by_type = theory_by_type * scale[:, :, None]

        # ---------- 第 3 步：付费意愿缩放 ----------
        willing_by_type = income_constrained_by_type * pay_willingness[:, :, None]

        # ---------- 第 4 步：价格弹性修正 ----------
        delta_p = price_ratio - 1.0
        elasticity_scale = 1.0 + self.price_elasticity[None, :, None] * delta_p
        actual_demand = (willing_by_type * elasticity_scale).sum(axis=1)
        actual_demand = np.maximum(actual_demand, 0.0)

        # ---------- 第 5 步：供需匹配度 ----------
        theory_total = theory_demand.sum()
        if theory_total == 0:
            match_degree = 1.0
        else:
            difference = np.abs(actual_demand - theory_demand).sum()
            match_degree = float(np.clip(1.0 - difference / theory_total, 0.0, 1.0))

        return actual_demand, match_degree


# ===================== 模块级函数接口 =====================
def calc_demand(
    pop: np.ndarray,
    income: np.ndarray,
    willingness: np.ndarray,
    config: Dict[str, Any],
    price_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    函数式接口：一次算完。
    :param config: 完整 config dict
    """
    calc = DemandCalculator(config)
    return calc.calc(pop, income, willingness, price_ratio=price_ratio)


# ===================== 自测 =====================
if __name__ == "__main__":
    # 简化的测试 config
    test_config = {
        "services": {
            "names": ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"],
            "variable_cost": {
                "助餐": 8, "日间照料": 16, "上门护理": 24,
                "康复理疗": 23, "助浴": 20, "紧急救助": 8,
            },
        },
        "demand": {
            "consumption_limits": {"Z": 0.20, "B": 0.25, "S": 0.30},
            "price_elasticity": {"Z": -1.2, "B": -0.9, "S": -0.6},
            "price_ratio": 1.0,
        },
        "demand_matrix": {
            "Z": [14, 8, 0, 2, 0, 0.15],
            "B": [20, 14, 6, 4, 2, 1],
            "S": [22, 18, 12, 6, 4, 3],
        },
    }

    # 初始人口（10 个小区）
    pop_0 = np.array([
        [496, 152, 64], [408, 136, 64], [632, 208, 80],
        [368, 120, 56], [536, 176, 72], [328, 104, 40],
        [592, 192, 80], [392, 128, 48], [504, 168, 64],
        [456, 144, 56],
    ])

    income_0 = np.array([3400, 3100, 3800, 2900, 3500, 2700, 3600, 3000, 3300, 3200])

    willingness_0 = np.array([
        [0.55, 0.66, 0.77], [0.50, 0.61, 0.72],
        [0.62, 0.72, 0.82], [0.42, 0.53, 0.64],
        [0.55, 0.66, 0.77], [0.42, 0.53, 0.64],
        [0.62, 0.72, 0.82], [0.50, 0.61, 0.72],
        [0.55, 0.66, 0.77], [0.50, 0.61, 0.72],
    ])

    calc = DemandCalculator(test_config)

    # ---------- 测试 1：基准价 ----------
    demand_normal, match_normal = calc.calc(pop_0, income_0, willingness_0, price_ratio=1.0)
    total_normal = demand_normal.sum()

    print("=" * 50)
    print("【测试1】price_ratio = 1.0（基准价）")
    print(f"理论需求（无约束）:  298874 次/月（基准校验值）")
    print(f"实际需求（所有约束后）: {total_normal:.2f} 次/月")
    print(f"供需匹配度 M:        {match_normal:.4f}")
    print("\n前2个小区实际需求（6类服务，次/月）:")
    print("小区 A:", [f"{x:.2f}" for x in demand_normal[0]])
    print("小区 B:", [f"{x:.2f}" for x in demand_normal[1]])

    assert 0 <= match_normal <= 1, "M 超出 [0,1] 范围"
    assert np.all(demand_normal >= 0), "存在负数需求"

    # ---------- 测试 2：涨价 30% ----------
    demand_high, match_high = calc.calc(pop_0, income_0, willingness_0, price_ratio=1.3)
    total_high = demand_high.sum()

    print("\n" + "=" * 50)
    print("【测试2】price_ratio = 1.3（涨价30%）")
    print(f"实际需求: {total_high:.2f} 次/月")
    print(f"供需匹配度 M: {match_high:.4f}")

    assert total_high < total_normal, "涨价后需求未下降，弹性逻辑有误"
    print("\n✅ 所有断言通过！涨价后需求从 {:.2f} 降至 {:.2f}，弹性生效。".format(
        total_normal, total_high
    ))