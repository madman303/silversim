"""
price_subsidy.py — 财务测算模块（接口式）

对外接口：
    finance = SiteFinance(config)
    result = finance.calc(scale, monthly_demand, price_ratio)

config 结构（对应 config.yaml）：
    services.names / revenue / variable_cost
    facility.build_cost / daily_fix_cost / daily_max_capacity / min_nurse_count
    finance.max_profit_ratio / depreciation_years / nurse_monthly_salary
    site.work_days_per_month
"""
from __future__ import annotations

from typing import Dict, Any


# ===================== 核心类 =====================
class SiteFinance:
    """站点财务测算器，所有参数从 config 注入。"""

    def __init__(self, config: Dict[str, Any]):
        svc_cfg = config["services"]
        fac_cfg = config["facility"]
        fin_cfg = config["finance"]
        site_cfg = config["site"]

        self.service_names = list(svc_cfg["names"])
        self.service_rev = dict(svc_cfg["revenue"])
        self.service_var_cost = dict(svc_cfg["variable_cost"])

        self.build_cost = dict(fac_cfg["build_cost"])
        self.daily_fix_cost = dict(fac_cfg["daily_fix_cost"])
        self.daily_max_capacity = dict(fac_cfg["daily_max_capacity"])
        self.nurse_count = dict(fac_cfg["min_nurse_count"])

        self.work_days_per_month = site_cfg["work_days_per_month"]
        self.max_profit_ratio = fin_cfg["max_profit_ratio"]
        self.depreciation_years = fin_cfg["depreciation_years"]
        self.nurse_monthly_salary = fin_cfg["nurse_monthly_salary"]

    # ---------- 月度容量上限 ----------
    def get_monthly_max_demand(self, scale: str) -> float:
        return self.daily_max_capacity[scale] * self.work_days_per_month

    # ---------- 月度财务测算 ----------
    def calc(
        self,
        scale: str,
        monthly_demand: Dict[str, float],
        price_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """
        计算单站点月度运营端财务。
        :param scale: "小型" | "中型" | "大型"
        :param monthly_demand: {服务名: 次数}
        :param price_ratio: 价格相对基准倍数
        :return: 财务指标 dict
        """
        if scale not in self.build_cost:
            raise ValueError(f"非法站点规模 {scale}")

        build_cost = self.build_cost[scale]

        # 月度固定成本 = 场地管理费 + 护理员工资
        monthly_fixed = (
            self.daily_fix_cost[scale] * self.work_days_per_month
            + self.nurse_count[scale] * self.nurse_monthly_salary
        )

        monthly_rev = 0.0
        monthly_var = 0.0
        total_cnt = 0.0
        for srv, cnt in monthly_demand.items():
            monthly_rev += cnt * self.service_rev[srv] * price_ratio
            monthly_var += cnt * self.service_var_cost[srv]
            total_cnt += cnt

        # 容量校验
        max_total = self.get_monthly_max_demand(scale)
        if total_cnt > max_total + 1e-6:
            raise ValueError(
                f"{scale}站点，月度总人次{total_cnt:.2f}超过容量上限{max_total:.2f}"
            )

        monthly_op_profit = monthly_rev - monthly_var - monthly_fixed
        profit_ratio = (
            monthly_op_profit / monthly_rev if abs(monthly_rev) > 1e-9 else 0.0
        )

        return {
            "scale": scale,
            "build_cost": build_cost,
            "monthly_revenue": round(monthly_rev, 2),
            "monthly_var_cost": round(monthly_var, 2),
            "monthly_fixed_cost": round(monthly_fixed, 2),
            "monthly_total_demand": round(total_cnt, 2),
            "monthly_max_capacity": round(max_total, 2),
            "monthly_profit": round(monthly_op_profit, 2),
            "profit_ratio": round(profit_ratio, 4),
        }

    # ---------- 财务约束校验 ----------
    def check_constraint(self, profit_ratio: float) -> tuple[bool, list[str]]:
        msgs = []
        ok = True
        if profit_ratio > self.max_profit_ratio + 1e-6:
            ok = False
            msgs.append(f"利润率 {profit_ratio:.2%} 超过上限{self.max_profit_ratio:.0%}")
        if profit_ratio < -1e-6:
            msgs.append(f"警告：站点亏损，利润率 {profit_ratio:.2%}")
        return ok, msgs


# ===================== 模块级函数接口 =====================
def calc_site_monthly_finance(
    scale: str,
    monthly_demand: Dict[str, float],
    config: Dict[str, Any],
    price_ratio: float = 1.0,
) -> Dict[str, Any]:
    finance = SiteFinance(config)
    return finance.calc(scale, monthly_demand, price_ratio)


def check_financial_constraint(
    profit_ratio: float,
    config: Dict[str, Any],
) -> tuple[bool, list[str]]:
    finance = SiteFinance(config)
    return finance.check_constraint(profit_ratio)


# ===================== 自测 =====================
if __name__ == "__main__":
    test_config = {
        "services": {
            "names": ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"],
            "revenue": {
                "助餐": 10, "日间照料": 20, "上门护理": 30,
                "康复理疗": 28, "助浴": 25, "紧急救助": 0,
            },
            "variable_cost": {
                "助餐": 8, "日间照料": 16, "上门护理": 24,
                "康复理疗": 23, "助浴": 20, "紧急救助": 8,
            },
        },
        "facility": {
            "build_cost": {"小型": 180000, "中型": 320000, "大型": 450000},
            "daily_fix_cost": {"小型": 2000, "中型": 3200, "大型": 4400},
            "daily_max_capacity": {"小型": 1000, "中型": 2000, "大型": 3000},
            "min_nurse_count": {"小型": 2, "中型": 5, "大型": 10},
        },
        "finance": {
            "max_profit_ratio": 0.08,
            "depreciation_years": 20,
            "nurse_monthly_salary": 5000,
        },
        "site": {
            "work_days_per_month": 22,
        },
    }

    finance = SiteFinance(test_config)

    demo_demand = {
        "助餐": 5000, "日间照料": 3000, "上门护理": 1200,
        "康复理疗": 800, "助浴": 300, "紧急救助": 200,
    }

    res = finance.calc("中型", demo_demand, price_ratio=1.0)
    ok, msgs = finance.check_constraint(res["profit_ratio"])

    print("【price_subsidy 自测】")
    for k, v in res.items():
        print(f"  {k:20s}: {v}")
    print(f"约束校验: {ok}，消息: {msgs}")