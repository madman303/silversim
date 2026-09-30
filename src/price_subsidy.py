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

        # ---------- 容量工时化参数（P0-1）----------
        # 与 site_optimize 使用同一套口径，避免"选址按工时算、财务按场地算"的校验冲突
        self.service_duration = dict(fac_cfg["service_duration"])
        self.nurse_daily_minutes = fac_cfg["nurse_daily_minutes"]
        self.service_names = list(svc_cfg["names"])
        self.avg_service_duration = float(
            sum(self.service_duration[s] for s in self.service_names) / len(self.service_names)
        )
        # 可选：小区级人力供应与设施适配系数（不提供时按 1.0）
        self.staff_supply = dict(config.get("labor", {}).get("staff_supply", {}))
        self.community_adapt = dict(config.get("communities", {}).get("adapt", {}))

    # ---------- 月度容量上限 ----------
    def get_monthly_labor_minutes(self, scale: str, cid: str | None = None) -> float:
        """
        月度护理员工时预算（分钟）= 人数 × 每日有效分钟 × 月工作日。
        传入 cid 时一并乘上该小区的人力供应与设施适配系数，与选址模块口径一致。
        """
        minutes = (
            self.nurse_count[scale] * self.nurse_daily_minutes * self.work_days_per_month
        )
        if cid is not None:
            minutes *= self.staff_supply.get(cid, 1.0) * self.community_adapt.get(cid, 1.0)
        return minutes

    def get_monthly_max_demand(self, scale: str, cid: str | None = None) -> float:
        """
        月度有效容量（等权口径服务次数）= min(场地容量, 人力容量)。
        人力容量 = 护理员工时预算 ÷ 等权平均单次工时。
        该值用于展示与规模选择；实际分配与超载校验以"护理员分钟"为准。
        """
        site_cap = self.daily_max_capacity[scale] * self.work_days_per_month
        labor_cap = self.get_monthly_labor_minutes(scale, cid=cid) / self.avg_service_duration
        return min(site_cap, labor_cap)

    # ---------- 月度财务测算 ----------
    def calc(
        self,
        scale: str,
        monthly_demand: Dict[str, float],
        price_ratio: float = 1.0,
        cid: str | None = None,
        max_capacity: float | None = None,
        labor_minutes_limit: float | None = None,
    ) -> Dict[str, Any]:
        """
        计算单站点月度运营端财务。
        :param scale: "小型" | "中型" | "大型"
        :param monthly_demand: {服务名: 次数}
        :param price_ratio: 价格相对基准倍数
        :param cid: 站点所在小区（用于按小区系数计算容量上限）
        :param max_capacity: 直接指定容量上限（等权口径服务次数）
        :param labor_minutes_limit: 直接指定月度护理员工时上限（分钟），优先用于超载校验
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
        total_labor_minutes = 0.0
        for srv, cnt in monthly_demand.items():
            monthly_rev += cnt * self.service_rev[srv] * price_ratio
            monthly_var += cnt * self.service_var_cost[srv]
            total_cnt += cnt
            total_labor_minutes += cnt * self.service_duration.get(srv, 0.0)

        # 容量校验：真实约束是护理员工时，故按分钟校验（与选址模块分配口径一致）
        if labor_minutes_limit is not None:
            limit_minutes = float(labor_minutes_limit)
        else:
            limit_minutes = self.get_monthly_labor_minutes(scale, cid=cid)
        if total_labor_minutes > limit_minutes + 1e-6:
            raise ValueError(
                f"{scale}站点，月度护理工时{total_labor_minutes:.2f}分钟"
                f"超过工时上限{limit_minutes:.2f}分钟"
                f"（合计{total_cnt:.2f}人次，等权容量上限"
                f"{self.get_monthly_max_demand(scale, cid=cid):.2f}人次）"
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
            # 容量口径（P0-1）：等权次数上限用于展示，工时上限才是真实约束
            "monthly_max_capacity": round(
                float(max_capacity) if max_capacity is not None
                else self.get_monthly_max_demand(scale, cid=cid), 2
            ),
            "monthly_labor_minutes": round(total_labor_minutes, 2),
            "monthly_labor_minutes_limit": round(limit_minutes, 2),
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
    cid: str | None = None,
    max_capacity: float | None = None,
    labor_minutes_limit: float | None = None,
) -> Dict[str, Any]:
    finance = SiteFinance(config)
    return finance.calc(
        scale, monthly_demand, price_ratio,
        cid=cid, max_capacity=max_capacity, labor_minutes_limit=labor_minutes_limit,
    )


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
            "service_duration": {
                "助餐": 15, "日间照料": 120, "上门护理": 60,
                "康复理疗": 40, "助浴": 60, "紧急救助": 30,
            },
            "nurse_daily_minutes": 480,
        },
        "finance": {
            "max_profit_ratio": 0.08,
            "depreciation_years": 20,
            "nurse_monthly_salary": 5000,
        },
        "site": {
            "work_days_per_month": 22,
        },
        "communities": {
            "adapt": {"A": 1.0},
        },
    }

    finance = SiteFinance(test_config)

    # 中型站人力容量 = 5 × 480 × 22 ÷ 54.17 = 975 次/月（场地容量 44000，人力才是瓶颈）
    demo_demand = {
        "助餐": 500, "日间照料": 150, "上门护理": 100,
        "康复理疗": 80, "助浴": 50, "紧急救助": 31,
    }

    res = finance.calc("中型", demo_demand, price_ratio=1.0)
    ok, msgs = finance.check_constraint(res["profit_ratio"])

    print("【price_subsidy 自测】")
    for k, v in res.items():
        print(f"  {k:20s}: {v}")
    print(f"约束校验: {ok}，消息: {msgs}")

    # 超容量必须报错（容量工时化后，旧口径放行的量会被拦下）
    over_demand = dict(demo_demand)
    over_demand["助餐"] = 3000
    try:
        finance.calc("中型", over_demand, price_ratio=1.0)
        print("❌ 超容量未拦截")
    except ValueError as e:
        print(f"✅ 超容量拦截生效: {e}")