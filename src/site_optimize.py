"""
site_optimize.py — 选址优化 + 方案约束校验（接口式 + 枚举算法）

对外接口：
    optimizer = SiteOptimizer(config)
    result = optimizer.optimize(demand, distance_matrix)   # 枚举最优
    sites = optimizer.greedy_allocate(demand)              # 保留贪心兼容

config 结构（对应 config.yaml）：
    services.names
    site.annual_build_budget / service_radius / adapt_threshold
         / new_site_efficiency / work_days_per_month
    facility.build_cost / daily_max_capacity
    communities.labels / adapt
    labor.staff_supply (若存在)
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Tuple


# ===================== 核心类 =====================
class SiteOptimizer:
    """选址优化器，枚举所有建站组合，返回评分最优方案。"""

    # 评分权重（可调整，会写进 config 里）
    DEFAULT_WEIGHTS = {
        "coverage": 0.4,
        "satisfaction": 0.3,
        "fulfillment": 0.3,
    }

    def __init__(self, config: Dict[str, Any]):
        self.service_names = list(config["services"]["names"])
        self.block_labels = list(config["communities"]["labels"])
        n = len(self.block_labels)
        self.n_blocks = n

        site_cfg = config["site"]
        self.total_budget_limit = site_cfg["annual_build_budget"]
        self.service_radius = site_cfg["service_radius"]
        self.scale_forbid_threshold = site_cfg["adapt_threshold"]
        self.new_site_efficiency = site_cfg["new_site_efficiency"]
        self.work_days_per_month = site_cfg["work_days_per_month"]

        fac_cfg = config["facility"]
        self.site_build_cost = dict(fac_cfg["build_cost"])
        self.site_daily_max_people = dict(fac_cfg["daily_max_capacity"])

        self.community_adapt = dict(config["communities"]["adapt"])

        # 人力供给（可选，缺省时全部视为 1.0）
        labor_cfg = config.get("labor", {})
        self.staff_supply = labor_cfg.get(
            "staff_supply",
            {cid: 1.0 for cid in self.block_labels},
        )

        # 评分权重
        self.weights = dict(self.DEFAULT_WEIGHTS)

    # ===================== 主入口：枚举最优 =====================
    def optimize(
        self,
        demand: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        枚举所有建站组合，返回评分最高的方案。
        :param demand: (n, 6) 需求矩阵
        :param distance_matrix: (n, n) 距离矩阵
        :return: dict，字段与旧版 optimize_sites 一致
        """
        demand = np.asarray(demand, dtype=float)
        distances = np.asarray(distance_matrix, dtype=float)

        if demand.shape != (self.n_blocks, len(self.service_names)):
            raise ValueError(f"demand 形状应为 ({self.n_blocks}, 6)")
        if distances.shape != (self.n_blocks, self.n_blocks):
            raise ValueError(f"distance_matrix 形状应为 ({self.n_blocks}, {self.n_blocks})")

        best_result = None
        best_score = -1.0
        n_evaluated = 0
        n_discarded = 0

        # 枚举所有非空子集（2^n - 1 种）
        for r in range(1, self.n_blocks + 1):
            for combo in combinations(range(self.n_blocks), r):
                # 建站方案：combo 中的索引建站
                result = self._evaluate_combo(combo, demand, distances)

                if result is None:
                    n_discarded += 1
                    continue

                n_evaluated += 1
                if result["_score"] > best_score:
                    best_score = result["_score"]
                    best_result = result

        if best_result is None:
            raise RuntimeError("枚举未找到可行方案（预算或适配度约束过严）")

        # 输出诊断信息
        print(f"[枚举选址] 共枚举 {2**self.n_blocks - 1} 种组合，"
              f"可行 {n_evaluated} 种，淘汰 {n_discarded} 种")
        print(f"[枚举选址] 最优得分 = {best_score:.4f}")

        # 清理内部字段
        best_result.pop("_score", None)
        return best_result

    # ===================== 评估单个组合 =====================
    def _evaluate_combo(
        self,
        combo: Tuple[int, ...],
        demand: np.ndarray,
        distances: np.ndarray,
    ) -> Dict[str, Any] | None:
        """
        评估一个建站组合：
        1. 为每个站点确定规模
        2. 检查预算
        3. 做需求分配
        4. 计算评分
        不满足约束时返回 None。
        """
        # ---------- Step 1: 为每个站点确定规模 ----------
        sites = []
        total_cost = 0
        covered_by_any = np.zeros(self.n_blocks, dtype=bool)

        for center in combo:
            cid = self.block_labels[center]
            adapt = self.community_adapt[cid]

            # 该站点半径内的所有小区
            in_radius = distances[center] <= self.service_radius
            regional_demand = float(demand[in_radius].sum())

            # 根据区域需求选最小可用规模
            scale = self._pick_scale(regional_demand, adapt)

            build_cost = self.site_build_cost[scale]
            total_cost += build_cost

            sites.append({
                "site_id": len(sites),
                "cid": cid,
                "center_idx": center,
                "scale": scale,
                "covered_blocks": [
                    self.block_labels[i] for i in np.flatnonzero(in_radius)
                ],
                "build_cost": build_cost,
                "effective_monthly_capacity": self._effective_capacity(center, scale),
                "monthly_demand_by_service": {s: 0.0 for s in self.service_names},
                "_in_radius": in_radius,
            })
            covered_by_any |= in_radius

        # ---------- Step 2: 预算检查 ----------
        if total_cost > self.total_budget_limit:
            return None

        # ---------- Step 3: 分配需求 ----------
        assignment, unmet_total, used_capacity = self._allocate(
            demand, distances, sites
        )

        # ---------- Step 4: 计算指标 ----------
        total_demand = float(demand.sum())
        assigned_total = float(sum(a["count"] for a in assignment))

        # 覆盖率
        coverage = float(covered_by_any.mean()) if self.n_blocks else 0.0

        # 满足率
        fulfillment = assigned_total / total_demand if total_demand > 0 else 1.0

        # 平均满意度
        avg_sat = self._avg_satisfaction(assignment, sites, used_capacity)

        # 评分
        score = (
            self.weights["coverage"] * coverage
            + self.weights["satisfaction"] * avg_sat
            + self.weights["fulfillment"] * fulfillment
        )

        # 加权距离
        weighted_distance = sum(a["count"] * a["distance"] for a in assignment)
        avg_distance = weighted_distance / assigned_total if assigned_total > 0 else 0.0

        # 利用率
        util = used_capacity / np.array(
            [s["effective_monthly_capacity"] for s in sites], dtype=float
        ) if sites else np.array([])
        util = np.clip(util, 0.0, 1.0)

        return {
            "selected_sites": sites,
            "assignment": assignment,
            "metrics": {
                "coverage_rate": coverage,
                "total_cost": float(total_cost),
                "remaining_budget": self.total_budget_limit - total_cost,
                "demand_satisfaction_rate": fulfillment,
                "avg_distance": float(avg_distance),
                "avg_satisfaction": float(avg_sat),
                "avg_utilization": float(util.mean()) if len(util) else 0.0,
                "unmet_demand": float(unmet_total),
            },
            "_score": float(score),
        }

    # ===================== 规模选择 =====================
    def _pick_scale(self, regional_demand: float, adapt: float) -> str:
        """根据区域需求选最小可用规模；适配度<阈值时禁大型。"""
        cap_small = self.site_daily_max_people["小型"] * self.work_days_per_month
        cap_medium = self.site_daily_max_people["中型"] * self.work_days_per_month

        if regional_demand <= cap_small:
            return "小型"
        elif regional_demand <= cap_medium:
            return "中型"
        else:
            # 需要大型，但检查适配度
            if adapt < self.scale_forbid_threshold:
                return "中型"  # 强行降级
            return "大型"

    # ===================== 有效容量 =====================
    def _effective_capacity(self, center_idx: int, scale: str) -> float:
        cid = self.block_labels[center_idx]
        daily_max = self.site_daily_max_people[scale]
        staff = self.staff_supply.get(cid, 1.0)
        adapt = self.community_adapt[cid]
        return (
            daily_max
            * self.work_days_per_month
            * staff
            * adapt
            * self.new_site_efficiency
        )

    # ===================== 需求分配 =====================
    def _allocate(
        self,
        demand: np.ndarray,
        distances: np.ndarray,
        sites: List[dict],
    ) -> Tuple[List[dict], float, np.ndarray]:
        """按距离就近原则分配需求到各站点。"""
        assignment: List[dict] = []
        used_capacity = np.zeros(len(sites), dtype=float)
        unmet_total = 0.0

        for block_idx in range(self.n_blocks):
            # 找出距离该小区 <= 半径的站点，按距离排序
            candidate = []
            for s_idx, site in enumerate(sites):
                d = distances[block_idx, site["center_idx"]]
                if d <= self.service_radius:
                    candidate.append((d, s_idx))
            candidate.sort(key=lambda x: x[0])

            if not candidate:
                unmet_total += float(demand[block_idx].sum())
                continue

            # 逐类服务分配
            for srv_idx, srv_name in enumerate(self.service_names):
                remaining = float(demand[block_idx, srv_idx])
                if remaining <= 0:
                    continue
                for d, s_idx in candidate:
                    available = sites[s_idx]["effective_monthly_capacity"] - used_capacity[s_idx]
                    assigned = min(remaining, max(available, 0.0))
                    if assigned <= 0:
                        continue
                    used_capacity[s_idx] += assigned
                    remaining -= assigned
                    sites[s_idx]["monthly_demand_by_service"][srv_name] += assigned
                    assignment.append({
                        "block": self.block_labels[block_idx],
                        "site": sites[s_idx]["cid"],
                        "service": srv_name,
                        "count": assigned,
                        "distance": float(d),
                    })
                    if remaining <= 1e-12:
                        break
                unmet_total += max(remaining, 0.0)

        return assignment, unmet_total, used_capacity

    # ===================== 平均满意度 =====================
    def _avg_satisfaction(
        self,
        assignment: List[dict],
        sites: List[dict],
        used_capacity: np.ndarray,
    ) -> float:
        if not assignment:
            return 0.0

        # 价格满意度（基准价时=1.0）
        price_score = 1.0

        site_index_by_cid = {s["cid"]: i for i, s in enumerate(sites)}
        total_count = 0.0
        weighted_sat = 0.0

        for a in assignment:
            s_idx = site_index_by_cid[a["site"]]
            capacity = sites[s_idx]["effective_monthly_capacity"]
            util = used_capacity[s_idx] / capacity if capacity > 0 else 0.0
            util = min(max(util, 0.0), 1.0)

            # 距离满意度
            d = a["distance"]
            if d <= 300:
                s1 = 1.0
            elif d <= 500:
                s1 = 0.9
            elif d <= 650:
                s1 = 0.75
            else:
                s1 = 0.6

            # 响应满意度
            if util <= 0.60:
                s2 = 1.0
            elif util <= 0.75:
                s2 = 0.93
            elif util <= 0.85:
                s2 = 0.85
            elif util <= 0.95:
                s2 = 0.72
            else:
                s2 = 0.60

            # 价格满意度
            s3 = price_score

            sat = 0.2 * s1 + 0.3 * s2 + 0.5 * s3
            weighted_sat += a["count"] * sat
            total_count += a["count"]

        return weighted_sat / total_count if total_count > 0 else 0.0

    # ===================== 保留贪心版（兼容旧代码） =====================
    def greedy_allocate(self, block_demand: np.ndarray) -> List[dict]:
        n = block_demand.shape[0]
        block_total = block_demand.sum(axis=1)
        idx_sorted = np.argsort(-block_total)
        site_list: List[dict] = []

        for bid in idx_sorted:
            cid_label = self.block_labels[bid]
            req_dict = dict(zip(self.service_names, block_demand[bid].tolist()))
            blk_sum = float(block_total[bid])

            placed = False
            for site in site_list:
                exist_sum = sum(site["monthly_demand"].values())
                cap = self.site_daily_max_people[site["scale"]] * self.work_days_per_month
                if exist_sum + blk_sum <= cap:
                    for srv in self.service_names:
                        site["monthly_demand"][srv] += req_dict[srv]
                    site["blocks"].append(cid_label)
                    placed = True
                    break
            if placed:
                continue

            scale = self._pick_scale(blk_sum, self.community_adapt[cid_label])
            site_list.append({
                "cid": cid_label,
                "scale": scale,
                "blocks": [cid_label],
                "monthly_demand": req_dict.copy(),
            })
        return site_list

    # ===================== 方案校验 =====================
    def check_scheme_valid(self, site_list: List[dict]) -> Tuple[bool, List[str]]:
        errors = []
        for s in site_list:
            cid = s["cid"]
            scale = s["scale"]
            a_i = self.community_adapt[cid]
            if scale not in self.site_build_cost:
                errors.append(f"小区{cid} 非法站点规模:{scale}")
                continue
            if scale == "大型" and a_i < self.scale_forbid_threshold:
                errors.append(f"小区{cid}适配度{a_i}<{self.scale_forbid_threshold}，禁止建设大型站点")
        return len(errors) == 0, errors


# ===================== 模块级函数接口 =====================
def optimize_sites(
    demand: np.ndarray,
    distance_matrix: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """函数式接口：枚举选址。"""
    optimizer = SiteOptimizer(config)
    return optimizer.optimize(demand, distance_matrix)


def greedy_site_allocate(
    block_demand: np.ndarray,
    config: Dict[str, Any],
) -> List[dict]:
    """函数式接口：贪心选址（兼容）。"""
    optimizer = SiteOptimizer(config)
    return optimizer.greedy_allocate(block_demand)


def check_site_scheme_valid(
    site_list: List[dict],
    config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """函数式接口：方案校验。"""
    optimizer = SiteOptimizer(config)
    return optimizer.check_scheme_valid(site_list)


# ===================== 自测 =====================
if __name__ == "__main__":
    test_config = {
        "services": {
            "names": ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"],
        },
        "site": {
            "annual_build_budget": 1200000,
            "service_radius": 1000,
            "adapt_threshold": 0.7,
            "new_site_efficiency": 0.8,
            "work_days_per_month": 22,
        },
        "facility": {
            "build_cost": {"小型": 180000, "中型": 320000, "大型": 450000},
            "daily_max_capacity": {"小型": 1000, "中型": 2000, "大型": 3000},
        },
        "communities": {
            "labels": list("ABCDEFGHIJ"),
            "adapt": {
                "A": 0.8, "B": 0.7, "C": 0.9, "D": 0.55, "E": 0.85,
                "F": 0.5, "G": 0.85, "H": 0.65, "I": 0.75, "J": 0.72,
            },
        },
    }

    distances = np.array([
        [0, 600, 1200, 900, 1500, 1800, 1300, 700, 1100, 500],
        [600, 0, 800, 500, 1100, 1400, 900, 400, 700, 300],
        [1200, 800, 0, 700, 600, 900, 500, 900, 600, 700],
        [900, 500, 700, 0, 800, 1100, 600, 300, 500, 400],
        [1500, 1100, 600, 800, 0, 500, 400, 1000, 500, 800],
        [1800, 1400, 900, 1100, 500, 0, 500, 1200, 700, 1100],
        [1300, 900, 500, 600, 400, 500, 0, 800, 400, 600],
        [700, 400, 900, 300, 1000, 1200, 800, 0, 600, 300],
        [1100, 700, 600, 500, 500, 700, 400, 600, 0, 400],
        [500, 300, 700, 400, 800, 1100, 600, 300, 400, 0],
    ])

    # 模拟需求
    demo_demand = np.array([
        [5000, 3000, 2000, 1500, 800, 500],
        [4000, 2500, 1600, 1200, 600, 400],
        [6000, 4000, 2500, 1800, 1000, 600],
        [3500, 2200, 1400, 1000, 550, 350],
        [5200, 3200, 2100, 1500, 850, 520],
        [3000, 1900, 1200, 900, 450, 300],
        [5800, 3800, 2400, 1700, 950, 580],
        [3800, 2400, 1500, 1100, 580, 380],
        [5000, 3100, 2000, 1400, 800, 500],
        [4500, 2800, 1800, 1300, 700, 450],
    ], dtype=float)

    optimizer = SiteOptimizer(test_config)
    result = optimizer.optimize(demo_demand, distances)

    print(f"\n最优方案：{len(result['selected_sites'])} 个站点")
    for s in result["selected_sites"]:
        print(f"  {s['cid']} {s['scale']} 成本={s['build_cost']} 覆盖={s['covered_blocks']}")
    print(f"\n指标：")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")