"""
site_optimize.py — 选址优化（枚举 + 站点生命周期）
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Tuple, Optional


class SiteOptimizer:
    DEFAULT_WEIGHTS = {"coverage": 0.4, "satisfaction": 0.3, "fulfillment": 0.3}

    def __init__(self, config: Dict[str, Any]):
        self.service_names = list(config["services"]["names"])
        self.block_labels = list(config["communities"]["labels"])
        self.n_blocks = len(self.block_labels)

        site_cfg = config["site"]
        self.total_budget_limit = site_cfg["annual_build_budget"]
        self.service_radius = site_cfg["service_radius"]
        self.scale_forbid_threshold = site_cfg["adapt_threshold"]
        self.new_site_efficiency = site_cfg["new_site_efficiency"]
        self.work_days_per_month = site_cfg["work_days_per_month"]
        self.salvage_ratio = site_cfg.get("salvage_ratio", 0.3)

        fac_cfg = config["facility"]
        self.site_build_cost = dict(fac_cfg["build_cost"])
        self.site_daily_max_people = dict(fac_cfg["daily_max_capacity"])

        self.community_adapt = dict(config["communities"]["adapt"])

        labor_cfg = config.get("labor", {})
        self.staff_supply = labor_cfg.get(
            "staff_supply",
            {cid: 1.0 for cid in self.block_labels},
        )

        self.weights = dict(self.DEFAULT_WEIGHTS)

    # ===================== 主入口 =====================
    def optimize(
        self,
        demand: np.ndarray,
        distance_matrix: np.ndarray,
        history_sites: Optional[List[dict]] = None,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        枚举所有建站组合，考虑站点生命周期。
        :param history_sites: 上一年保留的站点 [{"cid","scale","build_cost","build_year"}]
        :param budget: 年度净投资预算上限
        """
        demand = np.asarray(demand, dtype=float)
        distances = np.asarray(distance_matrix, dtype=float)

        if history_sites is None:
            history_sites = []
        if budget is None:
            budget = self.total_budget_limit

        history_dict = {s["cid"]: s for s in history_sites}
        history_cids = set(history_dict.keys())

        best_result = None
        best_score = -1.0

        for r in range(1, self.n_blocks + 1):
            for combo in combinations(range(self.n_blocks), r):
                result = self._evaluate_combo(
                    combo, demand, distances, history_dict, history_cids, budget,
                )
                if result is None:
                    continue
                if result["_score"] > best_score:
                    best_score = result["_score"]
                    best_result = result

        if best_result is None:
            raise RuntimeError("枚举未找到可行方案")

        # 站点变动汇总
        combo_cids = {s["cid"] for s in best_result["selected_sites"]}
        kept = combo_cids & history_cids
        new = combo_cids - history_cids
        closed = history_cids - combo_cids
        print(f"[站点变动] 保留 {len(kept)} | 新增 {len(new)} | 关停 {len(closed)}")
        print(f"[建设] 新增成本 {best_result['metrics']['total_new_cost']:.0f} "
              f"| 残值回收 {best_result['metrics']['total_salvage']:.0f} "
              f"| 净投资 {best_result['metrics']['net_invest']:.0f}")

        best_result.pop("_score", None)
        return best_result

    # ===================== 评估单个组合 =====================
    def _evaluate_combo(
        self, combo, demand, distances, history_dict, history_cids, budget,
    ):
        combo_cids = {self.block_labels[i] for i in combo}

        sites = []
        total_new_cost = 0.0
        covered_by_any = np.zeros(self.n_blocks, dtype=bool)

        for center in combo:
            cid = self.block_labels[center]
            adapt = self.community_adapt[cid]
            in_radius = distances[center] <= self.service_radius
            regional_demand = float(demand[in_radius].sum())
            scale = self._pick_scale(regional_demand, adapt)
            build_cost = self.site_build_cost[scale]

            if cid in history_cids:
                # 保留已有站点，无需新投资
                is_new = False
                actual_cost = 0.0
            else:
                # 新建站点
                is_new = True
                actual_cost = build_cost
                total_new_cost += actual_cost

            sites.append({
                "site_id": len(sites),
                "cid": cid,
                "center_idx": center,
                "scale": scale,
                "is_new": is_new,
                "covered_blocks": [self.block_labels[i] for i in np.flatnonzero(in_radius)],
                "build_cost": build_cost,
                "effective_monthly_capacity": self._effective_capacity(center, scale),
                "monthly_demand_by_service": {s: 0.0 for s in self.service_names},
            })
            covered_by_any |= in_radius

        # 关停残值
        closed_cids = history_cids - combo_cids
        total_salvage = sum(
            history_dict[cid].get("build_cost", 0) * self.salvage_ratio
            for cid in closed_cids
        )

        net_invest = total_new_cost - total_salvage
        if net_invest > budget:
            return None

        # 分配需求
        assignment, unmet_total, used_capacity = self._allocate(demand, distances, sites)

        total_demand = float(demand.sum())
        assigned_total = float(sum(a["count"] for a in assignment))
        coverage = float(covered_by_any.mean()) if self.n_blocks else 0.0
        fulfillment = assigned_total / total_demand if total_demand > 0 else 1.0
        avg_sat = self._avg_satisfaction(assignment, sites, used_capacity)

        score = (
            self.weights["coverage"] * coverage
            + self.weights["satisfaction"] * avg_sat
            + self.weights["fulfillment"] * fulfillment
        )

        weighted_distance = sum(a["count"] * a["distance"] for a in assignment)
        avg_distance = weighted_distance / assigned_total if assigned_total > 0 else 0.0
        util = used_capacity / np.array(
            [s["effective_monthly_capacity"] for s in sites], dtype=float
        ) if sites else np.array([])
        util = np.clip(util, 0.0, 1.0)

        return {
            "selected_sites": sites,
            "assignment": assignment,
            "metrics": {
                "coverage_rate": coverage,
                "total_new_cost": float(total_new_cost),
                "total_salvage": float(total_salvage),
                "net_invest": float(net_invest),
                "demand_satisfaction_rate": fulfillment,
                "avg_distance": float(avg_distance),
                "avg_satisfaction": float(avg_sat),
                "avg_utilization": float(util.mean()) if len(util) else 0.0,
                "unmet_demand": float(unmet_total),
            },
            "_score": float(score),
        }

    # ===================== 辅助方法 =====================
    def _pick_scale(self, regional_demand: float, adapt: float) -> str:
        cap_small = self.site_daily_max_people["小型"] * self.work_days_per_month
        cap_medium = self.site_daily_max_people["中型"] * self.work_days_per_month
        if regional_demand <= cap_small:
            return "小型"
        elif regional_demand <= cap_medium:
            return "中型"
        else:
            return "大型" if adapt >= self.scale_forbid_threshold else "中型"

    def _effective_capacity(self, center_idx: int, scale: str) -> float:
        cid = self.block_labels[center_idx]
        return (
            self.site_daily_max_people[scale]
            * self.work_days_per_month
            * self.staff_supply.get(cid, 1.0)
            * self.community_adapt[cid]
            * self.new_site_efficiency
        )

    def _allocate(self, demand, distances, sites):
        assignment = []
        used_capacity = np.zeros(len(sites), dtype=float)
        unmet_total = 0.0

        for block_idx in range(self.n_blocks):
            candidate = []
            for s_idx, site in enumerate(sites):
                d = distances[block_idx, site["center_idx"]]
                if d <= self.service_radius:
                    candidate.append((d, s_idx))
            candidate.sort(key=lambda x: x[0])

            if not candidate:
                unmet_total += float(demand[block_idx].sum())
                continue

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

    def _avg_satisfaction(self, assignment, sites, used_capacity):
        if not assignment:
            return 0.0
        site_idx_by_cid = {s["cid"]: i for i, s in enumerate(sites)}
        total_count = 0.0
        weighted = 0.0
        for a in assignment:
            s_idx = site_idx_by_cid[a["site"]]
            cap = sites[s_idx]["effective_monthly_capacity"]
            util = min(max(used_capacity[s_idx] / cap if cap > 0 else 0.0, 0.0), 1.0)
            d = a["distance"]
            s1 = 1.0 if d <= 300 else 0.9 if d <= 500 else 0.75 if d <= 650 else 0.6
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
            sat = 0.2 * s1 + 0.3 * s2 + 0.5 * 1.0
            weighted += a["count"] * sat
            total_count += a["count"]
        return weighted / total_count if total_count > 0 else 0.0