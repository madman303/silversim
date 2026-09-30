"""
site_optimize.py — 选址优化（枚举 + 站点生命周期）
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Tuple, Optional


class SiteOptimizer:
    # 评分权重默认值（可在 config.yaml 的 site.weights 覆盖）。
    # kept_ratio = 保留上一年已有站点的比例，用于抑制"拆旧建新"的系统性浪费。
    DEFAULT_WEIGHTS = {
        "coverage": 0.35,
        "satisfaction": 0.25,
        "fulfillment": 0.25,
        "kept_ratio": 0.15,
    }

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
        # 成本效率基准（次/万元）：本轮仅用于输出指标，尚未进入评分
        self.benchmark_cost_per_demand = float(
            site_cfg.get("benchmark_cost_per_demand", 11.4)
        )

        fac_cfg = config["facility"]
        self.site_build_cost = dict(fac_cfg["build_cost"])
        self.site_daily_max_people = dict(fac_cfg["daily_max_capacity"])
        self.nurse_count = dict(fac_cfg["min_nurse_count"])

        # ---------- 容量工时化参数（P0-1）----------
        # 场地容量回答"能进多少人"，人力容量回答"能干多少活"，两者取小值才是真实容量。
        self.service_duration = dict(fac_cfg["service_duration"])   # 单次服务护理员分钟
        self.nurse_daily_minutes = fac_cfg["nurse_daily_minutes"]   # 每名护理员每日有效分钟
        # 等权平均工时：用于把"护理员分钟预算"折算为"等权口径的服务次数"
        self.avg_service_duration = float(
            np.mean([self.service_duration[s] for s in self.service_names])
        )

        self.community_adapt = dict(config["communities"]["adapt"])

        labor_cfg = config.get("labor", {})
        self.staff_supply = labor_cfg.get(
            "staff_supply",
            {cid: 1.0 for cid in self.block_labels},
        )

        self.weights = {**self.DEFAULT_WEIGHTS, **site_cfg.get("weights", {})}
        # 工时不足时的分配顺序策略（见 _allocate 文档）
        self.allocation_mode = site_cfg.get("allocation_mode", "proportional")

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
        _bm = best_result['metrics']
        print(f"[工时] 月工时预算 {_bm['labor_minutes_available']:.0f} 分钟 "
              f"| 已用 {_bm['total_labor_minutes_used']:.0f} 分钟 "
              f"| 利用率 {_bm['avg_utilization']:.2%}")
        print(f"[成本效率] {_bm['cost_per_demand_wan']:.2f} 次/万元 "
              f"(基准 {self.benchmark_cost_per_demand:.1f})")
        if history_sites:
            print(f"[延续性] 保留老站 {_bm['kept_count']}/{len(history_sites)} "
                  f"= {_bm['kept_ratio']:.2%}")

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
            in_radius = distances[center] <= self.service_radius
            regional_demand = float(demand[in_radius].sum())

            # 先取规模（与站点新老无关），再按新老决定是否打折
            scale = self._pick_scale(center, regional_demand)
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

            # 站龄：新站为 0（建成当年），老站沿用历史，用于判定磨合期折减是否生效
            site_age = 0 if is_new else int(history_dict[cid].get("age", 1))

            sites.append({
                "site_id": len(sites),
                "cid": cid,
                "center_idx": center,
                "scale": scale,
                "is_new": is_new,
                "site_age": site_age,
                "covered_blocks": [self.block_labels[i] for i in np.flatnonzero(in_radius)],
                "build_cost": build_cost,
                "effective_monthly_capacity": self._effective_capacity(
                    center, scale, is_new, site_age
                ),
                # 容量工时化（P0-1）：容量以"护理员分钟预算"为真实约束
                "labor_minutes_budget": self._monthly_labor_minutes(
                    center, scale, is_new, site_age
                ),
                "labor_minutes_used": 0.0,
                # 容量瓶颈在"人力"还是"场地"（用于说明场地容量是否真的从未成为约束）
                "binding_constraint": self._binding_constraint(center, scale),
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
        assignment, unmet_total, used_capacity = self._allocate(
            demand, distances, sites, mode=self.allocation_mode
        )

        total_demand = float(demand.sum())
        assigned_total = float(sum(a["count"] for a in assignment))
        coverage = float(covered_by_any.mean()) if self.n_blocks else 0.0
        fulfillment = assigned_total / total_demand if total_demand > 0 else 1.0
        avg_sat = self._avg_satisfaction(assignment, sites, used_capacity)

        kept_cids = combo_cids & history_cids
        # 保留老站比例：无历史站点时视为 1.0（首年不存在"该不该保留"的问题）
        kept_ratio = len(kept_cids) / len(history_cids) if history_cids else 1.0

        score = (
            self.weights["coverage"] * coverage
            + self.weights["satisfaction"] * avg_sat
            + self.weights["fulfillment"] * fulfillment
            + self.weights.get("kept_ratio", 0.0) * kept_ratio
        )

        weighted_distance = sum(a["count"] * a["distance"] for a in assignment)
        avg_distance = weighted_distance / assigned_total if assigned_total > 0 else 0.0

        # ---------- 工时口径利用率（P0-1）----------
        # 真实约束 = 护理员分钟预算，故利用率 = 已用工时 / 月工时预算；饱和时为 100%
        labor_util_list, total_minutes_used = [], 0.0
        for s in sites:
            budget = s["labor_minutes_budget"]
            used_min = s["labor_minutes_used"]
            total_minutes_used += used_min
            if budget > 0 and used_min > 0:
                labor_util_list.append(used_min / budget)
        labor_util = float(np.mean(labor_util_list)) if labor_util_list else 0.0

        # ---------- 成本效率（本轮只入指标，不参与评分）----------
        # 口径：园区内所有选中站点的建设成本（老站按原值计入，保证跨年可比）
        build_cost_basis = float(sum(s["build_cost"] for s in sites))
        if build_cost_basis > 0:
            cost_per_demand = assigned_total / (build_cost_basis / 10000.0)  # 次/万元
            cost_efficiency = float(np.clip(
                cost_per_demand / self.benchmark_cost_per_demand, 0.0, 1.0
            ))
        else:
            cost_per_demand, cost_efficiency = 0.0, 0.0

        return {
            "selected_sites": sites,
            "assignment": assignment,
            "metrics": {
                "coverage_rate": coverage,
                "total_new_cost": float(total_new_cost),
                "total_salvage": float(total_salvage),
                "net_invest": float(net_invest),
                "build_cost_basis": build_cost_basis,
                "demand_satisfaction_rate": fulfillment,
                "avg_distance": float(avg_distance),
                "avg_satisfaction": float(avg_sat),
                "avg_utilization": labor_util,
                "unmet_demand": float(unmet_total),
                # ---------- 工时化新增指标（P0-1）----------
                "total_labor_minutes_used": float(total_minutes_used),
                "labor_minutes_available": float(sum(
                    s["labor_minutes_budget"] for s in sites
                )),
                "effective_labor_minutes": float(sum(
                    s["effective_monthly_capacity"] * self.avg_service_duration
                    for s in sites
                )),
                "assigned_count": float(assigned_total),
                "cost_per_demand_wan": float(cost_per_demand),
                "cost_efficiency": cost_efficiency,
                # ---------- 站点延续性（抑制拆旧建新）----------
                "kept_ratio": float(kept_ratio),
                "kept_count": int(len(kept_cids)),
            },
            "_score": float(score),
        }

    # ===================== 辅助方法 =====================
    def _pick_scale(
        self, center_idx: int, regional_demand: float, is_new: bool = True
    ) -> str:
        """
        选择站点规模：取"有效容量（场地与人力取小）能接住区域需求"的最小规模。
        若大型被适配度阈值禁建，则退到中型（容量不足的部分如实计入未满足需求）。
        """
        for scale in ["小型", "中型"]:
            if regional_demand <= self._effective_capacity(center_idx, scale, is_new):
                return scale
        # 大型：受小区设施适配度约束
        cid = self.block_labels[center_idx]
        if self.community_adapt[cid] >= self.scale_forbid_threshold:
            return "大型"
        return "中型"

    def _maturity(self, is_new: bool, site_age: int = 0) -> float:
        """
        磨合期容量系数：只在站点"建成当年"生效。

        判定依据是"站龄"而非"是否在本年列表中"——已运营满一年的站点即使规模不变，
        也不应继续被折减，否则会系统性低估老站、诱发拆旧建新。
        """
        return self.new_site_efficiency if (is_new and site_age < 1) else 1.0

    def _monthly_labor_minutes(
        self, center_idx: int, scale: str, is_new: bool = True, site_age: int = 0
    ) -> float:
        """
        站点月度护理员工时预算（分钟）——真实约束值。

        = 人数 × 每日有效分钟 × 月工作日 × 人力供应 × 设施适配度 × 磨合系数
        """
        cid = self.block_labels[center_idx]
        return (
            self.nurse_count[scale]
            * self.nurse_daily_minutes
            * self.work_days_per_month
            * self.staff_supply.get(cid, 1.0)
            * self.community_adapt[cid]
            * self._maturity(is_new, site_age)
        )

    def _effective_capacity(
        self, center_idx: int, scale: str, is_new: bool = True, site_age: int = 0
    ) -> float:
        """
        站点月度有效容量（等权口径的服务次数）——取场地容量与人力容量的较小值。

        场地容量 = daily_max_capacity × 月工作日（"能进多少人"）
        人力容量 = 护理员月工时预算 ÷ 等权平均单次工时（"能干多少活"）
        其中工时预算已含人力供应、设施适配度与磨合折减（见 _monthly_labor_minutes），
        故此处不再重复乘这些系数，恒等式为：有效容量 × 等权平均工时 = 工时预算。

        该值用于：① 规模选择 ② 结果展示 ③ 财务模块容量校验。
        实际需求分配不受此次数上限二次封顶，只受护理员分钟预算约束。
        """
        site_cap = self.site_daily_max_people[scale] * self.work_days_per_month
        labor_cap = (
            self._monthly_labor_minutes(center_idx, scale, is_new, site_age)
            / self.avg_service_duration
        )
        return float(np.floor(min(site_cap, labor_cap)))

    def _binding_constraint(self, center_idx: int, scale: str) -> str:
        """判断该站点的容量瓶颈在场地还是人力（用于诊断与展示）。"""
        site_cap = self.site_daily_max_people[scale] * self.work_days_per_month
        return "场地" if self._effective_capacity(center_idx, scale) >= site_cap else "人力"

    def _allocate(self, demand, distances, sites, mode: str = "proportional"):
        """
        按小区就近分配需求。

        容量口径（P0-1）：真实约束是护理员工时，故按"分钟"记账——
        每站有月度工时预算，分配一次服务即扣减该服务的工时定额（service_duration）。
        这样 15 分钟的助餐与 120 分钟的日间照料不再被当作等价的一次服务。

        分配顺序（mode）决定"工时不够时先满足谁"：
          "block"      —— 按小区顺序逐块满足（后到的小区/服务可能一点都拿不到）
          "service_rr" —— 小区轮转，每轮各小区取一次（仍可能被首个小区的首类服务吃光）
          "proportional"（推荐）—— 按各服务在覆盖范围内的需求工时占比，把全系统可用工时
                          切片给每类服务；再在服务内部按小区轮转分配。
                          保证"六类服务都按同一比例被削减"，不会出现某类服务为 0。

        :return: (assignment, unmet_total, used_capacity)
                 used_capacity: 各站已分配的服务次数（等权口径，用于展示/兼容）
        """
        assignment = []
        used_capacity = np.zeros(len(sites), dtype=float)
        unmet_total = 0.0
        if len(sites) == 0:
            return assignment, float(demand.sum()), used_capacity

        # 唯一真实约束 = 护理员分钟预算。
        # 不按"服务次数"二次封顶：次数上限是等权口径的展示指标，若拿来限制分配，
        # 短工时服务（助餐 15 分钟）会先把次数吃光，使工时远未用尽就被判定容量耗尽。
        # 场地容量在"能进多少人"上远大于人力容量，故真实约束始终是工时。
        remaining_minutes = np.array(
            [float(s["labor_minutes_budget"]) for s in sites], dtype=float
        )
        remaining = np.array(demand, dtype=float)  # (小区, 服务) 剩余需求

        # 每个小区按距离就近排序的候选站点
        candidates_by_block = []
        for block_idx in range(self.n_blocks):
            cands = [
                s_idx for s_idx, site in enumerate(sites)
                if distances[block_idx, site["center_idx"]] <= self.service_radius
            ]
            cands.sort(
                key=lambda s_idx: distances[block_idx, sites[s_idx]["center_idx"]]
            )
            candidates_by_block.append(cands)
            if not cands:
                # 无站点覆盖：需求整体计入未满足
                unmet_total += float(remaining[block_idx].sum())
                remaining[block_idx, :] = 0.0

        def serve(block_idx: int, srv_idx: int, minute_limit: float) -> float:
            """
            为某小区分配某类服务，按候选站点顺序扣减工时。
            :param minute_limit: 本次允许消耗的工时上限（比例分配模式下为服务切片余额）
            :return: 实际分配次数
            """
            srv_name = self.service_names[srv_idx]
            duration = float(self.service_duration[srv_name])
            served = 0.0
            for s_idx in candidates_by_block[block_idx]:
                if minute_limit <= 1e-9:
                    break
                want = float(remaining[block_idx, srv_idx])
                if want <= 0:
                    break
                # 同时受"剩余护理员工时"和"本服务工时切片"约束
                by_minutes = np.floor(remaining_minutes[s_idx] / duration)
                if np.isinf(minute_limit):
                    assigned = float(min(want, by_minutes))
                else:
                    assigned = float(min(
                        want, by_minutes, np.floor(minute_limit / duration + 1e-9)
                    ))
                if assigned <= 0:
                    continue

                used_minutes = assigned * duration
                remaining_minutes[s_idx] -= used_minutes
                minute_limit -= used_minutes
                used_capacity[s_idx] += assigned
                remaining[block_idx, srv_idx] -= assigned
                served += assigned

                site = sites[s_idx]
                site["monthly_demand_by_service"][srv_name] += assigned
                site["labor_minutes_used"] += used_minutes
                assignment.append({
                    "block": self.block_labels[block_idx],
                    "site": site["cid"],
                    "service": srv_name,
                    "count": assigned,
                    "distance": float(distances[block_idx, site["center_idx"]]),
                    "labor_minutes": float(used_minutes),
                })
            return served

        if mode == "proportional":
            # ---------- 两遍水分法 ----------
            # 第一遍：按各服务"可达需求"的工时占比，把可用工时切片给每类服务，
            #        保证六类服务按同一比例被削减，不会出现某类服务为 0。
            # 第二遍：把没被用掉的切片余额收回，按比例补给仍在"卡配额"的服务；
            #        反复循环直到工时用尽或无服务可分（消除切片分母与实际可达需求的偏差）。
            for _ in range(20):
                if remaining_minutes.sum() <= 1e-9 or remaining.sum() <= 1e-9:
                    break
                # 本轮可达需求工时（按当前剩余需求，避免给已满足的服务重复配额）
                demand_minutes = np.array([
                    float(remaining[:, srv_idx].sum()) * self.service_duration[srv_name]
                    for srv_idx, srv_name in enumerate(self.service_names)
                ])
                total_demand_minutes = float(demand_minutes.sum())
                if total_demand_minutes <= 1e-9:
                    break
                pool = float(remaining_minutes.sum())
                quota = pool * demand_minutes / total_demand_minutes  # 本轮切片额度
                used_by_service = np.zeros(len(self.service_names), dtype=float)
                exhausted = np.zeros(len(self.service_names), dtype=bool)
                for srv_idx, _ in sorted(
                    enumerate(self.service_names), key=lambda x: -demand_minutes[x[0]]
                ):
                    if quota[srv_idx] <= 1e-9:
                        continue
                    for block_idx in range(self.n_blocks):
                        if quota[srv_idx] <= 1e-9:
                            break
                        if not candidates_by_block[block_idx]:
                            continue
                        duration = float(self.service_duration[self.service_names[srv_idx]])
                        got = serve(block_idx, srv_idx, quota[srv_idx])
                        used_by_service[srv_idx] += got * duration
                        quota[srv_idx] -= got * duration
                    if quota[srv_idx] <= 1e-9:
                        exhausted[srv_idx] = True
                if used_by_service.sum() <= 1e-9:
                    break
        elif mode == "block":
            for block_idx in range(self.n_blocks):
                for srv_idx in range(len(self.service_names)):
                    serve(block_idx, srv_idx, float("inf"))
        else:
            # 小区轮转：每轮每个小区各取一次，直到所有小区都拿不到工时
            progress = True
            while progress:
                progress = False
                for block_idx in range(self.n_blocks):
                    if not candidates_by_block[block_idx]:
                        continue
                    for srv_idx in range(len(self.service_names)):
                        if remaining[block_idx, srv_idx] <= 0:
                            continue
                        if serve(block_idx, srv_idx, float("inf")) > 0:
                            progress = True

        unmet_total += float(remaining.sum())
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