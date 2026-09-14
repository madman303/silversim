"""
population.py — 人口演化模块

对外接口：
    model = PopulationModel(config)          # 用 config 初始化
    df = model.evolve(initial_pop, years=5)  # 演化 N 年，返回 DataFrame
    trans_p = model.sample_transition()      # 采样一年的转移概率
    next_pop = model.evolve_one_year(city_old, trans_p)  # 演化一年

config 结构（对应 config.yaml 的 population 段）：
    population:
        lambda_rate: 0.03
        transitions: {Z2B: 0.045, B2S: 0.10, B2Z: 0.12, S2B: 0.05, S2Z: 0.005}
        death: {Z: 0.035, B: 0.07, S: 0.16}
        migrate_out: 0.008
        random_seed: 42
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any


#状态定义（结构常量，不随情景变化）
STATE_Z = "Z"   # 自理
STATE_B = "B"   # 半失能
STATE_S = "S"   # 失能

TRANS_CN_NAME = {
    "Z2B": "自理→半失能",
    "B2S": "半失能→失能",
    "B2Z": "半失能→自理(康复)",
    "S2B": "失能→半失能(康复)",
    "S2Z": "失能→自理(康复)",
    "death_Z": "自理老人死亡",
    "death_B": "半失能老人死亡",
    "death_S": "失能老人死亡",
    "mig_out": "老人迁出",
    "Z2Z": "自理留在自理",
    "B2B": "半失能留在半失能",
    "S2S": "失能留在失能",
}


# ===================== 核心类 =====================
class PopulationModel:
    """人口演化模型，所有参数从 config 注入。"""

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: 完整的 config dict，内部读取 config["population"]
        """
        pop_cfg = config["population"]
 
        # 转移概率的均值（中心值），用于生成采样区间
        self.trans_center = {
            "Z2B": pop_cfg["transitions"]["Z2B"],
            "B2S": pop_cfg["transitions"]["B2S"],
            "B2Z": pop_cfg["transitions"]["B2Z"],
            "S2B": pop_cfg["transitions"]["S2B"],
            "S2Z": pop_cfg["transitions"]["S2Z"],
        }
        # 死亡概率
        self.death_center = {
            "death_Z": pop_cfg["death"]["Z"],
            "death_B": pop_cfg["death"]["B"],
            "death_S": pop_cfg["death"]["S"],
        }
        # 迁出率
        self.migrate_out_center = pop_cfg["migrate_out"]
        # 新增自理老人比例
        self.lambda_rate = pop_cfg["lambda_rate"]

        # 采样浮动区间：默认取中心值的 ±10%
        self.float_ratio = pop_cfg.get("float_ratio", 0.10)

        # 随机种子
        seed = pop_cfg.get("random_seed", None)
        if seed is not None:
            np.random.seed(seed)

    # ---------- 采样一年 ----------
    def sample_transition(self) -> Dict[str, float]:
        """采样一套完整的年度转移概率，保证每行概率和 = 1。"""
        p: Dict[str, float] = {}

        # 恶化类
        for key in ("Z2B", "B2S"):
            c = self.trans_center[key]
            p[key] = self._sample(c, self.float_ratio)

        # 康复类
        for key in ("B2Z", "S2B", "S2Z"):
            c = self.trans_center[key]
            p[key] = self._sample(c, self.float_ratio)

        # 死亡类
        for key, c in self.death_center.items():
            p[key] = self._sample(c, self.float_ratio)

        # 迁出
        p["mig_out"] = self._sample(self.migrate_out_center, self.float_ratio)

        # 计算留存概率，保证每行和 = 1
        self._fill_stay_prob(p)
        return p

    @staticmethod
    def _sample(center: float, ratio: float) -> float:
        """在 [center*(1-ratio), center*(1+ratio)] 内均匀采样，保留4位小数。"""
        low = max(0.0, center * (1 - ratio))
        high = min(1.0, center * (1 + ratio))
        return round(np.random.uniform(low, high), 4)

    @staticmethod
    def _fill_stay_prob(p: Dict[str, float]) -> None:
        """补齐 Z2Z / B2B / S2S，并校验不溢出。"""
        # 自理行
        z_out = p["Z2B"] + p["death_Z"] + p["mig_out"]
        p["Z2Z"] = round(1.0 - z_out, 4)
        if p["Z2Z"] < 0:
            raise ValueError(f"自理行概率溢出: {z_out:.4f} > 1")
        if p["Z2Z"] < 0.01:
            warnings.warn(f"自理留存概率过低: {p['Z2Z']:.4f}")

        # 半失能行
        b_out = p["B2S"] + p["B2Z"] + p["death_B"] + p["mig_out"]
        p["B2B"] = round(1.0 - b_out, 4)
        if p["B2B"] < 0:
            raise ValueError(f"半失能行概率溢出: {b_out:.4f} > 1")
        if p["B2B"] < 0.01:
            warnings.warn(f"半失能留存概率过低: {p['B2B']:.4f}")

        # 失能行
        s_out = p["S2B"] + p["S2Z"] + p["death_S"] + p["mig_out"]
        p["S2S"] = round(1.0 - s_out, 4)
        if p["S2S"] < 0:
            raise ValueError(f"失能行概率溢出: {s_out:.4f} > 1")
        if p["S2S"] < 0.01:
            warnings.warn(f"失能留存概率过低: {p['S2S']:.4f}")

    # ---------- 演化一年 ----------
    def evolve_one_year(
        self,
        city_old: List[Dict[str, float]],
        trans_p: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """
        演化一年，返回浮点数人口列表（未取整）。
        :param city_old: [{"name": "A", "Z": 496, "B": 152, "S": 64}, ...]
        :param trans_p: 由 sample_transition() 返回的概率字典
        :return: 新的人口列表（浮点数）
        """
        total_old = sum(d["Z"] + d["B"] + d["S"] for d in city_old)
        if total_old < 1e-9:
            return [{"name": d["name"], "Z": 0.0, "B": 0.0, "S": 0.0} for d in city_old]

        add_total_Z = total_old * self.lambda_rate
        result = []

        for blk in city_old:
            z0, b0, s0 = blk["Z"], blk["B"], blk["S"]

            # 状态转移（含恶化 + 康复）
            z_from_z = z0 * trans_p["Z2Z"]
            z_from_b = b0 * trans_p["B2Z"]
            z_from_s = s0 * trans_p["S2Z"]

            b_from_z = z0 * trans_p["Z2B"]
            b_from_b = b0 * trans_p["B2B"]
            b_from_s = s0 * trans_p["S2B"]

            s_from_b = b0 * trans_p["B2S"]
            s_from_s = s0 * trans_p["S2S"]

            z_new = z_from_z + z_from_b + z_from_s
            b_new = b_from_z + b_from_b + b_from_s
            s_new = s_from_b + s_from_s

            # 新增自理老人按小区占比分配
            share = (z0 + b0 + s0) / total_old
            z_new += share * add_total_Z

            result.append({"name": blk["name"], "Z": z_new, "B": b_new, "S": s_new})

        return result

    # ---------- 全局取整 ----------
    @staticmethod
    def round_total(city_float: List[Dict[str, float]]) -> List[Dict[str, int]]:
        """最大余数法取整，保证总量不漂移。"""
        def _alloc(values):
            total = sum(values)
            base = [int(np.floor(max(v, 0.0))) for v in values]
            need = round(total) - sum(base)
            fracs = sorted(
                [(max(v, 0.0) - np.floor(max(v, 0.0)), i) for i, v in enumerate(values)],
                reverse=True,
            )
            for k in range(need):
                base[fracs[k][1]] += 1
            return base

        z_list = _alloc([d["Z"] for d in city_float])
        b_list = _alloc([d["B"] for d in city_float])
        s_list = _alloc([d["S"] for d in city_float])

        return [
            {"name": d["name"],
             "Z": max(z_list[i], 0),
             "B": max(b_list[i], 0),
             "S": max(s_list[i], 0)}
            for i, d in enumerate(city_float)
        ]

    # ---------- 批量演化 ----------
    def evolve(
        self,
        initial_pop: List[Dict[str, float]],
        years: int = 5,
    ) -> pd.DataFrame:
        """
        从初始人口出发，演化 N 年，返回完整 DataFrame。
        :param initial_pop: [{"name": "A", "Z": 496, "B": 152, "S": 64}, ...]
        :param years: 演化年数
        :return: DataFrame，列 = [t, name, Z, B, S]
        """
        records = []

        # t = 0 快照
        for blk in initial_pop:
            records.append({"t": 0, **blk})

        current = initial_pop
        for t in range(1, years + 1):
            trans_p = self.sample_transition()
            city_float = self.evolve_one_year(current, trans_p)
            current = self.round_total(city_float)
            for blk in current:
                records.append({"t": t, **blk})

        return pd.DataFrame(records)


# ===================== 模块级函数接口（兼容旧代码调用） =====================
def evolve_population(
    initial_pop: List[Dict[str, float]],
    config: Dict[str, Any],
    years: int = 5,
) -> pd.DataFrame:
    """
    函数式接口：一步到位演化 N 年。
    :param initial_pop: 初始人口
    :param config: 完整 config dict
    :param years: 演化年数
    :return: DataFrame
    """
    model = PopulationModel(config)
    return model.evolve(initial_pop, years=years)


# ===================== 自测 =====================
if __name__ == "__main__":
    # 简化的测试 config（实际使用时从 config.yaml 加载）
    test_config = {
        "population": {
            "lambda_rate": 0.03,
            "transitions": {"Z2B": 0.045, "B2S": 0.10, "B2Z": 0.12, "S2B": 0.05, "S2Z": 0.005},
            "death": {"Z": 0.035, "B": 0.07, "S": 0.16},
            "migrate_out": 0.008,
            "random_seed": 42,
        }
    }

    init_pop = [
        {"name": "A", "Z": 496, "B": 152, "S": 64},
        {"name": "B", "Z": 408, "B": 136, "S": 64},
        {"name": "C", "Z": 632, "B": 208, "S": 80},
        {"name": "D", "Z": 368, "B": 120, "S": 56},
        {"name": "E", "Z": 536, "B": 176, "S": 72},
        {"name": "F", "Z": 328, "B": 104, "S": 40},
        {"name": "G", "Z": 592, "B": 192, "S": 80},
        {"name": "H", "Z": 392, "B": 128, "S": 48},
        {"name": "I", "Z": 504, "B": 168, "S": 64},
        {"name": "J", "Z": 456, "B": 144, "S": 56},
    ]

    df = evolve_population(init_pop, test_config, years=5)

    # 年度汇总
    agg = df.groupby("t").agg({"Z": "sum", "B": "sum", "S": "sum"}).reset_index()
    agg["total"] = agg["Z"] + agg["B"] + agg["S"]
    print("\n===== 人口演化汇总 =====")
    print(agg.to_string(index=False))