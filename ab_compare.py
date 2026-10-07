"""护理员编制对照：工时化容量口径下，不同编制方案的 t=0 / t=5 结果。

编制按站点规模档给出（小型/中型/大型）：
  A  课程作业原值            2/5/10
  B1 1:100 照料配比（当前）   3/10/30   （全区 43 人 / 6,864 老人 ≈ 1:160）
  B2 1:50                    6/20/60
  B3 1:30                   10/33/100
  B4 1:20                   15/50/150

注意：t=0 只建成 2 个站（受 120 万/年预算限制），t=5 才是建满后的稳态。
判读结论请用 t=5 列；只看 t=0 会把可行方案误判为不可行。

用法：.venv/bin/python ab_compare.py
"""
import io
import os
import re
import contextlib
import pandas as pd

from src.config_loader import load_config
from src.sim_main import run

SCENARIOS = {
    "A_原值_2_5_10": {"小型": 2, "中型": 5, "大型": 10},
    "B1_1比100": {"小型": 3, "中型": 10, "大型": 30},
    "B2_1比50": {"小型": 6, "中型": 20, "大型": 60},
    "B3_1比30": {"小型": 10, "中型": 33, "大型": 100},
    "B4_1比20": {"小型": 15, "中型": 50, "大型": 150},
}


def run_scenario(nurses: dict) -> dict:
    """跑完整 5 年，返回 t=0 / t=5 关键指标。"""
    cfg = load_config("config.yaml", overrides={"facility": {"min_nurse_count": nurses}})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(cfg, output_dir="simulation_output")
    out = buf.getvalue()

    fulfill = re.findall(r"满足率 = ([\d.]+)%", out)
    changes = re.findall(r"\[站点变动\] 保留 (\d+) \| 新增 (\d+) \| 关停 (\d+)", out)
    invests = re.findall(r"\| 净投资 ([\d.]+)", out)

    # 期末（t=5）站点构成：定位最后一年段落再解析，避免抓到中间年份
    parts = re.split(r"========== 仿真年份 t = (\d+) ==========", out)
    last_segment = parts[-1]
    sites = re.findall(r"\[(新|旧)\] (\w+) (\S+) 月利润=(-?[\d.]+)", last_segment)

    return {
        "t0满足率": float(fulfill[0]) / 100,
        "t5满足率": float(fulfill[-1]) / 100,
        "站点数": len(sites),
        "站点构成": "+".join(sorted({sc for _, _, sc, _ in sites})),
        "关停序列": [int(c) for _, _, c in changes],
        "累计净投资": sum(float(x) for x in invests),
        "站均月利润": sum(float(p) for *_, p in sites) / len(sites) if sites else 0.0,
    }


def main() -> None:
    rows = []
    for name, nurses in SCENARIOS.items():
        r = run_scenario(nurses)
        rows.append({
            "情景": name,
            "编制(小/中/大)": f"{nurses['小型']}/{nurses['中型']}/{nurses['大型']}",
            "t0满足率": round(r["t0满足率"], 4),
            "t5满足率": round(r["t5满足率"], 4),
            "站点数": r["站点数"],
            "站点构成": r["站点构成"],
            "关停总数": sum(r["关停序列"]),
            "累计净投资(元)": round(r["累计净投资"]),
            "站均月利润(元)": round(r["站均月利润"]),
        })

    df = pd.DataFrame(rows)
    os.makedirs("simulation_output", exist_ok=True)
    out_path = "simulation_output/scenario_AB_compare.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
