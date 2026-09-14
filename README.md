# SilverSim（银龄沙盘）

> 一个可复现的社区养老设施规划仿真工具包，模拟人口演化、需求计算、选址优化与财务测算的完整闭环。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## 一、项目背景

中国超过 90% 的城市老人选择居家社区养老，但供给端面临**空间不足、供需错配、经济不可持续、护理员短缺**四重矛盾。街道层面在规划养老服务站时，普遍缺少可以量化推演的工具。

**SilverSim 要回答的核心问题**：在人口结构动态变化、预算有限、人力受限的条件下，社区养老服务站应该建在哪里、建多大、如何定价？

---

## 二、核心方法

项目把社区养老供给系统抽象为一个**动态演化系统**，围绕这条主线构建：  
**人口演化 → 需求计算 → 选址分配 → 财务测算 → 反馈下一周期**  

| 模块 | 文件 | 回答的问题 |  
|---|---|---|  
| 人口演化 | `src/population.py` | 老年人健康状态如何逐年变化？ |  
| 需求计算 | `src/demand_calc.py` | 有多少服务需求能被释放？ |  
| 选址优化 | `src/site_optimize.py` | 站点建在哪、建多大？ |  
| 财务测算 | `src/price_subsidy.py` | 每个站能赚钱吗？ |  
| 主循环 | `src/sim_main.py` | 如何调度全部模块？ |  
| 可视化 | `src/visual_plot.py` | 结果如何呈现？ |  

### 关键技术点  

- **马尔可夫状态转移**：老年人口按自理/半失能/失能三状态演化，含分层死亡、康复、迁出、随机采样  
- **四层需求漏斗**：理论需求 → 消费约束 → 付费意愿 → 价格弹性  
- **贪心选址 + 动态重规划**：每 2 年触发一次站点重规划，支持关停与残值回收  
- **完整成本口径**：固定成本 = 场地管理费 + 护理员工资（含人力约束）  

---

## 三、快速开始  

### 环境要求  

- Python 3.10+  
- 依赖：`numpy`, `pandas`, `matplotlib`, `seaborn`  

### 安装  

```bash
git clone https://github.com/yourname/silversim.git  
cd silversim  
python -m venv venv  
source venv/bin/activate      # Windows: venv\Scripts\activate  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  