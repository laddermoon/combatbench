# STEPACE V1 — 步令验收 Harness 方案

> 状态：**已按本会话决策定稿**

## 0. 目的

以"正常调用方"的方式向步令策略发令，验收其系统级可用性：
事件门控发令 + 声明式重规划 + 三层记账（裸执行 / 闭环任务 /
纠偏量）。产出 per-task 指标表，支持多 policy 横向对比。

## 1. 与训练发令的差异（本会话确立）

| 维度 | 训练 | 验收 |
|---|---|---|
| 发令时机 | sim 内部时钟（到期轮转） | **事件门控**：落地/到达/超时三策略可选 |
| 规划 | 随机采样 | **重规划**：每令签发时按任务路径 P 纠偏 |
| 记账 | 令残差 | 三层：裸执行指标（不可被修正掩盖）+ 闭环任务指标 + 纠偏量本身（=系统偏差度量） |

纠偏器原则：**哑的、声明的、有界的**——P 式纠偏、clamp
±15°，不代表任何针对具体 policy 的补丁。可 `--no-replan`
关成开环对照。

## 2. 指令协议复述（本次澄清）

发令层写 `(foot, sag, lat, dyaw)` 目标参数 → 发令时刻锚定
世界系目标 → obs 每帧显示残差。**脚不动转躯干**用"名义动脚
目标=当前位置 + dyaw"表达（保持令 dyaw 被丢是已知小缺陷，
验收绕开不改代码）。

## 3. 五个任务

| 任务 | 令流 | 关键指标 |
|---|---|---|
| walk_fwd | 交替脚 sag U(0.35,0.5) lat~0.5 dyaw=0 ×12步 | 速度、横漂、d_mv、arr |
| walk_bwd | sag U(-0.40,-0.25) | 后退速度、同左 |
| turn_in_place | 交替脚 sag≈0 dyaw 同向 U(8,15)° ×12步 | 转角达成率、平移漂移 |
| torso_twist | 动脚目标=原位 + dyaw 交替 U(10,25)° ×6令 | 转角达成、脚锚漂移（成败判据） |
| turn_arc | sag U(0.3,0.45)+dyaw 同向 U(8,12)° ×10步 | 转向速度、轨迹半径 |

每任务门控默认 land（动脚触地即换令）+ dur 兜底。

## 4. 实现清单

| 件 | 内容 |
|---|---|
| `step_task_scripts.py` | TaskScript 基类（每帧 `should_issue`+`next_cmd`+`peek`）+ 5 任务 + P 纠偏器 + 修正日志 |
| `v2_stepace_simulator.py` | `V2StepAceSimulator(V2StepTvarSimulator)`：`step_script` 源；到期/门控晋升；`_settle`+`trunc` 标记；脚本时长覆盖 |
| `stepace_report_observer.py` | stepcmd 字段 + dur/res0/st_z/trunc |
| `v2_stepace_env.yaml` | steptvar 外壳换 ace sim |
| `step_accept.py` | CLI：--policy 别名/路径、--tasks、--gate、--no-replan、--episodes、--out md+json |

## 5. 验收对象

stepcmd u5000 / steparm 最新 / steptvar 最新 → `policy_exports/`
（act() 确定性动作）。输出至 `accept_reports/`。
