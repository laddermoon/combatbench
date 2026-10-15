# SAC 实验瓶颈解剖报告（诊断系统首次实战）

> 类型：调查记录 · 日期：2026-10-15 · 工具：WP1–WP4 切口系统（Contract Ledger + 十切口 + GradSig + recon/delta）
> 对象：`sac_standup`（瓶颈未解）与 `sac_balance`（已收敛）

## 0. 结论先行

| 实验 | 最终瓶颈 | 证据强度 |
|---|---|---|
| `sac_standup` | **阶段门跨越率（stage-gate crossing rate）**——势位函数在接触模式切换处设计了刻意的离散跳变，buffer 对下一阶段带近乎零覆盖；critic 对门上方是纯外推，actor 无梯度可跟。所有边际杠杆（更新数×3.1、actor LR×2、batch×2、episodes×2、explore+0.3）均不能加速 | 强（直方图悬崖 + 4 臂干预失效 + 逐样本证据） |
| `sac_balance` | 已被解决——此前瓶颈是 replay 容量（2M 修复）；无离散门结构（起始即站立），45min 收敛 held-out 0.984 | 强 |

**一句话**：standup 的长平台期不是优化器病、不是数据量、不是探索宽度——是"当前策略采样分布到下一阶段带之间的概率质量缺口"，只能被**结构性干预**（课程/相关噪声/门附近初始化/势位连续性）而非任何边际旋钮消除。

## 1. 证据链（全部可复现）

### 1.1 排除项（全部被切口数据证伪）

| 假说 | 证伪证据 |
|---|---|
| critic 欠拟合 | 切口⑤：EV≈0.985，TD~0.001——critic 精确拟合其目标，问题在目标/数据而非拟合 |
| 更新数不足 | 原 run round 63 打 `utd_ratio 0.125→0.5` 补丁（updates 3200→10000/round = 3.1×），爬坡速率 0 变化。同时证伪 actor 步数不足（每 tick 是 critic+actor 联合步进） |
| twin-min 退化 | `pair1_frac` mean=0.497（q10-q90: 0.31-0.69），联合 argmin 健康 |
| provenance 伪造 | C1.4 反而抓到 WP1.1 之前采集的 stale `regularizer_mode` 元数据——Ledger 工作正常 |
| 终态稀疏 | C2.2 WARN 是结构事实（standup 500K replay 中 `terminated` 恰为 0，全为 200 步超时；`bootstrap≡1`）——是任务形态，非 bug，且非瓶颈解释 |
| Shannon+auto_alpha 病态 | `s01_s42` 臂 Q≈20 是纯熵红利自举膨胀（奖励占 target 0.003%），该臂从未突破属配置级病态，已被 `u_floor`/`te0` 配方排除 |

### 1.2 正证据

**A. 爬坡是连续棘轮而非相变**：`final_potential_mean` 全程平滑爬升 0.04→0.84（reg_standup_ufloor_s42 链 rounds 1-124），eval success=1.0 只是越过 ~0.75 阈值的非线性读出。无突变事件。

**B. 阶段门悬崖（决定性）**：batch potential 直方图显示分布顶部存在**硬上界**。`standup_v2_r10.py` 的 stage 机是逐帧重算的接触模式判定（腹→手撑→手脚→单脚→双脚→双脚低速），band 边界 0.15/0.25/0.40/0.55/0.65/0.80/0.90，其中 0.85→0.90 是刻意的势位跳变（"potential gaps at stage boundaries to reward risky transitions"），其余悬崖是**策略可达性前沿**：

| dump | final_pot | batch 分布 | 悬崖位置 | 性质 |
|---|---|---|---|---|
| s01_s42@tick169K（深平台） | ~0.06 | 全部 <0.4 | ~0.4 | stage3 门（手离、仅脚触地） |
| ef03@695K（round~70） | ~0.60 | 质量堆 0.5-0.7 | ~0.72 | stage4 带内 h·u 可达性上限 |
| transition@795K（round~121） | ~0.73 | 质量堆 0.6-0.9 | ~0.9 | stage5 门（速度<2.0 + 设计 gap） |

平台期量化覆盖：242 个 episode 的抽样帧中，pot≥0.55 占 71%，**pot≥0.65（stage-4 门"双脚无手"）占 7%，pot≥0.8（stage-4.5）0%，pot≥0.9（stage-5）0%**。critic 在悬崖上方完全无数据。

**C. 干预实验（四臂同 ckpt resume 判别）**：统一从 `reg_standup_ufloor_s42_b/checkpoint_s00832000`（round 65, final_pot≈0.577）resume，各改一个杠杆，~15 轮：

终局对齐窗口（rounds 66-81，与基线 `_c` 同段位逐轮配对差）：

| 臂 | 杠杆 | r66-81 mean | **mean Δ vs BASE** | slope |
|---|---|---|---|---|
| `probe_ef03_s42` | `behavior_explore=0.3` | 0.600 | **−0.004** | +0.0025/r |
| `probe_lr2x_s42` | `learning_rate=6e-4`（2×） | 0.604 | +0.001 | +0.0034/r |
| `probe_ep128_s42` | `episodes_per_update=128`（2× 数据/轮） | 0.607 | +0.004 | +0.0032/r |
| `probe_b512_s42` | `batch_size=512`（梯度 SNR×2） | 0.604 | +0.001 | +0.0032/r |
| BASE（`_c` r68-81） | — | 0.607 | — | +0.0011/r |

逐轮噪声 σ≈0.02，n=10-14 → mean Δ 的 SE≈0.005：**全部臂与基线统计无差别**。加上历史 UTD×3.1 无效应，共五个边际杠杆（更新数、actor LR、数据量、batch SNR、探索宽度）在平台段位全部无效。机制性佐证：**ef03 扩宽 σ（mid-dim +36%）后其新轮前沿 max=0.661 仍不高于对照的 0.711**——更宽的各向同性噪声不产生更多跨越。

**D. 机制佐证**：
- `expectation_samples=1` + B=256 → 逐样本 actor 梯度信号弱漂移+强互斥（gradsig: `norm_G=0.0027`，41% 样本逆聚合方向）——扩散主导 regime
- u_floor 模式下 actor 目标**完全无熵项**（shannon 分支才有 α·logπ）；`U(s)` 逐维均值被钉在 0.498≈floor——但逐维 σ **双模态饱和**（min=0.05 塌缩维 / max=2.0 爆顶维），`explore_factor` 只影响未饱和维（实测 e=0.3 → σ_med×1.36 但 ε 只 +6%）
- critic 对前沿定价**正确地平**：pot 0.65→0.89 对应 target 0.658→0.684（+0.03）——"够到但守不住"在当前策略下确实不值钱，鸡生蛋结构

## 2. 瓶颈的精确表述

**Standup 的限速器是离散接触模式门的跨越率**：stage 3.5→4（双脚站立时松开手）与 4→5（低速稳定）要求协调的轨迹几何（边站边撤手、主动减速），每步高斯噪声对此是低效发生器。Buffer 零覆盖 → critic 纯外推 → actor 无方向 → 只能等偶发跨越样本；跨越样本一旦出现，tick 内机制（已被 Ledger 验证全部正确）会正常扩散其价值——所以爬升呈线性棘轮，速率 = 每轮新跨越事件的期望数。

**这不是 SAC 实现的缺陷**：所有切口断言通过、数学闭合逐位成立、修复与配方（u_floor/te0/replay）已解决所有已发现的实现级问题。剩余的是 off-policy 方法对离散技能链的经典困难。

## 3. 可操作的下一步（结构性，非边际）

1. **门附近初始化/课程**：`random_start_transitions` 目前是全随机；改为从"双脚触地+手撑"的靠近门位姿启动（需要环境侧复位插件支持位姿注入），直接把跨越事件频率提高一个量级。
2. **时间相关探索**：OU 噪声或持续若干步的动作扰动（如"抬手 10 步"原语）替代白噪声。
3. **逐维 U floor**：把 `u_floor` 从均值约束改为逐维约束，阻止 σ 双模态塌缩——塌缩维可能正是过门所需 DOF。
4. **势位连续性**：3.5→4 之间的 gap 是刻意的（奖励冒险转移）——可考虑把 stage-3.5 带（双脚+膝触地）的势位上限抬高使衔接更平滑，或给"短暂无手"加过渡带。需权衡设计意图。
5. **演示/离线数据播种**：向 replay 注入穿越轨迹（recon 证明回放管线位级精确，可放心注入）。

## 4. 本调查的自产工件

- 判别臂 run：`probe_ef03_s42` `probe_lr2x_s42` `probe_ep128_s42` `probe_b512_s42`（同 ckpt resume，各带 3 个 dump@655254/695254/735254）
- 新观测字段：`policy_det_action`（真实 ε 测量已生效，探测臂 dump 可用，旧 dump 诚实降级）
- 附带修复：s01_actor/tn_actor/transition/replay 端到端字段、C0.2 语义修正、viewer/explore 文档
