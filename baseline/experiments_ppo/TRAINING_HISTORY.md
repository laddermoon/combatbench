# 策略训练工作总结（PPO 主线）

> 截至 2026-10-15。记录 humanoid21 格斗策略的完整训练谱系、
> 关键机制、踩坑记录与当前状态。范围：PPO 策略训练主线；
> SAC 工作台与 mocap 资产建设是平行支线，只在相关处提及。

## 0. 流水线总览

```
standup ──→ step ──→ balance ──→ follow ──→ attack_prep ──→ strike ──→ fight
(已完成)   (已完成)   (已完成)    (已完成)    (已完成)         (迭代中)    (未达标)
   │                                                              ↑ 当前卡点
   └────── 新线（pure）：standup ──→ balance_pure / follow_pure ──┘
              （无迈步机制，迈步须从任务压力中涌现 + r_cross 引导）
```

训练契约：全部 `baseline/framework/train.py --experiment <name> --algo ppo`，
多 critic 通道 + EV-confidence 加权 + 课程学习 + 暖启动续训。

## 1. 各阶段结论

### Stage 1 — Standup（已完成）

- 收敛 run：`standup_mbs_ef05_s42`，checkpoint `u350`
- 终态：`success=0.992`，max_pot=0.968
- 架构：MBS actor（K=3 MoG + 有界 σ + σ=f(obs)），96 维观测，
  单通道 `r_potential`（4 阶段站立势）
- 该 checkpoint 是后续**所有**线的 warm-start 源

### Stage 2a — Step（老线，已完成）

- `step_mbs_s42` 收敛：`step=1.000`，cycles≈45/ep，alt≈0.80
- 机制：GaitClockSimulator（obs 96→99：cmd_L/cmd_R/窗进度）+
  每脚离地高度通道 + 时钟门控 + 交替 token + step-cycle bonus +
  swing 豁免 + σ×2 站立探索
- 教训密集（见 §3 踩坑）

### Stage 2b — Balance（老线，已完成）

- `balance_step_s42i` 收敛于 u5000：**level 11/11**
  （100N × 36-40 步推力），recovery=0.952，wallln=0.005
- 12 级推力课程（40N→100N，晋级线 recovery≥0.8×2 evals）

### Stage 3 — Follow（老线，已完成）

- `follow_s42b` u8565：speed 课程 13 级打满（1.5 m/s），
  hold=0.678，facing=0.999，推力钉在 level 3（40N×31-40 步）
- 遗留问题：转弯弱——目标一旦随机移动跟不上 → 新线 follow_pure
  的改进动机之一

### Stage 4 — Attack prep（已完成但教训深刻）

- `attack_prep_s42` u8795：能出左直拳/右上勾，但出现过
  **身体平移 + 手臂锁死**的 exploit（已修）

### Stage 5 — Strike（迭代中，老线的当前卡点）

演进顺序：

| 版本 | 设计 | 结果 |
|---|---|---|
| `strike_s42d` | 命中→结算→收手三段式，全奖励 | rollout 77.7% done；收手超时 13.7% |
| `strike_s42e` | follow 起点 + 手臂塑形 | 拔推力后 eval hit 0.75→**0.13 崩塌**（出拳耦合在抗推模态里） |
| `strike_sparse_s42c` | 无塑形 + 界内 σ×2 探索 | hit~0.6-0.7，retract_timeout ~25% |
| `strike_simple_s42b/c` | **删收手**：命中即 0.5s 结算即终止 | u35 峰值 hit≈0.98；**之后持续退化**（见 §5 活问题） |

关键洞见（dumpkit u234 诊断）：retract 失败 66% 不是手没收回，
而是**学习者在追一个远离的靶子**——r_retract 梯度 0.03 vs r_face ~18.5，
被 EV-confidence 加权放大 600 倍差距。简化版（命中即结算）绕过了
这个病态。

### 新线（pure，进行中）

赌"迈步应由任务压力涌现，而非奖励雕刻"：

- **`balance_pure`**（推力路线）：裸 96 维，2→3 通道
  （r_potential + r_fall + r_cross 引导）。s42a 无 r_cross 时已见
  涌现信号：level 5（100N×6-10步）cycles 0.05→0.45 且 recovery
  0.55→0.70 回升；s42b 带 r_cross 续训中，已过 level 6
- **`follow_pure`**（零扰动跳级路线）：standup→直接跟随机
  移动靶（方向随机 waypoint + 速度 U[0.3·smax,smax] + 永不朝向
  学习者）。s42b u295 已到 slevel 6（smax=0.6），hold=0.365

## 2. 训练框架沉淀的机制资产

| 机制 | 位置 | 用途 |
|---|---|---|
| 多 critic + EV-confidence 加权 | framework/ppo | 每奖励通道独立价值估计与梯度加权 |
| `explore_factor`（逐帧 σ 调度） | SamplingSpec | 状态/距离依赖的探索强度 |
| GaitClockSimulator | end2end | 步态时钟观测（老线，99 维） |
| StandingTriggeredForcePlugin | end2end | 站稳触发随机推力 + 12 级课程 |
| RandomMovePlugin / PoseMovePlugin | plugins | 脚本移动靶（方向/速度随机化） |
| StrikeSettlementPlugin | plugins | 命中结算窗状态机 |
| dumpkit/debug.py | framework/ppo | 逐帧轨迹/梯度/优势诊断，本项目的核心方法论工具 |
| 部署 adapter | policy/humanoid21/gait_clock | 99 维策略 → 96 维标准契约（时钟内部维护） |

## 3. 踩坑记录（对后续实验有价值的教训）

1. **塑形奖励的 exploit 面**：手臂塑形 → 手臂锁死+身体平移；
   墙 → wall-lean；脚高中点 → 脚尖摇摆假迈步（sole_clearance 修）；
   稠密步态奖励 → 同脚连步（交替 token 修）
2. **环境变更 = 分布变更**：拔推力导致 s42e 确定性 hit 崩塌——
   策略会把任务模态耦合进扰动分布
3. **稠密塑形淹没稀疏任务奖励**：confidence 加权会放大这个效应
   （r_face 梯度 / r_retract ≈ 600×）
4. **早停规则会继承**：exp_step 的 `no_improvement≥200 && u≥600`
   在 strike_simple 误杀上升期 run——长跑实验需显式禁用
5. **观测契约**：训练可自定义 obs（99 维），但部署/对战平台契约
   是 96 维——要么裸 simulator，要么 adapter 内封增广
6. **rollout 高完成率 ≠ 确定性策略健康**：随机采样命中兜底时
   确定性 eval 可能正在崩（s42e、s42c 均出现过）

## 4. 方法论

用户定的工作方式：**试 → 查 → 再试 → 再查**。"查"用 dumpkit：
假设 → 抓 dump → 逐帧分析（episode/trajectory/gradsig/timeline）
→ 改策略设计或改 Debug 系统本身（双重迭代）。视频审阅是收敛
判定的一部分，指标达标但视频里形态不对（前倾/乱挥）不算过。

## 5. 当前运行中的 run（2026-10-15）

| Run | GPU | 状态 |
|---|---|---|
| `strike_simple_s42c` | 0 | ⚠️ eval hit 从 0.95+ 单调退化至 ~0.3，建议停 |
| `balance_pure_s42b` | 1 | level 6（100N），recovery 0.68，cycles 0.77 爬坡中 |
| `follow_pure_s42b` | 2 | slevel 6（smax 0.6m/s），hold 0.365 爬坡中 |

## 6. 未决问题 / 下一步

- **strike 退化根因未明**：s42c 的 hit 持续下滑（机制待 dump 分析：
  是追 Q 牺牲命中率，还是 mixture 坍缩）
- **新线迈步涌现的最终形态**：cycles ~0.5-0.8 远低于 step 线的 ~45/ep
  —— 是"够用就行"还是"还没起来"，随课程深入看
- **stage 4 出拳**：mocap 先验奖励接入（近距离 AMP 式形态奖励）
- **stage 5 自对弈**：从未到达
- **吞吐**：CPU MuJoCo 是墙钟瓶颈（~10-25万物理步/s），
  `envs/batchframework`（mujoco-warp + `--collector device`）是
  在途的 GPU 批量路线，pure 线所需插件尚待迁移
