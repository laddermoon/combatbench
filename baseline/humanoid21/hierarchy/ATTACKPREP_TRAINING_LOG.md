# AttackPrep 子技能训练日志（96 维新路线）

> 最后更新：2026-10-26
> 当前状态：Stage 1（96 维从零训练）✅ 已完成

## 训练目标

bootstrip.md「总体规划」的子技能 **Follow And AttackPrep**（分左右，先训左侧）：

上层策略收到 96 维原始观测，判定需要时把 96 维原样转发给本子技能 → 子技能策略必须是 **96 维进、21 维动作出**，不走 CommandSimulator 66 维契约。

- **Stage 1**：倒地 → 起身 → 进入 jab_l 动态抱架（参照 `hier_s1_guard_cap`，66 维版先行验证）
- **Stage 2**：推力扰动下恢复并回到抱架（参照 `hier_s2_balance_s42i_b` / `balance_pure`）
- **Stage 3**：跟随并面向随机移动的对手（参照 `follow_pure`，已有 RandomMovePlugin 对手）

## 抱架 Clip 来源

- 源策略：`train_strike_kp_2phase_ppo_20261021_001547` 的**最后导出** `policy_exports/u09260`
  （注意：run 的 `policy/` 目录 = u08760 best-of-run，≠ 最终导出）
- 录制指令必须与 eval 一致，来源 `videos/video_options.json`（seed 997777）：
  robot_a `jab_l`（warmup `turn_right`×2）、robot_b `heavy_l`（warmup `step`×4）
- 管线：`round_runner + CoreClipPostRecorder` 录 100 步 → slice 去 warmup →
  `clip fix` 修穿地 → `clip check`（basic PASS + ref_action 回放 0 误差）→ `clip play` 出视频
- 特征库：`extract_standard_clips.py --npz`（新增单文件模式，跳过周期步态门控）→
  `assets/jab_l_guard_clips.npz`（7 × 40 帧 × 53 维）
- 抱架是**动态**的：承重差自相关 ~16 步周期 → `cadence_lag=8`；torso_z 实测 mean 1.223 / p90 1.235

资产：`assets/clips/policy_rollout/`（CoreClip + README）+ `assets/jab_l_guard_clips.npz`（特征库）

## 奖励结构（S1）

| 通道 | 奖励 | actor_weight |
|---|---|---|
| `r_potential` | `0.01 × φ(t) × I(h_torso ≤ 1.24)` | 1.0 |
| `r_std_clip` | `clip + pose + 0.5·cad + 0.5·h_term`（h 目标 1.21） | `φ(t)²` |

`_pot_h_cap`：父类 `HierS1StandupStepMBS` 上的默认关闭钩子，guard 实验设 1.24。
动机：`r_potential` 全程生效，h_score 在 torso 1.28 才饱和，而抱架 torso 只有 ~1.22——
会把姿态往"站直"方向拉。截断点取 clip p90（1.235）与站立 COM（1.236）之上一点。
**注意**：cap 用的是 torso COM z（rewarder 的 `h_torso`），不是 freejoint root z（≈1.28 会虚高）。

## 运行历史

| Run | 契约 | Updates | 结果 | 备注 |
|---|---|---|---|---|
| `train_hier_s1_guard_mbs_ppo_20261024_022220` | 66 维 | 1500 | success=1.0, pose=0.94, clip=0.68, cad=0.91, h=1.23, cycles≈11 | 未截断版，75×new_best |
| `hier_s1_guard_cap` | 66 维 | 1500 | success=1.0, pose=0.931, clip=0.639, cad=0.927, h=1.222, cycles≈10.8 | +`_pot_h_cap=1.24`，88×new_best；截断无损伤，h 略降符合预期 |
| **`hier_s1_guard_96d_s1`** | **96 维** | 1500 | **success=1.0, pose=0.922, clip=0.667, cad=0.832, h=1.221, cycles≈10.9** | ✅ **Stage 1 通过**（96 维从零，~4.5h，62×new_best） |

## 96 维版实现（commit `529cfca6`）

- `exp_hier_s1_guard_96d.py`：继承 `HierS1GuardMBS`，只覆盖 `obs_dim=96` /
  `actor_blueprint` / env 路径，奖励与评估原样继承
- `blueprints/hier_s1_guard_96d_env.yaml`：与 66 维版逐行同构，simulator 换
  `Humanoid21Simulator`（原生 96 维含对手观测）
- `init_policy_mbs_96d.yaml`：MBS 策略 96 维版（hidden 256 × 3 组分）

## 与 66 维版对比

| 指标 | 66 维 capped | 96 维 |
|---|---|---|
| success | 1.000 | 1.000 |
| pose | 0.931 | 0.922 |
| clip | 0.639 | **0.667** |
| cad | 0.927 | 0.832 |
| h | 1.222 | 1.221 |
| 收敛节奏 | success≈1.0 @ ~u520 | success≈1.0 @ ~u700 前后 |

96 维 obs 含对手状态（双方同时训练、互相可见），站立期略慢但终点指标持平，
clip 甚至略高。对手观测耦合未造成可感知的伤害。

## 踩坑记录

1. **`policy/` ≠ 最终导出**：run 的 `policy/` 是 best-of-run（u08760），最终逐
   update 导出在 `policy_exports/u09260`。两边 md5 不同，提取 clip 前必须确认。
2. **指令复现看 `videos/video_options.json`**：不能只凭技能名猜——jab_l 的
   eval 指令带 warmup（turn_right×2 / step×4）且 max_steps=30，换指令录出来的
   抱架形态完全不同。
3. **特征库 ≠ CoreClip**：`StandardClipObserver` 吃预提取的 53 维特征
   （q/qv 归一化 + 投影重力 + 脚相对骨盆 + 触地），需要 `body_xpos`/`feet_forces`
   ——CoreClip 里没有，要从 TrajectoryRecorder 录制走 `extract_standard_clips.py`。
4. **量高度要用 torso COM**：抱架"蹲"的观感主要来自屈膝和站架，torso_z 只比
   站立低 ~1.5cm；用 root freejoint z 会高估 ~5cm，cap 设错。
5. **smoke 期 KL warning 正常**：critic warmup 阶段常见，跑完 2 个 update
   链路通即可，不是代码问题。

## 下一步

- Stage 2（96 维）：`balance_pure` env 骨架 + `r_std_clip`(guard) + s2 的
  push_phase 门控，从 `hier_s1_guard_96d_s1/policy/` 热启动
- Stage 3（96 维）：`follow_pure` env（RandomMovePlugin 对手）+ guard clip 通道
- 右侧（jab_r）素材：u09260 的 jab_r 录制在 `/data1/dev/policy_capture`，
  可重新走特征提取建库，或直接镜像
