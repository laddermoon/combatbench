# AttackPrep 子技能训练日志（96 维新路线）

> 最后更新：2026-10-27
> 当前状态：S1 ✅ / S2 ✅（满级，继续磨） / S3 训练中

## 训练目标

bootstrip.md「总体规划」的子技能 **Follow And AttackPrep**（分左右，先训左侧）：

上层策略收到 96 维原始观测，判定需要时把 96 维原样转发给本子技能 → 子技能策略必须是 **96 维进、21 维动作出**，不走 CommandSimulator 66 维契约。

- **Stage 1**：倒地 → 起身 → 进入 jab_l 动态抱架（参照 `hier_s1_guard_cap`，66 维版先行验证）
- **Stage 2**：推力扰动下恢复并回到抱架（参照 `hier_s2_balance_s42i_b` / `balance_pure`）
- **Stage 3**：跟随并面向随机移动的对手，保持 [0.6, 1.0]m 距离带 + 抱架（参照 `follow_pure`）

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

## 奖励结构

**S1/S2**：

| 通道 | 奖励 | actor_weight |
|---|---|---|
| `r_potential` | `0.01 × φ(t) × I(h_torso ≤ 1.24)` | 1.0（S2 为 3.0） |
| `r_std_clip` | `clip + pose + 0.5·cad + 0.5·h_term`（h 目标 1.21） | `φ²`（S2 为 `2·φ²·(非受推中)`） |
| `r_fall`（仅 S2） | 摔倒 + 靠墙 + 受推窗口 | 2.0 |

`_pot_h_cap`：截断点取 clip p90（1.235）与站立 COM（1.236）之上一点。
**注意**：cap 用 torso COM z（`h_torso`），不是 freejoint root z（≈1.28 会虚高）。

**S3**（`exp_hier_s3_follow_guard_96d.py`，7 通道）：

| 通道 | 奖励 | actor_weight |
|---|---|---|
| `r_potential` | `0.01·φ·I(h≤1.24)` | 3.0 |
| `r_fall` | 摔倒 + 靠墙 + 受推窗口 | 2.0 |
| `r_cross` | 交替支撑引导 | `1.0·φ²` |
| `r_band_step` | **按步结算**：步窗净位移在 `d_hat=to_opp(t_end)` 上的径向分量；dist>1.0 奖接近、dist<0.6 奖远离、带内 0 | `3.0·out_band·φ²·(非受推中)`，权重稠密、奖励稀疏在落地帧 |
| `r_tan_step` | 同上切向分量恒惩罚 | `1.0·out_band·φ²·(非受推中)` |
| `r_face` | `cos(forward, to_opp)⁺` 逐帧 | `1.0·dist_gate·φ²·(非受推中)` |
| `r_std_clip` | 同 S1 | `3.0·φ²·I(dist≤1.0 & cos≥0.5)` —— **近且面向才有抱架约束**；远→自由追，背身→自由转，推中不放开 |

对手：`ArcMovePlugin`——弧长分段航向（段内 κ~U(−2,2) 半径≥0.5m 小弧切接、段间任意跳变可折返）+
时间分段速度（1–3s 段、v*~U[0,1]、|dv/dt|≤0.8 斜坡）+ **episode 级固定朝向**（50% 面向学习者 /
50% 随机固定 yaw）+ 场地内随机出生点（margin 0.4m、距学习者≥0.8m）+ 可朝学习者移动（0.4m 接触钳制）。

扰动：`StandingTriggeredForcePlugin` 只推学习者，固定 40N 顶档（duration 31–40 步）——S2 已验证不构成破架级别，姿态在推中保持。

## 运行历史

| Run | 契约 | Updates | 结果 | 备注 |
|---|---|---|---|---|
| `train_hier_s1_guard_mbs_ppo_20261024_022220` | 66 维 | 1500 | success=1.0, pose=0.94, clip=0.68, cad=0.91, h=1.23 | 未截断版 |
| `hier_s1_guard_cap` | 66 维 | 1500 | success=1.0, pose=0.931, clip=0.639, cad=0.927, h=1.222 | +`_pot_h_cap=1.24` |
| **`hier_s1_guard_96d_s1`** | **96 维** | 1500 | **success=1.0, pose=0.922, clip=0.667, cad=0.832, h=1.221** | ✅ **S1 通过**（从零） |
| **`hier_s2_guard_96d_s1`** | **96 维** | 2000+ | **level=11 满级，recovery=0.92–0.94，success=1.0** | ✅ **S2 判定通过**（热启动 S1/u1500；clip≈0.23 重推下让位，同参照 0.39 量级）；继续跑磨 recovery |
| `hier_s3_follow_96d_s2` | 96 维 | →440 | — | **废**：v_max=1.0 写死无课程，靶子满速追不上 |
| `hier_s3_follow_96d_s2b/c` | 96 维 | ~10 | — | **废**：b 补 9 级课程；c 补 level0 静止随机点 + episode 固定朝向 |
| `hier_s3_follow_96d_s2d` | 96 维 | →770 | facing=0.98, dist=1.16, slevel=2(0.3m/s), hold≈0.40 卡晋级线 | 从 S2/u825 热启动；**废**：机器人动得太慢——滑窗速度奖励信用分配差 |
| `hier_s3_follow_96d_s2e` | 96 维 | ~10 | — | 换步结算奖励验证 OK；**废**：为换 S2 更新起点 + 加扰动重启 |
| `hier_s3_follow_96d_s2f/g` | 96 维 | ~20 | — | f 从 S2/u1955 + 40N 扰动；g 加 clip 双门控；**废**：clip 推中门控误加（40N 不需破架） |
| **`hier_s3_follow_96d_s2h`** | **96 维** | 进行中 | — | 当前版本：步结算 + 40N 扰动 + clip 双门控（dist≤1.0 & cos≥0.5、推中不放开） |

## 与 66 维版对比（S1）

| 指标 | 66 维 capped | 96 维 |
|---|---|---|
| success | 1.000 | 1.000 |
| pose | 0.931 | 0.922 |
| clip | 0.639 | **0.667** |
| cad | 0.927 | 0.832 |
| h | 1.222 | 1.221 |

96 维 obs 含对手状态，站立期略慢但终点指标持平。对手观测耦合未造成可感知的伤害。

## 踩坑记录

1. **`policy/` ≠ 最终导出**：run 的 `policy/` 是 best-of-run（u08760），最终逐
   update 导出在 `policy_exports/u09260`。提取 clip 前必须确认。
2. **指令复现看 `videos/video_options.json`**：不能只凭技能名猜。
3. **特征库 ≠ CoreClip**：`StandardClipObserver` 吃预提取的 53 维特征，
   需要 `body_xpos`/`feet_forces`——CoreClip 里没有，要走 TrajectoryRecorder。
4. **量高度要用 torso COM**：抱架"蹲"主要来自屈膝和站架，torso_z 只比
   站立低 ~1.5cm；用 root freejoint z 会高估 ~5cm。
5. **速度课程必须从一开始就装**：S3 第一版 v_max=1.0 写死，靶子满速
   学习者完全跟不上——静止靶 → 慢速 → 逐级提速是必要曲线。
6. **对手朝向按 episode 采样**：按速度段重采样朝向会产生可观测的状态
   跳变干扰学习；50% 集内面向 / 50% 集内固定随机 yaw 才对。
7. **滑窗速度奖励信用分配差**：17 步滑窗把"是哪一步迈错"摊平了——
   换 `skill_step_mbs` 的按步结算（步窗净位移在结算时刻对手方向上
   分解径向/切向；奖励稀疏在落地帧、aw 保持稠密 φ²）信号才对得上。
8. **clip 门控要贴任务语义**：跟随中远距离/背身时抱架约束=自缚
   （走不快转不快）；但 40N 小扰动不构成破架理由，推中门控反而放掉
   了姿态——门控条件应只含距离+朝向。

## 下一步

- S3 `hier_s3_follow_96d_s2h`：盯 band/hold 爬升与速度课程晋级
  （hold≥0.5×2 升级），slevel 到 1.0 m/s 满级 + band 稳定为收敛标准
- 右侧（jab_r）素材：u09260 的 jab_r 录制在 `/data1/dev/policy_capture`，
  可重新走特征提取建库，或直接镜像
