# Skill 训练方案（初步技能阶段）

> 状态：**v3 待审阅**。S3/S4 判定失败终止，回到 S2 收敛策略
> （`hier_s2_balance_s42i_b` @ u1635，recovery=0.948）之上训练
> 离散单技能。**复合实验**：按技能族划分 env 变体，job 级
> episode_options 携带技能标签，build_trajectory 按标签挂通道。
> 全新实验文件，不改既有代码。

## 1. 目标

在 S2 基座上训练一组语义明确的单技能，每个技能独立可调用：

- 移动：前/后/左/右（最速，cmd 归一化 ±1，无速度参数）
- 原地转身：左/右（最速，±1）
- 原地踏步（含倒地起身——同一指令，不同初始状态）
- 打击：左/右刺拳、左/右重拳（纯贴合 mocap 形状）

机制（已验证框架原生支持，零改动）：

1. `Job.env_bp` per-job——rollouter 按 env 分组各自构建。
2. `Episode.episode_options` 持久化——job 写 `skill` 标签 +
   `move_schedule`，build_trajectory 按技能挂对应通道。
3. Trajectory 缺通道 → 该 critic 不训——每族技能免费专用 critic。

## 2. 指令编码

观测不变：57 proprio + 9 cmd = 66。指令值全部**归一化 ±1**
（schedule 物理单位写 ±V_MAX / ±W_MAX，cmd 列饱和 ±1.0）：

| 技能 | cmd 编码 | 语义 |
|---|---|---|
| move_fwd | VX=+1 | 最速前进 |
| move_back | VX=−1 | 最速后退 |
| move_left | VY=+1 | 最速左移 |
| move_right | VY=−1 | 最速右移 |
| turn_left | WZ=+1 | 最速原地左转 |
| turn_right | WZ=−1 | 最速原地右转 |
| step | 全零 | 原地踏步/起身（S2 自然指令） |
| jab_l | EFF=+1,SIDE=+1,TGT=(0.438,+0.391,0.498) | 左刺拳 |
| jab_r | EFF=+1,SIDE=−1,TGT=(0.438,−0.391,0.498) | 右刺拳 |
| heavy_l | EFF=+1,SIDE=+1,TGT=(0.622,−0.213,0.413) | 左重拳 |
| heavy_r | EFF=+1,SIDE=−1,TGT=(0.622,+0.213,0.413) | 右重拳 |

打击目标点 = 该 clip 自然落点（`_clip_meta` landing），
`resolve_clip` 按最近落点确定性绑定：jab=`hdm05_3146_88_2895_127`(_m)、
heavy=`cmu_144_20`(_m)`_seg01`。

## 3. Episode 时序结构（统一模板）

**所有技能 episode 统一为三段**（step 族只有前两段语义）：

```
[ lead-in：零指令原地踏步 ~10 步 (0.5s) ]   ← 从站姿初态收敛到标准步态
[ 技能段：技能指令，时长按技能 ]
[ settle：零指令原地踏步 ~12 步 (0.6s) ]   ← 仅 strike 需要（收势）
```

- **lead-in**：即使站姿初始，也先跑一段原地踏步指令，让机器人
  进入标准步态再发技能指令（用户指定）。时长 ~10 步可调。
- **strike 时序**（⚠ 需确认，见 §9-1）：
  - lead-in 10 步（零指令）→ **技能段 = 完整 clip 时长**
    （jab 22 步 / heavy 13 步，dyn_target 回放覆盖打出+收回全程）
    → settle ~12 步（零指令原地踏步，收势回标准姿势）
  - 依据：clip 本身含收回动作（jab 峰值在第 7/22 帧，
    heavy 在第 7/13 帧，后半段全是收回）。若在峰值处切掉
    打击指令，dyn_target 停播 + in_strike 关闭 → 收回过程
    无相位信号也无奖励引导。
  - 即"打出+收回"由 mocap 跟踪全程塑形，"回位到标准姿势"
    由零指令踏步收尾段完成。
- **step 族**：lead-in 即全部内容（零指令贯穿），时长 ~60 步；
  fallen 变体给 ~120 步（起身需要 ~50 步 + 稳定）。
- 摔倒（非 fallen 初始 episode）→ 提前终止。
- move/turn 技能段 ~70 步（+10 lead ≈ 80 总）；step ~60；
  strike ~10+13~22+12 ≈ 35-45；step_fallen ~120。

## 4. Env 变体（5 个 yaml，4 族）

| env | 技能 | 差异点 |
|---|---|---|
| `skill_move_env` | move_* ×4 | move_track 运动学 + standing_balance + wall_contact + foot_state |
| `skill_turn_env` | turn_* ×2 | 同 move 观测集，独立 yaml 便于调参 |
| `skill_strike_env` | jab/heavy ×4 | + `strike_dyn_target=true` + strike_track 观测 |
| `skill_idle_env` | step（站姿初始） | move_track + std_clip + balance/foot/wall |
| `skill_idle_fall_env` | step（倒地初始） | = idle_env + `RandomFallenStatePlugin`(both) |

⚠ 说明（§9-2）：step 与起身是**同一技能同一指令**（全零），
"倒地就起身"由初始状态决定。但 `RandomFallenStatePlugin` 是
无条件触发的，无法用选项关掉——所以零代码实现是同一族拆
两个仅差该插件的 yaml，build_jobs 按比例分配。替代方案：
给插件加 ~10 行 `fallen` episode-option 开关（additive），
则可字面单 env。

## 5. 奖励设计（按技能族挂通道）

公共：`r_potential`=0.01·φ（aw=1.0）、`r_fall`=摔倒+靠墙
（S2 组成，aw=2.0）。

### `r_move`（move/turn/strike/idle env 均挂，aw=3.0·φ²）

follow 式逐帧归因（滑窗 17 平滑 root_xy → 居中差分 → 分解）：

- 方向移动：`r = 2.0·v_along − 1.0·|v_off|`（线性=最速）
- step/lead-in/settle（cmd=0 时）：`r = exp(−(|v_xy|/0.15)²)·exp(−(|w|/0.5)²)`
  静止≈1，起身上升过程中 ≈0（不惩罚，由 φ/std_clip 引导）
- strike env：r_move 仅作用于非打击帧（乘 (1−in_strike) 掩码），
  负责 lead-in/settle 的原地收束

### `r_turn`（turn env，aw=3.0·φ²）

`r = 2.0·ω_signed − 2.0·|v_xy|`

### `r_strike`（strike env，aw=4.0·φ²·in_strike）

纯 mocap 形状（无命中/目标点奖惩）：

```
r = 0.35·pose + 0.10·vel + 0.25·kp + 0.10·height
    + 0.20·root + 0.35·hand_reach(σ=0.45)
```

### `r_std_clip`（idle env 双变体，aw=2.0·φ²）

标准站姿片段质量；step 与起身共用这个"回标准姿势"信号。
**不挂**在 move/turn/strike 任务段（防通道打架）。

## 6. 采样与初始

- 每 update job 分布（可调）：move×4+step ~35%、turn×2 ~15%、
  strike×4 ~35%、step_fallen ~15%。
- 双方同策略同练：a/b 各独立采样技能，共享权重双倍数据。
  initial_distance 2.0 不变。
- 站立初始 pose="standing"；fallen 由插件写回摔后姿态。

## 7. 网络与暖启动

obs_dim=66 不变；暖启动 `checkpoint_u01635.pt`（S2 最优）。
actor 全量；`r_potential`/`r_fall`/`r_std_clip` critic 同名继承；
`r_move`/`r_turn`/`r_strike` critic 冷启动。

## 8. 评测与验收

| 技能 | 指标 | 一期目标 |
|---|---|---|
| move_* | v_along / v_off / 摔倒率 | ≥0.4 m/s 不摔 |
| turn_* | ω_signed / \|v_xy\| | ≥1.5 rad/s |
| step | \|v_xy\|、yaw 漂移、std_clip | 基本不动 |
| step_fallen | 起身成功率、耗时 | ≥90% |
| jab/heavy | kp_err、q_rmse、hand_err、完成率 | kp≤0.10 动作可见 |

## 9. 待确认

1. **strike 技能段时长**：推荐覆盖完整 clip（含 clip 内收回，
   dyn_target 全程有相位信号），收尾用零指令踏步段——vs
   字面方案"打击指令只到出拳峰值、收回段零指令"（收回无引导，
   不推荐）。
2. **step/起身 env 实现**：两个仅差 fallen 插件的 yaml（零代码，
   推荐）vs 给插件加 fallen option（~10 行 additive，字面单 env）。
3. **lead-in 时长** ~10 步(0.5s)、settle ~12 步(0.6s)：可调。
4. **采样分布** §6：比例是否合适。

## 10. 已定案

- 转身双向两条指令；cmd 归一化 ±1=最速 ✓
- 重拳 `cmu_144_20`(_m)`_seg01` ✓
- B 同策略同练 ✓
- 起身/踏步同技能同指令（初始状态区分）✓
- 所有技能带零指令 lead-in；strike 带零指令 settle 收尾 ✓
- env 按技能族划分，job 级标签挂通道（复合实验）✓
