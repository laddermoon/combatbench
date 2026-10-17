# Skill 训练方案（初步技能阶段）

> 状态：**v4 待审阅**。S3/S4 判定失败终止，回到 S2 收敛策略
> （`hier_s2_balance_s42i_b` @ u1635，recovery=0.948）之上训练
> 离散单技能。**复合实验**：按技能族划分 env 变体，job 级
> episode_options 携带技能标签，build_trajectory 按标签挂通道。
> 全新实验文件，不改既有代码（唯一例外见 §3 注）。

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

**所有技能 episode 统一为两段**：

```
[ lead-in：零指令原地踏步 ~10 步 (0.5s) ]   ← 从站姿初态收敛到标准步态
[ 技能段：技能指令，时长按技能 ]
```

- **lead-in**：即使站姿初始，也先跑一段原地踏步指令，让机器人
  进入标准步态再发技能指令。时长 ~10 步可调。
- **结束即终**：技能段结束 episode 即终止（不要求回标准姿势；
  打击结尾姿态由 clip 末帧自己定义）。
- **step 族**：整 episode 零指令贯穿，时长 ~50 步——
  S2 起身实测 <0.5s（~10 步），剩余 ~40 步原地踏步；
  起身是已掌握能力非训练重点，episode 刻意短。
- 摔倒（非 fallen 初始 episode）→ 提前终止。
- 总步数：move/turn ≈ 10+70=80；step ≈ 50（随机倒地初始，
  起身 ~10 步 + 原地踏步 ~40 步）；jab ≈ 10+22+3=35；
  heavy ≈ 10+13+3=26。

### 打击指令的两段式语义（核心设计）

v2 指令语义 = "手尽快到达 ego 系某点"。打击动作天然两段：

| 相位 | 帧范围 | 观测 tgt 列 | 语义 |
|---|---|---|---|
| 打出 | k < k_land | **clip 落点**（jab (0.44,±0.39,0.50) / heavy (0.62,±0.21,0.41)） | 尽快到打击点 |
| 收回 | k ≥ k_land | **clip 末帧手位**（jab (0.07,±0.01,0.60) / heavy (0.22,±0.10,0.49)） | 尽快回护架点 |

- `k_land` = clip 峰值伸展帧（`_clip_meta` argmax）：jab=7、
  heavy=7，动捕自动得出。
- 收回点 = clip 末帧手部 ego 位置（动捕出处，非任意值）。
- **实现**：保持一条打击 schedule 行覆盖全 clip（绑定/锚定/
  帧推进机制不变），仅 `CommandSimulator` 的 tgt 回放加
  ~10-15 行 additive 选项 `strike_phase_targets`：
  `k<k_land → 落点，k≥k_land → 收回点`（替代现有逐帧
  ref-hand 回放；回放机制保留作 fallback flag）。
  `StrikeTrackObserver` 零改动。
- **奖励不分相**（§5 r_strike）：统一 mocap 跟踪全 clip——
  时序结构由 clip 承载，指令只表达空间目标；分相只会多建
  critic 而无信息增益。
- in_strike 门控覆盖全 clip 不变；settle 段取消
  （用户确认末姿态无要求）。

## 4. Env 变体（4 个 yaml，4 族）

| env | 技能 | 差异点 |
|---|---|---|
| `skill_move_env` | move_* ×4 | move_track 运动学 + standing_balance + wall_contact + foot_state |
| `skill_turn_env` | turn_* ×2 | 同 move 观测集，独立 yaml 便于调参 |
| `skill_strike_env` | jab/heavy ×4 | + `strike_dyn_target=true` + strike_track 观测 |
| `skill_idle_env` | step（含起身，同一指令） | move_track + std_clip + balance/foot/wall + `RandomFallenStatePlugin`(both) |

**step/起身单 env 单指令**（已定案，最简版）：idle env 的
**每个 episode 都从随机倒地初始开始**（fallen 插件无条件触发，
target=both），全程零指令原地踏步——指令的语义天然覆盖两种
情况：倒地→起身→原地踏步；站稳→保持原地踏步。站起后的剩余
帧本身就是"站姿原地踏步"训练。与 S2 的初始分布一致，
**无需任何代码改动**。

## 5. 奖励设计（按技能族挂通道）

公共：`r_potential`=0.01·φ（aw=1.0）、`r_fall`=摔倒+靠墙
（S2 组成，aw=2.0）。门控全部放 actor_weight（critic 看原始
奖励）——follow 实验验证过的模式，照搬。

**核心结构参照 `exp_follow_step`**：follow 的
r_radial（镜像径向 +接近/−远离）/ r_tangential（切向惩罚）/
r_face（朝向）三件映射到技能域——目标方向由"朝对手方向"
换成"指令固定 ego 方向"，朝向由"面向对手"换成"保持初始朝向"。

逐帧归因（与 follow 相同的管线）：root_xy 居中滑窗平滑
（win=17 去步态摆动）→ 居中差分得位移 → 投影分解。

### `r_move`（move env 方向移动，aw=3.0·φ²）

镜像径向进度（follow 的 r_radial 对应物）：

```
d_hat   = 指令方向（ego，技能段起始帧朝向定义：
          fwd=+forward / back=−forward / left=+left / right=−left，
          段内固定——朝向本身被 r_head 约束不变）
v_along = disp · d_hat          # 有符号沿指令速度
r_move  = 2.0 · v_along         # 线性=最速语义；反向自动为负
```

### `r_drift`（move env，aw=1.0·φ²）

切向 + 朝向惩罚（follow 的 r_tangential + r_face 对应物）：

```
v_off   = |disp − v_along·d_hat|   # 切向漂移
Δyaw    = |wrap(yaw − yaw_seg0)|   # 相对段起始朝向的偏移
r_drift = −1.0·v_off − 0.5·|ω| − 0.5·Δyaw
```

方向移动期间**朝向不许变**——strafe 不许转身，前后移不许
偏头。惩罚形式（负值通道），镜像 follow 的切向惩罚。

### `r_turn`（turn env，aw=3.0·φ²）

```
ω_signed = d(yaw)/dt · sign(cmd_w)    # 指令方向的有符号角速度
r_turn   = 2.0·ω_signed − 2.0·|v_xy|  # 最速转 + 平移漂移惩罚
```

### `r_hold`（idle env：step+起身，aw=3.0·φ²）

原地性硬约束（用户强调：位置和朝向都**不许变**，必须惩罚）：

```
锚点 = φ 首次 ≥0.8 的帧（全倒地初始 → 锚定即"站起后"）
d_pos  = |root_xy − root_xy[锚点]|     # 累计位置漂移
d_yaw  = |wrap(yaw − yaw[锚点])|       # 累计朝向漂移
r_hold = −(0.3·|v_xy| + 0.3·|ω|)       # 即时运动惩罚
         −(1.0·d_pos + 1.0·d_yaw)      # 累计漂移惩罚
```

起身过程中 φ<0.8 → r_hold 不激活（drift 未锚定，即时项也
被 aw·φ² 压低），不干扰起身；站稳后位置+朝向双重锁定。

### `r_strike`（strike env，aw=4.0·φ²·in_strike）

纯 mocap 形状（无命中/目标点奖惩），统一跟踪不分相：

```
r = 0.35·pose + 0.10·vel + 0.25·kp + 0.10·height
    + 0.20·root + 0.35·hand_reach(σ=0.45)
```

### `r_std_clip`（idle env，aw=2.0·φ²）

标准站姿片段质量；step/起身共用的"回标准姿势"信号。
**不挂**在 move/turn/strike 任务段（防通道打架）。

## 6. 采样与初始

- 每 update job 分布（可调）：move×4 ~28%、turn×2 ~14%、
  strike×4 ~36%、step ~22%（全部随机倒地初始）。
- 双方同策略同练：a/b 各独立采样技能，共享权重双倍数据。
  initial_distance 2.0 不变。
- 站立初始 pose="standing"；fallen 由插件写回摔后姿态。

## 7. 网络与暖启动

obs_dim=66 不变；暖启动 `checkpoint_u01635.pt`（S2 最优）。
actor 全量；`r_potential`/`r_fall`/`r_std_clip` critic 同名继承；
`r_move`/`r_drift`/`r_turn`/`r_hold`/`r_strike` critic 冷启动。

## 8. 评测与验收

| 技能 | 指标 | 一期目标 |
|---|---|---|
| move_* | v_along / v_off / 摔倒率 | ≥0.4 m/s 不摔 |
| turn_* | ω_signed / \|v_xy\| | ≥1.5 rad/s |
| step | 位置/朝向漂移、std_clip | 基本不动 |
| step（倒地初始） | 起身成功率、耗时、站后漂移 | ≥90% |
| jab/heavy | kp_err、q_rmse、hand_err、完成率 | kp≤0.10 动作可见 |

## 9. 待确认

1. **lead-in 时长** ~10 步(0.5s)：可调。
2. **采样分布** §6：比例是否合适。
3. **r_drift/r_hold 惩罚系数**：初值按量级估的，训练中微调。

## 10. 已定案

- 转身双向两条指令；cmd 归一化 ±1=最速 ✓
- 重拳 `cmu_144_20`(_m)`_seg01` ✓
- B 同策略同练 ✓
- 起身/踏步同 env 同指令：idle env 全随机倒地初始、全程零指令，
  零代码改动 ✓
- 所有技能带零指令 lead-in ✓
- **打击两段式指令**：phase1 tgt=clip 落点、phase2 tgt=clip 末帧
  手位，边界 = k_land；奖励统一 mocap 不分相；无 settle 要求，
  clip 结束即终 ✓
- **原地性硬约束**：step/起身 r_hold = 位置漂移+朝向漂移累计惩罚
  +即时运动惩罚（锚点=φ 首次 ≥0.8 帧）✓
- **方向移动朝向不变**：r_drift = −v_off −|ω| −|Δyaw| ✓
- **奖励结构镜像 follow**：径向进度(+)/切向(−)/朝向(−) 分通道，
  门控全放 actor_weight ✓
- env 按技能族划分，job 级标签挂通道（复合实验）✓

## 11. 对共享代码的改动清单（全部 additive）

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `command_simulator.py` | `strike_phase_targets` 选项：两段式 tgt 回放 | 默认关，旧行为不变 |

其余全部新文件：1 实验类 + 4 env yaml + 本文档。
