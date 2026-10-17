# Skill 训练方案（初步技能阶段）

> 状态：**v2 待审阅**。S3/S4 判定失败终止，回到 S2 收敛策略
> （`hier_s2_balance_s42i_b` @ u1635，recovery=0.948）之上训练
> 离散单技能。**复合实验**：4 种 env 变体 × 按技能挂通道，
> 全新实验文件，不改既有代码。

## 1. 目标

在 S2 基座上训练一组语义明确的单技能，每个技能独立可调用：

- 移动：前/后/左/右（最速，无速度参数）
- 原地转身：左/右（最速）
- 原地踏步（默认站姿保持）
- 打击：左/右刺拳、左/右重拳（纯贴合 mocap 形状）
- 倒地恢复（维持 S2 能力）

核心机制（已验证框架支持，零改动）：

1. `Job.env_bp` 是 per-job 的——rollouter 按 env 分组各自构建。
2. `Episode.episode_options` 持久化——job 里写 `skill` 标签 +
   `move_schedule`，build_trajectory 时按技能挂对应奖励通道。
3. Trajectory 缺通道 → 该 critic 不在这条 traj 上训练——
   **每类技能免费获得专用 critic**。

## 2. 四种 env 变体

同 simulator（CommandSimulator v2），仅插件/参数差异——全部是
纯 yaml 配置，零新代码：

| env | 用途 | 专属配置 |
|---|---|---|
| `skill_move_env.yaml` | move_* + step | move_track_a/b（运动学观测）+ standing_balance(φ) + wall_contact + foot_state + standard_clip；无 strike observer；`strike_dyn_target` 关 |
| `skill_turn_env.yaml` | turn_l/r | 同 move env 的观测集（转向指标从同一运动学源取），独立 yaml 便于单独调参（如 max_steps） |
| `skill_strike_env.yaml` | jab/heavy | `strike_dyn_target=true` + strike_track_a/b + standing_balance + wall_contact + foot_state |
| `skill_fallen_env.yaml` | recover | `RandomFallenStatePlugin`（写回摔后姿态）+ move_track + standing_balance + wall_contact + foot_state + standard_clip |

每个 env 的 observer 集合决定它产出什么奖励信号——move env
不跑 strike observer，strike env 不跑 clip observer，各取所需。
`build_jobs` 按技能给 job 指派对应 `env_bp` + 写入
`episode_options={"skill": <name>, "move_schedule": {...}}`。

## 3. 技能清单与指令编码

观测不变：57 proprio + 9 cmd = 66。编码沿用 v2 字段：

| 技能 | env | 指令（物理量） | 长度(步) |
|---|---|---|---|
| move_fwd | move | vx=+1.5（列饱和 +1.0=最速） | 80 |
| move_back | move | vx=−1.5 | 80 |
| move_left | move | vy=+1.5 | 80 |
| move_right | move | vy=−1.5 | 80 |
| step | move | 全零 | 60 |
| turn_left | turn | w=+2π | 80 |
| turn_right | turn | w=−2π | 80 |
| jab_l | strike | eff=+1,side=+1,tgt=(0.438,+0.391,0.498) | 10+37 |
| jab_r | strike | eff=+1,side=−1,tgt=(0.438,−0.391,0.498) | 10+37 |
| heavy_l | strike | eff=+1,side=+1,tgt=(0.622,−0.213,0.413) | 10+28 |
| heavy_r | strike | eff=+1,side=−1,tgt=(0.622,+0.213,0.413) | 10+28 |
| recover | fallen | 全零 + 摔倒初始 | 120 |

- 转身双向两条指令（cmd 支持 ±w，高层可选方向）。
- 打击目标点 = 该 clip 的自然落点（`_clip_meta` landing），
  `resolve_clip` 按最近落点确定性绑定片段，指令语义 =
  "向该拳型的典型落点出拳"。jab=`hdm05_3146_88_2895_127`(_m)，
  heavy=`cmu_144_20`(_m)_seg01（已定案）。
- 打击段结构：`[10 步 STOP lead]` + `[n_ctrl+15 pad strike 行]`，
  经 `move_schedule` 走既有通道，simulator/observer 零改动。
  ego 锚定在段首步自动建立。

## 4. 奖励设计（按技能挂通道）

公共：`r_potential`=0.01·φ（aw=1.0）、`r_fall`=摔倒+靠墙惩罚
（沿用 S2 组成，aw=2.0）。摔倒在非 recover episode 提前终止。

### `r_move`（move env：move_* / step，aw=3.0·φ²）

follow 式归因：居中滑动平均去步态摆动 → 居中差分 → 投影分解：

- 方向移动：`r = 2.0·v_along − 1.0·|v_off|`
  （线性无封顶=最速语义；反向自动为负惩罚）
- step：`r = exp(−(|v_xy|/0.15)²)·exp(−(|w|/0.5)²)`（静止≈1）

### `r_turn`（turn env，aw=3.0·φ²）

`r = 2.0·ω_signed − 2.0·|v_xy|`（转得越快越好 + 平移漂移惩罚，
ω_signed = d(yaw)/dt·sign(cmd_w)，方向错自动为负）

### `r_strike`（strike env，aw=4.0·φ²·in_strike）

纯 mocap 形状跟踪（`StrikeTrackObserver` 现成输出）：

```
r = 0.35·pose + 0.10·vel + 0.25·kp + 0.10·height
    + 0.20·root + 0.35·hand_reach(σ=0.45)
```

hand_reach 是"手 vs 参考手"的贴合项（形状的一部分，
非命中奖励）。观测 tgt 列由 dyn_target 回放参考手轨迹
（隐式相位，已验证有效）。**无目标点/命中奖惩**。

### `r_std_clip`（fallen env recover + move env step，aw=2.0·φ²）

标准站姿片段质量。**不挂**在 move/strike/turn 任务上——
避免 S3 时代步态奖励与任务奖励打架。

## 5. Episode 结构与采样

- 单技能 episode = 单条 trajectory，无切分需求。
- 采样比例（每 update 的 job 分布，可调）：
  - move_*×4 + step：~35%（每方向 ~7%，step ~7%）
  - turn_*×2：~15%
  - strike×4：~35%（每种 ~9%）
  - recover：~15%（fallen env，维持起身能力）
- 双方同策略同练：robot_a/robot_b 各自独立采样技能，共享
  权重、各自产轨迹（双倍数据）。initial_distance 2.0 不变；
  技能是 ego 系的，碰撞偶发且是真实物理。
- 摔倒（非 recover）→ 提前终止，短 episode 失败廉价。
- 帧数估算：~50-100 步/ep × 512 eps ≈ 25-50k 帧/update，
  比 S3/S4（400 步 ep）快 5-10 倍。

## 6. 网络与暖启动

- obs_dim=66 不变，`init_policy_mbs_66d.yaml` 不变。
- 暖启动 `checkpoint_u01635.pt`（S2 最优，recovery=0.948）。
- Actor 全量加载；`r_potential`/`r_fall`/`r_std_clip` critic
  同名继承；`r_move`/`r_turn`/`r_strike` 新 critic 冷启动。

## 7. 评测与验收

per-skill eval（每技能独立若干 episode）：

| 技能 | 指标 | 一期目标 |
|---|---|---|
| move_* | v_along / v_off / 摔倒率 | ≥0.4 m/s 不摔 |
| turn_* | ω_signed / \|v_xy\| | ≥1.5 rad/s |
| step | \|v_xy\|、yaw 漂移、std_clip | 基本不动 |
| jab/heavy | kp_err、q_rmse、hand_err、完成率 | kp≤0.10 且动作可见 |
| recover | 起身成功率、耗时 | ≥90% |

## 8. S3/S4 失败教训 → 本方案对策

| 失败 | 根因 | 对策 |
|---|---|---|
| S3 v_err 卡 ~0.2、跟随率 30% | 多技能混排 400 步 ep，信用稀释+通道竞争 | 单技能短 ep + 每 ep 只挂相关通道 |
| S4 出拳动作不出现 | 常量指令相位盲+折中静止 | dyn_target 回放保留（已验证） |
| S4 strike_r 涨但无动作 | 静态地板淹没出拳梯度 | 专用 critic 吸收地板 |
| 移动奖励稀疏 | 段级标量广播 | 逐帧位移投影（follow 式） |
| 步态 vs 任务互相侵蚀 | clip 通道挂在所有任务上 | clip 只挂 step/recover |

## 9. 文件计划（全部新增）

- `baseline/experiments_ppo/exp_skill_train_mbs.py` —— 实验类
- `baseline/humanoid21/hierarchy/blueprints/skill_move_env.yaml`
- `baseline/humanoid21/hierarchy/blueprints/skill_turn_env.yaml`
- `baseline/humanoid21/hierarchy/blueprints/skill_strike_env.yaml`
- `baseline/humanoid21/hierarchy/blueprints/skill_fallen_env.yaml`
- 复用：CommandSimulator、StrikeTrackObserver、
  StandardClipObserver、MoveTrackObserver、standing_balance、
  wall_contact、foot_state、RandomFallenStatePlugin、
  follow 的平滑/分解逻辑（新实验文件内改写为指令方向版）

## 10. 已定案

- 转身：双向两条指令 ✓
- 重拳：`cmu_144_20_seg01`/`_m_seg01`（已定案）✓
- 机器人 B：同策略同练 ✓
- env 划分：move / turn / strike / fallen 四种 ✓
