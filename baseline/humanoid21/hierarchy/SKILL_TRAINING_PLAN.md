# Skill 训练方案（初步技能阶段）

> 状态：**v7 实施中**。S3/S4 判定失败终止，回到 S2 收敛策略
> （`hier_s2_balance_s42i_b` @ u1635，recovery=0.948）之上训练
> 离散单技能。**复合实验**：按技能族划分 env 变体，job 级
> episode_options 携带技能标签，build_trajectory 按标签挂通道。
>
> v7 两大结构性修订：
> 1. **episode 内 lead-in 废除** → `WarmupSteppingPlugin` 在 reset
>    阶段直接把机器人初始化到"步态中间态"——episode 本体 100%
>    是纯技能，奖励/观测无任何门控。
> 2. **打击只学"打出去"** → 打分窗口 = 打出段+余量帧，打完后的
>    一切（摔倒、乱动）不计分；"收招回位"留作后续独立技能。

## 1. 目标

在 S2 基座上训练一组语义明确的单技能，每个技能独立可调用：

- 移动：前/后/左/右（最速，cmd 归一化 ±1，无速度参数）
- 原地转身：左/右（最速，±1）
- 原地踏步（含倒地起身——同一指令，不同初始状态）
- 打击：左/右刺拳、左/右重拳（只学打出去：贴合 mocap + 越快越好）

机制（已验证框架原生支持，零改动）：

1. `Job.env_bp` per-job——rollouter 按 env 分组各自构建。
2. `Episode.episode_options` 持久化——job 写 `skill` 标签 +
   `move_schedule` + `warmup_policy_cls`，build_trajectory 按技能
   挂对应通道。
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
| jab_l | EFF=+1,SIDE=+1,TGT=(0.438,+0.391,0.498) | 左刺拳打出 |
| jab_r | EFF=+1,SIDE=−1,TGT=(0.438,−0.391,0.498) | 右刺拳打出 |
| heavy_l | EFF=+1,SIDE=+1,TGT=(0.622,−0.213,0.413) | 左重拳打出 |
| heavy_r | EFF=+1,SIDE=−1,TGT=(0.622,+0.213,0.413) | 右重拳打出 |

打击指令 = "手尽快到达 ego 系打击点"，**单相位全程指向落点**
（无收回段——v6 的两段式设计废弃，`strike_phase_targets`
已从 CommandSimulator 移除，不留死代码）。

打击目标点 = 该 clip 自然落点（`_clip_meta` landing），
`resolve_clip` 按最近落点确定性绑定：jab=`hdm05_3146_88_2895_127`(_m)、
heavy=`cmu_144_20`(_m)`_seg01`。

## 3. Episode 时序结构

**初始状态即步态中间态**（WarmupSteppingPlugin，详见 §3.1），
episode 本体 = 纯技能段：

| 族 | schedule | 时长 |
|---|---|---|
| move | [70, ±V_MAX, 0, 0] 等 | 70 步 |
| turn | [70, 0, 0, ±W_MAX] | 70 步 |
| step | [80, 0,0,0] | 80 步 4s（随机倒地初始） |
| strike | [k_land/rate+margin, ..., eff,sid,tgt] | jab 11+13=24 / heavy 12+13=25（rate=0.5）|

### 3.1 WarmupSteppingPlugin（替代 lead-in，核心修订）

参照 `ImpulsePerturbationPlugin`/`RandomFallenStatePlugin` 的
**内部 sim 写回**模式，新增实验侧插件
`baseline/humanoid21/plugins/warmup_stepping.py`：

1. `on_pre_episode`：真实环境 core state → 内部 Humanoid21Simulator。
2. 内部 sim 跑 ~10 动作步（0.5s）原地踏步：obs = 本体 57 维 +
   全零 cmd9（v2 零指令=踏步语义），两个机器人同步驱动。
3. 结束 core state 写回真实环境 → episode 第 0 帧就是步态中间态。

**引导策略 = 当前 post-update 导出策略**（每次都不一样，且始终
跟随训练中步态）——实验在 `build_jobs` 里把
`policy_bp.cls`（file:<export>/policy.py:<Cls>）写入
`episode_options["warmup_policy_cls"]`，插件惰性加载、按 cls
字符串缓存（上限 8 条）。采样动作（`policy.sample`）提供
相位多样性；无策略注入时退化零 action（PD 站姿）。

收益：奖励无门控（r_move/r_turn/r_drift 全帧激活、dyaw 锚定
第 0 帧）、第 0 帧观测即"步态中间态+技能指令"、episode 各短
10 帧。

### 3.2 step/起身（不变）

idle env 每个 episode **随机倒地初始**（fallen 插件无条件触发，
target=both），全程零指令——倒地→起身→原地踏步。时长 80 步
4 秒（起身 ~10 步已掌握 + 原地踏步 ~70 步）。

### 3.3 打击：只学"打出去"（核心修订）

- **打分窗口 = [0, k_land + margin]**：k_land = clip 峰值伸展帧
  （`_clip_meta` argmax 实测：jab=11、heavy=6）；margin=13 给慢速
  尝试留足完成时间。重拳参考时钟半速（`clip_rates=0.5`，
  12 帧到 k_land）——开立步站架需要 ~1 步位移，6 帧物理上不可达。
- **余量帧的参考 = 钳在 k_land 的保持打击姿态**
  （`StrikeTrackObserver` 新 `ext_only` 选项，参考帧
  `min(t, k_land)`）——**速度自动变成"在目标姿态上停留的时间"**：
  打得快→余量帧全程拿满跟踪分；打得慢→余量帧还在够，分少。
  无需显式速度项。
- **重心完全放开（v8）**：strike ep 的 r_fall/r_potential 全程
  actor_weight=0——允许扑出去打、允许打完摔倒；"怕摔"曾压制
  CoM 前移（实测收敛策略全程 φ≈1.0）。r_strike 也去掉 φ² 门控
  （窗口内恒 4.0），重心下沉不再衰减打击梯度。
- 比 clip 还快的提前到位不额外加分（frame-锁定跟踪
  只奖励"按时+保持"），先不做更复杂的到达即冻结参考。

### 3.4 后续规划（暂不实现）

打出去学会后，把"打完瞬间状态"（手臂伸展、重心不稳）作为
初始状态，把"收招回位到标准姿态"当**另一个技能**学——
同一套 WarmupSteppingPlugin 模式：内部 sim 跑完打拳，取终态
作为收招技能的 episode 初始状态。

### 3.5 阶段二：技能衔接（`skill_trans_mbs`，进行中）

EXP_CURVED_MOTION（`EXP_CURVED_MOTION.md`）实测：阶段一技能只在
"站立起步"分布下可靠——fwd/turn 穿插与 vx+w 组合下 10-42%
episode 时间处于失衡低位。阶段二目标：**任意技能的任意阶段都可
作为任意技能的初始状态**，补齐衔接瞬态。

机制（对阶段一零侵入）：

- `WarmupSteppingPlugin` 新增技能热身路径：
  `episode_options["warmup"][robot_id] = {skill, steps, schedule}`。
  有 spec 时内部 sim 用 `CommandSimulator(strike_dyn_target=True)`
  按 `move_schedule` 驱动（打击动态目标/锚点自动生效）；无 spec
  时走原零指令踏步路径——**eval、视频渲染、阶段一训练完全不变**。
- 新实验 `exp_skill_trans_mbs.py`（`SkillTransMBS`）继承
  `SkillTrainMBS`，**只**重写 stochastic `build_jobs`：每 robot
  独立采热身技能（全技能均匀）+ 时长（均匀 [0,cap]，0=无热身）。
  Eval 路径（stochastic=False）调用父类不变。
- 排除规则：仅"热身打击 == 目标打击"（同 clip 相位冲突）；
  其余允许，含同技能延续（move→move）与异打击衔接（jab→heavy）。
- 热身时长上限：move 35 / turn 30 / idle 40 / strike 20 步
  （打击窗口短 → cap 小）。

技能集合 = 11 个（move×4 / turn×2 / step / jab×2 / heavy×2）。
"12 种"的说法以 11 为准（`SKILLS` 表）。

## 4. Env 变体（4 个 yaml，4 族）

| env | 技能 | 差异点 |
|---|---|---|
| `skill_move_env` | move_* ×4 | WarmupStepping + move_track + standing_balance + wall + foot |
| `skill_turn_env` | turn_* ×2 | WarmupStepping + 同 move 观测集 |
| `skill_strike_env` | jab/heavy ×4 | WarmupStepping + strike_track(ext_only) |
| `skill_idle_env` | step（含起身，同一指令） | fallen 插件 + move_track + std_clip + balance/foot/wall |

## 5. 奖励设计（按技能族挂通道）

公共：`r_potential`=0.01·φ（aw=3.0）、`r_fall`=摔倒+靠墙
（aw=2.0）。门控全部放 actor_weight（critic 看原始奖励）——
follow 实验验证过的模式，照搬。

**核心结构参照 `exp_follow_step`**：r_radial（镜像径向）/
r_tangential（切向惩罚）/ r_face（朝向）映射到技能域。

逐帧归因管线：root_xy 居中滑窗平滑（win=17）→ 居中差分得位移
→ 投影分解；位移上限 0.10 m/step（2 m/s 物理帽）。

### `r_move`（move env，aw=3.0·φ²）

```
d_hat   = 指令方向（ego，第 0 帧朝向定义；段内固定）
v_along = disp · d_hat
r_move  = 2.0 · v_along              # 线性=最速；反向自动为负
```

### `r_drift`（move env，aw=2.0·φ²）

```
v_off   = |disp − v_along·d_hat|
Δyaw    = |wrap(yaw − yaw[0])|
r_drift = −1.0·v_off − 0.5·|ω| − 1.5·Δyaw
```

方向移动期间**朝向不许变**。系数经两轮热修：Δyaw 0.5→1.5
（dyaw 曾平台 ~1.2-1.6 rad）、权重 1→2（加权优势曾被 r_move
打平）。

### `r_turn`（turn env，aw=3.0·φ²）

```
ω_signed = d(yaw)/dt · sign(cmd_w)   # clip ±8
r_turn   = 0.5·clip(ω_signed,−4,4) − 1.0·|v_xy|
```

### `r_hold`（idle env，aw=3.0·φ²）

```
锚点 = φ 首次 ≥0.8 的帧（倒地初始 → 锚定即"站起后"）
d_pos  = clip(|root_xy − root_xy[锚点]|, 0, 3)
d_yaw  = |wrap(yaw − yaw[锚点])|
r_hold = −(0.3·|v_xy| + 0.3·|ω|) − (d_pos + d_yaw)
```

### `r_strike`（strike env，aw=4.0·in_strike）

纯 mocap 形状，**单调进度跟踪**（monotone progress tracking）：

- 进度指针 k 从 0 起，每步在 `[k, k+2]` 内取完整 reward 最高的
  ref 帧；只进不退——打多快时钟走多快，速度不被 clip 限速；
  缩手则姿态与当前帧不匹配自动掉分（天然禁止收回）。
- 指针到达 `k_land`（出拳伸展位）后，每剩余一步额外 +0.25
  hold bonus：落地越早赚得越多（speed = time-at-target）。
- 站架慢的尝试不罚时（指针会等），episode 长度 = 到达预算 +
  margin=13 保持帧。

单帧 reward（候选帧共用同一函数）：

```
r = 0.35·pose + 0.10·vel + 0.50·arm + 0.25·kp + 0.10·height
    + 0.20·root + 0.50·hand_reach(σ=0.45)   [+0.25 if landed]
```

strike ep 的 `r_potential`/`r_fall` actor_weight 恒 0（重心完全放开）。

### `r_std_clip`（idle env，aw=2.0·φ²）

标准站姿片段质量；不挂在 move/turn/strike 段（防通道打架）。

## 6. 采样与初始

- 族分布：move 30%、turn 15%、strike 35%、idle 20%。
- 族内弱技能过采样（S2 步态右利手性）：
  turn_left ×2.5、move_left ×1.5、jab_l ×1.5、heavy_r ×3.0。
- 双方同策略同练：a/b 各独立采样技能，共享权重双倍数据。
- episodes_per_update = 1536（episode 短，翻倍吞吐）。

## 7. 网络与暖启动

obs_dim=66 不变；暖启动 `checkpoint_u01635.pt`（S2 最优）。
actor 全量；`r_potential`/`r_fall`/`r_std_clip` critic 同名继承；
`r_move`/`r_drift`/`r_turn`/`r_hold`/`r_strike` critic 冷启动。

## 8. 评测与验收

| 技能 | 指标 | 一期目标 |
|---|---|---|
| move_* | v_along / v_off / dyaw / 摔倒率 | ≥0.5 m/s、dyaw<0.3 |
| turn_* | ω_signed / \|v_xy\| | ≥1.5 rad/s |
| step | stood / d_pos / d_yaw | 起身 100%、漂移 <0.1 |
| jab/heavy | kp_err / q_rmse / hand_err / land_t（in_strike 内） | kp≤0.10 动作可见 |

quality = 各族归一化均值（move v/0.6、turn w/2.0、
strike 1−mean(kp,hand)/0.30、idle 1−d_pos/0.5），best-of-run 需
falls<0.2。`land_t` = 进度指针首次到达 k_land 的步号（出拳速度）。

## 9. 待定

1. margin=13 是否合适（慢速尝试的完成余量 vs 保持姿态的负担）。
2. r_drift/r_hold 系数继续按指标微调。
3. 左弱技能是否还有空间（可能步态物理上限）。

## 10. 已定案

- 转身双向两条指令；cmd 归一化 ±1=最速 ✓
- 重拳 `cmu_144_20`(_m)`_seg01` ✓
- B 同策略同练 ✓
- 起身/踏步同 env 同指令：idle env 全随机倒地初始、全程零指令 ✓
- **WarmupSteppingPlugin 替代 lead-in**：引导策略=当前导出策略，
  episode 本体纯技能 ✓
- **打击只学打出去**：打分窗=[0,k_land+margin]、参考帧钳 k_land、
  窗口后一切不计分、指令单相位指落点 ✓
- 不留死代码：`strike_phase_targets` 移除 ✓
- 原地性硬约束 + 方向移动朝向不变 + follow 式三件套 ✓
- env 按技能族划分，job 级标签挂通道（复合实验）✓

## 11. 对共享代码的改动清单

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `command_simulator.py` | 移除 `strike_phase_targets`（两段式回放废弃） | 唯一使用者即本实验 |
| `strike_track_observer.py` | 新增 `ext_only` 选项：参考帧钳 `k_land`；`clip_land_frame` 保留供窗口计算 | 默认关，旧实验不变 |

新文件：实验类 + 4 env yaml + `warmup_stepping.py` + 本文档 +
`render_skill_videos.py`（带 warmup_policy_cls 注入）。
