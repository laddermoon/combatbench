# 精细化打击技能方案（Precision Strike）—— 阶段三

> 状态：**方案待审阅**（2026-10-19）
> 前置：阶段一技能学习（`skill_train_mbs`，u9860 收敛）、阶段二技能衔接
> （`skill_trans_mbs`，进行中）

## 1. 目标

打击技能从"模仿一条参考 clip"升级为"命中一个被指令参数化的目标点"。

结果导向的两个独立评价维度：

- **精度**：出手过程中手中心轨迹（空间曲线）与目标点的最近距离 `d_min`
- **速度**：到达最近点的用时 `t*`（action steps）

塑形不丢：参考 clip 的姿态/关节链跟踪项保留为辅助奖励，防止纯末端
约束退化成"戳指"。

**初始姿态泛化内嵌**：本阶段建立在技能衔接训练（`skill_trans_mbs`）
之上并继承其全部机制——训练 episode 仍以"任意技能任意阶段"热身
起手（warmup spec 全覆盖）。因此目标打击的掌握标准是：**从移动中、
转身中、另一打击的中/后段等任意初始姿态出发，仍能命中指定点**。
不单独做"站立起步下的精确打击"中间态。

## 2. 关键概念：世界锚定目标

指令 `cmd[6:9]` 的 target 是 **ego 坐标**（相对骨盆当前位姿）。出拳中
机器人转胯/上步，同一 ego (x,y,z) 对应的世界点每帧都在漂。

**解法（沿用现有 anchor 机制）**：打击段开始时（该段 `move_schedule`
生效的第一帧），把指令 ego 点烘成**世界系锚点**：

```
p_world = root_pos(t0) + Rz(yaw(t0)) · p_ego        # 段首捕获一次
p_ego(t) = Rz(-yaw(t)) · (p_world - root_pos(t))    # 每帧重投影回 ego → cmd[6:9]
```

`command_simulator._strike_target_ego` / `seg['anchor']` 现有的
"段首锚定 → 逐帧 ego 重投影"管线原样复用——现在它锚的是 clip 手
轨迹回放，本方案换成锚**被指令的自由目标点**。cmd 合同不变
（观测仍是一个会随机器人移动而在 ego 里漂的目标点）。

### 2.1 移动目标（阶段 3b）

锚点从常数变为时变轨迹：

```
p_world(t) = p0 + v_drift · (t - t0) · dt          # 缓速匀速漂移
```

出手续 ~25 步 ≈1.25s，v_drift ∈ [0, 0.4] m/s 随机方向 → 目标在出手
过程中漂移 ≤0.5m。重投影逻辑不变，只把 `p0` 换成 `p0(t)`。

## 3. 奖励设计（新增两个惩罚通道）

手的**伸展段曲线**：段首 → `t_reach`（段内 `||hand−pelvis||` 取最大值
的步号，即出手最远点；之后的收手/保持段不计入曲线）。在
`build_trajectories` 里事后计算——与现有各通道一致，不占实时观测。

由曲线算出：

```
d_min = min_{t∈[seg_start, t_reach]} ||hand_world(t) − p_world||
t*    = argmin ...                                             (相对段首)
```

| 通道 | 定义 | 类型 | 语义 |
|---|---|---|---|
| `r_tgt_dist` | 在 `t*` 处一次性 `-w_d · d_min`（稀疏终点罚） | 惩罚 | 精度：轨迹离目标有多近 |
| `r_tgt_time` | `t ∈ [seg_start, t*]` 每步 `-w_t`（t* 之后归零） | 惩罚 | 速度：累计 = `-w_t·t*`，越晚接近罚越多 |
| `r_strike` | **保留**：clip 姿态/关节链跟踪（塑形正则） | 已有 | 防退化形态 |
| `r_hit`（可选） | `d_min ≤ ε` 时在 `t*` 处 `+B` | 奖励 | 命中标记，利于 hit_rate 梯度 |

设计取舍说明：

- `r_tgt_dist` 选**稀疏终点罚**（在 t* 记 `-d_min`）而非每步 `-dist(t)`
  稠密罚——后者把"手离目标远的时间"也计入（出手前段手在护架必然远），
  与"最近距离=精度"的定义不严格对应；稀疏罚语义干净，GAE 可回传。
- `r_tgt_time` 用稠密 `-1/步` 直到 t*——等效于 `-t*` 的 episode 总罚，
  且对 PPO 是稠密梯度。与 strike 的 land_t/hold bonus 同向（都快），
  但 `t*` 基于**自由目标**而非 clip 指针，语义独立。
- `r_strike` 保留但 actor_weight 可考虑随课程半径增大而下调
  （目标偏离 clip 落点变远时，姿态跟踪与终点命中会拉扯）。

## 4. 课程：锥形放宽

### 4.1 原点：实测命中分布（P0 先做）

在当前策略（`skill_trans` 最新 checkpoint）上对四个打击技能各跑
N 个 eval episode，**带 warmup 泛化起手**（与训练同分布），记录
`t_reach` 时刻手的世界坐标 → 得到每技能的命中点分布
`c_skill ± σ_skill`（世界系，相对出手段首机器人位姿）。

这就是"机器人现在会打到的 xyz"——课程的锥尖。泛化起手下的分布
会比纯站立起步更宽，锥心取条件均值，初始半径需覆盖泛化方差。

### 4.2 采样与放宽

```
目标采样（ego，段首）：
  p_ego ~ c_skill + Uniform(ball(r_curr))
  r_curr = r_min + (r_max − r_min) · level/L_max      # 锥形放宽
```

- `r_min ≈ 0.05m`（贴着现有命中点）→ `r_max ≈ 0.5-0.6m`（臂展 0.85m
  + 弓步余量内的宽范围）
- 采样域约束：身体前方半球锥（出拳可达域）；过低/过近躯干的点剔除
- clip 绑定仍由 punch type（skill 名/side+eff_type）决定——目标点
  不影响选哪条 clip，只影响打到哪里

### 4.3 晋级判据

沿用 follow_step 的 eval 门控晋升模式：

```
每 eval：按当前 r_curr 采样 K 个目标 → 测 hit_rate = P(d_min ≤ ε)
hit_rate ≥ θ_hit（如 0.6）连续 patience 次 eval → r_curr 进一档
```

`ε` 建议 0.08m（≈拳面直径）；`θ_hit`、`patience` 待训后调。
课程状态（`r_curr`、hit 计数）进 `state()/load_state()`，随
checkpoint 续训恢复。

### 4.4 阶段 3b：移动目标

静态锥形打满（r_max 处 hit_rate 达标）后，开放 `v_drift ∈ [0, v_max]`，
同样 eval 门控放宽 v_max。

## 5. 实施清单

| # | 改动 | 文件 | 说明 |
|---|---|---|---|
| 1 | 段首目标锚定 | `command_simulator.py` | strike 段的 `tx,ty,tz` 语义从"选 clip 的 intent"改为"ego 目标点"；新增 `strike_target` 处理：段首锚定 + 逐帧 ego 重投影写入 cmd[6:9]；支持可选 `v_drift` |
| 2 | 目标规格传递 | episode_options | 新增 `episode_options["tgt_strike"][rid] = [{a_step, p_ego, v_drift}]`，与 move_schedule 平行（不改行格式）；CommandSimulator 和 observer 共用 |
| 3 | 手轨迹记录 | `strike_track_observer.py` | 每步增输出 `hand_wx/y/z`（世界系手中心）、`tgt_wx/y/z`（当前锚点世界系）；`build_trajectories` 用它们重建伸展段曲线 |
| 4 | 新通道 | `exp_skill_precise_mbs.py`（新） | 继承 `SkillTransMBS`（**warmup spec 全部保留**——每 episode 仍先采热身技能+时长，strike 目标打击在泛化起手分布上训练）；strike ep 加挂 `r_tgt_dist`/`r_tgt_time`（+可选 `r_hit`）；job 构造时对 strike 技能采锥形目标，与热身采样相互独立 |
| 5 | 课程状态 | 同上 | `r_curr`、`v_drift` 档位、hit_rate 窗 → `state()/load_state()` |
| 6 | eval 指标 | 同上 | 每技能报 `tgt_d_min / tgt_t / hit_rate`，**分两种起手统计**：无热身（对齐旧 eval）+ 带热身泛化（泛化起手下的精度才是本阶段目标）；quality 复合分加入精度项 |
| 7 | 诊断先行 | 新脚本 | P0：实测四技能命中点分布 → 锥心表（写进本文件附录） |

**零侵入约束**（沿用阶段二原则）：

- `skill_train_mbs` / `skill_trans_mbs` 代码不改——新实验类独立存在
- `strike_dyn_target` 旧路径（无 `tgt_strike` spec 时）行为完全不变：
  eval、视频渲染、旧 run resume 不受影响
- `WarmupSteppingPlugin` 技能热身照旧工作（其内部 sim 的 strike
  段没有 tgt spec → 走原 clip 回放热身，合理）

## 6. 风险与开放问题

1. **形态与终点拉扯**：r_max 放宽后目标点可能远离 clip 落点锥——
   姿态跟踪(clip)与终点命中(target)梯度冲突。预案：r_strike 的
   actor_weight 随 r_curr 线性衰减（1.0→0.4），或只保留姿态项中
   与落点无关的部分（arm 链形状，去掉 hand 端点项）。
2. **伸展段终点定义**：`t_reach = argmax ||hand−pelvis||` 对"没出手
   就跌倒"的 episode 会给出无意义曲线——若段内 max reach <
   reach_min（如 0.3m）判为未出手，d_min 按 WORST_DIST 计罚。
3. **目标过界**：ego 采样需剔除不可达域（||p_ego||>0.8m 或
   x<0 背后）；锥形放宽后期视情况加可达性 clip。
4. **d_min 稀疏罚的信用分配**：t* 处单点罚离目标时刻近、离出手
   决策远——若学习信号弱，退化为"在 t* 前后 ±2 步窗口分摊罚"。
5. **与衔接阶段共存（本阶段的核心约束）**：新实验继承
   SkillTransMBS，warmup spec 在所有 episode 保留——strike ep 也从
   任意技能中段起手（移动中出拳、转身中出拳、一拳打一半接目标打
   击）。目标锚定在**打击段首**执行：段首时刻的世界系锚点对任意
   起手姿态都成立，与热身无机制冲突。需留意两类形态：
   (a) 热身=异/同技能打击中段 → episode 起手时手臂已伸出，伸展段
   曲线从"已伸展"开始，`t_reach` 可能极早——精度语义仍成立
   （曲线仍是有序空间轨迹），但 t* 的"速度"含义弱化为"已在前",
   属可接受退化；
   (b) 热身=倒地恢复中段 → 起手失衡，φ² 门控下 tgt 通道梯度本来就
   弱，等站稳后打击仍记完整曲线——曲线起点用"段内首次 φ≥0.8"
   截断更干净（待定，P0 数据看后定）。

## 7. 验收标准（草案）

- 阶段 3a（不动点）：r_max 放宽至 ≥0.4m 时，**带热身泛化起手**的
  eval 采样目标 hit_rate(ε=0.08) ≥ 0.6，t* 中位数 ≤ 8 步；无热身
  eval 同水位（泛化起手不弱于站立起步）
- 四拳形态不退化（`jab_*_q`/`heavy_*_q` 不低于阶段一终态 90%）
- 阶段 3b（移动点）：v_drift 0.3m/s 下 hit_rate ≥ 0.5
- 回归：move/turn/step/hold/衔接指标不因新通道退化

## 附：P0 实测命中分布（2026-10-19，u10135 导出，带 warmup 泛化起手）

数据：`hit_distribution_p0.json`。锥心取 **φ0≥0.8 站立起手**条件化
统计（倒地起手的野点已剔除——其 land 落在身后，属恢复过程乱序数据）：

| skill | 命中点 mean (ego, m) | std (m) | t_reach 均值 |
|---|---|---|---|
| jab_l | (0.17, 0.30, 0.54) | (0.25, 0.24, 0.03) | 16.6 步 |
| jab_r | (0.37, −0.32, 0.53) | (0.27, 0.23, 0.03) | 12.0 步 |
| heavy_l | (0.51, −0.13, 0.52) | (0.32, 0.41, 0.08) | — |
| heavy_r | (0.42, 0.14, 0.48) | (0.25, 0.34, 0.05) | — |

- 泛化起手下命中点自然散布 ±0.2–0.4m —— 正是本阶段要收敛的量
- 实测命中点与 clip 落点有系统偏移（如 jab_l mean.x=0.17 vs
  clip 0.44）—— 泛化起手下起手姿态影响出拳落点，印证"锥心取
  实测分布而非 clip 落点"的决定

## 附：实施状态（2026-10-19）

已实现并启动训练 `train_skill_precise_mbs_ppo_20261019_141450`
（GPU2，从 `skill_trans_s2b_u9860b/checkpoint_u10170.pt` 续训）：

- `command_simulator.py`：`tgt_anchor` option + `tgt_drift` 规格；
  段首锚定 `p_anchor = pelvis + R·tgt_ego`，逐帧重投影回 ego 写入
  cmd[6:9]（与 policy 所见一致）；无 spec 时旧路径不变
- `strike_track_observer.py`：clip 显式覆盖（`strike_clip` option）+
  每步输出 `hand_ex/y/z`、`tgt_ex/y/z`（ego 系，经 accessor
  `get_observation` 读到的逐帧显示目标——与 policy 输入完全一致，
  无需双端重建锚点）
- `exp_skill_precise_mbs.py`：10 通道（+`r_tgt_dist`/`r_tgt_time`），
  `SkillTransMBS` 子类保留 warmup；锥形目标采样 + r_curr 课程
  （eval hit_rate 门控晋级，state/load_state 持久化）；eval 报
  `{skill}_tgt_{dmin,t,hit}` 指标

验证记录：

- build_jobs：strike ep 带 `tgt_anchor`+`tgt_spec`，eval 同；move/
  turn ep 无锚——✓
- 锚定实测：25 步内重投影世界点与 p_anchor 逐位一致（机器人弓步
  前移 0.27m 而世界目标不动）——✓
- smoke：10 通道全部产出，r_tgt_* 在 strike traj 上激活 1086 条——✓
- 首 eval(u10175)：falls=0.026 quality=0.797（warm-start 保留）；
  jab_r_tgt_dmin=0.077 hit=0.611，jab_l 0.154，heavy_l 0.174，
  heavy_r 0.235 —— 课程将从 r≈0.1m 起步放宽
