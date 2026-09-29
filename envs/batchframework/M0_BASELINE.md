# M0 基线与任务契约：standup_floor04

日期：2026-09-29。落实 [ROADMAP.md](ROADMAP.md) §3 的 M0 阶段，遵循 [discuss.md](discuss.md) 的原则。

本文是后续所有转换与验证工作的**冻结输入清单**。状态标注约定：

- **[冻结]**：已从代码/历史 Run 核实，可作为转换依据。
- **[候选]**：有推荐值，需用户确认后才成为正式协议。
- **[协议]**：已定义测量/验收方法，**本阶段未执行**（当前机器负载高，性能基线与 CPU 复评延后）。
- **[缺口]**：已识别的现有 MJX 原型与目标契约之间的差距。

## 1. 参考版本冻结

### 1.1 任务侧代码版本（转换源）

**结论：任务侧（`envs/`）在两个候选 Run 的 snapshot 与当前 HEAD 之间完全一致，HEAD 可以直接作为任务语义的转换源。**

已核实 `git diff`：

| 范围 | bf589247→HEAD | 5bdf8625→HEAD |
|---|---|---|
| `envs/framework`、`envs/humanoid21` | **无变化** | **无变化** |
| `baseline/experiments_ppo/exp_standup.py` | 仅 plumbing：`extract_explore_factor`→`extract_sampling_ctx`，新增 `post_update` 在线指标 | 同左 |
| `baseline/experiments_ppo/base.py` | +209 行（SamplingSpec/reference-delta/post_update artifacts 机制） | 同左 |
| `baseline/framework/rollout/*`、`ppo/trainer.py` | 大幅重构（SamplingContext 化、版本流 policy export） | 同左（差异更小） |
| `envs/batchframework/*` | 新增（discuss.md / ROADMAP.md / 原型） | 同左 |

含义：

- **任务语义转换源 = 当前 HEAD**。simulator、插件、rewarder、blueprint 与历史 Run 所用版本一致，无需 checkout snapshot 来读任务语义。
- **训练数据契约转换源 = 当前 HEAD**（M5 集成目标）。`Job`/`Episode`/`SamplingContext` 契约在两个 snapshot 之后被重构过，必须按 HEAD 对齐，而不是按 snapshot。
- 历史 Run 的用途是**行为与指标参照**（学习曲线、最终能力、计时），不是代码来源。

### 1.2 依赖版本（转换源环境）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.10 | `/usr/bin/python3` |
| MuJoCo | 3.8.0 | CPU 参考路径 |
| mujoco-mjx | 3.8.0 | 加速路径；`warp`/`mujoco-warp` 未安装（MJX-JAX 可用，import 时打印 warp 警告） |
| JAX / jaxlib | 0.6.2 | `jax_enable_x64=True`（现有原型要求） |
| PyTorch | 2.7.1+cu126 | 训练与策略推理 |
| NumPy | 2.2.6 | |

### 1.3 历史参考 Run（候选，未正式选定）

| | Run A | Run B（**[候选] 推荐为主参考**） |
|---|---|---|
| 路径 | `baseline/runs/train_standup_floor04_ppo_20260916_103522` | `baseline/runs/train_standup_floor04_ppo_20260920_164819` |
| snapshot commit | `bf589247`（base `06fb1caf`） | `5bdf8625`（base `009d7462`） |
| snapshot 分支 | `exp/train_standup_floor04_ppo_20260916_103522_20260916_103522` | `exp/train_standup_floor04_ppo_20260920_164819_20260920_164819` |
| seed / workers | 42 / 96 | 42 / 96 |
| 首个 eval success ≥0.9 | ~update 385 | ~update 380 |
| 末段 eval（最后 10 次均值） | success 0.9984 / max_pot 0.9989 / final_pot 0.9986 | success 0.9976 / max_pot 0.9982 / final_pot 0.9981 |
| eval@1500 | success=1.000, max_pot=1.000, final_pot=0.999, max_stage=4.0, max_h=1.282 | （同量级，见 train.log） |
| 稳态计时中位数 | rollout 10.85s / ppo 0.82s / total 12.19s | rollout 9.01s / ppo 3.03s / total 12.74s |

两个 Run 的 `config.json` 已核实：实验参数完全一致（见 §2.6）。差异仅在框架版本（Run B 的 ppo_params 含 `early_stop_kl_window` 等较新字段）与随机实现细节。

**推荐 Run B 为主参考**（snapshot 距 HEAD 更近、框架字段更全），Run A 作旁证。**[候选] 待用户确认。**

### 1.4 参考策略探针（已存在于 Run 目录）

| 探针 | 来源 | 用途 |
|---|---|---|
| 未训练策略 | `init_policy_truncated_normal.yaml` 新建实例（随机初始化权重，seed 由 `set_seed(42)` 控制） | V3 基线：随机行为下两后端的观测/奖励一致性 |
| 中期策略 | `policy_exports/u00360`…`u00558`（每 Run 多个导出版本） | V3：阶段 3/4 附近行为、奖励阈值敏感性 |
| 成熟策略 | `policy/`（best-of-run 导出）或 `checkpoints/checkpoint_u01500.pt` | V3：站立行为、长时间接触、Stage 4 平台区 |

`policy_exports/uNNNNN` 为 post-update-N 版本的独立导出目录（policy.py + model.pt + MANIFEST），可直接用 `PolicyBlueprint` 加载，不依赖训练进程。

### 1.5 种子与随机性协议 **[冻结]**

主路径随机性全部由 `SeedSequence.spawn` 派生（`envs/framework/SEED.md`）。训练侧的派生规则（`loop.py` + `base.py`，HEAD 版本）：

```
训练 seed = 42
rollout_seed(u)   = 42 + u * 512              # 第 u 个 update 的采样批种子
job_seed(i)       = rollout_seed + i          # 每个 episode 一个 int（算术派生，job 层）
episode 内部      = SeedSequence(job_seed).spawn(...) → simulator/policy/plugin 各自 int
eval_seed(u)      = 42 + 100_000 + u * 97     # eval 批种子，stochastic=False
initial_distance  = default_rng(rollout_seed).uniform(1.5, 3.5)   # 逐 episode
```

要点：

- Job 层用算术派生 `base_seed + i`（在 `spawn` 之前）；episode 内部一律 `spawn`。加速路径若要复现**同分布**初态，只需要复现 episode 级语义；若要逐 episode 对照，必须复现这条派生链（M1 记录为固定 fixture 生成规则）。
- `initial_distance ∈ U[1.5, 3.5]` 是**逐 episode 随机量**，不属于 blueprint 默认值（blueprint 里 `initial_distance=2.0` 只作 simulator 构造缺省，实际每次 reset 被 `episode_options` 覆盖）。
- eval 是 `stochastic=False` 的确定性 `act()`；训练 rollout 是 `stochastic=True` 的 `sample()`。两者的 SamplingContext 路径不同，都要在契约内。

## 2. 任务契约清单

以 HEAD 代码为权威来源。所有字段名、单位、坐标系以下面为准；加速路径输出必须在语义上逐项对应。

### 2.1 环境与时间

| 项 | 值 | 来源 |
|---|---|---|
| XML | `envs/humanoid21/battle_circular_v2.xml`（24 墙圆形场地 + ground/ceiling） | `simulator.py:35` |
| 物理步长 dt | 0.002 s（500 Hz） | `meta.py:200` |
| 每动作步物理子步 | 25（动作频率 20 Hz，每 episode 5000 物理步 = 10 s 模拟时间） | blueprint `phy_steps_per_action` |
| 最大动作步 | 200（TimeoutPlugin） | blueprint `max_steps` |
| strict | true（插件异常直接抛出） | blueprint `runtime.strict` |
| 机器人 | robot_a（后缀 `_a`）、robot_b（`_b`，reset 时 yaw +180° 面向 a） | `meta.py:145` |
| 受控关节 | 21 DOF，`CONTROLLED_JOINTS` 顺序即 action 维度顺序 | `meta.py:134` |

### 2.2 动作契约 **[冻结]**

- 输入：`{"robot_a": (21,), "robot_b": (21,)}`，接受 ndarray/torch/list/`None`（`None` 跳过该机器人，沿用上次目标）。
- 语义：归一化目标关节位置，**逐物理子步**做 PD：`torque = KP*(action*scale + ref − qpos) − KD*qvel`，`ctrl = clip(torque/gear, ctrl_lo, ctrl_hi)`。KP/KD 为 21 维固定表（`meta.py:203-227`）。
- 裁剪：action 先 clip 到 [-1, 1]；ctrl 再 clip 到 actuator ctrlrange。
- **MJX 侧必须每个物理子步重新计算 PD**（不能只算一次 ctrl 后 scan 25 步）。

### 2.3 观测契约（96 维）**[冻结]**

`get_observation()` → `{robot_id: (96,) float32}`，按序拼接（`_get_robot_view`）：

| 段 | 维度 | 内容 | 变换 |
|---|---|---|---|
| joint_pos_norm | 21 | 归一化关节位置 | 线性 `(pos−ref)/scale` |
| joint_vel | 21 | 归一化关节速度 | `sign(v)*sqrt(|v|)/2`（观测专用非线性） |
| projected_gravity | 3 | 重力在机体系的单位向量 | 机体系 |
| height | 1 | torso 世界系 z | — |
| linear_vel | 3 | 根线速度 | **机体系**（世界系 × R^T） |
| angular_vel | 3 | 根角速度 | 机体系 + `sign*sqrt(|v/2|)` |
| feet_forces | 2 | 双足地面受力 | **除以体重 m·g 归一化** |
| arena_center_local | 3 | 世界原点在机体系的位置 | 机体系 |
| opp relative_pos | 3 | 对手 torso 相对位置 | 自机体系 |
| opp relative_vel | 3 | 对手 torso 线速度 | 自机体系 |
| opp face_vector | 3 | 对手朝向 (+x) 在自机体系 | 自机体系 |
| opp keypoint pos | 15 | head/hand_r/hand_l/foot_r/foot_l 相对位置 | 自机体系 |
| opp keypoint vel | 15 | 同 5 关键点线速度；head 不变换，其余 `sign*sqrt(|v|)/2` | 自机体系 |

注意：观测中的 sqrt 压缩只作用于平铺向量，derived_state 里的原始字段不带变换。两者都要在加速路径可区分地提供。

### 2.4 核心状态契约 **[冻结]**

`get_core_state()` → 每机器人：

| 字段 | 形状 | 定义 |
|---|---|---|
| `root_pos` | (3,) | torso 世界系位置 (m) |
| `root_rot` | (4,) | torso 世界系四元数 **[w,x,y,z]** |
| `root_vel_local` | (3,) | 根线速度，**机体系**（MuJoCo qvel[0:3] 是世界系，读时乘 R^T；写回时乘 R） |
| `root_angular_vel_local` | (3,) | 根角速度，机体系（qvel[3:6] 本来就是机体系，直通） |
| `joint_pos_norm` | (21,) | 归一化关节位置 |
| `joint_vel_norm` | (21,) | 归一化关节速度（`vel/scale`，无 ref 偏移） |

`set_core_state` 语义：按 robot 和字段**部分更新**；`root_vel_local` 用**当前四元数**做机体系→世界系转换；写完统一 `mj_forward` 刷新。

### 2.5 reset 与插件语义 **[冻结]**

**`simulator.reset(options)`**：

1. `mj_resetData`；可选 `initial_distance` / `initial_pose_a` / `initial_pose_b`（standup 用 standing/standing + 逐 episode 距离）。
2. robot_a 放 x=−d/2，robot_b 放 x=+d/2 且四元数乘 z 轴 180°。
3. qvel、xfrc_applied、qfrc_applied 清零；按实际 joint_pos 反算 action 目标。
4. `mj_forward` 刷新（读状态立即可用）。

**`RandomFallenStatePlugin`（on_pre_episode，本任务配置 target_robots=[a,b]，max_phy_steps=1000，height_threshold=0.3，reset_interval=5）**：

1. 建一个**内部 CPU Humanoid21Simulator**，reset 后写入真实环境的当前 core state。
2. 每个目标机器人给**一个** uniform[-1,1] 随机 action（整段摔倒过程不变）。
3. 逐物理步推进内部 sim，**两个目标机器人在同一场景中同时摔倒、互相可见**。
4. 早停条件：每步取目标机器人 root 高度的**最小值**，`< 0.3` 即停——即**任一方先低于阈值就结束整段摔倒**，另一方可能仍在下落途中。
5. 把目标机器人的完整 core state 写回真实环境；记 `*_fallen_init_steps/height/threshold` metrics。

注意两点精确语义：

- 本任务 `target_robots` 覆盖 a+b，因此 `non_target_state` 为空，**`reset_interval` 周期重置机制处于休眠**（代码路径存在但不被触发）。转换时该机制仍需保留语义位置，单目标配置会激活它。
- 早停是"min over targets"，不是"all targets 均低于阈值"——MJX 版若改成逐机器人独立早停会改变初态分布。

**这是 M0 识别的最大转换难点**：内部 sim 的单环境 CPU 循环无法直接上设备。M4 的合规做法（先相同初态对照、再设备端原生 reset）必须保持：随机 action 分布、双机同场摔倒、min-height 早停、只写回目标机器人、metrics 字段名。

**外力契约**：`apply_external_force` 累加进 `xfrc_applied[body]`，**下一个物理步生效，每个 physical_step 结束后清零**——要持续施力必须每个子步重放。standup_floor04 当前无运行时扰动插件，但契约属于 simulator 公共能力，需在能力矩阵登记。

### 2.6 奖励/观测插件契约 **[冻结]**

`StandingBalance4StageRewarder`（每个 robot 一个实例，observer key：`standing_balance_a/b`）：

- 输入：`derived_state['contacts']`（SoA）+ `derived_state[agent]`（body_xpos/body_xquat）+ `static_data`（keypoint 名、id→name/aff 映射）。
- 接触过滤：`aff` 一侧为环境(0)一侧为目标机器人(1/2)，且环境 geom 名 == `'ground'`，且 `force_mag ≥ 1.0 N`；按 **body** 去重计 `extra_contact_count`（set 语义）。
- 常量：`H_HAND_MAX=H_FOOT_MAX=0.3, D_MAX=1.0, D_MIN=0.2, OTHER_PENALTY_K=0.5, F_ENTER=0.8, D_GATE=0.6, H_CROUCH=0.15, H_STAND=1.28, F_LOAD_MIN=10.0`。
- 输出 9 字段：`stage, potential, f_score, contact_score, d_score, d_hf, w_foot, h_score, h_torso`。
- 训练侧消费：仅 `potential` → `r = 0.01*φ`；`stage`/`h_torso` 进 eval 指标；其余字段经 observer_outputs 进 dump。

**终止语义**：本任务无早停——所有 episode 跑满 200 步以 timeout 结束（两个 Run 的 `termination_reasons` 均为 `{timeout: 1024}`）。`Trajectory.is_terminated=False` 恒成立，timeout 走 bootstrap 路径。加速路径仍须实现通用 per-agent termination 契约，只是本任务不触发。

### 2.7 Rollout 数据契约（M5 边界，HEAD 版本）**[冻结]**

`Job`（冻结 dataclass）：`policy_a_bp, policy_b_bp, env_bp, seed:int, episode_options, sampling_a/b:SamplingSpec, stochastic:bool`。`SamplingSpec`：`explore_factor`（standup=0.0）、`reference/delta_factor/delta_mix`（standup 未启用，`delta_mix=0`）。

`Episode`（冻结 dataclass）：`base_seed, episode_index, blueprint_hash, num_frames, episode_options, agent_termination_proposal_records, observations{agent:(T,96)}, actions{agent:(T,21)}, action_extras{agent:{key:(T,...)}}, explore_factors{agent:(T,)}, observer_outputs{stacked}, final_observation{agent:(96,)}, episode_metrics`。
- `num_frames=T` 动作步数；`final_observation` 是 `obs_{T+1}`，用于 bootstrap。
- `sampling_contexts` 是派生视图：`explore_factor` + `action_extras` 中所有 `sctx__` 前缀键。
- 真终止判定：`agent_termination_proposal_records[agent][0].reason != "timeout"`。

**每 update 数据量**：512 episodes × 2 agents = 1024 trajectories × 200 steps = 204,800 transitions。eval：64 episodes × 2 agents = 128 条 agent 轨迹，指标按 agent 轨迹计（success 分母 128）。

### 2.8 指标与判定 **[冻结]**

- 训练指标：`exp.online_success`（末帧 φ≥0.9 占比）、`exp.final_potential_mean`；PPO 侧 uncertainty/std/KL/confidence 等。
- eval 指标（`on_eval`）：`max_pot, final_pot, max_stage, max_h, success`（max_pot≥0.9）。
- 历史参照：见 §1.3。两个 Run 都在 ~update 380-385 达到 eval success≥0.9，末期 success≈1.0。

## 3. 能力矩阵与现有 MJX 原型缺口

| 能力 | standup 必需 | 现有原型状态 | 备注 |
|---|---|---|---|
| 正确 XML（battle_circular_v2） | 是 | **[缺口]** 加载 battle_v1.xml | M2 第一优先；模型不一致时一切对照无效 |
| PD 控制（逐子步、KP/KD/gear/ctrlrange） | 是 | 已实现，待对冻结契约验证 | 需确认每子步重算而非 scan 外计算 |
| core state 读/写（含机体系速度换算） | 是 | 部分实现 | **[缺口]** 写后即时刷新、`root_vel_local` 换算路径待核 |
| 96 维观测（含 sqrt 变换、机体系投影、feet_forces 归一化） | 是 | 名义 96 维 | **[缺口]** 字段布局/变换与 §2.3 逐段对齐待核 |
| derived_state：contacts SoA + body/joint 数组 | 是 | 部分（含 Python 循环提取） | **[缺口]** 力提取需 efc_force+锥分解对齐；热路径须设备化 |
| `set_core_state` 部分写入 + mj_forward 等价刷新 | 是 | 未验证 | MJX 需 `mjx.forward` 等价物 |
| `apply_external_force`（次步生效、每步清零） | 否（任务未用，契约要求登记） | 未验证 | **[缺口]** xfrc 语义对齐待做 |
| reset（双侧站位、b 侧 yaw 翻转、逐 episode 距离） | 是 | 部分实现 | 待按 §2.5 核 |
| RandomFallenStatePlugin（内部 sim 随机摔倒） | 是 | **不存在** | **[缺口]** 最大单项：需设备端等价实现，分布级验证 |
| StandingBalance4StageRewarder | 是 | 不存在 | **[缺口]** 依赖接触语义先行 |
| Timeout/termination/bootstrap 契约 | 是 | 不存在 | batchframework 需 per-agent 终止语义 |
| SeedSequence 派生链 | 是 | 不存在 | 加速路径需等价 RNG 协议（分布等价即可，不要求同序列） |
| Episode/Job/采样契约输出 | 是 | 不存在 | M5 边界 |
| 渲染 `get_broadcastview_image` | 否（训练热路径不需要） | 不存在 | 允许用 CPU renderer 回放录制状态 |
| `get_sensor_data`（当前恒 `{}`） | 是（契约面） | 返回空即可但须显式存在 | 不许"漏字段"式省略 |

## 4. M1 最小测试输入范围（冻结建议）

供 M1 建 fixture/比较器使用，分六类：

- **F1 初始态**：reset 后 standing 双机，覆盖 initial_distance ∈ {1.5, 2.0, 3.5}。
- **F2 摔倒初态**：CPU 插件在若干 episode seed 下的输出 core state 快照（M1 从主路径采集落盘，作为无条件分布样本与逐点对照输入）。
- **F3 中间姿态**：成熟策略 rollout 中的 stage2/3/4 代表帧（CPU 录制，含多接触、手脚支撑、半蹲）。
- **F4 阈值边界**：构造态覆盖 rewarder 门槛两侧 ±ε——`force_mag`≈1N、`d_hf`≈0.52m（D_GATE 边界）、`f_score`≈0.8、`h_torso`≈0.15/1.28、`F_foot+F_hand`≈10N。
- **F5 动作序列**：零动作、站姿保持 action（INITIAL_POSES.standing.action）、±1 交替、历史 rollout 实采 action 片段。
- **F6 步进粒度**：单物理步、25 子步（1 动作步）、多动作步短程（≤10 步），区分"逐点对齐窗口"与"混沌发散窗口"。

比较器要求：按字段（qpos/qvel/观测 96 维逐段/contacts 语义匹配/奖励 9 字段）分 tolerance 报告；接触按 (geom pair, position) 语义匹配，不比数组顺序；报告逻辑差异与数值差异分开。

**容差数值属于 [候选]**：现有测试在 fp64 下单步 qpos/qvel ~1e-14；正式容差待 M1/M2 按 fp32/fp64 两档标定后冻结。

## 5. 待确认与延后事项

### 5.1 待用户确认 **[候选]**

1. 主参考 Run：推荐 **Run B（20260920_164819）**，Run A 旁证。
2. 验收门槛候选值（讨论值，未冻结）：eval success 差距 ≤2pp、final_pot 差距 ≤0.02、样本量容忍 1.25×、端到端加速目标 2×。
3. GPU 分配：8×4090 中训练用卡与空闲基线测试卡的划分。

### 5.2 本阶段未执行（负载原因，协议已定义）**[协议]**

1. **CPU 复评**：用 Run B 成熟策略在 HEAD 跑 eval 协议（64 eps，stochastic=False），确认 `success` 仍 ≈1.0，排除"HEAD 上 eval 已退化"的隐性前提。建议在高负载缓解后执行，约分钟级。
2. **CPU 成本分解**：模型初始化 / reset（含 RandomFallen 内部 sim）/ 物理推进 / 观测+奖励 / 策略推理 / 进程通信 / buffer / PPO / eval 分段计时。现有 `time.*` 只有 rollout/buffer/ppo/eval 四段，需要在 worker 内插桩细拆——**特别注意 RandomFallenStatePlugin 的内部 sim 每 episode 最多 1000 物理步，可能占 reset 成本大头**。
3. **性能计时口径**：稳态取连续非 eval 更新的中位数；单独报告冷启动（JIT 编译）与含 eval 的 update；样本效率用 agent transitions 计数而非 update 数。

### 5.3 M0 出口检查

- [x] 参考版本与依赖清单（§1）
- [x] 任务契约清单（§2）
- [x] 能力矩阵与缺口（§3）
- [x] M1 测试输入最小范围（§4）
- [ ] 主参考 Run 正式选定 — 待用户确认 §5.1-1
- [ ] 验收门槛冻结 — 待用户确认 §5.1-2
- [ ] CPU 复评与成本分解 — 延后（§5.2）
- [ ] 评估种子/留出集固化 — 随 M1 fixture 生成时一并冻结

**M0 结论**：任务侧转换源可直接用 HEAD；主要缺口集中在 XML 对齐、接触语义、摔倒初始化、设备端热路径四点。在 §5.1 确认后进入 M1。
