# 上下文：时间相关探索噪声（Temporally Correlated Exploration）

**状态**：上下文整理完成，待细化方案
**关联文档**：`TODO_temporally_correlated_exploration.md`（设计草案）、`GUIDE.md`
**整理日期**：2026-09-02

---

## 1. 问题证据

### 1.1 当前训练观测

`train_standup_step_v3_ppo_20260902_153131`（从 standup u710 checkpoint resume）：

| 指标 | u711（起步） | u822（~110 updates 后） | 趋势 |
|---|---|---|---|
| entropy | -3.264 nats | -5.42 nats | 单调下降，无企稳 |
| std_mean | 0.224 | 0.205 | 缓慢收缩 |
| std_min | 0.082 | 0.082 | 卡在 log_std_min 下界 |
| KL | ~0.05（触发 early stop） | ~0.019（跑满 4 epochs） | 策略移动量变小 |
| clip_frac | 0.35 | 0.28 | 下降 |
| early_stop | 几乎每 update 触发 | | |
| 4 channel EV | r_potential=0.95 | 全部 >0.95 | critic 已学好 |
| 4 channel confidence | r_potential=0.98 | 全部 >0.97 | critic 已学好 |

**诊断**：策略在快速收缩探索范围（等效体积缩小到 ~1/9），但没有发现新行为（仍然 timeout=200 原地平衡）。critic 已经学好，actor 在 critic 指导下收敛到确定性解。

### 1.2 根因

白噪声（每步独立采样）的功率谱密度平坦，低频段（步态所需 1-2Hz）能量不足。策略发现大 std 只带来破坏性高频抖动，理性选择是收缩 std。这不是 reward/观测/策略族问题，是**探索噪声的频谱结构**问题。

### 1.3 文献

- *Pink Noise Is All You Need* (ICLR 2023)：连续控制上 pink/OU 噪声普遍优于白噪声，尤其步态任务。
- 经典 RL 中 OU 噪声用于 off-policy DDPG 已有共识；on-policy PPO 中少用但根因相同。

---

## 2. 架构全景：数据流

### 2.1 完整路径（policy → rollout → trajectory → PPO update）

```
experiment.exploration(u) → ExplorationSpec(temperature, entropy_coef, ...)
    │  loop.py:464-470
    ▼
actor.set_exploration(spec)
    │  tanh_gaussian_mlp.py:197-225 或 tanh_squashed_base.py:334-356
    │  存储 temperature → log_std_offset, entropy_coef → _entropy_coef
    ▼
actor.to_blueprint(stochastic=True)
    │  tanh_gaussian_mlp.py:252-288 或 tanh_squashed_base.py:399-432
    │  调用 export_actor_policy_artifacts / export_generic_policy_artifacts
    │  生成 model.pt + policy.py + policy_blueprint.yaml
    │  PolicyBlueprint(cls="file:.../policy.py:ExportedPolicy", config={"stochastic": True})
    ▼
ParallelRollouter.collect(jobs)
    │  parallel_rollouter.py:187-261
    │  序列化: policy_bp.to_dict() → picklable dict
    │  分组: 按 (policy_a, policy_b, env) blueprint hash 分组
    │  分发: ProcessPoolExecutor.map(_run_chunk, chunks)
    ▼
Worker 进程 (_run_job_batch, parallel_rollouter.py:74-141)
    │  PolicyBlueprint.from_dict(dict).build()
    │  → _resolve_policy_class (envs/framework/policy.py:214-249)
    │  → importlib 动态加载 policy.py
    │  → ExportedPolicy(**config) → 内部加载 model.pt, 构造原始 Policy 类
    │  EnvBlueprint.from_dict(dict).build() → EnvRuntime
    │  EpisodeRunner(policy_a, policy_b, runtime)
    ▼
EpisodeRunner.run_episode(seed, want_extras=True)
    │  envs/framework/episode_runner.py:187-258
    │  policy.reset(seed)  ← 每集重置 (line 368-371)
    │  循环:
    │    obs = runtime.get_observation()
    │    action_a, extra_a = policy_a.act(obs_a, want_extra=True)
    │      → act_numpy (tanh_squashed_base.py:379-393)
    │      → sample_action (tanh_gaussian_mlp.py:126-133)
    │      → Normal(mean, std).rsample() → tanh → log_prob
    │      → extra = {"log_prob": float(log_prob)}
    │    runtime.step(action_a, action_b, action_a_extra=extra_a, ...)
    │      → env_runtime.py:332-370
    │      → _RuntimeCore.step: phy_steps_per_action × physical_step
    │      → EpisodeRecorder.on_post_action_step: 记录 frame
    │  返回 Episode
    ▼
Episode (baseline/common/rollout/episode.py:216-281)
    │  observations: {agent_id: (T, 96)}  ← _stack_agent_field
    │  actions: {agent_id: (T, 21)}      ← _stack_agent_field
    │  action_extras: {agent_id: {key: (T, ...)}}  ← _stack_action_extras
    │  observer_outputs: {...}           ← _stack_observer_outputs
    │  final_observation: {agent_id: (96,)}
    ▼
experiment.build_trajectories(episodes)
    │  exp_standup_step_v3.py:417-425
    │  从 Episode 提取 obs/actions/reward/termination → Trajectory
    │  Trajectory(obs, actions, last_obs, channels, importance, mode)
    │  (trajectory.py:73-102) — 注意：log_prob 字段已在 A3 清理中删除
    ▼
PPOBuffer(trajectories, actor, ...)
    │  trainer.py:112-263
    │  拍平所有 trajectory → all_obs, all_actions
    │  frame_modes 处理 (trainer.py:160-170)
    │  **关键调用**:
    │  ev = actor.evaluate_actions(all_obs_t, all_acts_t, want_stats=True, **kwargs)
    │    (trainer.py:182-183)
    │  → tanh_gaussian_mlp.py:139-191
    │  → atanh(actions) → Normal(mean, std).log_prob - tanh_jacobian
    │  → 返回 ActorEval(log_prob, regularizer, stats)
    │  存储 old_log_prob = ev.log_prob
    ▼
ppo_update(actor, critics, buffer, ...)
    │  trainer.py: 后续
    │  minibatch 循环: 重新 evaluate_actions → ratio → clip_loss → KL early stop
    │  critic 更新: 继续到所有 epoch 结束 (B1 fix)
    │  返回 UpdateStats(diagnostics=[...])  (B8 fix)
    ▼
experiment.on_update(stats)
    │  累积状态供 exploration() 读取
```

### 2.2 关键约束

**约束 1：log_prob 必须是 (obs, actions) 的纯函数**

PPOBuffer 在 trainer.py:182-183 把所有 trajectory 拍平成一个 batch，在 θ_old 下重算 log_prob。这要求 `log π(a|o)` 不依赖采样时的随机数流。

朴素 OU（策略内部维护隐状态）会破坏这个约束：`evaluate_actions` 重算时拿不到 `x_t`，log_prob 算错，ratio 失真，PPO 静默失效。

**约束 2：Worker 进程独立构建策略**

`ParallelRollouter` 通过 `ProcessPoolExecutor` 分发任务。Worker 进程通过 `PolicyBlueprint.from_dict().build()` 重新构建策略。策略内部的任何可变状态（如 OU 过程状态）不会跨进程传递——它必须在 worker 内部从初始状态开始演化。

**约束 3：Episode 内策略复用**

`_run_job_batch` (parallel_rollouter.py:74-141) 在同一 batch 内复用同一个 Policy 实例跑多个 episode。`EpisodeRunner.run_episode` 在每集开始时调用 `policy.reset(seed)` (line 368-371)。OU 状态应在 `reset` 时归零。

**约束 4：传送（set_core_state）无回调**

`Humanoid21Simulator.set_core_state` (simulator.py:1111-1178) 是纯物理状态覆写，不发出任何事件。`RandomMovePlugin` (random_move.py:113-190) 每步可能传送对手。如果 OU 状态需要在传送时重置，需要新增回调机制或让策略自己检测状态跳变。

---

## 3. 精确改动点清单

### 3.1 策略层

#### TanhGaussianMLPPolicy（baseline，不继承 base）

**文件**：`baseline/common/policies/tanh_gaussian_mlp.py`

| 位置 | 行 | 当前 | 改动 |
|---|---|---|---|
| `__init__` | 56-69 | 接收 log_std_min/max, entropy_coef | 增加 `ou_theta=0, ou_sigma=0, ou_kappa=0` 参数 |
| (新增) `reset` | — | 继承 Policy.reset 空实现 | 初始化 `self._ou_state = zeros(action_dim)` |
| `sample_action` | 126-133 | `Normal(mean, std).rsample()` | `mean_shifted = mean + κ·x_t`, 采样后 `x_{t+1} = θ·(-x_t) + σ_ou·ξ` |
| `evaluate_actions` | 139-191 | `Normal(mean, std).log_prob(atanh(a))` | 接收 `noise_state` kwarg, `mean_shifted = mean + κ·noise_state` |
| `set_exploration` | 197-225 | 存储 temperature, entropy_coef | 增加 `noise_correlation` 字段处理 |
| `act` / `act_numpy` | 230-304 | 返回 (action, extra={"log_prob":...}) | extra 中增加 `"noise_state": x_t` |
| `to_blueprint` | 252-288 | 导出 model.pt + policy.py | extra_payload 增加 OU 参数 |

**决策**：baseline 是否支持 OU？

- TODO 建议 `κ=0` 时行为不变，baseline 默认关闭。
- 但用户明确要求"不改 baseline"。
- **方案**：在 `TanhSquashedPolicyBase` 实现 OU，baseline `TanhGaussianMLPPolicy` 不动。新策略族继承 base 自动获得 OU 能力。如需在 baseline 上验证，创建一个 `TanhGaussianOUMLPPolicy` 继承 base + Gaussian hooks + OU mixin。

#### TanhSquashedPolicyBase（四个新族的基类）

**文件**：`baseline/common/policies/tanh_squashed_base.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `__init__` | 73-90 | 增加 OU 参数 (theta, sigma_ou, kappa) |
| (新增) `reset` | — | 初始化 OU 状态 |
| `_raw_sample` | 96-105 | 子类 hook 签名不变；OU 移位在 `sample_action` 层做 |
| `sample_action` | 180-204 | 在 `_raw_sample` 返回后，raw_action 不变；OU 移位加在 mean 上：`mean_shifted = mean + κ·x_t`，然后 `Normal(mean_shifted, std).rsample()`。但 `_raw_sample` 内部已经调用了 `forward(obs)` 得到 mean 并采样——需要重构。 |
| `evaluate_actions` | 214-301 | 接收 `noise_state` kwarg, 在 log_prob 计算时 `mean_shifted = mean + κ·noise_state` |
| `set_exploration` | 334-356 | 处理 `noise_correlation` |
| `act` / `act_numpy` | 362-393 | extra 中返回 `noise_state` |
| `to_blueprint` | 399-432 | extra_payload 增加 OU 参数 |
| `export_config` | 145-152 | 增加 OU 参数 |

**关键设计问题**：`_raw_sample` 的签名。

当前 `_raw_sample(obs) → (raw_action, extras)` 内部调用 `forward(obs)` 得到分布参数并采样。OU 移位需要修改 mean，但 mean 在子类内部计算。

两个选择：
- **A**：`_raw_sample` 签名增加 `mean_shift` 参数，子类在采样前加上。
- **B**：拆分 `forward(obs) → params` 和 `sample_from(params, mean_shift)`，`sample_action` 在中间插入移位。

选择 B 更干净，但改动面大。选择 A 改动小但侵入子类 hook。

#### ExportedPolicy（生成的 policy.py）

**文件**：`baseline/common/policies/export_generic.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `build_generic_export_payload` | 41-68 | extra_payload 增加 OU 参数 |
| `build_generic_export_policy_code` | 71-152 | 生成的 `ExportedPolicy.__init__` 传入 OU 参数；`act` 返回 `noise_state` in extra |

### 3.2 数据管道层

#### Episode

**文件**：`baseline/common/rollout/episode.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `Episode` dataclass | 216-281 | 无需新增字段——`action_extras` 已支持任意 key |
| `_stack_action_extras` | 156-210 | 无需改动——自动 stack `noise_state` key |
| `from_buffer_frames` | 287-347 | 无需改动 |

**结论**：Episode 层零改动。`noise_state` 作为 `action_extras[agent_id]["noise_state"]` 自动流转。

#### EpisodeRecorder

**文件**：`baseline/common/rollout/episode_recorder.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `on_post_action_step` | 114-150 | 无需改动——已记录 `action_extras` |

**结论**：Recorder 层零改动。

#### EpisodeRunner

**文件**：`envs/framework/episode_runner.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `run_episode` | 187-258 | 无需改动——`policy.act(obs, want_extra=True)` 返回的 extra 包含 noise_state |
| `policy.reset(seed)` | 368-371 | 无需改动——调用策略的 reset |

**结论**：Runner 层零改动，前提是策略 `reset` 正确初始化 OU 状态。

#### ParallelRollouter

**文件**：`baseline/common/rollout/parallel_rollouter.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `collect` | 187-261 | 无需改动——blueprint 序列化已包含 OU 参数 |
| `_run_job_batch` | 74-141 | 无需改动——策略重建时 OU 参数从 blueprint config 传入 |

**结论**：Rollouter 层零改动。

### 3.3 PPO 训练层

#### Trajectory

**文件**：`baseline/framework/ppo/trajectory.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `Trajectory` dataclass | 73-102 | 增加 `noise_state: Optional[np.ndarray] = None` |

#### Experiment build_trajectories

**文件**：各 `exp_*.py`

| 改动 |
|---|
| 从 `episode.action_extras[agent_id]["noise_state"]` 提取，传入 `Trajectory(noise_state=...)` |

每个 experiment 的 `_build_agent_trajectory` 需要修改。当前有 ~20 个 experiment 文件。

**简化方案**：在 base class 提供一个 helper `_extract_noise_state(episode, agent_id)`，各 experiment 调用。

#### PPOBuffer

**文件**：`baseline/framework/ppo/trainer.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `PPOBuffer.__init__` | 112-263 | 检查 trajectory 是否有 noise_state，如有则拼接并传入 evaluate_actions |
| batched evaluate_actions | 182-183 | `kwargs["noise_state"] = all_noise_t` |
| ppo_update minibatch | ~852-857 | minibatch 的 noise_state 切片传入 |

**模式**：与 `frame_modes` 完全相同的模式——检测是否存在，如存在则拼接并传入 kwargs。

#### ExplorationSpec

**文件**：`baseline/framework/ppo/experiment.py`

| 位置 | 行 | 改动 |
|---|---|---|
| `ExplorationSpec` dataclass | 229-261 | 增加 `noise_correlation: Optional[float] = None` |

#### loop.py

**文件**：`baseline/framework/ppo/loop.py`

| 位置 | 行 | 改动 |
|---|---|---|
| exploration 调用 | 464-470 | 无需改动——`set_exploration` 接收新字段 |

### 3.4 改动量总结

| 层 | 文件数 | 改动量 | 说明 |
|---|---|---|---|
| 策略层 | 3-4 | 中 | base class + export_generic + 可能的新 OU policy |
| 数据管道 | 0 | 零 | Episode/Recorder/Runner/Rollouter 全部零改动 |
| PPO 训练 | 3 | 小 | Trajectory + trainer + experiment.py |
| Experiments | ~20 | 小 | 每个 build_trajectories 加一行 noise_state 提取 |
| 测试 | 2-3 | 中 | 新增 OU 过程测试 + log_prob 一致性测试 |

**关键优势**：数据管道层（Episode/Recorder/Runner/Rollouter）完全不需要改动，因为 `action_extras` 已经是通用 dict 通道。

---

## 4. 核心设计决策

### 4.1 已决策（来自 TODO）

| 决策 | 理由 |
|---|---|
| 方案 B：OU 扰动均值 `a ~ TanhGaussian(μ(o)+κ·x_t, σ)` | 保持 TanhGaussian 解析性，log_prob 可重算 |
| noise_state 物化进 Trajectory | 满足 PPO on-policy 约束 |
| baseline 默认关闭 OU (κ=0) | 保持 baseline 行为兼容 |
| Stage 0 先做离线验证 | 避免在未验证根因假设下投入完整实现 |

### 4.2 待决策

**Q1：OU 实现在 base class 还是新 mixin？**

| 选项 | 优点 | 缺点 |
|---|---|---|
| 在 TanhSquashedPolicyBase 实现 | 四个新族自动获得 | base class 变复杂 |
| 新建 OUMixin | 模块化 | 多继承复杂度 |
| 只在新 TanhGaussianOUMLPPolicy 实现 | 最简单 | 其他族需要时再改 |

**Q2：`_raw_sample` 签名如何修改？**

当前 `_raw_sample(obs) → (raw_action, extras)` 内部完成 forward + sample。OU 需要在 mean 上加移位。

| 选项 | 改动面 | 侵入性 |
|---|---|---|
| A: `_raw_sample(obs, mean_shift=0)` | 每个子类改签名 | 中 |
| B: 拆分 forward + sample_from | 重构 base | 大 |
| C: 在 sample_action 层做移位，_raw_sample 返回 mean | 改 sample_action + 子类返回值 | 中 |

**Q3：noise_state 的形状和存储**

- per-step per-agent: `(T, action_dim)` float32
- 存在 `Trajectory.noise_state`
- 传入 `evaluate_actions(noise_state: torch.Tensor)` 时形状 `(B, action_dim)`

**Q4：传送时 OU 重置**

当前无传送回调。选项：
- 不处理（OU 状态跨传送延续，可能产生不一致）
- 策略检测 obs 跳变（不可靠）
- 新增 `on_teleport` 回调（改动 env 框架）
- **建议**：Stage 0-3 不处理，Stage 5 生产化时再加

**Q5：OU 参数调度**

`ExplorationSpec.noise_correlation` 是 OU 的 θ。是否需要 `noise_kappa` 和 `noise_sigma_ou` 也进 ExplorationSpec？

- 简单方案：只暴露 θ，σ_ou 和 κ 在 config 里固定
- 完整方案：三个参数都进 ExplorationSpec，支持退火调度
- **建议**：先只暴露 θ，验证假设后再扩展

---

## 5. Stage 0 离线验证方案细化

### 5.1 目标

验证"频谱结构是根因"假设。不写生产代码，故意破坏 log_prob 正确性。

### 5.2 方法

在 `to_blueprint` 导出时，把 σ 替换成一个离线生成的 OU 序列的等效值。具体：

```python
# 临时实验脚本
# 1. 正常导出 blueprint
# 2. 在 ExportedPolicy.act 中替换采样逻辑：
#    不用 Normal(mean, std).rsample()
#    而用 mean + ou_process.next()  （ou_process 预生成或在线递推）
# 3. log_prob 仍然用原始 Normal(mean, std) 计算（故意不一致）
```

### 5.3 验证标准

- 跑 50-100 updates
- 观察：策略是否开始迈步（episode 行为变化，foot reward 上升）
- 如果策略开始迈步 → 频谱假设成立 → 投入 Stage 1-3
- 如果策略仍然原地平衡 → 假设不成立 → 重新诊断

### 5.4 风险

- log_prob 不一致会导致 PPO ratio 偏差，但 epoch 0 的 ratio 仍接近 1（θ_old = θ）
- 第一个 epoch 的梯度方向大致正确
- 跑太多 updates 后 ratio 偏差累积，结果不可信
- **限制**：只看前 50 updates 的行为变化趋势

---

## 6. 风险地图

| 风险 | 严重度 | 检测方式 | 缓解 |
|---|---|---|---|
| log_prob 重算与采样不一致（静默失效） | **高** | Stage 2.5 单测：固定 (o,a,x_t) 验证 log_prob 数值一致 | 强制测试 |
| OU 参数需要调参 | 中 | Stage 0 给初始值，Stage 4 网格搜索 | 从文献推荐值开始 |
| 传送时 OU 状态不重置 | 中 | 暂不处理，Stage 5 再加 | — |
| 条件熵 vs 边际熵混淆 | 中 | regularizer 用条件熵，单测验证 | 明确用 H(a\|o,x_t) |
| _raw_sample 签名修改影响四个子类 | 中 | 每个子类单测 | 选择最小侵入方案 |
| worker 进程 OU 状态序列化 | 低 | OU 参数在 blueprint config，x_0=0 固定 | 不传 x_t，每集 reset |

---

## 7. 文件索引

### 核心文件

| 文件 | 角色 | 关键行 |
|---|---|---|
| `baseline/common/policies/tanh_gaussian_mlp.py` | baseline 策略（不改） | sample:126, eval:139, set_exploration:197, to_blueprint:252 |
| `baseline/common/policies/tanh_squashed_base.py` | 新族基类 | sample:180, eval:214, set_exploration:334, act:362, to_blueprint:399 |
| `baseline/common/policies/export_generic.py` | 策略导出 | payload:41, code_gen:71 |
| `baseline/common/policies/checkpoint.py` | checkpoint 导出 | export_actor_policy_artifacts:156 |
| `baseline/common/rollout/episode.py` | Episode 数据结构 | dataclass:216, stack_extras:156 |
| `baseline/common/rollout/episode_recorder.py` | 帧记录 | on_post_action_step:114 |
| `baseline/common/rollout/parallel_rollouter.py` | 并行 rollout | collect:187, _run_job_batch:74 |
| `envs/framework/episode_runner.py` | 单集运行 | run_episode:187, reset:368 |
| `envs/framework/env_runtime.py` | 环境运行时 | step:332, _RuntimeCore.step:148 |
| `envs/framework/policy.py` | PolicyBlueprint | dataclass:252, build:274, resolve:214 |
| `envs/humanoid21/simulator.py` | 96 维观测 + set_core_state | obs:503, set_core_state:1111 |
| `envs/humanoid21/disturbance_plugins.py` | 传送插件 | RandomFallenState:815, RandomMove:1047 |
| `baseline/framework/ppo/experiment.py` | ExplorationSpec + TrainablePolicy | ExplorationSpec:229, evaluate_actions protocol:319 |
| `baseline/framework/ppo/trajectory.py` | Trajectory | dataclass:73 |
| `baseline/framework/ppo/trainer.py` | PPOBuffer + ppo_update | buffer:112, evaluate:182, update:后续 |
| `baseline/framework/ppo/loop.py` | 训练主循环 | exploration:464, set_exploration:467 |
| `baseline/experiments_ppo/base.py` | experiment 基类 | exploration():175 |

### 测试文件

| 文件 | 内容 |
|---|---|
| `baseline/framework/ppo/tests/test_trainer.py` | PPO trainer 测试（57 tests） |
| `baseline/common/policies/test_policy_families.py` | 策略族测试 |
| `envs/humanoid21/tests/test_data_interfaces.py` | 数据接口测试 |
