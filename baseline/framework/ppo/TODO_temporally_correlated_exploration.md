# TODO: 时间相关探索噪声（Temporally Correlated Exploration）

**状态**：待办（设计草案，未实现）
**优先级**：高 —— 直击"学不出节律行为"的根因
**前置阅读**：`GUIDE.md`、`experiment.py`（`ExplorationSpec` / `TrainablePolicy`）、`trajectory.py`、`loop.py`
**关联文档**：`baseline/common/policies/DESIGN_OVERVIEW.md`

---

## 1. 动机

### 1.1 现象

`exp_basic_balance_step` 的 docstring 自己记录了策略卡在"原地平衡"局部最优，学不出迈步这类有节律的、时间上延展的行为。多次 reward 改造和观测变换（sqrt velocity, commit `d595081`）都没能改变这个困境。

### 1.2 根因诊断

当前所有策略族（`TanhGaussianMLPPolicy` 及四个新族）的探索噪声都是**时间不相关**的：每个 action step 独立从 `π(a|o)` 采样，20Hz 控制频率下相当于注入**白噪声**。

白噪声的功率谱密度是平的，**低频段几乎没有能量**。而步态是 1–2Hz 的低频周期信号。用高频白噪声去撞低频周期解，命中概率极低——这就是策略学不出迈步的物理本质。

这不是 reward shaping 问题，不是观测维度问题，不是策略族问题，是**探索噪声的频谱结构**问题。所有策略族都受这个限制。

### 1.3 文献支撑

- *Pink Noise Is All You Need* (ICLR 2023)：在连续控制 benchmark 上，pink/OU 噪声普遍显著优于白噪声，尤其针对步态类任务。
- 经典 RL 文献中 OU 噪声用于 off-policy DDPG 等已有共识；on-policy PPO 中相对少用，但根因相同。

---

## 2. 为什么不能简单地在采样端加 OU

### 2.1 PPO 的 on-policy 约束

PPO 的核心是重要性比 `r = exp(log π_θ(a|o) − log π_θ_old(a|o))`。`PPOBuffer` 在 `loop.py` 里做这件事：

```python
# trainer.py / experiment.py
all_obs   = concat([t.obs      for t in trajectories])
all_acts  = concat([t.actions  for t in trajectories])
old_eval  = actor.evaluate_actions(all_obs, all_acts, want_stats=True)  # θ_old
```

它把所有 trajectory 的所有帧拍平成一个 batch，**在 θ_old 下重算 log_prob**。这要求：

> **`log π(a|o)` 必须是 `(o, a)` 的纯函数，与采样时的随机数流无关。**

### 2.2 朴素 OU 注入会破坏这个约束

如果在策略里维护一个 OU 状态 `x_t`，采样时 `a = tanh(μ(o) + σ·x_t)`，那么 `log π(a|o)` 依赖于 `x_t`，而 `x_t` 是采样时随机数驱动的、不可从 `(o, a)` 复现的。`evaluate_actions` 重算时拿不到 `x_t`，log_prob 算错，PPO ratio 失真。

**症状**：epoch 0 的 ratio 恒为 1，`clip_frac=0`，看起来完全健康，但梯度方向是错的。这正是上次 review 中 B4 描述的静默失效模式——非常危险。

### 2.3 结论

OU 噪声过程的状态**必须物化进 Trajectory**，让 `evaluate_actions` 能在重算时拿到完全一致的 `x_t`。否则不能做。

---

## 3. 设计

### 3.1 核心思路：噪声状态显式化

把时间相关噪声建模成**策略分布的一部分**，而不是采样端的副作用。具体地：

- 策略维护一个 per-step 的噪声状态 `x_t ∈ R^D`（D = action_dim）。
- 采样时：`a = tanh(μ(o) + σ·x_t)`，然后 `x_{t+1} = φ(x_t, ξ_t)`，`ξ_t ~ N(0, I)`。
- **关键**：`x_t` 和 `ξ_t` 都要存进 Trajectory，让 `evaluate_actions(o, a, x_t)` 能重算 `log π(a | o, x_t)`。

这样 log_prob 就是 `(o, a, x_t)` 的纯函数，on-policy 约束满足。

### 3.2 概率分布的精确定义

引入噪声状态后，策略分布不再是简单的 `TanhGaussian(μ(o), σ)`。需要明确：

**方案 A（推荐）：条件高斯，噪声状态作为"似然辅助变量"**

定义 `a = tanh(μ(o) + σ · x_t)`，其中 `x_t` 是 OU 过程的当前状态，**在给定 `x_t` 的条件下**：

```
π(a | o, x_t) = TanhGaussian(μ(o) + σ·x_t, 0)   # 即退化为 tanh 上的 delta-like
```

但这退化成确定性映射，没有熵。不行。

**方案 B（推荐）：噪声状态平移均值**

```
a ~ TanhGaussian(μ(o) + κ·x_t, σ)
```

`x_t` 是 OU 过程状态，`κ` 是噪声耦合强度。给定 `(o, x_t)`，`a` 的分布仍是 TanhGaussian，log_prob 可解析计算。OU 过程的 `x_t` 本身不参与 log_prob（它是采样辅助，不是分布参数），但**必须存进 trajectory 以便重算**。

这是 ACT / Diffusion Policy 之外、on-policy 设定下最干净的做法。本质上是"用 OU 过程给均值加一个时间相关的扰动，但扰动量被记录下来"。

**方案 C（更激进）：把 OU 状态当分布参数**

把 `x_t` 视作策略的一部分，`π(a|o, x_t) = TanhGaussian(μ(o, x_t), σ(o))`，`x_t` 由策略网络自己预测。这接近 recurrent policy，复杂度高，不在本 TODO 范围。

**采用方案 B。**

### 3.3 数据流改动

```
rollout 端：
  policy.act(o) → (a, x_t)        # 同时返回当前噪声状态
  Episode 增加 noise_state 字段    # (T+1, D) 或 (T, D)

trajectory 端：
  Trajectory 增加 noise_state: np.ndarray  # (T, D) float32
  build_trajectories 时从 Episode 拷贝

buffer / trainer 端：
  evaluate_actions(obs, actions, noise_state=...) → ActorEval
  TrainablePolicy.evaluate_actions 签名增加可选 noise_state 参数
  log_prob = TanhGaussian(μ(o) + κ·noise_state, σ).log_prob(a)
```

### 3.4 接口改动清单

| 组件 | 改动 | 兼容性 |
|---|---|---|
| `Trajectory` | 增加 `noise_state: Optional[np.ndarray]` 字段 | 旧 trajectory 无此字段 → 视为 0，行为不变 |
| `TrainablePolicy.evaluate_actions` | 增加可选 `noise_state` 参数 | 旧策略不接收此参数 → 行为不变 |
| `TrainablePolicy.act` / blueprint 采样 | 策略内部维护 OU 状态，导出时序列化初始状态 | 旧策略无 OU → 行为不变 |
| `PolicyBlueprint` | 增加 `noise_process` 配置段（OU 参数 θ, σ_ou, κ） | 旧 blueprint 无此段 → 行为不变 |
| `Episode` | 增加 `noise_states` 字段（per-agent） | 旧 episode 无此字段 → 视为 0 |
| `ExplorationSpec` | 增加 `noise_correlation` 字段（OU θ 或 pink β） | 旧 spec 无此字段 → 行为不变 |
| `ParallelRollouter` | 透传 noise_state，无需特殊处理 | 透明 |

**所有改动都是可选字段 + 默认值回退，旧 checkpoint / 旧 blueprint / 旧 experiment 完全兼容。**

### 3.5 Checkpoint 兼容性

- 旧 checkpoint 加载：`noise_state` 字段不存在 → 策略 OU 状态初始化为 0，等价于无时间相关噪声。**checkpoint 不失效。**
- 新 checkpoint 保存：OU 过程参数（θ, σ_ou, κ）是策略超参，已存在于 config；per-step `x_t` 不需要持久化（每个 episode reset 时重新初始化）。
- 这是本方案相对于"全量帧堆叠"和"action chunking"的关键优势：**不废 checkpoint**。

### 3.6 Entropy / 正则化

引入 OU 扰动均值后，策略的**条件熵** `H[π(·|o, x_t)]` 仍由 σ 决定，与 baseline 一致。但**边际熵** `H[π(·|o)] = E_{x_t}[H[π(·|o, x_t)]] + H[x_t]` 会因为 `x_t` 的扩散而增大。

PPO 的 entropy 正则项应该用**条件熵**（在给定 `x_t` 下计算），否则会错误地奖励"OU 过程的随机性"而不是"策略本身的探索性"。`ActorEval.regularizer` 里返回的 entropy 必须是基于 `(o, x_t, a)` 的条件熵，不能对 `x_t` 做边际化。

### 3.7 与 `Trajectory.mode` 的关系

`Trajectory.mode` 已经是 per-trajectory 标量，用于路由到不同子网络。本方案需要的是 **per-step 向量**（`x_t` 每步不同），所以不能复用 `mode` 字段。但设计思路一致：把采样时的"事实"物化进 trajectory，让重算成为纯函数。

可以考虑把两者统一成一个 `per_step_extras: Dict[str, np.ndarray]` 字段，但当前规模下分开更清晰，留待后续重构。

---

## 4. 实施步骤（建议顺序）

### Stage 0：可行性验证（不写生产代码）

- [ ] **0.1** 写一个最小脚本：在 `exp_basic_balance_step` 上把 rollout 采样噪声从白噪声换成离线生成的 OU 序列（直接替换 `to_blueprint` 导出的 σ），观察是否能学到迈步。
  - 这一步**故意破坏 log_prob 正确性**，只为验证"频谱结构是根因"这个假设。
  - 如果验证通过（策略开始迈步），再投入做正确的实现。
  - 如果验证不通过，说明根因不在噪声频谱，本 TODO 暂停，重新诊断。

### Stage 1：接口扩展（向后兼容）

- [ ] **1.1** `Trajectory` 增加 `noise_state: Optional[np.ndarray]` 字段，默认 None。
- [ ] **1.2** `TrainablePolicy.evaluate_actions` 签名增加 `noise_state: Optional[torch.Tensor] = None`，默认 None 时行为与现在完全一致。
- [ ] **1.3** `Episode` 增加 `noise_states` 字段（per-agent dict），`build_trajectories` 透传。
- [ ] **1.4** `ExplorationSpec` 增加 `noise_correlation: Optional[float] = None`（OU 的 θ，或 pink noise 的 1/f 指数）。
- [ ] **1.5** 单测：旧 trajectory / 旧 episode / 旧 blueprint 走新代码路径，行为不变。

### Stage 2：策略实现

- [ ] **2.1** 在 `TanhSquashedPolicyBase` 或新 mixin 中实现 OU 过程：`x_{t+1} = θ·(0 − x_t) + σ_ou·ξ_t`（zero-mean OU）。
- [ ] **2.2** 采样路径：`a = tanh(μ(o) + σ·ε + κ·x_t)`，`ε ~ N(0,I)`，记录 `x_t` 到 episode。
- [ ] **2.3** `evaluate_actions` 路径：`log_prob = TanhGaussian(μ(o) + κ·x_t, σ).log_prob(a)`，使用传入的 `noise_state`。
- [ ] **2.4** `to_blueprint` 导出 OU 参数（θ, σ_ou, κ）和初始 `x_0`（通常为 0）。
- [ ] **2.5** 单测：
  - 给定固定 `(o, a, x_t)`，`evaluate_actions` 的 log_prob 与采样时记录的 log_prob 数值一致（within float tolerance）。
  - OU 过程的 `x_t` 序列与离线 reference 实现一致。
  - 旧策略（无 OU）走新接口，log_prob 与旧实现一致。

### Stage 3：rollout 集成

- [ ] **3.1** `ParallelRollouter` 透传 `noise_state`，确认 worker 进程能正确序列化/反序列化 OU 初始状态。
- [ ] **3.2** Episode reset 时 OU 状态归零（或按 config 初始化）。
- [ ] **3.3** `disturbance_plugins.py` 的 `set_core_state`（传送）发生时，OU 状态是否需要重置？—— **建议重置**，避免传送后的动作基于传送前的噪声状态。需要 observer/plugin 联动接口。
- [ ] **3.4** 端到端 smoke training：`--smoke` 跑通，log_prob 重算与采样时一致（ratio 在 epoch 0 应接近 1，但不是恒等于 1）。

### Stage 4：实验验证

- [ ] **4.1** 在 `exp_basic_balance_step` 上开启 OU 噪声，对照 baseline（白噪声），跑 3 个 seed。
- [ ] **4.2** 指标：是否学到迈步、episode 长度、reward 曲线、entropy 曲线、clip_frac。
- [ ] **4.3** 如果 OU 显著优于白噪声，推广到其他 exp_*。
- [ ] **4.4** 如果 OU 不显著，尝试 pink noise（1/f^β，β=1）—— 频谱更集中在低频。

### Stage 5：生产化

- [ ] **5.1** `ExplorationSpec.noise_correlation` 的调度：让 experiment 在 `exploration()` 里根据 `on_update` 状态动态调整 θ（早期强相关、后期趋白）。
- [ ] **5.2** 决策日志：记录 OU 参数选择、调度策略、对照实验结果。
- [ ] **5.3** 文档：更新 `GUIDE.md` 的探索章节，说明时间相关噪声的语义和配置。

---

## 5. 风险与未决问题

### 5.1 已识别风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| log_prob 重算与采样不一致（静默失效） | **高** | Stage 2.5 单测强制验证；rollout 时记录 log_prob 并与重算对比 |
| OU 参数需要调参（θ, σ_ou, κ） | 中 | Stage 0 离线验证给出初始值；Stage 4 网格搜索 |
| 传送（`set_core_state`）时 OU 状态联动 | 中 | Stage 3.3 显式处理 |
| 条件熵 vs 边际熵混淆 | 中 | Stage 3.6 明确用条件熵；单测验证 regularizer 数值 |
| worker 进程序列化 OU 状态 | 低 | Stage 3.1 验证 |

### 5.2 未决问题

- **Q1**：OU 噪声应该在**所有策略族**上支持，还是只在新策略族上？建议在 base class 实现，所有族继承，但 baseline `TanhGaussianMLPPolicy` 默认关闭（`κ=0`），保持 baseline 行为兼容。
- **Q2**：`noise_state` 是 per-agent 独立的 OU 过程，还是共享？—— **独立**，两个机器人各自探索。
- **Q3**：是否需要 pink noise（1/f）而非 OU？OU 是指数衰减相关，pink 是幂律衰减。pink 在频谱上更"粉"，但实现稍复杂（需要 Voss-McCartney 算法或 FFT 滤波）。建议先 OU，再 pink。
- **Q4**：`evaluate_actions` 的 `noise_state` 参数是否应该成为 `TrainablePolicy` Protocol 的正式成员？—— 是，但作为可选参数，旧实现忽略它。

---

## 6. 为什么优先做这个（而不是帧堆叠 / action chunking）

| 方案 | 直击根因 | checkpoint 兼容 | 改动范围 | 预期收益 |
|---|---|---|---|---|
| **时间相关噪声（本 TODO）** | ✅ 频谱结构 | ✅ 不失效 | 中（接口+策略） | 高 |
| 全量帧堆叠 k=3 | ❌ 不解决频谱 | ❌ 失效 | 中（observer） | 低 |
| action chunking | ❌ 不解决频谱 | ❌ 失效 | 高（重写 MDP） | 负（开环对抗有害）|
| health 入观测 | 部分解决可观测性 | ❌ 失效 | 低 | 中 |

时间相关噪声是唯一**同时满足"直击根因 + 不废 checkpoint + 收益高"**的方案。

---

## 7. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-01 | 立项本 TODO | 诊断出"学不出节律"的根因是探索噪声频谱结构，而非观测/动作/reward 设计 |
| 2026-09-01 | 采用方案 B（OU 扰动均值）而非方案 A/C | A 退化无熵，C 过于复杂；B 保持 TanhGaussian 解析性，log_prob 可重算 |
| 2026-09-01 | Stage 0 先做离线验证 | 避免在未验证根因假设的情况下投入完整实现 |
| 2026-09-01 | baseline 默认关闭 OU（κ=0） | 保持 baseline 行为兼容，符合"不改 baseline"原则 |
