# TODO: 时间相关探索噪声（Temporally Correlated Exploration）

**状态**：已实现（Stage 1–5 完成，A/B 对照实验待跑）
**优先级**：高 —— 直击"学不出节律行为"的根因
**前置阅读**：`GUIDE.md`、`experiment.py`（`ExplorationSpec` / `TrainablePolicy`）、`trajectory.py`、`loop.py`
**关联文档**：`baseline/common/policies/DESIGN_OVERVIEW.md`、`CONTEXT_temporally_correlated_exploration.md`
**实现提交**：见 git log `feat(policies): OU noise shift` 系列

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

> **实现修正（2026-09-02）**：原方案 B 写作 `a ~ TanhGaussian(μ(o) + κ·x_t, σ)`，
> 暗示要改分布的均值参数。实际实现采用了更精确的等价形式：**raw 空间平移**。
>
> 对任意分布 `p`，若 `z ~ p(·|o)` 且 `raw = z + s`（`s` 给定），则 `raw` 的密度是
> `p(raw - s | o)`。所以：
>
> ```
> 采样：z = _raw_sample(obs);   raw = z + s;   a = tanh(raw)
> 打分：raw = atanh(a);          raw_log_prob = _raw_log_prob(obs, raw - s)
> ```
>
> 这对 Gaussian、low-rank、MoG、RealNVP **全部数学精确**（平移一个密度而非近似），
> 并且 `_raw_sample` / `_raw_log_prob` / `_raw_log_prob_per_dim` / `_raw_mode`
> 四个 hook 的签名和实现**全部不变**——平移只发生在 `TanhSquashedPolicyBase`
> 的 `sample_action` 和 `evaluate_actions` 两处。
>
> 原方案 B 的概念仍然正确（"用 OU 过程给均值加一个时间相关的扰动，但扰动量被
> 记录下来"），只是实现方式从"改分布参数"简化为"平移 raw 空间"。

**方案 A**：退化为 delta，无熵，不行。

**方案 B（采用，以 raw 空间平移实现）**：见上方修正。

**方案 C**：recurrent policy，复杂度高，不在本 TODO 范围。

### 3.3 数据流改动

> **实现修正（2026-09-02）**：存 `s_t = noise_scale * x_t`（已施加的平移量），
> 而非 OU 原始状态 `x_t`。契约变成"rollout 告诉你它究竟加了多少偏移"。
> 训练侧 `evaluate_actions` **完全不需要任何 OU 参数**，只做一次减法。
> κ 不一致这类 bug 从结构上不可能发生。字段命名为 `noise_shift`。

```
rollout 端：
  policy.act(o) → (a, extras={"log_prob": ..., "noise_shift": s_t})
  Episode.action_extras[agent_id]["noise_shift"]  # (T, D) float32

trajectory 端：
  Trajectory 增加 noise_shift: Optional[np.ndarray]  # (T, D) float32
  build_trajectories 时从 episode.action_extras 提取并 [:T] 截断

buffer / trainer 端：
  evaluate_actions(obs, actions, noise_shift=...) → ActorEval
  TrainablePolicy.evaluate_actions 签名增加可选 noise_shift 参数
  raw_actions = atanh(actions) - noise_shift
  log_prob = _raw_log_prob(obs, raw_actions) + tanh_jacobian
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

> **实现确认（2026-09-02）**：此问题不存在。`evaluate_actions` 的正则项路径
> 用的是 `_raw_sample` + `_raw_log_prob` 配对的 raw 空间估计，而**微分熵对平移
> 不变**，所以该路径无需任何改动就已经是正确的条件熵。闭式正则项
> （`Normal.entropy()`）同理。**该项工作量为零。**

原分析（保留供参考）：引入 OU 扰动均值后，策略的**条件熵** `H[π(·|o, x_t)]` 仍由 σ
决定，与 baseline 一致。但**边际熵** `H[π(·|o)]` 会因为 `x_t` 的扩散而增大。PPO 的
entropy 正则项应该用**条件熵**（在给定 `x_t` 下计算），否则会错误地奖励"OU 过程的
随机性"而不是"策略本身的探索性"。

### 3.7 与 `Trajectory.mode` 的关系

`Trajectory.mode` 已经是 per-trajectory 标量，用于路由到不同子网络。本方案需要的是 **per-step 向量**（`x_t` 每步不同），所以不能复用 `mode` 字段。但设计思路一致：把采样时的"事实"物化进 trajectory，让重算成为纯函数。

可以考虑把两者统一成一个 `per_step_extras: Dict[str, np.ndarray]` 字段，但当前规模下分开更清晰，留待后续重构。

---

## 4. 实施步骤（建议顺序）

### Stage 0：可行性验证（不写生产代码）

- [x] **0.1** ~~写最小脚本验证~~ → **跳过**（用户决策：直接做正确实现）

### Stage 1：策略层（OU 能力）— ✅ 完成

- [x] **1.1** `TanhSquashedPolicyBase` 增加 OU 参数（`noise_tau_steps`, `noise_scale`）、AR(1) 步进、`reset(seed)`、`_next_noise_shift()`。
- [x] **1.2** `sample_action` / `evaluate_actions` 支持 `noise_shift` 参数（raw 空间平移）。
- [x] **1.3** `act` / `act_numpy` 线程化 noise_shift，`want_extra=True` 时返回 `{"log_prob": ..., "noise_shift": s_t}`。
- [x] **1.4** `set_exploration` 接收 `noise_tau_steps` / `noise_scale`，`to_blueprint` 导出。
- [x] **1.5** `export_generic.py` 生成的 `ExportedPolicy` 透传 OU 参数，`reset` 转发给内部策略。
- [x] **1.6** 新建 `FixedSigmaGaussianMLPPolicy`（参数名与 baseline 一致，checkpoint 兼容）。
- [x] **1.7** 新建 `init_policy_fixed_sigma_gaussian.yaml`。

### Stage 2：PPO 数据通道 — ✅ 完成

- [x] **2.1** `ExplorationSpec` 增加 `noise_tau_steps` / `noise_scale` 字段。
- [x] **2.2** `TrainablePolicy.evaluate_actions` protocol 签名增加 `noise_shift`。
- [x] **2.3** `Trajectory` 增加 `noise_shift: Optional[np.ndarray]` 字段。
- [x] **2.4** `PPOBuffer` 检测、校验（不一致直接 raise）、拼接、线程化 `noise_shift`。
- [x] **2.5** `ppo_update` minibatch 调用点用 kwargs dict 线程化 `noise_shift`。

### Stage 3：实验接线 — ✅ 完成

- [x] **3.1** `CombatExperimentPPOBase` 增加 `noise_tau_steps` / `noise_scale` 类属性，`exploration()` 带入 spec。
- [x] **3.2** `extract_noise_shift` helper（从 `episode.action_extras` 提取并 `[:T]` 截断）。
- [x] **3.3** `exp_basic_balance_step.py` 的 `_build_agent_trajectory` 传入 `noise_shift`。
- [x] **3.4** 新建 `exp_basic_balance_step_ctrl.py`（FixedSigma, noise_scale=0）和 `exp_basic_balance_step_ou.py`（FixedSigma, noise_scale=0.3, tau=10）。

### Stage 4：测试 — ✅ 完成

- [x] **4.1** 退化等价（noise_scale=0 → 与 baseline bit-identical）。
- [x] **4.2** 采样/打分一致性（最关键：sample with shift → score with shift → match）。
- [x] **4.3** OU 统计量（稳态方差 ≈ 1，lag-1 自相关 ≈ exp(-1/τ)）。
- [x] **4.4** reset 决定性（同 seed 同序列，reset 归零）。
- [x] **4.5** 导出往返（OU 参数存活，act 返回 noise_shift，reset 转发）。
- [x] **4.6** 端到端 PPOBuffer log_prob 一致。
- [x] **4.7** 不一致输入报错（混合有/无 noise_shift → raise）。
- [x] **4.8** 向后兼容（120 existing tests pass）。

### Stage 5：对照实验 — ⏳ 待跑

- [ ] **5.1** `basic_balance_step_ctrl` 与 `basic_balance_step_ou` 各跑 3 个 seed。
- [ ] **5.2** 判读指标：是否出现迈步、episode 长度、entropy 曲线、clip_frac、KL。
- [ ] **5.3** 若 OU 胜出 → 推广到其他实验。
- [ ] **5.4** 若不显著 → 尝试更大 noise_scale 或 pink noise。

---

## 5. 风险与未决问题

### 5.1 已识别风险

| 风险 | 严重度 | 状态 | 缓解 |
|---|---|---|---|
| log_prob 重算与采样不一致（静默失效） | **高** | ✅ 已消除 | `test_ou_exploration.py` 采样/打分一致性 + PPOBuffer 端到端测试；不一致输入直接 raise |
| `ExportedPolicy.reset` 不转发导致 OU 跨 episode 延续 | **高** | ✅ 已修复 | `export_generic.py` 的 `reset` 现在转发给内部策略；测试覆盖 |
| OU 参数需要调参 | 中 | ⏳ 待调 | 单位方差归一化使 `noise_scale` 与 σ 可比，从 0.3 起 |
| 平移放大 tanh 饱和 | 中 | ⏳ 待监控 | 平移零均值不引入偏置；监控 `tanh_sat_frac` |
| 传送（`set_core_state`）时 OU 不重置 | 低 | 📋 后续处理 | `basic_balance_step` 不使用逐步传送，暂无暴露 |
| ~~`_raw_sample` 签名修改影响四个子类~~ | ~~中~~ | ✅ 已消除 | raw 空间平移设计使子类 hook 完全不变 |
| ~~条件熵 vs 边际熵混淆~~ | ~~中~~ | ✅ 已消除 | 微分熵对平移不变，正则项路径无需改动 |

### 5.2 未决问题

- **Q1**：~~OU 噪声应该在所有策略族上支持？~~ → **已决策**：在 `TanhSquashedPolicyBase` 实现，所有新族继承；baseline `TanhGaussianMLPPolicy` 不改，用 `FixedSigmaGaussianMLPPolicy` 作为 OU-enabled 替代。
- **Q2**：~~per-agent 独立还是共享？~~ → **已决策**：独立。`Policy.reset(seed)` 每个 agent 独立调用。
- **Q3**：是否需要 pink noise？ → **待定**：先 OU，若不显著再 pink。
- **Q4**：~~`noise_shift` 是否成为 Protocol 正式成员？~~ → **已决策**：是，作为可选参数，旧实现忽略它。

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
| 2026-09-02 | 跳过 Stage 0，直接做正确实现 | 用户决策：不浪费时间去验证一个故意错误的实现 |
| 2026-09-02 | 方案 B 实现为 raw 空间平移而非均值参数移位 | 对任意分布精确（平移密度 ≠ 近似），四个子类 hook 零改动，风险表删除该项 |
| 2026-09-02 | 存 `noise_shift = noise_scale * x_t` 而非 OU 原始状态 `x_t` | 训练侧不需要任何 OU 参数，κ 不一致 bug 从结构上不可能 |
| 2026-09-02 | 新建 `FixedSigmaGaussianMLPPolicy` 而非改 baseline | baseline 不动（用户要求）；新类参数名与 baseline 一致，checkpoint 兼容 |
| 2026-09-02 | `ExportedPolicy.act` 委托给内部策略的 `act` | 消除两处采样逻辑漂移风险；OU 步进和 extras 由基类统一负责 |
| 2026-09-02 | `ExportedPolicy.reset` 转发给内部策略 | 修复 OU 状态不按 episode 归零的 bug |
| 2026-09-02 | `_compute_stats` 不覆盖子类已提供的 `entropy` | 让闭式熵（Normal.entropy）优先于 score-function 估计，与 baseline 一致 |
| 2026-09-02 | OU 参数化为 `(noise_tau_steps, noise_scale)` 而非 `(θ, σ_ou, κ)` | τ 以 policy step 为单位有物理直觉（20Hz 下 τ=10 ≈ 0.5s）；`noise_scale` 与 σ 可比 |
| 2026-09-02 | `PPOBuffer` 对不一致 `noise_shift` 输入直接 raise | 静默填零 = 错误的 log_prob = 最危险的静默失效模式 |
| 2026-09-02 | A/B 对照用同一策略族（FixedSigma），只差 `noise_scale` | 排除"新策略族本身带来差异"这个混淆因素 |
