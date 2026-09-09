# TODO: 参考策略差分探索（Reference-Policy Delta Exploration）

**状态**：**未实现 / 已主动搁置**（2026-09-09）
**优先级**：中 —— 机制有理论支撑，但**不解决当前瓶颈**（见 §7）
**搁置原因**：当前瓶颈是"发现新行为"，本方案是"加速收敛"，失配。等 stepping 出现后再评估。
**前置阅读**：`DESIGN_unified_exploration_control.md`、`GUIDE.md`、`experiment.py`（`TrainablePolicy` / `ActorEval`）、`stochastic_policy.py`、`trajectory.py`
**关联文档**：`policies/todo/DESIGN_low_rank_gaussian.md`（强协同，见 §9.5）、`policies/todo/TODO_temporally_correlated_exploration.md`
**关联实验**：`experiments_ppo/exp_standup_step_v3.py`（phase-dependent `explore_factor` 已验证同类数据流可行）

---

## 1. 核心想法

PPO 的策略是一轮一轮迭代的。**相邻两代策略在同一状态下的动作差值，本身就是一个免费的、advantage 已加权过的信号。** 现在这个信号被完全丢弃了。

Rollout 时：

1. 对每个 obs，**确定性**调用参考策略（上一代 / EMA）→ `a_ref`（21 维）
2. 当前训练中的策略 → `μ_new`（21 维）
3. 逐维取差 → `Δ = μ_new - a_ref`（21 维）

`Δ` 近似是策略在动作空间的**位移速度**。而策略梯度本身 ≈ advantage 加权的 `∇_a Q`，所以

```
Δ(s) ≈ η · ∇_a Q(s, μ(s))
```

这意味着 `Δ` 不是随机量，而是**已经被一整轮 PPO 梯度平均过的改进方向**。信噪比远好于从单个 batch 内部现算"噪声-advantage 相关性"。

> **注意**：必须**确定性**调用参考策略（`act()` 而非 `sample()`）。否则 `Δ` 里混入参考策略自身的采样噪声，信号失效。

---

## 2. 两种用法

### 2.1 用法 1：Δ 当标准差（推荐优先）

```
σ_i = c · |Δ_i|          # 逐维
```

`c = 1` 时，参考策略的动作正好落在 1σ 处。

语义：**在策略最近变动大的维度上多探索，在已收敛的维度上少探索。** 完全自适应，无需手调。

某维 `|Δ_i|` 过小时 fallback 到策略自身的 `σ_i`（见 §5.1，这是**必需项**不是可选项）。

### 2.2 用法 2：Δ 当均值平移（提前探索）

```
μ_sample = μ_new + Δ
σ         = 2c · |Δ|
```

语义：**上一轮有效的方向，下一轮很可能仍然有效**，所以提前沿这个方向多走一步再采样。

`σ = 2|Δ|` 使参考动作仍落在 1σ 处，两种用法的几何性质一致，设计自洽。

### 2.3 一个优雅的共同性质

`Δ → 0` 随策略收敛而**自动发生**（`μ_new ≈ a_ref`）：

- 训练早期：位移大 → σ 大 / 平移激进 → 探索强
- 训练后期：位移趋零 → 退回策略自身分布 → 自然收敛

**这是自退火的**，不需要像现在的 `uncertainty_floor` + `phase2_threshold` 那样手调阈值、手动切 phase。这是本方案最有价值的性质。

---

## 3. 理论定位：不是凭空的启发式

| 已有方法 | 外推/调节的对象 | 方向来源 | 与本方案的关系 |
|---|---|---|---|
| **CMA-ES** rank-one update | 协方差 `C` | 演化路径 `p_c`（均值位移累积） | **用法 1 的直接祖先**，见下 |
| **OAC** (Ciosek 2019) | 采样分布均值 | `∇_a Q` 的上置信界 | **用法 2 的直接对应**，本方案是其 critic-free 廉价近似 |
| **Adaptive param noise** (Plappert 2018) | 噪声 σ | 测得的动作空间差异 | 同属"用动作差异定噪声尺度" |
| **Nesterov / Lookahead** | 参数 θ | 参数梯度历史 | 同为速度外推，但在**参数**空间且用于梯度计算，不是采样 |

### 3.1 与 CMA-ES 的对应（重要）

CMA-ES 的 rank-one covariance update：

```
C ← (1 - c₁) · C + c₁ · p_c p_cᵀ
```

其中 `p_c` **就是均值位移的累积**。CMA-ES 的核心信念正是"沿着均值最近在移动的方向把搜索分布放宽"——**和用法 1 完全同源**。用法 1 是它的**对角近似**（用逐维 `|Δ_i|` 代替完整 rank-one 外积）。

CMA-ES 是最成功的无梯度优化器之一。这个对应关系是本方案最强的理论背书。

---

## 4. 与 PPO importance ratio 的兼容性：干净

只要把 `a_ref` 当作**外生逐帧输入**（和 `explore_factor` 完全同构）：

```
π_θ(a | s, a_ref) = N(a;  μ_θ(s),  (c·|μ_θ(s) - a_ref|)²)      # 用法 1
```

- rollout 时 `θ = θ_old`，行为策略即 `π_{θ_old}(· | s, a_ref)`
- 训练时 ratio = `π_θ(a|s,a_ref) / π_{θ_old}(a|s,a_ref)` —— **完全正确**
- `a_ref` 逐帧记录、训练时回放

这条链路 `exp_standup_step_v3` 的 per-frame `explore_factor` 已经实测验证过（2026-09-08，见该文件的 `_verify_explore_factor_flow`）。唯一区别是 `explore_factor` 是标量、`a_ref` 是 21 维向量。

---

## 5. 算法分析：用法 1 的三个真实问题

### 5.1 σ 归零是吸收态（必须处理）

某维收敛后：`Δ_i → 0 → σ_i → 0 → 无探索 → 无梯度信号 → Δ_i 永远为 0`。

**自我强化的死锁。** 所以真实形态必须是：

```
σ_i = max(c · |Δ_i|,  σ_floor_i)
```

**代价**：`σ_floor` 是新超参，且**训练后期实际起作用的机制会退化成 floor**。本方案的价值集中在训练中期。这一点要有预期，不要指望它替代掉所有 floor 逻辑。

### 5.2 `|Δ|` 没有量纲校准

`|Δ|` 的大小取决于 learning rate、batch size、advantage 尺度。**LR 翻倍 → Δ 翻倍 → 探索翻倍**。

这把探索强度和优化器超参耦合在一起，难以独立推理，且 `c` 跨实验不可移植（换 LR 就要重标定）。不致命，但要记录在案。

### 5.3 单步差分噪声太大 —— 参考策略应该用 EMA

CMA-ES 用的是**累积**演化路径（`p_c` 是均值位移的 EMA），不是单步差分，原因正是单步噪声大。

对应建议：参考策略**不应固定为"上一个 update 的策略"**，而应是过去若干 update 的 **EMA / slow weights**：

```
θ_ref ← (1 - τ) · θ_ref + τ · θ_new
Δ = μ_θ(s) - μ_θ_ref(s)
```

这样 `Δ` 是平滑的、有累积意义的位移。**这是对本方案最重要的一条算法改进建议**，且直接影响工程接口设计（见 §10.3）。

---

## 6. 算法分析：用法 2 的风险，及两者的关键不对称

| | **用法 1**（改 σ） | **用法 2**（改均值） |
|---|---|---|
| 部署时无 `a_ref` | **无影响** —— 部署走 `act()` 取均值，σ 不参与 | **train/deploy 失配** —— 训练的是平移后的策略 |
| 策略震荡时 | Δ 大 → σ 大 → "这里不确定，多探索" — **响应合理** | Δ 方向交替 → 平移交替 → **放大震荡** |
| 与已有梯度的关系 | 只改探索宽度，不动改进方向 | **重复计步** —— PPO 已在往该方向走，再外推等于偷偷放大有效步长 |
| KL 护栏 | 有效 | `target_kl` 算的是同 `a_ref` 下的 `π_θ` vs `π_θ_old`，**看不见平移带来的行为分布变化** |

**结论：用法 1 严格更稳健。** 机制应设计成两者都能支持，但**先只上用法 1**，用法 2 单独作为对照实验，不要混在一起。

---

## 7. 适用性判断：为什么不解决当前的 stepping 问题

这是搁置本方案的核心原因，务必先读。

**用法 1 恰好指向所需的反方向。**

`exp_standup_step_v3` 的实测（2026-09-08）：`adv_mean_r_left_foot ≈ 0.0002`，`adv_std ≈ 0.06`——腿部维度上策略基本没在有效移动。那么：

```
腿部 Δ 小  →  σ 小  →  探索更少  →  更发现不了 stepping
```

两种探索哲学的分野：

| 哲学 | 规则 | 适用场景 |
|---|---|---|
| **σ ∝ 进展**（用法 1 / CMA-ES） | 在正在学的地方多探索 | **加速收敛**，局部优化 |
| **σ ∝ 1/进展**（optimism / curiosity） | 在卡住的地方多探索 | **跳出局部最优，发现新行为** |

CMA-ES 用第一种，因为它是**局部优化器**。当前的 stepping 问题是**发现问题**，需要第二种。

**定位**：本方案是**自适应退火器 / 收敛加速器**，不是**发现机制**。它的价值在 stepping 出现**之后**——用有原理的自适应 σ 替代现在手调的 `uncertainty_floor` + `phase2_threshold`（那套确实很脆：二次 hinge 在接近目标时无力，已不得不把 floor 目标 0.35 和触发阈值 0.30 解耦，见 commit `57df509`）。

---

## 8. 一个具体的数学陷阱（实现时必踩）

若 `σ = c·|μ_θ(s) - a_ref|`，则 **σ 依赖于正在被训练的 `μ_θ`**。此时

```
log p(a) = -(a - μ)² / (2σ²) - log σ - const
```

对 `μ` 求导，`-log σ` 项贡献 `-c · sign(μ - a_ref) / σ`。含义：**把 μ 往 `a_ref` 方向移动可以提高 log_prob**（σ 变小、分布变尖）。

**后果**：PPO 最大化 `ratio × A` 时，可以**不通过改进动作、而是通过把 μ 塌向 `a_ref` 来抬高 ratio**。这是一个与动作质量无关的作弊通道，可能导致策略冻结在参考策略上。

**解法**：把 σ 的计算从计算图里摘出来

```python
sigma = c * (mu.detach() - a_ref).abs().clamp_min(sigma_floor)   # σ 视为外生量
log_prob = Normal(mu, sigma).log_prob(a).sum(-1)                 # 只对 μ 求导
```

**框架的 docstring 必须明确警告这一点**——它足够不直观，其它策略族的实现者一定会踩。

---

## 9. 工程设计方案

### 9.1 核心原则：框架编排，策略解释

框架提供 `a_ref`，**不解释它的含义**。怎么用是策略自己的事。这与 `explore_factor` 已有的模式完全一致（框架传 `[-1,1]` 标量，策略自己映射成 `exp(ef·ln3)` 的 σ 缩放）。

好处：
- 对高斯是 σ / 均值平移；对混合是分量重加权；对流是基分布平移
- 任何策略族随时可插拔，框架不需要知道策略族细节

### 9.2 数据流（对照 `explore_factor` 的既有链路）

```
experiment.<switch on>  →  Job.reference_policy_bp = <blueprint>
  → ParallelRollouter: stochastic=True → 在 ExploratoryPolicy 外/内再套一层
  → 每帧: a_ref = ref_policy.act(obs)            # 确定性
         action, extra = policy.sample(obs, ctx(explore_factor, a_ref))
  → action_extras["reference_action"] 记录每帧 (action_dim,)
  → Episode.<新字段>  →  Trajectory.<新字段>  (T, action_dim) float32
  → PPOBuffer 拼接  →  evaluate_actions(obs, acts, ctx)
  → ppo_update 每 minibatch 切片传入
```

关键不变量与 `explore_factor` 相同：**rollout 采样和 PPO log_prob 重算用同一个 `a_ref`**。

### 9.3 接口方案：先把外生输入收口成一个类型（重要）

**不要继续加参数。** 否则会变成：

```python
sample(obs, *, explore_factor, reference_action, want_extra)
evaluate_actions(obs, actions, explore_factor, reference_action, *, want_stats)
Trajectory(..., explore_factor=..., reference_actions=...)
```

再来第三个外生输入就是四个参数。而这块**已经改过两轮名字**（`explore_intensity`→`explore_factor`、`EiSpec`→`EfSpec`、`Episode.explore_intensities`→`explore_factors`），每次都要动所有策略实现和测试——`policies/todo/test_blueprint_episode.py` 导入已删除符号、以及四个 plugin 文件导入已删除的 `DEFAULT_LOG_STD_MAX`（commit `00d1bc9`）都是这么来的。

**建议实现本方案前先做这一步收口**：

```python
@dataclass(frozen=True)
class SamplingContext:
    """策略采样 / 评估的外生逐帧输入。

    由框架在 rollout 时构造并记录，训练时原样回放，
    保证 PPO importance ratio 的采样分布与评估分布一致。
    """
    explore_factor: Any                    # 标量 或 (B,)
    reference_action: Optional[Any] = None # (action_dim,) 或 (B, action_dim)

sample(obs, ctx, *, want_extra=False)
evaluate_actions(obs, actions, ctx, *, want_stats=False)
Trajectory(..., sampling_context=...)
```

一次性改动，之后加字段不动接口。**这一步的独立收益比本 feature 本身还大。**

### 9.4 用可选字段，不要加新接口 —— 且不支持时必须 raise

新接口（如 `sample_with_history`）的问题：每个策略两条代码路径、框架要分支、能力探测只能靠 `hasattr`。而 **`hasattr` 绕过类型检查是 `CLAUDE.md` 的 "Fail Loud" 明令禁止的**。

用可选字段，并且：

> **策略收到非 `None` 的 `reference_action` 却不支持时，必须 `raise NotImplementedError`，绝不能静默忽略。**

否则就是 `CLAUDE.md` 里最典型的那类 bug——实验以为功能开着，策略默默忽略，训练用错误语义跑完全程且不报错。

### 9.5 各分布的对应语义（含一个强协同）

框架不解释 `a_ref`，所以非高斯策略族天然可扩展：

| 策略族 | `a_ref` 的自然用法 |
|---|---|
| Truncated Normal / 对角高斯 | σ（用法 1）或均值平移（用法 2） |
| 混合高斯 | 平移 / 重加权分量 |
| RealNVP | 平移基分布均值 |
| **低秩高斯 `Σ = diag + UUᵀ`** | **把 `Δ` 放进 `U` —— 这就是 CMA-ES 的 rank-one update 本身，不是近似** |

最后一条值得单独注意：**本 feature 和 `policies/todo/DESIGN_low_rank_gaussian.md` 是互相强化的**。低秩高斯 + `Δ` 作为 rank-1 方向 = 完整的 CMA-ES 协方差自适应。合起来比单独任何一个都更有说服力。

---

## 10. 工程注意事项

### 10.1 rollout 缓存逻辑要动

`parallel_rollouter.py` 的 `_run_job_batch` 现在按 env / policy_a / policy_b 三个 key 做细粒度复用（`policy_a == policy_b` 时共享同一实例）。加第四个策略要扩展这套缓存。**这是本改动里最容易出隐蔽 bug 的地方。**

### 10.2 算力开销：可能可接受，但必须实测

rollout 现在占墙钟 ~75%（14s 里 10–11s）。多一次 `96→256→21` 的 CPU 前向，相对每步 25 个 MuJoCo physics step 应该很小，估计 **< 10%**。但要实测确认，不要假设。

注意：**不需要跑两遍 rollout**，只需要对**同一批观测**做两次前向。这个区别很关键——前者让 rollout 成本翻倍，后者几乎免费。

### 10.3 `Job` 字段语义要通用，不要绑定"上一个 update"

结合 §5.3 的 EMA 建议：字段应是通用的 `reference_policy_bp: Optional[PolicyBlueprint]`，框架只负责"用它确定性求值"。谁来维护这个 blueprint 是训练循环的事。这样 **单步差分 → EMA 是 drop-in**，不动契约。

现成条件：`policy_exports/uNNNNN/` 每轮都在导出，上一轮策略磁盘上就有。EMA 需要额外维护 + 导出一个对象。

### 10.4 eval 路径要绕过

eval job 是 `stochastic=False`，`a_ref` 应当无关。要确保 wrapper 像 `ExploratoryPolicy` 一样被 `_wrap_policy` 绕过。简单但容易忘。

### 10.5 update 0 / resume 后第一轮没有参考策略

行为必须**显式定义**（按 fail-loud 原则，不要静默默认）：
- 要么明确用当前策略当参考（`Δ = 0`，全维 fallback 到策略自身 σ）
- 要么明确跳过该轮并打日志

### 10.6 `uncertainty` 语义不变

`ActorEval.uncertainty` 应继续报告**策略自身分布**的不确定度（不含 `a_ref` 带来的 σ 修正），与 `explore_factor` 的现有约定一致（`std_mean` vs `eff_std_mean`）。否则 `uncertainty_floor` 的语义会被污染。

---

## 11. 建议的推进顺序

### 阶段 0：只做诊断，不改行为（约 1/10 工作量）—— **下次继续时从这里开始**

在 rollout 侧加载参考策略、算 `Δ`、**只记录统计量**：

- `‖Δ‖` 整体，以及**按维度分组**（腿部关节 vs 上肢）
- `Δ` 的**方向一致性**：连续两轮 `Δ` 的 cosine 相似度 —— 直接区分"稳定改进"与"震荡"

不动 `Trajectory`、不动 trainer、不动策略接口。

**它能回答决定性问题**：腿部维度的 `Δ` 里到底有没有可用信号？

- 有信号 → 完整机制值得做
- 是噪声 → 用法 1 会把 σ 压小，等于做一个反向优化的功能

**附带价值**（这条本身就值得做）：现在 phase 2 后 uncertainty 从 0.30 掉到 0.18，**无法区分**"在收敛到学到的行为"还是"卡住了只在压方差"。`‖Δ‖` + 方向一致性直接把这两种情况分开，比现在基于 `uncertainty` 阈值的间接信号强得多。

### 阶段 1：`SamplingContext` 收口重构

独立于本 feature，先做收益也独立。见 §9.3。

### 阶段 2：实现用法 1

σ 用 `detach`（§8）、参考策略用 EMA（§5.3）、带 `σ_floor`（§5.1）。

### 阶段 3：用法 2 作为独立对照实验

不要和用法 1 混在一起上。

---

## 12. 决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-09 | **搁置本方案**，优先解决"如何发现新行为" | 本方案是收敛加速器而非发现机制，与当前瓶颈失配（§7） |
| 2026-09-09 | 若恢复，**从阶段 0 诊断开始**，不直接实现 | 诊断是完整实现的严格子集，且能先验证 `Δ` 是否携带可用信号 |
| 2026-09-09 | 优先用法 1，用法 2 降级为对照实验 | 用法 1 无 train/deploy 失配、震荡响应合理（§6） |
| 2026-09-09 | 参考策略语义定为通用 blueprint，非"上一个 update" | 为 EMA 留出 drop-in 空间（§10.3） |
