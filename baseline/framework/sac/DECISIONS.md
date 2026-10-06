# SAC 阶段一详细计划：设计边界与验收口径

> 状态：本文件顶部为阶段一执行计划，尚未完成调查、实验验证与设计裁决。
> 总体路线图：[PLAN.md](PLAN.md)，已获用户批准。原始需求：[bootstrip.md](bootstrip.md)。
> 下方旧 Implementation Decision Log 为历史参考，不是本轮已采纳决定。

## 1. 目标、范围与产物

阶段一要回答：新 SAC 的目标函数是什么、数据如何流动、各层负责什么、怎样证明正确，以及怎样判断两个真实任务训练成功。

本阶段允许代码阅读、依赖核查、数学推导、现有证据核验和必要的小规模验证；不进行生产训练器重写、大规模训练、八格全量迁移或完整 viewer 开发。本次落文仅制定计划，不代表这些工作已执行。

阶段结束时应形成以下可审阅产物，可继续记录在本文件的裁决区，并由总体路线图链接：

| 编号 | 产物 | 必须回答的问题 |
|---|---|---|
| A1 | 现状与依赖/复用矩阵 | 哪些经过验证可用，哪些仅为历史说法，哪些需复制适配或拒绝 |
| A2 | 两任务语义及验收协议 | 任务是否等价、如何计数、怎样才算训练完成 |
| A3 | SAC 数学与多通道规范 | 单通道基线、多通道 Q/熵/权重/掩码各自含义 |
| A4 | 策略与探索能力矩阵 | 八格如何训练，控制作用于什么分布，哪些风险何时解决 |
| A5 | 数据、实验及恢复契约 | transition/replay/采集/实验/导出/恢复之间如何连接 |
| A6 | debug 最小捕获与溯源规范 | 首版必须记录什么，怎样重算和定位一个样本 |
| A7 | 验证矩阵与阶段二工作拆分 | 每项设计如何证伪，哪些阻塞项必须先解决 |

每条结论注明来源、代码位置/版本、验证状态和适用条件。区分「已批准约束」「代码观察」「候选方案」「验证结论」，不得将本计划的候选项写成既定事实。

## 2. 工作包与执行顺序

主依赖：W1 → W2 → W3 → W4/W5 → W6 → W7 → W8。接口和数学之间允许迭代；后一个工作包发现问题时回溯修正，不为维持顺序保留错误假设。

### W1：核实当前资产与独立性边界

**调查入口：**

- `baseline/framework/sac/{experiment,trainer,replay,networks,loop}.py`、`tests/` 和 `baseline/experiments_sac/`。
- `baseline/framework/ppo/{experiment,trajectory,sampling_context,stochastic_policy}.py`、`policies/`、`dumpkit/CONTEXT.md`。
- `baseline/framework/rollout/`、`baseline/framework/train.py`、`envs/framework/{policy,episode_runner,recorder}.py`。
- 两个 PPO 实验所引用的环境蓝图、observer、初始化和终止插件。

**具体动作：**

1. 记录调查所用 git revision 和工作区状态，建立代码事实基线。
2. 列出训练、采集、registry、CLI、策略导出、debug 的直接及传递依赖。
3. 特别核对 `rollout/exploratory_policy.py` 等共享模块对 PPO `SamplingContext` 的依赖，以及顶层导入是否隐式加载 PPO。
4. 按「环境中立能力可继续使用／算法相关代码复制适配／需改写／不采用／证据不足」分类；不预设整个 rollout 层直接复用。
5. 检查旧测试究竟验证了什么；旧测试通过不代表新契约正确。需要执行时先检查资源及副作用，记录未执行项。
6. 旧 run 的性能或失败解释仅列为待核验证据，不将历史注释视为因果结论。

**出口：** A1 完整；独立性有可测试定义；列出预计修改范围，不触碰 PPO 内核以实现 SAC。

### W2：固定两个任务的语义与评估协议

**基准事实（需沿蓝图和实现继续核实）：**

| 项目 | standup | basic_balance |
|---|---|---|
| PPO 入口 | `baseline/experiments_ppo/exp_standup.py` | `baseline/experiments_ppo/exp_basic_balance.py` |
| 环境蓝图 | `standup_4stage_dense_v2_env.yaml` | `basic_balance_v2_phi_dual_env.yaml` |
| 初始状态 | 双 agent 随机倒地初始化 | 双 agent 站姿 |
| 奖励通道 | `r_potential = 0.01 × φ` | `r_fall = 0.01 × φ` 与 cross-support 奖励 |
| actor 权重 | 单通道 1 | `r_fall=3`，`r_cross=φ²` |
| 终止语义 | timeout 需要 bootstrap | imbalance 真终止不 bootstrap；timeout bootstrap |
| 现有主评估 | 最大 potential ≥0.9 的 agent 比例 | 非 imbalance 终止的 agent 比例 |

**具体动作：**

1. 核实 horizon、物理步/动作步、观测维度、动作范围、observer 输出时间和两个 agent 的边界。
2. 对每个奖励/权重标注来自 `s_t`、`a_t`、`s_{t+1}` 还是历史窗口；尤其 φ² 由 post-action observer 生成时，不直接假定它就是 `w(s_t)`。
3. 明确末帧、零物理增量退化帧、单方提前终止时的处理。保留合法边界切片，拒绝用截断掩盖数据长度错误。
4. 明确「同等功能」对齐任务事实与评估，不复制静默回退或不一致注释，也不要求复制 PPO 的 GAE、归一化和超参数。
5. 设计同批 episode 语义对拍 fixture，包括 timeout、单方倒地、子步终止及缺失字段的负例。
6. 核实可比 PPO run 的代码、配置和训练条件；无法核实时明确缺失，不用历史记录的步数直接算速度倍数。

**验收协议待冻结字段：**

- 两任务各至少三个训练 seed；可候选 42/43/44，正式登记后不因结果差替换。
- 验证 seeds 与最终保留评估 seeds 分离；固定评估 episode 数、评估模式及 checkpoint 选择规则，避免用最终测试集调参。
- ≥90% 为成功率/存活率候选阈值；明确连续达标窗口、最终 checkpoint 标准和 seed 级通过规则。
- standup 增加最终站立质量与视频核查；basic_balance 检查通道行为但不改成 step 验收。
- 环境交互预算、有效 agent transition 预算、梯度步预算、资源预算与停止/失败处理，全部在正式训练前冻结。
- 默认策略的三 seed 验收，与至少一种结构不同替代策略在两个任务的训练级验证分开列出；八格都需接口/梯度/短训验证。
- 同时报告环境步、agent transition、计算时间，不以 PPO update 与 SAC 梯度步直接比较效率。

**出口：** A2 不再留影响「是否成功」的模糊项。资源或参考结果不足时列为待用户确认的阻塞，而非自行假设。

### W3：建立单通道基线并裁决多 critic 数学

**基线锚点：** 以标准单步、双 Q、最大熵 SAC 为参照，先定义：

```text
a_next ~ πθ(. | s_next)
y = r + γ × bootstrap_mask × [min(Q1_target, Q2_target)(s_next, a_next)
                              - α × log πθ(a_next | s_next)]
L_Qi = E[(Qi(s, a_replay) - stop_gradient(y))²]
a_new ~ πθ(. | s)
L_actor = E[α × log πθ(a_new | s) - min(Q1, Q2)(s, a_new)]
```

写清自动温度目标、熵符号约定、梯度隔离、更新顺序和 target 软更新时钟；不能只有接口名而没有公式。

**必须裁决：**

1. 各通道 Q 表示纯任务 return 还是含熵的 soft return；熵是否单独建模，如何避免组合时重复或不一致计入。
2. state-dependent 权重是在每步奖励里定义，还是只调制当前 actor 更新；二者并非同一目标，不能声称等价。
3. 回放帧的权重如依赖已执行动作或下一状态，能否用于当前新动作的 actor 更新；需要额外条件、重算还是限制契约。
4. 通道有效性、bootstrap mask、actor weight、sample weight 各自独立。负 actor 权重不得进入 critic MSE 作为负损失权重。
5. 不同 γ、通道缺失、全零 actor 权重、负权重的含义；负权重与 clipped double-Q 的悲观方向不能机械组合。
6. 先确定一个可解释多通道基线。梯度尺度归一化、共享 Q trunk、按 γ 分组等均为候选，而非默认正确。
7. 若采用动作梯度尺度诊断，区分 `∂Q/∂a`、参数梯度、通道间方向冲突及优化器后的参数位移；不能将归一化权重直接宣称为实际梯度占比。
8. 记录双 agent 自对弈的非平稳性和观测不完全带来的建模限制；不能将任意对手历史数据当然视为固定 MDP 下可交换样本。
9. 默认先以 1-step 建立可信基线。若后续启用 n-step，另行论证 off-policy 偏差及中间熵项，不把它当作 GAE λ 的直接替代。

**出口：** A3 有公式、支持矩阵及反例。单通道标准退化、常量权重对照、零权重与缺失通道都有预期行为；不支持的组合明确拒绝。

### W4：八格策略与探索设计可行性

**具体动作：**

1. 建立八格清单，对照网络结构、密度、确定性 act、采样、σ 参数化、uncertainty、私有 RNG 和导出元数据。
2. 单分量检查逆 CDF 重参数化对均值/σ 的梯度，覆盖极小/极大 σ、边界、数值 clamp 和低概率区。
3. 混合策略保留「一条动作向量共享分量索引」语义，比较分量枚举加权、带正确项的随机梯度估计等候选，不悄悄改成逐维独立 mixture。
4. 设计可解析 toy Q：比较均值、σ、混合 logits 的期望梯度；使用固定噪声有限差分或高精度参考，不能仅检查有梯度。
5. 区分行为策略 β、训练策略 π、确定性评估策略。SAC 的 actor/target 使用当前目标分布，不为模仿 PPO 而重放旧采样上下文来计算重要性 ratio。
6. 设计采集探索、训练熵控制、优化控制三类旋钮，明确每个旋钮的单位、范围、生效位置、调度时钟、日志与恢复状态。
7. 不将八格 uncertainty 当作 Shannon 熵。评估有界 σ 下的可达熵范围、混合分布熵估计与自动温度目标；不照抄 `-action_dim` 或旧 α 下限。
8. uncertainty floor/reference delta 暂作候选；若保留，说明与 α 的职责、作用分布和相互影响，禁止无论证叠加。

**出口：** A4 明确阶段二首个策略、全八格适配路径及梯度验证方案。混合策略若尚未最终裁决，必须提供独立可行性门槛且不污染单分量接口；在阶段四扩齐前解决。

### W5：实验、transition/replay、时钟与恢复契约

**具体动作：**

1. 划分实验职责：环境/job、奖励与通道、数据准入、actor 控制、评估、状态；划分框架职责：replay、目标/梯度计算、调度、导出和持久化。
2. 为 transition 拟定 shape/dtype/必需性/时间语义：`obs`、执行动作、`next_obs`、逐通道 reward/有效性/bootstrap、权重来源与版本。
3. 明确单方提前结束的 `next_obs` 必须对应该 agent 的真实边界；不能无条件使用整个 episode 最后 observation。
4. 为样本定义稳定来源：run/episode/agent/frame、行为策略版本、采集时间/步数、必要环境与奖励语义版本；replay 槽位覆盖不能改变身份。
5. 区分历史事实与当前课程/权重，决定初版冻结、重算或拒绝跨版本混用；不因旧方案有 relabel 就强制实现全量重标注。
6. 定义采集 round、已执行环境动作步、有效 agent transition、critic step、actor step、temperature step；UTD 写清分子/分母，并记录请求值与限额后的实际值。
7. 明确评估、导出、checkpoint、热参数变更的时钟和边界；小数更新预算是否累计、warmup 是否计入必须有一致约定。
8. 拟定完整恢复状态清单：在线/target 模型、优化器、α、replay 内容/游标/ID、所有 RNG（含策略私有 generator）、实验、调度及计数器。
9. 区分 exact resume 与 warm-start；定义可恢复的一致性边界、设备/确定性前提、格式不兼容的失败行为和存储成本。

**出口：** A5 包括接口草案、字段表和状态清单。接口名允许在实现前微调，含义不能依赖隐式时序或无声默认。

### W6：从第一版就可诊断的捕获契约

**具体动作：**

1. 将 SAC 分成采集、转换、replay、target、critic、actor、α/target 更新、评估八类切面。
2. 定义常规标量与按需重型数据：阶段二必须具备基础标量和可追溯 minibatch；阶段五补足完整 viewer 和重型梯度分析。
3. 指定关键截面所需内容：样本 ID、抽样概率/权重、来源、模型版本、target 构成、Q/TD、actor 各项、温度、生效配置、更新前后时点。
4. 离线重算需保存相应参数状态与采样噪声或 RNG 状态；只保存 loss 不构成可重现截面。
5. replay 覆盖后，dump 中已选样本与必要来源仍自包含；视频重演的可用性和物理重放限制显式声明。
6. 设计诊断隔离 RNG 与不改变模型状态的约束，缺失数据不补零；CLI/HTTP/UI 使用同一套 SAC 分析函数和指标定义。

**出口：** A6 有最小 schema、捕获时点、来源映射及开销分层；首版即能回答一次更新为什么得到该 target 和 loss。

### W7：建立风险与验证矩阵

每项记录验证层级、输入、预期、数值容差、失败含义、证据位置及所属阶段。至少覆盖：

| 风险/契约 | 预期验证 | 主要执行阶段 |
|---|---|---|
| 单通道 SAC 数学 | 手算 target、梯度隔离、α 调节方向、target 更新 | 2 |
| TruncNorm 数值与梯度 | 固定噪声有限差分/解析参考、边界与极端参数 | 2、4 |
| mixture logits 学习 | toy Q 的期望梯度对照与分量退化等价 | 4；可行性方案在 1 |
| replay/终止边界 | 环形覆盖、单方结束、timeout、退化帧、缺字段 | 2、3 |
| 多通道含义 | 单通道退化、零 actor 权重、缺失通道、熵记账 | 3 |
| 实验语义一致 | 相同 episode 的奖励/边界/评估对拍 | 3 |
| 探索与优化控制 | 控制单变量，核对生效值、分布与调度恢复 | 4 |
| debug 正确性 | 截面重算、故障注入、开关诊断结果一致 | 2 起，5 完整 |
| 完整恢复 | 连续训练与 checkpoint 分段训练对照 | 2 起，7 完整 |
| 独立性与回归 | 直接/传递依赖检查、PPO 基线路径回归 | 2 起，7 完整 |
| 任务收敛与替代策略 | 预注册 seed/预算下训练与独立评估 | 6 |

阶段一负责建立验证设计并执行必要的低成本可行性核查，不声称已通过后续阶段的实现测试。训练曲线、梯度范数、Q1/Q2 差异均不可单独充当因果证明或真实 Q 误差。

**出口：** A7 风险可追踪，每项都有验证位置；将阶段二拆成小规模可测试工作包，而不是直接开启长训练。

### W8：汇总裁决并提交阶段一评审

**评审包：** A1–A7、候选方案取舍、未决项、支持矩阵、阶段二具体实施顺序。

每条新裁决使用与历史 N1–N11 不混淆的编号，例如 `SAC-R1-D01`，内容包含：问题、候选、证据、选择、拒绝原因、适用范围、验证办法和重开条件。

未决项分三类：

- **阻塞阶段二**：单通道目标/梯度、transition 边界、首个策略、独立接口、恢复与计数口径等；不解决不能进入实现。
- **阻塞后续阶段**：例如 mixture 估计器的最终优化、复杂多通道扩展；必须有可行路径与最晚解决阶段，不可无限后延。
- **可选增强**：PER、REDQ、异步等，不纳入初版承诺，须有新证据才启动。

若验收预算、资源或数学设计缺乏足够依据，明确报告阻塞与备选，不为通过评审给出虚假确定性。

## 3. 阶段一完成门槛

- [ ] A1–A7 均有可审阅内容，事实、候选和验证结果分开。
- [ ] 单通道标准 SAC 退化明确，多通道核心语义与支持边界可解释。
- [ ] 两任务奖励/权重/终止的时间对齐已核对，具备对拍方案。
- [ ] 八格保留范围、SAC 梯度差异和探索控制分层明确。
- [ ] 数据、依赖、时钟、完整恢复及最小 debug 捕获契约可直接指导阶段二。
- [ ] 训练级验收 seeds、阈值、持续窗口、预算与失败处理已冻结或明确列为阻塞。
- [ ] 旧决策逐项判定，没有将历史规划的结论自动继承为本轮结论。
- [ ] 阶段二工作包与测试门槛明确，并获得用户对阶段一结果的审阅确认。

上述复选框当前均未完成。批准总体路线图和要求落文，不等于阶段一已经通过。

---

# 历史参考区：旧实现决策（不作为本轮决策）

以下保留旧文全文，供调查取证。其 Phase 编号、默认参数、共享方式、性能判断及因果解释不自动生效；被本轮采用时必须另立 `SAC-R1-*` 决策并说明证据。

# SAC V2 Implementation Decision Log

Chronological record of design decisions made during implementation.

---

## [2026-08-27] N1 — Memory budget & buffer storage structure

**Decision:** In-memory only, no disk persistence for replay buffer. Default capacity 500K transitions (configurable per experiment). On resume, buffer re-warmups from scratch — model weights are checkpointed, buffer is not.

**Rationale:**
- Machine has 1TB RAM. A 500K-transition buffer for `fight` (9 channels, obs=96, act=21) costs ~1GB. Trivial.
- Disk persistence of a 1GB+ buffer adds IO complexity and checkpoint bloat for marginal benefit (warmup is ~10K transitions, a few seconds of rollout).
- Thread-safe write interface designed from the start (for future async), but Phase 1 is synchronous.

**Per-transition storage layout:**
- `obs`: (obs_dim,) float32
- `action`: (action_dim,) float32
- `next_obs`: (obs_dim,) float32 — stored explicitly (not computed from trajectory) for O(1) sampling
- `done`: (n_channels,) bool — per-channel termination flag
- `reward`: (n_channels,) float32 — per-channel reward at this step
- `actor_weight`: (n_channels,) float32 — per-channel actor weight at this step
- `tags`: Dict[str, float32] — per-transition tags (phase, source, etc.)
- `reward_features`: Dict[str, float32] — raw features for relabeling (optional)
- `traj_id`: int32 — which trajectory this transition belongs to (for n-step continuity)
- `traj_step`: int32 — position within trajectory (for n-step continuity)

**Trajectory-segment storage:** Buffer stores transitions flat but tracks trajectory boundaries via (traj_id, traj_step) pairs. This enables n-step return computation without storing full trajectory arrays.

---

## [2026-08-27] N2 — Relabel strategy

**Decision:** Full batch relabel with version tagging. When `experiment.relabel()` is called (e.g. on curriculum advance), scan entire buffer, recompute rewards and actor_weights from stored `reward_features`. Tag each transition with a `relabel_version` so we can detect stale data.

**Rationale:**
- Full scan of 500K transitions takes <1 second (pure numpy).
- Simpler than lazy relabel — no per-sample computation overhead during training.
- Requires `reward_features` to be stored, which costs ~200 bytes/transition extra. Acceptable.
- If an experiment doesn't use relabeling, `reward_features` can be empty — zero overhead.

**Interface:** `experiment.relabel(features, tags, ctx) -> (rewards, actor_weights)` is optional. Default: no relabeling (returns None, buffer keeps original values).

---

## [2026-08-27] N3 — Q network trunk grouping

**Decision:** Experiments declare trunk groups explicitly via `SACRewardChannel.trunk_group`. Channels with the same `trunk_group` share a trunk network with per-channel heads. Default: auto-group by `gamma` (channels with same gamma share a trunk). Single-channel exclusive trunk allowed by setting a unique group name.

**Rationale:**
- Auto-grouping by gamma is semantically correct: gamma determines the effective time horizon, and channels with similar horizons benefit from shared representations.
- Explicit override allows semantic grouping (e.g. "all damage-related channels share a trunk").
- Escape hatch: a critical channel like `r_fall` can get its own trunk.

**Architecture per group:**
```
Trunk: Linear(obs+act, hidden) → ReLU → Linear(hidden, hidden) → ReLU
Head_c: Linear(hidden, 1)  # one per channel in the group
```
Twin Q: each group has two independent trunk+heads (Q1 and Q2 for clipped double-Q).
Target networks: deep copy of each Q, soft-updated.

**For the MVP (sac_balance, 2 channels):** Both channels share gamma=0.99, so they share one trunk with 2 heads. Total networks: 2 Q (trunk+2heads) + 2 target = 4 networks. Very lightweight.

---

## [2026-08-27] N4 — Async collection

**Decision:** Phase 1 is synchronous. `TaggedReplay` write interface is designed to be thread-safe (using a lock), but no async rollout in the initial implementation. Measure synchronous rollout/train ratio first, then decide.

**Rationale:**
- Async adds significant complexity (concurrent buffer writes, policy version tracking, staleness observability).
- Need baseline measurements before justifying the complexity.
- SAC's UTD ratio means training time dominates rollout time in many configs, so async's benefit may be smaller than expected.

---

## [2026-08-27] N5 — log_std_min vs auto-alpha

**Decision:** SAC experiments use a wide log_std range (-10, 2) by default. Exploration is controlled entirely by alpha. The `entropy_coef` field from `ExplorationSpec` is not used by SAC — alpha replaces it. `target_entropy` defaults to `-action_dim` (-21) but is configurable per experiment and can be scheduled.

**Rationale:**
- In SAC, alpha IS the exploration controller. Hard-clamping log_std fights alpha's调节.
- The existing PPO experiments' tight log_std bounds (-1.8 to -2.5) are PPO-specific hacks that don't transfer.

---

## [2026-08-27] N7 — Action gradient normalization

**Decision:** Implement as the primary actor loss mechanism, with a fallback to naive weighted Q sum. The fallback is controlled by a `SACParams` flag (`use_grad_norm=True/False`). Validation against the fallback happens in the `sac_balance` experiment.

**Implementation:**
- Every K steps (K=10 default), estimate `ŝ_c = running_RMS(||∂Q_c/∂a||)` on a subsample of the batch using `torch.autograd.grad`.
- Actor loss: `α·logπ - Σ_c w_c(s) · Q_c(s,a) / ŝ_c`
- Normalize `Σ_c w_c` to 1.0 so the effective Q scale is constant.
- Log per-channel gradient share as diagnostic.

**Fallback:** `use_grad_norm=False` → actor loss = `α·logπ - Σ_c w_c · Q_c` (naive weighted sum, matching V1 SAC).

---

## [2026-08-27] N6 — Per-channel sampling vs multi-head Q

**Decision for MVP:** Use shared batch sampling (all channels see the same batch). Per-channel sampling is a Phase 2 feature. The multi-head Q architecture still provides per-channel Q values from a single forward pass.

**Rationale:**
- Per-channel sampling with multi-head Q is architecturally conflicting (different batches can't share a trunk forward pass).
- For `sac_balance` (2 channels, both dense), per-channel sampling provides no benefit — both channels are active everywhere.
- For `sac_fight` (sparse damage channels), per-channel sampling matters more, but that's Phase 2.
- Shared batch + per-channel actor_weight masking (aw=0 frames don't contribute to that channel's Q loss) is the MVP approach.

---

## [2026-08-27] Architecture decision — package structure

```
baseline/framework/sac/
├── __init__.py
├── PLAN.md           (planning document)
├── DECISIONS.md      (this file)
├── replay.py         (TaggedReplay buffer)
├── networks.py       (MultiHeadQCritic, trunk+heads architecture)
├── trainer.py        (sac_update: per-channel n-step TD, auto-alpha, grad norm)
├── experiment.py     (ExperimentSAC ABC, SACParams, SACRewardChannel, data types)
├── loop.py           (train_sac: synchronous loop, env_step clock, diagnostics)
└── tests/
    ├── test_replay.py
    └── test_trainer.py
```

SAC experiments live in `baseline/experiments_sac/` (separate from V2 PPO experiments):
```
baseline/experiments_sac/
├── __init__.py       (registry, auto-discovery)
├── base.py           (CombatExperimentSACBase — shared combat defaults)
└── exp_sac_balance.py
```

---

## [2026-08-27] Implementation scope for first iteration

**In scope (MVP):**
1. TaggedReplay with trajectory-continuous storage, n-step targets, per-channel done
2. Multi-head Q critic (trunk + heads, twin Q, soft target update)
3. sac_update with per-channel n-step TD, clipped double-Q, auto-alpha
4. Action gradient normalization (with fallback)
5. ExperimentSAC interface with data_sources, build_slices, relabel, replay_plan
6. Synchronous training loop with env_step clock, diagnostics, divergence guardrails
7. train.py --algo sac dispatch
8. sac_balance experiment (2 channels, basic_balance env)
9. Unit tests for replay flattening, n-step, per-channel done
10. Smoke test + real training run

**Out of scope (Phase 2+):**
- Async rollout collection
- Per-channel sampling
- Buffer-based env reset
- Stratified retention (MVP uses uniform sampling with optional tag filtering)
- Opponent-pool self-play data ingestion
- Multiple data sources (MVP uses single source: learner rollout)
- DroQ (Dropout + LayerNorm for Q networks)

**Simplifications for MVP:**
- `data_sources()` returns a single `SelfRollout` source (learner's own rollout)
- `replay_plan()` returns uniform sampling (no stratification)
- `relabel()` not used (no curriculum in sac_balance)
- `tags` stored but not used for sampling in MVP
- `reward_features` stored but not used in MVP
- `core_state` not stored in MVP

---

## [2026-08-27] N8 — Training stability: alpha collapse & reward scale

**Context:** First real training runs of `sac_balance` revealed two critical
stability issues that required parameter tuning.

**Issue 1: Alpha collapse (v2 run)**
- With `target_entropy=-21` (= -action_dim) and `alpha_lr=3e-4`, alpha
  collapsed from 0.2 to 0.003 in <20 rounds (2000 grad steps/round).
- This caused policy collapse: episode lengths crashed from 36 to 8.
- Q values went from +3.7 to -5.6 in a death spiral.

**Fix:**
- `target_entropy`: -21 → -10 (less aggressive, allows earlier exploitation)
- `alpha_lr`: 3e-4 → 1e-4 (3x slower alpha convergence)
- `log_alpha_min`: -10 → -5 (alpha floor ≈ 0.007, prevents total collapse)
- `q_layer_norm`: False → True (stabilizes Q estimates)

**Issue 2: Q overestimation divergence (v6 run)**
- With `reward_scale=200`, the policy learned successfully (survived=14
  at 1.1M env steps, 27x more sample-efficient than PPO's 27M env steps).
- But Q losses grew from 200 to 1138, causing divergence at 1.55M env steps.
- The high reward scale made TD errors too large for stable Q learning.

**Fix:**
- `reward_scale`: 200 → 50 (4x reduction in TD error magnitude)
- `critic_learning_rate`: 3e-4 → 1e-4 (3x slower Q learning for stability)

**Key insight:** SAC with 1-step TD needs reward scaling for small per-step
rewards (~0.005), but the scale must be balanced against Q stability.
PPO doesn't need this because GAE naturally amplifies credit assignment.

---

## [2026-08-27] N9 — Training scale: matching PPO's env step budget

**Context:** PPO `basic_balance` requires ~27M env steps to first reach
survival_rate=1.0 (update 295, 1024 episodes/update, 96 workers).

**Decision:** SAC `sac_balance` configured with:
- `max_env_steps`: 10M (SAC should be more sample-efficient than PPO)
- `episodes_per_update`: 256 (PPO uses 1024, but SAC reuses data)
- `rollout_workers`: 96 (match PPO's parallelism)
- `utd_ratio`: 0.25 (1 grad step per 4 new transitions)
- `max_grad_steps_per_round`: 2000 (caps round time to ~52s)
- `replay_buffer_size`: 1M (allows long-term data reuse)
- `eval_interval`: 100K env steps

**Rationale:** SAC's off-policy data reuse should need fewer env steps
than PPO. The UTD ratio of 0.25 with a 1M buffer provides effective
data reuse of ~26x per transition (buffer_size / batch_size × rounds_in_buffer).
The grad step cap keeps wall-clock time reasonable (~52s/round).
