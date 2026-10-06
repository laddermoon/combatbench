# SAC 阶段一详细计划：设计边界与验收口径

> 状态：阶段一 W1–W3 已完成，对应 A1–A3 见下方裁决区；W4–W8 待执行。算法实现与真实任务训练尚未开始。
> 总体路线图：[PLAN.md](PLAN.md)，已获用户批准。原始需求：[bootstrip.md](bootstrip.md)。
> 下方旧 Implementation Decision Log 为历史参考，不是本轮已采纳决定。

## 1. 目标、范围与产物

阶段一要回答：新 SAC 的目标函数是什么、数据如何流动、各层负责什么、怎样证明正确，以及怎样判断两个真实任务训练成功。

本阶段允许代码阅读、依赖核查、数学推导、现有证据核验和必要的小规模验证；不进行生产训练器重写、大规模训练、八格全量迁移或完整 viewer 开发。执行证据在各 A 节登记；公式探针、旧测试回归和新契约实现验收严格区分。

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

# 裁决区（本阶段产出）

## A1 — W1 执行结果：现状与依赖/复用矩阵（2026-10-06）

> 本节的每条结论均标注证据类型：**〔代码事实〕** 直接读码/实测可得；**〔运行实测〕** 本次调查实际执行；**〔历史证据〕** 来自旧 run/旧文档，仅作待核验线索。

### 1. 调查基线

- Git：`772e5dca`（`main`，含本阶段计划提交），工作区干净。
- 环境：`instance-1f1igpaq`，Python 3.10，torch+CUDA 可用。
- 方法：全量读码 + 选择性实测（import 链、PolicyBlueprint.build、Job 解包、旧测试），未启动任何训练。

### 2. 依赖事实（实测确认）

**2.1 SAC 模块的直接依赖**

| 文件 | 导入 | 性质 |
|---|---|---|
| `sac/experiment.py:28` | `baseline.framework.ppo.TrainablePolicy` | **SAC→PPO 直接耦合** |
| `sac/experiment.py:282` | `Job = Tuple[...]` 别名，遮蔽第 29 行导入的 rollout `Job` dataclass | 幽灵定义，死代码且误导 |
| `sac/loop.py:32` | `baseline.framework.rollout.{Episode, ParallelRollouter}` | 经 `rollout/__init__` 传递加载 `exploratory_policy` → `ppo.sampling_context` |
| `experiments_sac/base.py:24` | `baseline.framework.ppo.TrainablePolicy` | 直接耦合 |
| `experiments_sac/exp_sac_balance.py:33` | `rollout.{extract_per_step_field, extract_per_step_scalar}` | 纯 numpy 工具，中立 |
| `experiments_sac/__init__.py` | `sac.experiment.ExperimentSAC`；注册 glob 为 `exp_sac_*.py` | 新实验命名须含 `sac` 中缀，否则不注册 |
| `sac/{replay,networks,trainer}.py` | 仅互相依赖 + numpy/torch | 内部自洽 |

**2.2 传递依赖（import 实测）**

- `import baseline.framework.rollout.job`（任意叶子模块）→ 必然执行 `baseline/framework/__init__.py` → `from .ppo import ...` → 加载 `ppo.experiment/sampling_context/stochastic_policy`。**框架根 `__init__.py` 是全局性 PPO eager import。**
- `rollout/__init__.py` 还经 `parallel_rollouter → exploratory_policy → ppo.sampling_context` 形成第二条传递边；`SamplingContext`/`StochasticPolicy`/`SamplingSpec`/`SamplingPolicy`/`Job` 这条链本质上是 PPO 的采样探索体系。
- `train.py:19-20` 顶层 eager import 两个 registry → 跑 PPO 也加载 `experiments_sac`（进而 sac.experiment→ppo），跑 SAC 也加载 `experiments_ppo`。
- `ppo/experiment.py:130` 反向 import `rollout.job.Job`；`ppo/loop.py` 内 lazy import `envs.batchframework.device_rollouter`（`--collector device`，PPO 独占，SAC 已在 `train.py:451` 禁用）。

**2.3 关键结论：SAC 当前不可能"名义独立"**

即使删掉 `sac/experiment.py:28` 的直接 import，`baseline/framework/__init__.py` 与 `rollout/__init__.py` 两条传递边仍会使 SAC 路径加载 PPO 模块。要达成可测试的独立性，必须处理这三条边（见 §5）。

### 3. 当前 SAC 的健康度（代码事实，非"可用基线"）

| # | 发现 | 证据 | 影响 |
|---|---|---|---|
| F1 | 默认 actor blueprint `init_policy.yaml` 指向 `ppo.policies.tanh_gaussian_mlp:TanhGaussianMLPPolicy`，该类已于 `f232b8c5`(2026-09-05) 移入 `policies/todo/`；`PolicyBlueprint.build()` 实测 `ModuleNotFoundError` | 〔运行实测〕 | **当前 SAC 无法构建 actor，训练入口已死**；v1–v7 旧 run 早于该移动 |
| F2 | `sac/loop.py:605` `v_p_a, v_p_b, v_env, v_seed, v_options = eval_jobs[0]` 把 rollout `Job` dataclass 当五元组解包；`54fc8f54`(2026-09-05) 把 Job 改成 8 字段 frozen dataclass 后此处必 `TypeError` | 〔运行实测〕 `cannot unpack non-iterable Job object` | 首个视频渲染（sac_balance 约 500K env_step 处）即崩溃 |
| F3 | `sac/loop.py` 调用 `actor.to_blueprint(dest_path=, stochastic=)`、`hasattr(actor, "export_policy_artifacts")`、`actor.sample_action(obs)`；`TrainablePolicy` ABC 只声明 `to_blueprint(dest_path)`——`stochastic` kwarg 仅存在于 todo/ 版 tanh_gaussian，八格 TruncNorm 的 `to_blueprint` 均无该参数 | 〔代码事实〕 | SAC 的 actor 事实契约未被 ABC 表达，且与八格签名不兼容 |
| F4 | `trainer.py` 中 `actor_weight` 同时加权 critic 回归损失与 actor loss | 〔代码事实〕 | 混同"通道是否学习"与"通道如何影响 actor"两个语义，违反本阶段计划预设的分离原则 |
| F5 | `replay.py sample_nstep` 沿 `(start_idx+k) % capacity` 线性取后续帧，环形覆盖后可跨轨迹拼接（靠 done 掩码截断只是多数情况下的侥幸） | 〔代码事实〕 | n-step>1 时可能产生跨轨迹污染的 target，W5 需定案 |
| F6 | replay 采样用全局 `np.random`；checkpoint 不含 replay/RNG/rollout 状态；`loop.py:442` 有 `*0` 死代码 | 〔代码事实〕 | "resume" 实际只是模型 warm-start，非完整续训 |
| F7 | `exp_sac_balance.py:140-141` obs/acts/fin_obs 为 `None` 时静默 `return []` | 〔代码事实〕 | 违反 fail-loud；丢数据不报错 |
| F8 | 旧测试 `tests/` 16/16 通过 | 〔运行实测〕 | 只覆盖 replay 机械行为与 trainer smoke，不覆盖契约正确性；不能作为新 SAC 依据 |
| F9 | `sac_balance_real_v7`：10M env_step / ~1.03M grad_step，末段 survival≈0.56；r_cross 的 q1_mean≈-177、q1_loss≈62、旧 grad_share≈0.2% | 〔历史日志线索；W3 修正解释〕 | 未达到任务门槛。reward_scale=50 作用于 reward target，且 Q 含熵，不能用未缩放纯奖励界判断 Q 越界。旧 grad_share 实为 `(mean(aw)/scale)²` 的占比，不含真实通道梯度；aw≈0.72 对 3.0 也只是系数比例，不是期望梯度份额。**撤回据此认定实际梯度压制或训练失败原因的推断**；需新诊断和消融，见 A3.9。 |

### 4. 资产分类矩阵

| 资产 | 当前角色 | 分类 | 处置方向 |
|---|---|---|---|
| `envs.framework.*`（Policy/PolicyBlueprint/EpisodeRunner/Recorder/插件体系） | 运行时底座 | **环境中立，继续用** | 不动 |
| `envs.humanoid21.*`（含两个目标 env blueprint：balance 的 `basic_balance_v2_phi_dual_env.yaml` 与 PPO 实验同名一致；standup 的 `standup_4stage_dense_v2_env.yaml`） | 任务定义 | **环境中立，继续用** | 不动 |
| `rollout/episode.py` Episode / `episode_recorder.py` / `episode_collection.py` | 采集数据契约 | **中立但 import 链被 `rollout/__init__` 污染** | 内容可直接用；归置方式见 §5 决策点 |
| `rollout/job.py`（Job/SamplingSpec/ReferenceSpec）、`exploratory_policy.py`、`parallel_rollouter.py`、`remote_policy.py`、`inference_server.py`、`observer_utils.py` | 采集与采样包装 | **机制中立、承载 ppo 类型** | 同上决策点；`SamplingPolicy` 的探索体系（ef/reference/delta σ-floor）属 PPO 语义，SAC 拷贝时应裁剪为 SAC 自己的探索契约 |
| `ppo/policies` 八格 TruncNorm | actor 候选族 | **复制适配** | `sample_action` 为逆 CDF 重参数化采样，天然满足 SAC 的 d a/d θ 需求〔代码事实〕；需 SAC 化接口：移除/替换 `evaluate_actions`+SamplingContext 语义、统一 `to_blueprint(stochastic=)`、补 `export_policy_artifacts` 等价物；ef 约定（truncnorm ef∈[-1,1]，0 中性）与旧 tanh_gaussian（ef∈[0,1]，0.5 中性）不一致，须在 SAC 内统一口径 |
| `ppo/policies/_export_template*.py` + `file:` blueprint 机制 | 部署导出 | **复制适配** | 模板自包含（无 baseline 依赖），逐族拷贝到 sac/；`ExportedTruncNormPolicy.sample(ctx)` 按鸭子类型读 ctx 字段 |
| `ppo.sampling_context.SamplingContext` / `ppo.stochastic_policy.StochasticPolicy` | rollout 侧探索容器/接口 | **复制适配** | 代码量小（~70+~49 行）；SAC 若要独立探索字段（如 noise injection、ou 噪声等），拷贝后改造 |
| `TrainablePolicy`/`ActorEval`（ppo.experiment） | actor 契约 | **需改写（SAC 自有）** | SAC 需自有 ABC：`sample_action(obs)→(a,logπ)` 可微、`deterministic_action`、`act`、`to_blueprint(stochastic)`、导出面；`evaluate_actions`/uncertainty 属 PPO 语义不进 SAC |
| `dumpkit/`（~7.7K 行：frame_access/dump_capture/dump_analysis/metric_catalog/viewer）+ `ppo/debug.py` | debug 系统 | **仅设计参照，不移植** | 直接依赖 `PPOBuffer/Trajectory/UpdateStats/algos`；继承其"沿计算链下钻+单样本溯源+CLI/viewer 同源"的架构思想，SAC 自建 |
| `code_snapshot.py`、`train.py` 的 `--background/--resume/--seed/--set/--list-experiments`/`_setup_logging` | 工程外壳 | **中立可共用** | train.py 需把两个 registry 改 lazy import；PPO 专属 flag（`--param`/`--dump-at`/collector）对 SAC 目前是静默丢弃，需补 gate 或接线 |
| `baseline/framework/__init__.py` | 包 init | **需改写** | 去掉 eager `from .ppo import`（外部无 `from baseline.framework import X` 消费方，已核查） |
| `critic_mlp.py`（framework 根） | 共享 critic MLP | **证据不足/倾向不用** | sac 现有 networks 未使用；W3 再定 |
| `sac/{experiment,trainer,replay,networks,loop}.py` 现状 | 旧实现 | **逐件裁决：结构可参照，语义需重写** | 通道对象/transition 契约思路保留为候选；actor 契约、aw 语义、n-step、relabel、checkpoint 按 §3 问题清单重写 |
| `experiments_sac/` 现状 | 旧实验 | **改写** | base.py 保留骨架风格；`exp_sac_balance` 的奖励构成（r_fall=0.01φ、r_cross、φ² 门控 aw）与 PPO `basic_balance` 语义对应关系待 W2 对拍确认 |
| PPO 专属机制（PPOBuffer/GAE/confidence/dual-clip/param_patches/uncertainty floor/DeviceRollouter/analyze_training.py——后者文件已不存在） | — | **不采用** | 不进 SAC |

### 5. 独立性定义与关键决策点

**建议的可测试标准（待评审冻结）：**

- C1（import 洁净）：`import baseline.framework.sac` 与 `import baseline.experiments_sac` 后，`sys.modules` 中不出现 `baseline.framework.ppo*`。
- C2（删除存活）：将 `baseline/framework/ppo/` 暂时挪走后，SAC 单测与 `train.py --algo sac --smoke` 路径可运行。
- C3（语义边界）：SAC 训练数据路径上不出现 PPO 对象（SamplingContext、重要性权重、GAE 等）；探索控制走 SAC 自有契约。

按现状，C1/C2 均不成立（三条传递边 + 两处直接 import + 已死的 actor blueprint）。

**阻塞性决策点（W1 不能替用户定，列明选项）：**

**D-ROLLOUT：采集/采样层的归置。** 选项：
- (a) `rollout/` 视为共享中立层，仅把 `SamplingContext/StochasticPolicy` 提升出 ppo（改 ppo import 指向新位置，违反"不动 PPO"约束，不推荐）；
- (b) **SAC vendor 一份采集层**（`job/episode/recorder/rollouter/采样wrapper` 拷贝进 `sac/`，裁剪 reference/delta σ-floor 等 PPO 探索语义）——符合"拷贝加修改"原则，`rollout/` 对 PPO 完全不动，SAC 获得自有探索契约的落点。**W1 倾向此项**；
- (c) 仅 vendor `sampling_context` 等价物 + 继续用 `ParallelRollouter`（靠 `file:` 导出策略鸭子类型消费 ctx）——最小改动，但 `Job.stochastic` 语义、spec 字段、`sctx__` extras 仍由 PPO 体系定义，C3 不达标。

**D-PKG-INIT：`baseline/framework/__init__.py` 去 eager ppo。** 外部无 `from baseline.framework import X` 消费方（已核查），清空为惰性/空 init 属中立改动、不动 PPO 内部。C1 需要。

**D-CLI：`train.py` 两个 registry 改 lazy import**，SAC 分支补 `--param`/`--dump-at` 等的 gate 或接线。

### 6. 预计修改范围（不含 PPO 内部任何改动）

- `baseline/framework/sac/`：`experiment.py`（自有 actor ABC、Job 别名清除）、`loop.py`（Job 解包/导出契约/`hasattr` 收敛）、`trainer.py`（aw 语义按 W3 裁决重写）、`replay.py`（n-step 跨轨迹防护、RNG 独立化、checkpoint 含 replay 待定）、`networks.py`（按 W3 保留或简化）、新增 `sac/policies/`（八格拷贝适配）与按 D-ROLLOUT 决定的 `sac/` 内采集层。
- `baseline/experiments_sac/`：`base.py`（actor 契约/蓝图默认）、新增两个目标实验 `exp_sac_*.py`。
- `baseline/framework/train.py`：registry lazy import、SAC flag gate。
- `baseline/framework/__init__.py`：去 eager ppo。
- `baseline/humanoid21/blueprints/`：新增指向 sac policies 的 `init_policy_sac_*.yaml`。
- **不动**：`baseline/framework/ppo/**`、`envs/**`、`rollout/`（若选 D-ROLLOUT-b）。

### 7. 未决问题与证据缺口

1. v1–v7 的未收敛归因（reward_scale=50 × critic_lr、alpha 钳制、grad_norm 实际权重公式、UTD 0.25×batch256）仅有日志表面信号，W2/W3 才做因果核查。
2. 旧 SAC 的 `n_critics/in_target_min`（REDQ 字段）、`relabel`、`DataSource`/`ReplayPlan` 多为未被消费的接口——保留价值待 W5。
3. `TaggedReplay` 环形缓冲与 traj_id 生命周期在覆盖下的正确性需构造性测试。
4. 八格中 mixture 族的离散分量在 SAC actor loss 下的梯度估计方案，W4 专查。
5. `--collector device`（batchframework）对 SAC 的必要性评估未做（初版默认 cpu collector）。

### 8. W1 出口自检

- [x] 依赖图与直接/传递依赖已列出（§2）
- [x] rollout 共享模块对 `ppo.sampling_context` 的依赖已确认（exploratory_policy.py:34；framework/\_\_init\_\_.py eager ppo；train.py:19-20 eager registry）
- [x] 旧测试实际覆盖面已核实（§3-F8），不作契约正确性证据
- [x] 独立性有候选可测定义（§5），阻塞决策点已列出供裁决
- [x] 预计修改范围明确，且不需要改动 PPO 内部（§6）

**W1 遗留待用户确认**：D-ROLLOUT 采集层归置选型（建议 b）、是否接受 §5 的 C1–C3 作为冻结的独立性标准、以及 `baseline/framework/__init__.py` 与 `train.py` 两处中立改动是否在授权范围内。

### W1 裁决记录（用户确认，2026-10-06）

- **D-ROLLOUT = (b)**：采集/采样层 vendor 进 `sac/` 自有命名空间（job/episode/recorder/rollouter/采样 wrapper 拷贝加修改），裁剪 PPO 的 reference/delta σ-floor 等探索语义；`baseline/framework/rollout/` 对 PPO 路径保持不动。
- **独立性标准冻结**：C1（`import baseline.framework.sac` / `baseline.experiments_sac` 后不出现 `baseline.framework.ppo*` 模块）、C2（`ppo/` 暂时挪走后 SAC 单测与 `--algo sac --smoke` 路径可运行）、C3（SAC 训练数据路径无 PPO 对象）全部接受。
- **中立改动授权**：`baseline/framework/__init__.py` 去 eager `ppo` import、`train.py` 双 registry 改 lazy import，均在授权范围内（不属 PPO 内部改动）。

## A2：两个目标任务的语义冻结与验收协议（W2 完成稿，2026-10-06）

本节区分代码事实、历史日志线索和验收设计；历史 run 的源码一致性、续训谱系及资源条件尚未全部核验，不能称为严格可比基准。「冻结」字段在验收前不得因结果不理想而修改。W3 修订：post-action 权重仅是历史事实，其 SAC 应用规则以 A3 为准；用户已确认 basic_balance 改用动作前 actor 门控，见 SAC-R1-D04。

### A2.1 两任务共同事实（已核实）

- Humanoid21 双 agent：obs_dim=96、act_dim=21（[-1,1] 归一化 PD 目标）、`phy_steps_per_action=25`（dt=2ms → 每动作步 50ms，控制频率 20Hz）。两个实验 `agent_used="both"`，self-play 同一份策略，**每个 episode 产出 2 条 agent 轨迹**。
- 两个实验都以 `max_steps=200` materialize 蓝图（`experiments_ppo/base.py:254,507`），即 **horizon = 200 动作步 = 5000 物理步 = 10s**。`basic_balance_v2_phi_dual_env.yaml` 蓝图默认值 600 被实验覆盖，实际 horizon 也是 200。
- Episode 计数语义（`env_runtime.py:148-171`、`context.py:278-311`、`episode.py:394-419`）：
  - `episode_step` 每进入一个 `step()` 无条件 +1（在 post_action_step hooks 之前递增），`physics_step` 只计实际执行的物理子步；某帧 `physics_step` 增量=0 是「动作未物理生效」的退化帧。
  - 终止提议记录 `(reason, episode_step)`，其中 `episode_step = 帧索引+1`。
  - `agent_frame_boundary[aid] = min(该 agent 首次终止提议的 episode_step, 最后一个 physics 增量>0 的帧号+1)`；`obs[:T]` 切片**包含提出终止的那帧**（其动作已物理执行）。
  - `final_observation` 在 `on_post_episode` 捕获 = **整段 episode 末帧后的观测**，两个 agent 的采集时刻相同，但各自观测向量不同（`episode_recorder.py:152-159`）。
  - `request_termination(reason)` 无 agent_id → 全局终止双方（TimeoutPlugin 走此路径）；带 `agent_id` → 逐 agent 终止（DualImbalanceTerminationPlugin 走此路径）。
  - 逐 agent 终止后 episode 为存活方继续；终止方 policy 仍被逐帧查询（`EpisodeRunner.post_termination_action="policy"` 默认值），其后帧照常记录但被 boundary 切片丢弃。

### A2.2 观测者时序 —— 奖励/权重的 s_{t+1} 对齐（W2 重点核实项）

三个 reward/weight 相关 observer 全部在 `on_post_action_step` 计算输出（`standing_balance_4stage.py:145`、`height_phi_observer.py:49`、`cross_support.py:149`）。因此：

| 数据 | 记录位置 | 语义 |
|---|---|---|
| `observations[t]` | 帧 t | s_t（动作前） |
| `actions[t]` | 帧 t | a_t |
| `observer_outputs[*][t]` | 帧 t | **s_{t+1} 的函数**（post-action 状态）；`r_cross` 还依赖 rewarder 内部 FSM 历史（接触/换脚计时器），是 episode 窗口信号而非纯 s_{t+1} 函数 |
| `rewards_c[t]`（实验推导） | transition t | 转移 (s_t,a_t)→s_{t+1} 的回报，按下标 t 对齐 —— 与标准 RL 约定一致，无需移位 |
| `actor_weight[t]`（实验推导） | transition t | **post-action 权重**：`aw_cross[t] = φ²(s_{t+1})`，即按该转移的结果状态评估；不是 `w(s_t)`。含义为「对结局处于直立状态的转移，r_cross 通道对 actor 目标的影响更大」。存同一数组只能保留历史转移事实，不能保证用于新动作更新时语义不变。W3 已裁决：保留 `actor_weight_post_reference` 供对拍；SAC 实际 actor 使用动作前状态门控（A3.5），不可混称 |
| `final_observation` | episode 级 | 末帧之后的观测；仅当 `T == num_frames` 时才是该 agent 末转移的 s_{T+1} |

### A2.3 逐任务语义冻结表

**standup（`exp_standup.py` ↔ 蓝图 `standup_4stage_dense_v2_env.yaml`）**

| 项 | 冻结值 |
|---|---|
| 初始化 | `RandomFallenStatePlugin` 双机器人随机倒地（`target_robots: both`、`max_phy_steps: 1000`、`height_threshold: 0.3`、`reset_interval: 5`；逐 episode RNG 经 `set_episode_seed` 重建）；`initial_distance ∈ U[1.5,3.5]` |
| 终止 | 无逐 agent 终止插件；唯一终止 = 全局 `timeout`（step 200） |
| 奖励 | `r_potential[t] = 0.01·potential[t]`；potential ∈[0,1] 分段：stage1 翻身 `0.10·f_score`、stage2 支撑 `0.10+0.10·contact_score`、stage3 手脚 XY 距离 `0.20+0.10·d_score`、stage4 `0.30+0.70·w_foot·h_score`（w_foot=脚载重比，h_score=躯干高度 0.15→1.28m 归一） |
| actor 权重 | 恒 1.0 |
| is_terminated | 恒 False（末转移 bootstrap，`fin_obs` 有效） |
| 评估 | 64 eval episodes → 128 agent 轨迹；每 agent `success = max_pot ≥ 0.9`，success_rate = 达标 agent 比例；报告 `max_pot / final_pot / max_stage / max_h`；best-of-run 以 mean `max_pot` 选优 |

**basic_balance（`exp_basic_balance.py` ↔ 蓝图 `basic_balance_v2_phi_dual_env.yaml`）**

| 项 | 冻结值 |
|---|---|
| 初始化 | 双机器人站姿（无倒地插件）；`initial_distance ∈ U[1.5,3.5]` |
| 终止 | `DualImbalanceTerminationPlugin`（`force_threshold=1.0`N、`tolerance=1`、`min_height=0`）：post-action-step 粒度检测非脚部 body↔地面接触 ≥1N，单帧即对该 agent 提议 `imbalance_robot_X`；episode 到双方均终止才结束（另一方继续到倒地或 timeout） |
| 奖励 | `r_fall[t] = 0.01·φ[t]`（`HeightPhiObserver`：`φ = uprightness · h/1.28`）；`r_cross[t] = CrossSupportBalanceRewarder` 标量（默认参数下 ≤0：首次单脚支撑宽限 30 步后 `−0.25·excess/30`、单脚段 <4 步 `−0.45·deficit/4`、A→B 换脚 >18 步 `−0.25·excess/18`；无正奖励项） |
| actor 权重 | `r_fall` 恒 3.0；`r_cross = φ²(s_{t+1})`（A2.2 的 post-action 语义） |
| is_terminated | 首条终止提议以 `imbalance` 开头 → True（末转移 done，不 bootstrap）；仅 timeout → False |
| 评估 | 16 eval episodes → 32 agent；`survived` = 首条终止原因不以 `imbalance` 开头的 agent 数（timeout 计为存活）；survival_rate；best 按 survived 数 |
| 死字段 | `posture_a/b`（PostureRewarder）接入蓝图并被记录，但实验的 `posture_key` 参数在 `_build_agent_trajectory` 内从未使用 —— 不参与奖励/评估；SAC 侧可保留为诊断观测，不计入任务语义 |

### A2.4 PPO 代码中的静默行为 —— SAC 不复制（fail-loud 分歧点，有意为之）

1. `exp_standup._build_agent_trajectory`：`potential` 字段缺失 → `np.zeros(T)` 静默回退（`exp_standup.py:149`）。**SAC 改为 KeyError。**
2. `exp_basic_balance`：`cross_support_*` observer 缺失 → `extract_per_step_scalar` 返回 zeros（`observer_utils.py:60-61`），且其 `if r_cross is not None` 分支是死代码（该函数对缺失 observer 从不返回 None）。**SAC 改为 KeyError。**
3. `exp_basic_balance`：`phi` 缺失 → KeyError（正确范式，SAC 沿用）。
4. `coerce_per_step`：leaf 为 None → zeros；长度不匹配 → ValueError（保留后者）。
5. PPO 对 `obs/actions/fin_obs` 缺失 → `return []` 静默丢整条轨迹。**SAC 改为 raise**：畸形 episode 是 bug 而非稀疏数据。

### A2.5 SAC 数据契约冻结（实验侧切片语义）

- 转移 i 字段：`(obs[i], act[i], r_c[i], next_obs[i], done_c[i], aw_c[i])`，`next_obs[i] = obs[i+1]`（i+1 < num_frames 时）否则 `final_observation`。**禁止**对 `T < num_frames` 的 agent 用 `final_observation` 充当下一个状态（那是别的时刻的观测）。
- 逐 agent 切片：T = `agent_frame_boundary`（含提议帧、排除尾部零物理增量退化帧）；所有通道数组长度恒等于 T；`done_c[T-1]=True` 当且仅当首条提议原因为真终止（本任务族即 `imbalance*`）；timeout → 全部 done=False。
- 单方先终止：该 agent 切片止于其提议帧；其后为另一 agent 记录的帧一律不得入池；其末转移 next_obs = `obs[T]`（该帧在 episode 中存在），done=True 屏蔽 bootstrap。
- 中途（物理子步内）终止：本子步增量 ≥1 → 真实帧计入；proposal 记在该帧；runtime 支持，fixture 覆盖。
- 缺声明的 observer/字段、长度不匹配、缺 obs/act/fin_obs → 一律 raise。
- `done`、动作前门控事实及动作后参考权重均按 transition 存储；W3 已在 A3 区分实际 actor 权重与 `actor_weight_post_reference`，不得把两个字段互相代用。

### A2.6 PPO 参照 run 锚点（仅任务事实锚，不作效率基准）

| run | seed | 配置 | 收敛锚点 |
|---|---|---|---|
| `train_standup_ppo_20260821_001328`（日志含 u1–2320，续训谱系未核验） | 42 | 512 eps/upd、eval 64 eps/5 upd | W2 日志扫描估算 success 从 u1225 持续 ≥0.9，累计约 **125.4M env steps**；后期接近 1.0（并非每次恒等 1.0） |
| `train_basic_balance_ppo_20260908_180553`（u0–2165） | 42 | 1024 eps/upd、eval 16 eps/5 upd | survival 首次持续 ≥0.9 ≈ u160 ≈ **7.8M env steps**（ep_len 由 ~27 渐升至 200）；其后 2000+ update 恒 1.0 |

PPO update 数与 SAC 梯度步数不可直接比较；以上仅登记 W2 对日志的扫描结果，不证明代码/初始化/资源条件与当前实验一致。此前用单空格正则只匹配到 u1000 以后是日志解析错误（早期格式为 `[eval    5]`），不能据此认定续训。正式对比前还需核验 code_snapshot、参数覆盖、日志完整性和 checkpoint 谱系，不据这些数字承诺 SAC 样本效率。

### A2.7 验收协议冻结字段

- **任务指标**：定义与 PPO 完全一致 —— standup `success_rate`（agent 粒度 max_pot≥0.9 比例）；balance `survival_rate`（非 imbalance 终止比例）。
- **训练 seeds**：`42/43/44`，本文件登记，不因结果差替换。
- **评估流**：确定性策略（`stochastic=False` 等价物）；训练中评估 = 验证流，W2 登记的 base_seed 方案为 `train_seed + 100_000 + eval_index·97`，最终保留评估为 `train_seed + 200_000` 起，只对最终 checkpoint 跑一次。**W3 更正：偏移本身不保证互斥**；W5 必须冻结实际 seed manifest/调度并检查训练、验证、保留集无重叠，冲突时在正式训练前修订生成方案，不在看过保留集结果后换 seeds。
- **评估 episode 数**：standup 64（128 agent）、balance **32**（64 agent —— 有意从 PPO 的 16 加倍以提高分辨率，指标定义不变、口径仍可比）。
- **通过阈值**：两任务指标均 ≥0.90；**通过判定** = 连续最后 ≥3 次验证评估 ≥0.90 且最终 checkpoint 在保留评估上 ≥0.90。
- **standup 附加质量闸**：mean `final_pot` ≥0.80 且 mean `max_h` ≥1.20，且对最终 checkpoint 抽 ≥2 episode 视频目检为真实站立收尾（排除瞬间触高即倒）。
- **balance 附加报告**（非闸门）：r_cross 通道统计、eval φ 均值轨迹、视频交替支撑行为 —— 用于确认通道确实起作用，不改成步数指标。
- **checkpoint 规则**：固定 env-step 预算跑满后的**最后 checkpoint 为唯一验收 checkpoint**（不做 best-of-run 挑选）；best 导出仅作诊断工件。
- **预算（用户已确认的 per seed 硬上限）**：standup ≤40M env steps / ≤80M agent transitions / ≤6M 梯度步 / ≤24h wall；balance ≤12M env steps / ≤24M agent transitions / ≤2M 梯度步 / ≤8h wall。任一上限到达即停止训练，使用停止时的最后 checkpoint 验收，不挑 best；达到预算本身不是失败，未满足指标才判失败。它们是资源约束，不是「SAC 应优于 PPO」的保证；W5 必须明确末批预算调度及评估开销预留。
- **报告字段（每 seed 必报）**：首次达标的 env steps、agent transitions、梯度步数、wall-clock、终值指标；不输出 PPO↔SAC 步数对比。
- **策略验收分层**：默认策略（单分量 TruncNorm SAC actor）两任务各 3 seeds 全量验收；≥1 个结构不同族（mixture 或 state-σ 变体）两任务各 1 seed 训练级验证；八格全部完成接口/梯度/短训验证。
- **停止/失败**：NaN 或经预登记规则确认的数值发散 → 中止并留诊断快照；预算触顶且验收未达标 → fail；基础设施中断单列，不得当作成功 seed 或静默换 seed。仅凭 soft Q 超出纯奖励界不判发散。

### A2.8 同批 episode 语义对拍 fixture 设计（W3 前置，供实现期落地）

以合成 `Episode` 为主（确定性、零环境成本），另加 1 条真实录制 episode 做双路径对拍：

| Fixture | 构造 | 期望 |
|---|---|---|
| FX-1 双方 timeout | T_full=200，双 agent 提议 `("timeout",200)` | boundary=200/200；done 全 False；末转移 next_obs=final_observation |
| FX-2 单方倒地 | A 提议 `("imbalance_robot_a",80)`、B `("timeout",200)` | A boundary=80、done[-1]=True、aw_cross=φ[:80]²；B boundary=200、done 全 False；A 的末转移 next_obs=obs[80]（非 fin_obs） |
| FX-3 中途帧+退化尾帧 | physics_steps 末物理帧 delta=7、随后尾部一帧 delta=0 | 增量=7 帧计入、退化帧排除出 boundary |
| FX-4 缺必需字段 | 分别去掉 height_phi_a.phi、standing_balance_a.potential、cross_support_a | phi 缺失时 PPO/SAC 均 raise；potential/cross 缺失时仅旧 PPO 静默补零，SAC 均 raise |
| FX-5 observer 长度错 | phi 数组长度 ≠ num_frames | ValueError（沿用 coerce_per_step 语义） |
| FX-6 真实对拍 | 1 条录制的 basic_balance episode（含单 agent 倒地）分别过 PPO `build_trajectories` 与 SAC 切片构造 | `(T,r_fall,r_cross,done)` 及 post-action 参考权重相等；SAC 实际 actor 权重按 A3 动作前门控单独核对，**不要求等于 PPO 的 aw_cross** |

### A2.9 用户确认与遗留证据缺口

1. 用户在「OK，继续进行 W3」中确认 balance eval=32 和 A2.7 各预算数值；不再是待确认项。
2. 用户在 W3 选项中明确选择「动作前 actor 门控（推荐）」；该项替代 A2 对 post-action actor 权重可直接迁移的推断，见 A3.5。
3. standup 视频质量检查的执行方式在 W6 细化；固定数值门槛不后移。posture_a/b 接线保留为诊断，不新增奖励。
4. `final_observation` 对早终止 agent 不是正确边界观测；真终止掩码只消除 Bellman bootstrap 影响，不免除 next_obs 的数据正确性要求。
5. W5 仍须补齐评估时钟和实际 seed manifest 的互斥验证：仅给不同 base_seed 偏移不能证明派生 episode seeds 永不重叠。该项阻塞正式训练，不能再称「天然分离」。
6. W2 的 FX-1～6 是 fixture 设计，尚未实现或执行；真实 episode 对拍不得由本次公式验证替代。

## A3：SAC 基线与多 critic 数学裁决（W3）

**范围与状态：** 这是阶段一的数学设计，不是总体路线图的阶段三实现。基于 `f77cf82e` 工作树核查旧实现，完成公式推导、反例及 CPU 数值核验；未改任何算法代码、未启动训练。所有下述契约需在后续实现期变成永久单测。用户已确认 A3.5 的动作前门控适配。

### A3.1 本轮决策登记

| 编号 | 决策 | 理由与适用边界 |
|---|---|---|
| SAC-R1-D01 | 首版 1-step、两个独立 Q、当前策略采样、固定或自动单一 α | 标准 SAC 锚点；不继承旧 n-step/REDQ/梯度归一化默认值 |
| SAC-R1-D02 | 通道 Q 为含未来熵的 soft Q；actor 权重非负且逐样本和为 1；组合通道共同 γ、共同 bootstrap 语义 | 精确通道值中的未来熵是公共项，凸组合只计一次；不支持任意 γ/done 混用 |
| SAC-R1-D03 | **先按通道合成，再选较小的 twin；各通道 target 使用同一个 twin 索引** | 常量权重下合成的 Bellman target 精确等于标量 SAC；逐通道各取 min 不具此性质 |
| SAC-R1-D04 | basic_balance 使用动作前 φ² 的 actor 门控，原始奖励不变 | **用户已选定**；保留站稳时重视 cross 的意图，不把旧动作的结果权重用于新动作；明确是局部多目标更新规则 |
| SAC-R1-D05 | actor weight、数据有效性、bootstrap、sample weight 四者分离 | 零 actor weight 不关闭 critic；缺字段不等于无效数据；MSE 不接受负权重 |
| SAC-R1-D06 | 初版无通道梯度尺度归一化、无共享 trunk、无自动按 γ 分组 | 优先建立可解释的目标与隔离诊断；有消融证据再增加优化机制 |
| SAC-R1-D07 | γ 不同、bootstrap 不同、负 actor 权重、全零权重、n-step>1 均在首版显式拒绝 | 不把未论证的组合包装成支持；两个目标任务不需要这些组合 |
| SAC-R1-D08 | 当前历史 grad_share 不作梯度份额证据；诊断分别报告动作梯度、参数梯度、冲突及真实参数位移 | 已找到指标定义与名称不符，撤回 W1 的过强归因 |

### A3.2 单通道标准基线：定义、熵与温度

先在固定 Markov 环境中定义理论；实际机器人只提供观测 o，POMDP/self-play 限制见 A3.10。下标 t 从 0 开始，转移为 `(s_t,a_t,r_t,s_{t+1})`，N 条转移的最终状态是 s_N。`m=1-d`，d 是真终止，timeout 的 m=1；零物理增量帧不构成转移。`sg` 表示 stop-gradient。

```text
πθ：训练策略；β：采集行为策略（允许不同）；D：回放状态/转移分布
ℓθ(s,a) = log πθ(a|s)，对整条 21 维动作向量的联合密度取自然对数
H(π(.|s)) = E[-ℓθ(s,a)]，是 differential entropy，可以为负

Qπ,α(s,a) = E[r_t + Σ_{k≥1} γ^k (r_{t+k} - α ℓθ(s_{t+k},a_{t+k}))]
a' ~ πθ(.|s')
y = sg[r + γ m (min_j Qbar_j(s',a') - α ℓθ(s',a'))]
L_Qj = mean[(Q_j(s,a_replay)-y)²]
a_new ~ πθ(.|s)
L_actor = mean[sg(α) ℓθ(s,a_new) - min_j Q_j(s,a_new)]
```

Q 不包含当前给定动作的熵，actor 中当前熵恰好一次。target 一律无梯度；actor 步冻结 Q 参数但**保留 Q 对动作的导数**，不可把整个 Q forward 放进 no_grad。π 的采样和 log_prob 必须对应同一归一化动作分布：TruncNorm 包含截断归一化常数；tanh 族包含变换 Jacobian；mixture 用联合混合密度而非被选中分量的密度（梯度估计由 W4 验证）。不使用 PPO ratio/GAE，也不把 β 的探索上下文带进 π 的评分。

温度约束 `E_D[H] ≥ H_target`，令 `η=log α`：

```text
L_α_exact(η) = exp(η) · sg(E_D[-ℓθ] - H_target)
首版采用常见 log-temperature surrogate：
L_η = - E_D[η · sg(ℓθ + H_target)]
∂L_η/∂η = E_D[H] - H_target
```

两式驻点与调整方向相同，梯度大小相差 α，**不是同一个优化器轨迹**。实测熵低于目标 → 梯度下降增大 η/α；高于目标 → 减小。温度步不回传 actor；actor/critic 不回传 η。自动温度使用与 actor 相同的有效状态集合及归一化 sample weights（默认全 1）。固定 α 模式不创建温度 optimizer；α=0 仅作显式无熵对照，不取 log(0)。

归一化动作立方体 [-1,1]^21 上理论最大熵为 `21 ln 2 ≈14.556 nats`；参数受限的八格策略未必达到它。不得无依据把 uncertainty 当 H，或直接继承 `H_target=-21`、旧 α 下限。首个 oracle 用固定 α；自动温度目标及分布可达范围由 W4 确定。数值非有限即 fail；任何后续 α 边界都要登记为约束并报告饱和，不用静默 clamp 掩盖不可达熵目标。

### A3.3 多通道 soft-Q 方案与常量权重证明

设同一 actor cohort 有 C 个通道，各通道两套独立网络 `Q_{c,1},Q_{c,2}` 和 target，pair 索引 j 在所有通道间一致。默认不共享参数，两个 pair 也不共享。共同 γ∈[0,1)、同一 transition 的 m 在 cohort 内一致；原始 reward 不因 actor 门控改写。

定义 `g_c(s)≥0, Σg_c(s)>0`，`w_c(s)=g_c(s)/Σg(s)`。权重是固定实验函数，不对 θ 或新动作求导。常量配置使用同一归一化规则；总体奖励单位通过显式固定 `reward_scale>0` 定义（基线=1），而不是靠把全部 g 乘常数偷偷改变。

```text
F_j(s,a;w) = Σ_c w_c(s) Q_{c,j}(s,a)
j_next = argmin_{j∈{1,2}} Σ_c w_c(s') Qbar_{c,j}(s',a')
y_c = sg[r_c + γ m (Qbar_{c,j_next}(s',a') - α ℓθ(s',a'))]
L_Qc,j = 有效数据上的加权 MSE(Q_{c,j}(s,a_replay), y_c)

j_actor = argmin_j F_j(s,a_new;w)
L_actor = E_D[α ℓθ(s,a_new) - Σ_c w_c(s) Q_{c,j_actor}(s,a_new)]
```

所有通道使用同一个 a' 和联合 log_prob；actor 同理。tie 固定选 pair 1；不对 argmin 索引反传，在非切换点沿选中分支求导。终止行 m=0 不需计算 next policy/Q，避免 `0×NaN`；timeout 则用真实边界 next_obs。

**标准退化证明（权重 w 恒定时）：** 令 `r_w=Σw_c r_c`，因为 `Σw=1`、γ/m 相同，

```text
Σ_c w_c y_c
 = r_w + γ m [Σ_c w_c Qbar_{c,j_next}(s',a') - α ℓθ(s',a')]
 = r_w + γ m [min_j Σ_c w_c Qbar_{c,j}(s',a') - α ℓθ(s',a')]
```

这恰是标量 r_w 的 clipped-double-Q SAC target；actor 合成式同样成立。C=1、g>0 → w=1，退化为标准单通道。这里只证明**target 与 actor 目标的代数等价**；多网络逐通道 MSE 不等于对 F 的单个 MSE，因此不保证训练轨迹与单个标量网络相同。

精确 policy evaluation 下，每个通道 `Q_c = R_c + E_entropy`；公共的未来熵项因凸组合恰计一次。实际不同近似器会有不同熵估计误差，不能据此宣称数值上完美分解，也不能把单个 Q_c 称作纯任务 return。所选 Q_c 分支未必是该通道最小值；悲观操作针对 F 而非每个通道分别生效。

### A3.4 其他候选与明确不采用的等价说法

| 候选 | 判断 |
|---|---|
| 各通道纯任务 Q + actor 当前熵 | 缺少未来熵对当前行动的价值，α>0 时不能标准退化；不作为默认 |
| 纯任务 Q + 独立熵 continuation critic | 可以另行设计，但增加网络及 α 变化下的记账要求；两个任务暂不需要 |
| 各通道各取 min 再加权 | 取最小与求和不交换，不能称为标量 twin-SAC 的相同 target |
| 未归一化的 soft Q 直接 3:1 求和 | 未来熵变为 4 倍，但 actor 即时熵仍一次；不采用 |
| 把 actor 权重乘 reward 再学 Q | 是另一种可定义目标，但不是只调制当前 actor 的规则；用户未选择 |
| 用归一化动作梯度替代 Q 单位 | 改变了目标、相对 α 的尺度及历史依赖；作为以后消融，不进可信基线 |

### A3.5 basic_balance：用户已确认的动作前门控

**被替代的旧假设：** `w_post` 来自 β 执行 a_old 后的 s'_old；在同一 s 上新抽 a_new，不能假定其结果仍是 s'_old。简单 detach 不消除此偏差：它只切断导数，不改变权重来自旧行为结果的事实。即使 β=π，独立的旧/新动作也丢失了结果与动作的相关性；importance ratio 不是这里可直接套用的补丁。

登记的实际规则：

```text
φ_pre  = clip(与 obs_t 同时刻的 height · uprightness / 1.28, 0, 1)
φ_post = clip(当前 transition 的 HeightPhiObserver.phi, 0, 1)
g_fall(s_t) = 3
g_cross(s_t) = φ_pre²
w_fall = 3/(3+φ_pre²)，w_cross = φ_pre²/(3+φ_pre²)
r_fall = 0.01 φ_post，r_cross = 原 CrossSupportBalanceRewarder 输出
下一状态 target 分支的权重使用 φ_post，即 s' 在 a' 之前的门控。
```

`φ_post²` 另以 `actor_weight_post_reference` 保存，只用于对拍和时间差诊断；不能在 trainer 里作为实际 actor 权重的后备值。φ_pre=0 时 cross 不贡献该样本的 actor Q 项，但其 critic 仍学习；不把 g=3 解释为 75% 的实际参数梯度。

**采集契约交给 W5 实现：** SAC 自有采集器在取 obs 时，同一快照经正式 accessor 取门控事实并记录 `phi_pre/phi_post`、时序与版本；无需改环境/PPO。不从归一化 obs 的未知下标猜 φ。普通连续 episode 可用 `[initial_phi, phi_post[:-1]]` 做独立一致性校验，但不能对缺数据静默移位补齐；重置插件顺序、额外状态改写和退化帧必须先核验。有效末转移的 φ_post 与 next_obs 必须对应同一物理时刻。

**不能宣称的等价性：** 状态变化后 w 也变。`Σw_c(s) Q_c(s,a)` 使用当前偏好评价各通道未来 return，不等于沿未来每一步使用 `w(s_k)` 的奖励和；也不保证存在固定标量奖励的全局策略改进定理。动态门控是明确标识的 **local_actor_gate** 扩展；常量权重是 **constant_scalarization** 基线。分别记录 objective_mode；不能用常量模式的证明为动态模式背书。

验收保留环境、初始化、原始奖励、边界和评估；actor 时间点的变化是用户接受的 SAC 适配，**不是 PPO 训练目标完全等价**。FX-6 对拍只比较原始任务事实及 post-action 参考权重，SAC 的实际 w_pre 另设测试。

### A3.6 四种权重/掩码及支持矩阵

令 i 为 batch 行、v_ic 是显式通道有效性、b_i 是非负有限 sample weight（基线=1）：

```text
L_Qc,j = Σ_i b_i v_ic (Q_c,j - sg(y_ic))² / Σ_i b_i v_ic
actor 可用行 A = 所有 cohort 通道有效的行（首版保守策略）
L_actor = Σ_{i∈A} b_i [α ℓ_i - F_min,i] / Σ_{i∈A} b_i
```

v 表示 reward/边界语义可用；m 表示能否 bootstrap；w 只表达通道偏好；b 改变样本分布，不是 β/π importance ratio。它们的作用点不得互换。字段必须完整且 shape/dtype/finite 合法，显式 v=0 不等于允许缺 observer、NaN 或坏长度进入 batch。

| 组合/边界 | 首版行为 |
|---|---|
| 单通道、正 g、1-step | 标准 SAC，g 的总体倍数被归一化掉 |
| 多通道常量非负 g，共同 γ/m | 支持；满足 A3.3 代数对照 |
| 动态动作前 g(s) | 支持 local_actor_gate，明确局部更新语义及 POMDP 限制 |
| 动作后/执行动作相关 g(s,a_old,s'_old) 用于新动作 | 拒绝作为 actor 门控；仅保留历史标签 |
| 某通道 w=0、v=1 | critic 正常学；actor 不贡献该通道 |
| 某行所有 g=0 | ValueError；不靠 ε 除法悄悄转为熵-only 训练；两目标任务始终有正权重 |
| g<0 或非有限 | ValueError；可将成本预定义为负 reward（新语义版本），但不允许负 MSE 权重 |
| 显式某通道 v=0 | 该通道 critic 不用此行；该行不进 actor/α；其他有效通道 critic 可用。记录有效样本计数 |
| 某通道有效加权分母=0 | 跳过该通道 optimizer 及其 target 更新，不能让 Adam 动量产生空数据更新；不计该通道成功 step |
| actor 无有效加权行 | 跳过 actor/α，不伪造成功 step；全 batch 无可学习数据则 fail loud |
| 缺 required 通道、observer 或 shape 不符 | raise；不能用 v=0 自动吞掉 |
| 不同 γ、不同 bootstrap mask 的同一 cohort | 配置/数据校验拒绝；不静默按 γ 分组或删除熵项 |
| n-step>1、REDQ/PER、relabel、共享 trunk | 初版不支持；对应配置必须拒绝，不留接受但不生效的旋钮 |

为使 target 的合成定义完整，首版所有 cohort Q 网络始终存在；缺少某行通道测量不意味着删除网络。两个目标任务的每个合法转移均应全通道有效；v 的边界测试用于检验隔离，而非掩盖实际采集错误。

### A3.7 更新顺序、梯度隔离与时钟

一次成功 critic tick k：

1. 校验 batch/权重，冻结本 tick 的 θ_k、α_k、门控版本及 target 快照。用当前 πθ_k 在 s' 抽 a'，构造无梯度的 y_c，缓存选中 twin、log_prob、m 和逐通道 target。
2. 清零 critic 梯度，按通道独立 MSE 反传并优化；初版梯度裁剪阈值为显式配置，记录裁剪前后范数。actor/温度均不得有新梯度。
3. 默认每个 critic tick 做一次 actor tick：冻结已更新 critic 参数（不是 action 图），以 πθ_k 在 s 抽 a_new；用合成 F 和 α_k 计算 loss，更新 θ。Q 不更新、不积累 actor 步梯度。
4. 自动温度复用步骤 3 的**更新前 actor** log_prob，detach 后更新 η；使用相同 actor 有效行/b_i，不重新采样，不偷看 θ_{k+1}。固定温度则跳过。分别记录 α_used=α_k 与 α_after。
5. 成功优化的 critic 执行 `Qbar ← (1-τ)Qbar + τQ_online`，τ∈(0,1]；时钟为 critic optimizer tick，不是 env step/rollout round/actor tick。暂停或无效通道不更新其 target；增加延迟 actor 更新时仍按这个时钟。

保存 critic/actor/temperature/target 四类计数；默认都是每 critic tick 一次，固定 α 除外。checkpoint 恢复这些时钟，避免 target 半衰期或探索调度改变。actor/target 两次采样使用可恢复的独立 RNG 流；诊断只复用缓存或 fork RNG，不改变训练流。W5/W6 细化序列化与捕获字段。

### A3.8 反例：支持边界为什么必要

1. **min 不可交换**：pair1 的两个 Q 为 `(0,10)`，pair2 为 `(10,0)`，w=(0.5,0.5)。`Σw·min_each=0`，`min_pair ΣwQ=5`；独立各取最小拼出任何 twin 都没预测过的组合。
2. **未来熵重复**：零 reward、γ=0.9、α=1、后续状态熵恒 1.7，则每通道 Q=15.3。soft Q 直接 3+1 求和变成 61.2；凸组合仍是 15.3。旧代码非 grad_norm 分支已有 L1 归一化，故不能笼统指控它在正恒权重下必然重复计熵；问题要按具体分支判断。
3. **旧动作结果不是新动作门控**：单状态两动作 0/1，Q_cross(a)=a，结果 φ=a，fall Q=0，归一化 cross 权重为 0/0.25。π(1)=p、β(1)=q，当前动作结果目标为 `0.25p`，旧结果乘新 Q 的期望为 `0.25qp`；对 p 的导数分别 0.25 和 0.25q。q=0.25 时差 4 倍。此例说明相关性丢失，不是在宣称 PPO 本身恰优化这个标量目标。
4. **actor 门控不是奖励门控**：s0 偏好=(1,0)，下一状态 s1 偏好=(0,1)，s1 只有通道2奖励10，γ=0.9。当前偏好合成纯任务未来 Q 得0，按各时刻门控奖励回传得9；加公共熵不消除差异。
5. **负权重翻转悲观方向**：Q1=1、Q2=3，`-min(Q1,Q2)=-1`，但负目标的悲观估计是 `min(-Q1,-Q2)=-3`。signed actor 权重还破坏本方案熵系数和=1的证明，因此初版直接拒绝。
6. **不同 γ/终止破坏公共熵项**：零 reward、相同熵，γ1=0.9 时未来熵=9αH，γ2=0.5 时=αH；一通道终止而另一通道继续也不共享 entropy continuation。不能继续用 A3.3 的公共项证明。
7. **缺有效样本不等于零目标**：v=0 应无该通道梯度；补 reward=0 并回归会真的把 Q 拉向0。w=0 则相反，应保留该通道有效学习。

### A3.9 旧实现审计与新的诊断语义

代码事实（均为 W3 基线版本的相对路径/行号）：

- `trainer.py:181-203` 给每个通道 target 加熵；`215-227` 用 actor_weight 加权 critic MSE，会把结果门控变成 critic 训练分布；零权重停止学习，且 signed 配置不再是合法非负回归。
- `trainer.py:264-301`：scale 来自 `mean_batch ||∂Q/∂a||` 的时间 EMA 二阶统计，不是 `sqrt(E_batch ||g||²)`；旧 grad_share 是 `(mean(aw)/scale)² / Σ(...)²`，没有乘上梯度、actor Jacobian 或逐行归一化分母。
- 反例：两正交动作梯度范数为1/100，aw=3/1，scale=1/100。旧 proxy 的 cross 比例≈0.00001111，而实际加权归一化后的动作梯度平方范数比例为0.10；参数空间再经 actor Jacobian 后还会变化。**旧 grad_share 不能解释成实际梯度占比。**
- `trainer.py:287-291` 用 `w_c/scale_c` 乘含熵 Q，使未来熵系数成为 `Σw_c/scale_c`，而即时熵仍 α；即使单通道，scale≠1 也不满足本轮标准退化目标。这是可定位的公式偏差，不等于已经找到 v7 失败的因果原因。
- `trainer.py:317-340` actor 接受 batch_weights、α 未用相同权重；`331-340` 自动默认 -action_dim 并 clamp；`319-324` 未冻结 critic 参数，actor backward 会给 critic 累积梯度（下一次 zero_grad 清除，不等于这些梯度被 optimizer 应用）。
- 旧 tests 主要检查更新可运行、字段存在和 proxy 和≈1；不能证实梯度份额、熵分解或配置支持正确。

新 debug 规范：记录 g/w 分布、每通道 Q/target/TD 残差、pair 分歧及 j 的选择率；熵即时项 αlogπ 与 Q 总项分别报告。通道贡献使用**共同选中分支**：

```text
G_c = ∇θ E_D[-w_c(s) Q_{c,j_actor}(s,aθ)]
G_H = ∇θ E_D[α logπθ(aθ|s)]
G_total = Σ_c G_c + G_H（裁剪之前）
```

同时测动作梯度范数、G_c 参数范数、成对 cosine、裁剪前后总范数、优化器后的 `Δθ`。范数份额只表示该定义下的范数比例，向量有相消，不能称作可加的任务贡献百分比；Adam 的参数位移也不能按各 loss 单独更新后相加。Q 的纯奖励界不能直接当 soft Q 上界，尤其连续熵可负、α 在变化。

### A3.10 replay、self-play 和部分观测的限制

- 标准 1-step SAC 允许 β≠π，但依赖转移/奖励机制固定和覆盖充分；不是「任意旧数据都同样可信」。本项目 self-play 的对手也更新，过去对手使有效转移核变化。记录本方/对方 policy_version、样本年龄、reward/objective_version；按 age/version 分桶分析 TD 与评估。容量/保留窗口由 W5 决定，不静默把旧样本当当前 MDP 的精确样本。
- 96 维观测不包含 CrossSupport rewarder 的 FSM 计时器，物理状态也不完整；相同 o 可能有不同回报历史。上述 Markov 推导是理论锚，Q(o,a) 是部分观测近似，不保证 Bellman closure。动作前门控若使用未包含在 o 的状态事实，也是显式的训练侧辅助信息，不谎称完全可由 o 唯一恢复。若因此训练失败，再以独立版本评估增加历史/观测，不偷偷更改两个基准实验。
- timeout bootstrap 是继续任务的学习约定，200 步是数据/评估截断，并非把任务价值在10秒强制归零；不跨 reset 取 next_obs。两个 agent 不算两倍 env step，分别登记有效 agent transitions。
- 初版 n=1。n>1 会引入 β 中间动作的 off-policy 偏差，soft return 还需正确处理中间熵项（本 Q 定义下 k=1...n-1 的熵以及末端 bootstrap 熵）；旧「奖励累加+末端熵」不是自动正确扩展。另立方案与测试后才支持，不类比 GAE λ。

### A3.11 本次验证证据与实现期测试门槛

**已执行：** CPU、torch float64、固定 seed=314159、1线程；两次内存公式探针共17项断言（M16 两个分支），没有写入 replay、加载训练 checkpoint 或运行 MuJoCo。

| 探针 | 输入/判据 | 实测 |
|---|---|---|
| M01～03 | C=1/2/5、B=31，随机 Q/r/logp；w∝1...C，α=.17、γ=.99、每3行一条 terminal；比较 Σw y_c 与标量 SAC y | 最大误差 0 / 2.220e-16 / 2.776e-16 |
| M04～05 | A3.8 的 min 交换及零 reward 熵计数反例 | 0≠5；15.3≠61.2 |
| M06～07 | H_target=-2，H=-3/-1，η=-1；检查温度梯度及 policy detach | dL/dη=-1/+1；log_prob 无梯度 |
| M08～09 | 旧动作后门控反例 q=.25；动态偏好两状态反例 | 梯度 .25≠.0625；价值 0≠9 |
| M10 | 正交 g=(1,100)、aw=(3,1)、scale=(1,100) | proxy_cross=.00001111，动作梯度平方范数比例=.1 |
| M11 | Q_pred=(2,4)、target=1、v=(1,0)、b=(2,1)，actor 权重不进入 MSE | critic grad=(2,0) |
| M12～14 | signed 悲观方向；Q(a)=2a、a=tanh(.3) 冻结 Q；target初值1向online=3按τ=.1更新4次 | -1≠-3；actor grad=-1.83027392/Q参数无梯度；target=1.6878 |
| M15 | P=((.7,.3),(.2,.8))、r=((1,-2),(3,.5))、H=(.4,1.1)、γ=.9、α=.2、w=(.75,.25)，解线性 Bellman 方程 | soft returns 合成误差1.776e-15 |
| M16 | 固定样本 actor surrogate，single/constant 两配置；中心差分ε=1e-6，误差<1e-8 | autograd/差分分别为 -2.32、-1.47 |

核心对照可在项目目录以 `python3 -B` 执行以下自包含代码复现（只依赖现有 torch）：

```python
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(314159)
for c in (1, 2, 5):
    b = 31
    w = torch.arange(1, c + 1, dtype=torch.float64)
    w /= w.sum()
    q, r, lp = torch.randn(b, 2, c), torch.randn(b, c), torch.randn(b)
    m = (torch.arange(b) % 3 != 0).double()
    f = (q * w).sum(-1)
    j = f.argmin(-1)
    y = r + .99*m[:, None]*(q[torch.arange(b), j] - .17*lp[:, None])
    scalar = (r*w).sum(-1) + .99*m*(f.min(-1).values - .17*lp)
    assert ((y*w).sum(-1) - scalar).abs().max() < 1e-12
p = torch.tensor([[.7, .3], [.2, .8]])
r = torch.tensor([[1., -2.], [3., .5]])
h, w = torch.tensor([.4, 1.1]), torch.tensor([.75, .25])
a = torch.eye(2) - .9*p
q = torch.linalg.solve(a, r + .9*.2*(p@h)[:, None])
scalar = torch.linalg.solve(a, r@w + .9*.2*(p@h))
assert (q@w - scalar).abs().max() < 1e-12

def objective(t, w):
    q = torch.stack((torch.stack((2*t+t*t, -t+.2)),
                     torch.stack((t+t*t+.5, -2*t+1))))
    return .2*(-.3+t*t) - (q*w).sum(-1).min()

for w in (torch.tensor([1., 0.]), torch.tensor([.75, .25])):
    t = torch.tensor(.2, requires_grad=True)
    grad = torch.autograd.grad(objective(t, w), t)[0]
    fd = (objective(t.detach()+1e-6, w)-objective(t.detach()-1e-6, w))/2e-6
    assert abs(grad-fd) < 1e-8
print('SAC scalarization and actor-gradient formula probes passed')
```

**旧测试回归：** `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider baseline/framework/sac/tests` → **16 passed**，只有 pynvml deprecation warning。这是旧实现回归，不是新契约测试通过。

**实现期永久测试必须新增：** 上述公式及反例；所有拒绝组合的异常；零 w 不冻结 critic；v=0 和空 batch 的 optimizer/target 时钟；actor/Q/α/target 梯度隔离；动作前/后门控时刻；同批 episode 的 FX-1～6；有限噪声下连续训练/恢复及 debug 开关一致性。单分量/mixture 的真实分布梯度由 W4 设计，不能用 M16 的解析 Q 替代。

### A3.12 W3 出口与后续边界

- [x] 单通道公式、温度方向、密度单位、梯度隔离、更新/target 时钟已明确。
- [x] 多通道 Q/熵记账、常量标准退化、动态门控非等价性均有公式和反例。
- [x] 支持矩阵覆盖零/负权重、缺失/无效通道、不同 γ/m、n-step；不支持项 fail loud。
- [x] 动作前 actor 门控已获用户确认，W2 对拍预期同步修订；PPO/环境不改。
- [x] 数学探针和旧测试已执行；没有声称新 trainer、策略或机器人任务已验证。

**W3 无新增待用户裁决项。** W4 接续解决八格重参数化/mixture 梯度及可达熵；W5 落实门控事实采集、replay/version/时钟、评估 seed 互斥。历史 v7 失败归因仍未确定，需要实现期新诊断和受控消融。共享 trunk/梯度归一化等优化保持候选，不自动继承。

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
