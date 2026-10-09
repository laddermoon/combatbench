# SAC 阶段一详细计划：设计边界与验收口径

> 状态：阶段一 W1–W8 已完成，对应 A1–A8 见下方裁决区；阶段一设计边界已获用户批准，可进入阶段二实现。Shannon 基线与用户批准的 uncertainty 替代路线已分开定义。算法实现与真实任务训练尚未开始。
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
| A8 | 阶段一收口与用户批准 | 历史决策如何处置，阶段二入口条件是什么 |

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

#### W5 执行拆解与产物顺序

**边界：** W5 只做契约设计、代码事实核查和必要的低成本 fixture/序列化探针；不改生产训练路径、不启动训练。所有结论以 A5 落文，接口名可再改，语义不能只靠调用顺序暗示。

| 子项 | 工作内容 | 主要产物 / 门槛 |
|---|---|---|
| W5.0 现状链路复核 | 沿 `train.py → ExperimentSAC → rollout/job → Episode → build_slices → replay → trainer → eval/checkpoint` 画出真实调用链；列出仍存在的 PPO 依赖、死契约和旧字段。 | 数据流图与缺口表；不得把旧 `Job`/`TrajectorySlice` 当成已批准接口。 |
| W5.1 职责边界 | 明确实验侧拥有环境蓝图、job/seed、任务事实、评估、agent 切片、通道语义、数据准入；框架侧拥有采集执行、transition 校验、replay、更新、调度、导出、checkpoint。 | 责任矩阵；每个字段/动作只有一个 owner；未声明实验能力显式 unsupported。 |
| W5.2 transition schema | 冻结逐 agent transition 草案：`obs_t/action_t/next_obs`、逐通道 reward/valid/terminated/bootstrap、`phi_pre`、`phi_post_reference`、termination reason、physics delta、样本来源与版本、行为策略/探索参数。标注 dtype、shape、单位、允许缺失性和时间对齐。 | 字段表 + `TrajectorySlice`/batch 边界草案；缺必需字段 fail loud，`phi_pre` 不从归一化 obs 猜下标。 |
| W5.3 replay 契约 | 定义容量单位、插入顺序、覆盖语义、均匀采样 RNG、稳定 `sample_id`、trajectory/source metadata、age/staleness 统计、batch 返回内容。首版 n-step/PER/stratify 是否支持须显式裁决。 | replay 写入/覆盖/采样契约；环形覆盖不得让同一 `traj_id,traj_step` 指到别条轨迹；身份不等于数组下标。 |
| W5.4 时钟与 UTD | 分别定义 env action steps、physics steps、episodes、rounds、有效 agent transitions、requested/actual critic steps、actor/temperature/target ticks、eval/export/checkpoint ticks；写清 UTD 分子分母、小数累计、warmup 与上限处理。 | 时钟表 + round 内事件顺序；请求值和实际生效值都可记录、恢复。 |
| W5.5 配置与状态分层 | 区分 run config、experiment static semantics、可变 schedule state、runtime counters；定义配置指纹和语义版本。热参数修改必须显式作用域与生效时点。 | config/state manifest 草案；不允许 checkpoint 静默覆盖当前 config 或被当前 config 静默覆盖。 |
| W5.6 exact resume 与 warm-start | 拟定 checkpoint manifest：actor、critic online/target、全部 optimizer、α/λ 及 optimizer、replay 内容/游标/source ID/采样 RNG、行为/eval/训练 RNG、计数器、调度器、实验 state、代码/环境/格式版本。 | 状态清单和兼容性矩阵；exact resume 要定义同设备/内核确定性条件，格式不符直接报错；warm-start 明确“加载了什么、丢弃了什么”。 |
| W5.7 采集与入口边界 | 决定 SAC 自有 rollout/job 的最小接口、policy export 版本、worker seed 派生、失败传播、episode 返回数据的最小集合；保证不引入 `SamplingContext`、ratio、GAE 等 PPO 对象。 | SAC collection contract 草案；并满足 C1–C3 独立性边界。 |
| W5.8 验证设计 | 把 A2 的 FX-1～6 扩展为 W5 fixture：缺字段、边界 next_obs、replay wraparound、乱序 episode、重复/冲突 sample_id、恢复状态缺失、跨版本 resume。每项指定输入、期望、失败含义和所属实现阶段。 | W5 测试矩阵草案；设计期可先做纯 Python/合成 Episode 探针，不把 fixture 通过等同于生产实现完成。 |
| W5.9 A5 汇总评审 | 合并字段表、责任矩阵、时钟表、replay 契约、恢复清单、接口草案和未决项；按“阻塞阶段二 / 后续阶段 / 可选增强”分类。 | A5 决策稿；列出必须由用户确认的默认口径与资源/存储取舍。 |

**必须先裁决的问题（进入实现前不能留默认值）：**

1. replay 容量和 UTD 是否统一以**有效 agent transition**计；推荐如此，但需确认。
2. 每个样本是否强制携带行为策略版本、探索参数和最低限度采样元数据；推荐强制，训练不用也可供 debug/审计。
3. 首版是否禁止 n-step/PER/relabel/stratified retention，并把相关配置显式拒绝；推荐禁止未验证项。
4. exact resume 是否默认持久化完整 replay；推荐是，另设显式 warm-start/no-replay 模式。
5. `phi_pre` 的来源是 observer 显式输出还是由 episode 前向构造；必须能从同一状态快照取得，且缺数据报错。
6. checkpoint 恢复是否允许覆盖当前学习率/调度/实验名；推荐默认禁止，除非用户在 warm-start 中显式声明。

**出口：** A5 包括接口草案、字段表和状态清单。接口名允许在实现前微调，含义不能依赖隐式时序或无声默认。

### W6：从第一版就可诊断的捕获契约

**执行顺序与边界：** 本工作包仍只做设计和最小验证，不移植 PPO dumpkit、不改生产训练路径、不启动训练。PPO `dumpkit`/`debug.py` 只作为能力参照；SAC 的数据模型是 collection round + 大量 critic tick + replay minibatch，不能沿用 PPO 的「一个 update = 一批 on-policy trajectory + 若干 epoch」截面。

#### W6.0 核对 PPO debug 能力与 SAC 差异

以 `ppo/dumpkit/CONTEXT.md`、`DATA_FLOW.md`、`dump_capture.py`、`dump_analysis.py` 为证据，列出可参照的架构能力：run 总览、指标目录、scheduled/on-demand dump、episode/trajectory 下钻、单样本溯源、离线重算、CLI/HTTP/UI 同一份分析函数。逐项判定「架构思想采用 / 字段语义改写 / 不采用」；不复制 GAE、ratio、clip、epoch early-stop 等 PPO 专有意义。

#### W6.1 冻结 SAC 诊断因果链

把首版分析链固定为：

```text
CollectedEpisode → TransitionSlice → Replay admission/overwrite
→ sampled minibatch(sample_id/source_key)
→ Bellman target 分解 → critic update
→ actor objective 分解（Q、Shannon/U、gate）
→ temperature/target-network update
→ post-update policy/export delta → deterministic eval
```

每个环节给出输入、输出、可观测字段、失败时归因点；明确 eval 不写 replay、诊断不改训练状态。

#### W6.2 定义指标命名空间与双时钟 x 轴

设计 `__RAW_STATS__`/metrics schema：至少区分 `round.*`、`collect.*`、`replay.*`、`batch.*`、`target.*`、`critic.*`、`actor.*`、`regularizer.*`、`update.*`、`eval.*`、`cost.*`、`debug.*`。同一指标必须标注主时钟（`collection_round`、`env_step`、`agent_transition`、`critic_tick`）和次要时钟；不允许把 env step 与 agent transition 混作 UTD 或 replay 覆盖率分母。

#### W6.3 定义常态指标、tick 聚合和捕获分层

裁决四类开销：

- **L0 常态标量**：每 round/评估必备，成本接近零；
- **L1 常态结构摘要**：每 round 的 replay 年龄/来源/通道/行为参数分布和更新统计；
- **L2 预约截面**：指定 `collection_round`、`critic_tick` 或 tick range，保存足以重算该截面的 batch/model/optimizer/RNG/config；
- **L3 重型诊断**：逐样本梯度、逐通道贡献、action Jacobian、mixture 分量分解、更新前后参数 delta、完整或抽样 replay 快照，仅 dump-only。

给出各层触发方式、保存目录、保留策略、默认关闭项和成本预算。

#### W6.4 定义 SAC cross-section dump schema

一个 dump 不能只叫 `uNNNNN`；设计以 `(collection_round, tick_range)` 或单个 `critic_tick` 为主键的目录/manifest。内容至少覆盖：collection/job provenance、admitted transition manifest、sampled batch 的 `sample_id/source_key` 和必要字段、行为策略 fingerprint、actor/critic/target/regularizer 版本与参数状态、Bellman target 分解、Q/TD/actor 各项、α/λ、optimizer、RNG、EffectiveConfig、更新前后摘要。

#### W6.5 定义离线重算与证据等级

区分三档：

1. **解释级**：仅展示已保存标量/数组，不声称重算；
2. **逻辑重算级**：相同 schema/代码下对同一 batch 重新计算 target/loss，给数值容差；
3. **bitwise 级**：仅在 config-lock、同代码、同设备、同库版本、确定性内核和完整 RNG 下追求。

裁决每个截面必须保存多少模型状态与采样噪声；历史 tick 若未预约且未 checkpoint，只能解释级，不伪造可重算性。

#### W6.6 定义样本溯源与回放边界

基于 A5 的 `sample_id/source_key/slice_id` 建立从 batch 样本回到 collection round/job/agent/frame 的查询；dump 必须自包含所选样本，即使 replay 已覆盖。区分：符号字段回放、基于 `core_state` 的物理重演、视频渲染；首版只承诺哪些可离线验证，哪些需要 checkpoint 中额外保存状态，缺件显示不可用而不是补零。

#### W6.7 定义诊断隔离、安全和资源约束

诊断使用独立 RNG stream，不消耗训练采样/训练更新 RNG；除被明确标记的 dry-run 外不修改模型、optimizer、replay、schedule。规定 worker 内诊断禁项、显存/内存上限、序列化失败处理、慢诊断对训练吞吐的测量方式，以及「诊断失败是否终止训练」的 fail-loud/fail-isolated 边界。

#### W6.8 定义 CLI/分析函数/UI 数据面

冻结 SAC 自有 metric catalog、dump manifest schema、离线分析 API 的形状；CLI、HTTP、viewer 必须调用同一分析实现，不允许多份解析。阶段二最小出口可只交付结构化 JSON/CLI；阶段五再完成 viewer，但字段与身份从首版就不能缺。独立性检查要求 SAC debug 不 import `ppo.dumpkit` 或 PPO buffer/trajectory 类型。

#### W6.9 设计故障注入与验证矩阵

至少覆盖：bootstrap/terminated 错、timeout 错、缺 observer/fact、source_key 冲突、replay 覆盖后 dump 溯源、陈旧 policy 版本过多、采样 RNG 恢复、α 失控、Q/TD 异常、actor gate 全零、通道压制、mixture logits 无梯度、U floor/bonus 误接、optimizer mismatch、checkpoint bundle 缺件、诊断 RNG 污染、dump 重算不一致。每项给出注入方法、预期可见指标、应定位到的模块和所属阶段。

#### W6.10 汇总 A6 与阶段二门槛

形成 A6：最小捕获 schema、触发与保留、身份映射、重算等级、开销分层、CLI/UI 同源边界、故障矩阵。列出必须先完成的决策项；只有 A6 能让首版回答「这次 critic tick 为什么得到该 target/loss、用了哪些历史样本、这些样本来自哪里」。

**必须先裁决的问题（进入实现前不能留默认值）：**

1. dump 主键采用 `collection_round`、`critic_tick`，还是 `(round, tick_range)`；推荐支持单 tick 与显式 tick range，不把所有 round 内更新压成一个截面。
2. 常态日志是每 tick 落盘、内存 ring buffer、还是按 round 聚合；推荐 round 聚合 + 最近 N tick ring + dump 时才完整落盘。
3. L2 截面是否默认保存 optimizer state；推荐保存，否则无法声明可恢复/复算更新。
4. dump 默认保留数量与磁盘上限；推荐 latest-N 加显式 pin，后台训练不得无限增长。
5. 是否要求首版支持物理重演；推荐先支持符号字段与可选渲染，逐帧物理重演只在保存 core-state 的 dump 中承诺。
6. 诊断失败边界：数据/schema 类失败 fail loud；重型可视化或离线渲染失败是否允许隔离失败而不终止训练，需要确认。

**出口：** A6 有最小 schema、捕获时点、来源映射、重算等级、开销分层及测试门槛；首版即能回答一次指定更新为什么得到该 target 和 loss，并能把所用样本追回 collection/job/agent/frame。

### W7：建立风险与验证矩阵

**执行顺序与边界：** W7 是阶段一的验证设计与实施拆分收口，不补写生产测试、不启动训练。输入是 A1–A6 的代码事实、数学裁决、数据/恢复契约和诊断契约；输出 A7 必须能让阶段二按依赖顺序开工，而不是把风险写成不可执行的原则。

#### W7.0 汇总决策与证据索引

从 A1–A6 抽出全部 `SAC-R1-D*`、FX/DX fixture、阶段二包和已知代码缺口，建立唯一 ID 索引。每项记录：决策内容、适用边界、证据类型（代码事实/运行实测/数学推导/用户裁决/设计假设）、当前验证状态、关联文件与行号。发现编号冲突、遗漏或语义冲突时先修正文档，不进入矩阵。

#### W7.1 冻结风险矩阵 schema

每条风险使用统一字段：

```text
risk_id, decision_refs, contract, failure_mode, blast_radius,
likelihood, detectability, severity, validation_level,
test_id/fixture, input_or_injection, expected, tolerance,
evidence_artifact, owner_phase, gate, status
```

`severity` 至少区分：blocker（使目标错误/不可恢复）、major（可能错误训练但可检测）、minor（诊断/可用性问题）。`validation_level` 分层为：静态/schema、单元、property、集成、smoke、短训、多 seed 正式验收。

#### W7.2 按风险域枚举，而不是按文件枚举

至少覆盖十个风险域：

1. **MATH**：Bellman target、熵/U 记账、梯度隔离、温度方向、target update。
2. **POLICY**：S01 数值边界、actor ABC、导出/私有 RNG、未来 mixture logits。
3. **DATA**：transition schema、时间对齐、terminated/truncated/bootstrap、phi_pre/post。
4. **REPLAY**：sample_id/source_key、FIFO 覆盖、uniform sampling、RNG 与持久化。
5. **COLLECT**：SACJob/CollectedEpisode、worker 失败、行为参数、PPO 独立性。
6. **CLOCK**：collection round、agent transition、各类 tick、UTD credit、eval/checkpoint 边界。
7. **STATE/RESUME**：bundle、manifest、白名单覆盖、config-lock、warm-start 区分。
8. **DEBUG**：events、capture、溯源、重算、诊断 RNG 隔离、CLI/HTTP 同源。
9. **TASK**：standup/basic_balance 语义、评估协议、指标、预算和停止规则。
10. **INTEGRATION**：train.py/registry/import、资源使用、后台模式、PPO 回归。

#### W7.3 把决策映射到可证伪测试

为每个风险指定一个或多个测试 ID。命名建议：`MATH-*`、`POL-*`、`DATA-*`、`RPL-*`、`COL-*`、`CLK-*`、`CKPT-*`、`DBG-*`、`TASK-*`、`IND-*`、`PERF-*`。每项必须写输入、预期、容差/判定、失败含义和证据文件；不能只写“观察曲线正常”。对已有 FX/DX 项直接映射，不重复造编号。

#### W7.4 定义验证层级与通过门槛

明确每类证据能证明什么：

- 静态/schema：接口、字段、import、配置拒绝。
- 单元：单一函数/类在受控输入下的行为。
- property：随机/合成数据上的不变量。
- 集成：collector→slice→replay→trainer→metrics 的端到端小路径。
- smoke：真实环境少量 episode/updates。
- 短训：有限 budget 下确认机制趋势和恢复，不作为任务收敛证明。
- 多 seed 正式验收：阶段六任务成功标准。

同一风险若低风险只需要低层验证；blocker 必须有能直接证伪的测试，不允许只用训练曲线证明。

#### W7.5 冻结容差与证据产物规则

为数值检查定义默认精度原则：整数/布尔/schema exact；纯 NumPy 公式可用 tight tolerance；torch/GPU 只承诺 dtype/device/算法声明容差；bitwise 只在 `config-lock + 同代码/设备/库/确定性内核` 下作为目标。每个测试必须写明证据落点：pytest 断言、dump artifact、metrics event、run manifest 或评估日志。

#### W7.6 统一阶段二工作包与依赖 DAG

合并 A5/A6 已列出的 `P2-*` 包，消除重叠并给出依赖顺序。目标顺序草案：

```text
P2-IND-0 package/lazy registry/train.py 边界
  ├─ P2-COLL-1 SAC collection/job/runner
  ├─ P2-DATA-1 transition/slice/validator
  │    └─ P2-REPLAY-1 replay identity/FIFO/persist/sample
  │         └─ P2-CKPT-1 bundle/manifest/RNG/resume
  ├─ P2-DBG-1 metrics/events/catalog
  ├─ P2-DBG-2 tick ring + L0/L1
  └─ P2-TRAIN-1 S01 actor + SAC trainer math
       └─ P2-LOOP-1 clocks/UTD/eval/checkpoint loop
            ├─ P2-DBG-3/4 dump/access/analysis
            └─ P2-ENV-1 fake/small env integration → real env smoke
```

每个包必须有输入契约、改动范围、测试门槛、出口 artifact、不可做事项；任何包不得以“先跑通再补测试”为由跳过 gate。

#### W7.7 标出阶段门与最晚解决点

把风险分成：

- **进入实现前必须确认**：数学公式、transition/schema、首个 actor、独立性、replay/UTD 口径。
- **阶段二完成前必须通过**：collection/data/replay/checkpoint/loop/metrics 的单元与集成 gate。
- **阶段三前必须解决**：两个实验语义、phi_pre、边界、真实 env smoke、基础诊断。
- **阶段四前必须解决**：八格策略与探索/regularizer 扩展验证。
- **阶段五前必须解决**：完整 dump/access/CLI/viewer 同源。
- **阶段六前必须解决**：训练预算、评估 seeds、warm-start/resume 与长训监控。
- **可选/暂缓**：PER、n-step、relabel、stratified retention、异步、GPU inference server。

#### W7.8 定义 W7 内的低成本核查

仅执行不改生产代码的检查：文档编号一致性、决策引用完整性、测试命名冲突、阶段包依赖是否成环、已有测试是否可作为某些项的证据。禁止把现有 PPO/旧 SAC 测试通过直接计作新契约通过。

#### W7.9 汇总未决项与重开条件

把未决项分为阻塞阶段二、阻塞后续阶段、可选增强。每项写明为什么未决、最晚决策点、需要的新证据、推翻现有决策的条件。用户已确认的口径标成 `approved`；助手推荐但未审阅的标成 `proposed`，不能混写。

#### W7.10 汇总 A7 并更新路线图

形成 A7：完整风险矩阵、测试命名/层级、阶段二 DAG、阶段门、未决项和证据索引。同步 PLAN.md 的状态与阶段二入口条件；提交推送后由 W8 做最终阶段一评审包。

**必须先裁决的问题（进入实现前不能留默认值）：**

1. 风险严重度与 gate 是否采用 `blocker/major/minor` + 七级验证层级；推荐采用。
2. 阶段二是否以 fake/small env integration 作为真实 env smoke 前置 gate；推荐采用，避免第一版直接依赖昂贵 MuJoCo 长跑。
3. 阶段二是否允许并行开发 collector/data 与 metrics/debug 基础层；推荐允许，但接口按 A5/A6 schema mock，不反向改契约。
4. 训练曲线能否作为任何 blocker 的唯一证据；推荐禁止，blocker 必须有独立可证伪测试。
5. 旧测试是否能改造成新契约测试；推荐只有在输入/断言/身份语义明确映射时才允许，否则新建测试。

**出口：** A7 风险可追踪，每项都有验证位置、证据产物、阶段门和失败含义；阶段二工作包有明确 DAG，而不是直接进入长训练。

### W8：汇总裁决并提交阶段一评审

**评审包：** A1–A7、候选方案取舍、未决项、支持矩阵、阶段二具体实施顺序。

每条新裁决使用与历史 N1–N9 及未编号历史条目不混淆的编号，例如 `SAC-R1-D01`，内容包含：问题、候选、证据、选择、拒绝原因、适用范围、验证办法和重开条件。

未决项分三类：

- **阻塞阶段二**：单通道目标/梯度、transition 边界、首个策略、独立接口、恢复与计数口径等；不解决不能进入实现。
- **阻塞后续阶段**：例如 mixture 估计器的最终优化、复杂多通道扩展；必须有可行路径与最晚解决阶段，不可无限后延。
- **可选增强**：PER、REDQ、异步等，不纳入初版承诺，须有新证据才启动。

若验收预算、资源或数学设计缺乏足够依据，明确报告阻塞与备选，不为通过评审给出虚假确定性。

## 3. 阶段一完成门槛

- [x] A1–A7 均有可审阅内容，事实、候选和验证结果分开。
- [x] 单通道标准 SAC 退化明确，多通道核心语义与支持边界可解释。
- [x] 两任务奖励/权重/终止的时间对齐已核对，具备对拍方案。
- [x] 八格保留范围、SAC 梯度差异和探索控制分层明确。
- [x] 数据、依赖、时钟、完整恢复及最小 debug 捕获契约可直接指导阶段二。
- [x] 训练级验收 seeds、阈值、持续窗口、预算与失败处理已冻结或明确列为阻塞。
- [x] 旧决策逐项判定，没有将历史规划的结论自动继承为本轮结论。
- [x] 阶段二工作包与测试门槛明确，并获得用户对阶段一结果的审阅确认。

上述门槛在 W8 收口时全部完成。这里的「完成」仅表示设计契约获批准，不表示实现、测试或训练已经通过。

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

## A4：八格策略、梯度估计与两条正则路线（W4，2026-10-08）

**授权与范围：** 用户确认「uncertainty 作为基线之外的另一条路」后继续 W4。Shannon 熵 SAC 仍是基线；U 不是 Shannon 熵的别名，也不默认与熵叠加。调查代码基线为 `e578f1b4`（期间其他任务提交已推进仓库）；本次只改 SAC 设计文档，保留既有未提交的 `baseline/humanoid21/end2end/README.md`，不改 PPO/环境/生产 trainer。以下是阶段一的可行性与接口设计，不是总体路线图阶段四已实现。

### A4.1 决策摘要

| 编号 | 决策 | 证据/边界 |
|---|---|---|
| SAC-R1-D09 | 保留 2×2×2 八格的网络、分布和整向量共享分量语义，复制到 SAC 自有命名空间 | 不继承 PPO ABC、SamplingContext、ratio 或 GAE；SAC 内可以复用自己的分布内核 |
| SAC-R1-D10 | mixture actor 默认枚举 K 个分量，各分量逆 CDF 采样，再按可微概率加权；target 同样采用分量枚举 | 修复 Q 对 logits 无梯度；固定噪声及独立积分验证通过。采集仍 categorical 抽一个整向量分量 |
| SAC-R1-D11 | SAC 需要自有稳定 TN 数值内核，不原样搬入现有单分量的 CDF 差/固定裁剪 | 已复现大 σ 密度不归一和窄边界采样塌缩；N-SAC-01 是实现验收门槛，不修改 PPO 来修 SAC |
| SAC-R1-D12 | 三个互斥 regularizer_mode：`shannon`（基线）、`u_bonus`、`u_floor`（替代路线的两个子模式） | U 模式同步改变 actor/target/系数控制；不自动继承标准熵理论保证 |
| SAC-R1-D13 | β 的探索、π 的正则、优化器时钟三层分开；eval 确定性模式独立 | actor/target 不重放 β 的探索上下文；reference/delta 不进入首版 |
| SAC-R1-D14 | 阶段二首个 actor：single/shared/bounded；完整八格在总体阶段四扩齐 | 采用已有 BoundedStdTruncatedNormalPolicy 架构和界值，不另改网络。是阶段一对首个单分量的细化，不改变任务事实 |
| SAC-R1-D15 | 八格与正则模式分层验收，数值/梯度先行，收敛最后 | 小规模探针不是训练成功证据；GPU、SAC 导出、短训和真实任务验收仍待实现 |

### A4.2 八格能力矩阵及不变项

缩写 U_peak=`mean_d[1/(2 max p_d)]`，U_L2=`mean_d[1/(2∫p_d²)]`；两者均为动作边际有效宽度，不是联合 Shannon 熵。表中类/文件来自 `baseline/framework/ppo/policies/`，将复制适配，不能在 SAC 中 import 原类。

| 格 | 源类 / 文件 | σ 来源 | σ 映射 | 原生 U |
|---|---|---|---|---|
| S00 | TruncatedNormalPolicy / truncated_normal_mlp.py | `(D,)` 参数 | exp(logσ)，数值安全 clamp ±20 | peak |
| S01 | BoundedStdTruncatedNormalPolicy / bounded_std_truncated_normal_mlp.py | `(D,)` v 参数 | sigmoid-logσ | peak |
| S10 | StateTruncatedNormalPolicy / state_truncated_normal_mlp.py | trunk+σ head | exp(logσ) | peak |
| S11 | StateBoundedStdTruncatedNormalPolicy / state_bounded_std_truncated_normal_mlp.py | trunk+v head | sigmoid-logσ | peak |
| M00 | SharedMixtureTruncatedNormalPolicy / shared_mixture_truncated_normal_mlp.py | `(K,D)` 参数 | exp(logσ) | L2 |
| M01 | SharedMixtureBoundedStdTruncatedNormalPolicy / shared_mixture_bounded_std_truncated_normal_mlp.py | `(K,D)` v 参数 | sigmoid-logσ | L2 |
| M10 | MixtureTruncatedNormalPolicy / mixture_truncated_normal_mlp.py | trunk+σ head | exp(logσ) | L2 |
| M11 | StateMixtureBoundedStdTruncatedNormalPolicy / state_mixture_bounded_std_truncated_normal_mlp.py | trunk+v head | sigmoid-logσ | L2 |

- shared/state 只指 σ 来源；mixture 的 logits/μ 始终依赖状态。单分量 μ=tanh(mean head)；mixture 是 K 组 D 维对角 TN 的混合，不是每维各自抽 K。
- 保留 trunk 的两层 Tanh MLP、各 head 布局及已有初始化：σ=e^-1；state σ/v head 零 weight、对应常量 bias；mixture 初始 logits 均匀、分量均值微扰破对称。K 默认3，D=21，obs=96；隐藏宽度显式登记（参考蓝图256）。
- bounded：`σ=exp(log(.05)+log(2/.05)·sigmoid(v))`，v_init≈.164422，探索 raw shift 系数 κ≈1.199339。此 κ 不是 SAC 温度 α；σ 是底层 Normal 的尺度参数，不是截断后实际标准差，σ_min 也不能直接称为动作标准差下限。
- 确定性评估：single 返回 μ；mixture 返回最大概率分量的 μ 向量，tie 选首分量。μ 是未截断 Normal 的中心，**不等于截断分布期望**；不把多模态动作均值替换成确定性 act。
- 联合 log_prob：`logsumexp_k(log p_k + Σ_d log TN_kd(a_d))`；不能用被抽中分量的 log_prob 或逐维 logsumexp 之和。
- 原策略有私有 generator、文件式自包含导出及分布/σ/U 元数据。SAC 需继承能力而非旧训练类型；同一权重用于两个 agent 时 RNG 流仍应按 agent 隔离，不复制旧 collector 的共享实例/后一次 reset 覆盖语义。

### A4.3 SAC-native actor 契约草案

接口命名可在 W5 整合，以下语义不可省略：

1. `distribution(obs)`：返回当前 π 的参数；不接受 PPO SamplingContext，训练 π 的 e=0。
2. `expectation_samples(obs, noise)`：返回 actions `[B,K,M,D]`、联合 log_prob `[B,K,M]`、可微 integration_weights `[B,K,M]`。single K=1；mixture 权重为 p_k/M。M 首版=1（每分量每状态一个噪声样本），全部权重沿 K/M 求和为1。**integration_weights 不是 replay/sample weight，也不是 actor 通道权重。**
3. `sample_behavior(obs, behavior_spec, rng)`：β 的随机动作；mixture categorical k 作用于整向量，输出实际动作、行为版本/探索设置，log_prob 仅作诊断，不参与 PPO ratio。
4. `deterministic_action(obs)`：按 A4.2 约定；不是 e=-1，后者仍有随机性。
5. `uncertainty(obs, kind)`：对 π 计算可微 U，显式度量种类；可返回逐维 U 用于塌缩诊断。β 的有效 U/熵另列指标。
6. 显式噪声或 RNG state 注入、读取和恢复；actor/target/collector/eval/diagnostic 各自分流。不能只存一个 seed 冒充保存 generator 当前位置。
7. 自包含导出记录架构、K、σ 来源/参数化/界值、U_kind、数值内核版本、确定性动作语义；载入时 strict 校验。checkpoint 另记正则模式/目标/系数、RNG 和 optimizer；切换模式或架构不是 exact resume。

trainer 消费同一种 expectation sample 结构，不按八个类名写分支。策略数学与采集导出留在 SAC 内部，不建设 PPO/SAC 通用层。密度仍是必备能力，即便 U 路线暂不把 log_prob 用进 loss。

### A4.4 mixture actor：为什么枚举及其梯度

令 `p_k=softmax(z)_k`，`a_k=f_k(ε_k;μ_k,σ_k)`，F 为 A3 的通道合成并选 twin 后的 Q：

```text
Shannon actor：L = E_s Σ_k p_k E_ε[α log π(a_k|s) - F(s,a_k)]
U actor：      L = E_s Σ_k p_k E_ε[-F(s,a_k)] - E_s Bθ(s)
```

- outer p_k **不得 detach**；inner logπ 仍是完整混合密度，对所有分量参数可微。α、Q 参数和 A3 actor 门控参数冻结，但动作路径/分量权重保持梯度。
- 对 Q-only，logits 导数包含 `p_k(E[-Q|k]-E[-Q])`。现有 `torch.multinomial → gather → Q` 路径没有该项；即使熵/U 能给 logits 梯度，也没有替代任务价值的分量偏好学习。
- 枚举只是对离散分量积分，仍用抽样估计分量内连续期望；不是同时执行 K 条动作、不改变策略分布、不是逐关节选分量。K=1 退化到普通重参数化。
- 同样可用 categorical score-function：采样 k 后，在条件重参数化/直接密度导数之外，补 `sg(f-b)∇log p_k`（f 为样本目标，b 不依赖本次 k）。它有较低每步计算量但更高方差和 baseline/双计数风险；保留为大 K 后续候选，首版不用 straight-through/Gumbel 近似，也不悄悄漏 score 项。
- full mixture log density 在每个 a_k 处需要评估所有分量，典型密度成本 O(B M K² D)，Q 成本 O(B M K)；K=3/M=1 先实现，不能声称与单分量同速。超出显存预算报错或显式分块，不自动切换到错误梯度路径。

**target 枚举：** a'_k 由当前 π 在 replay 的 next_obs 生成；每个 a'_k 各自按 A3 选共同 twin `j_k`（先 min 再对 k 求期望，不能交换次序）：

```text
Shannon：y_c = r_c + γm Σ_k p'_k [Qbar_c,j_k(s',a'_k) - α logπ(a'_k|s')]
U：      y_c = r_c + γm [Σ_k p'_k Qbar_c,j_k(s',a'_k) + Bθ(s')]
```

整条 target 无梯度。actor 同样按每个候选动作选择 twin，不选一个分量再复用于整个 batch。理论上 categorical target 在 no_grad 下仍可无偏估计，但首版为统一核验采用枚举。噪声/权重/选中 twin 需支持 dump 重算。

### A4.5 单分量和数值内核：N-SAC-01

已核实当前单分量 `sample_action` 支持 u 注入（`truncated_normal_mlp.py:430-473`），但其 `Z=Φ(b)-Φ(a)` 且 clamp_min=1e-8，inverse CDF 概率 clamp=1e-6，最后动作裁成 `[-1+1e-6,1-1e-6]`。mixture 使用较稳定的 erf-sum Z 和 erf 空间逆变换，但仍有 erfinv/动作固定 clamp，**也不能原样宣称全域精确**。

CPU 实测旧 single：

| 参数 | 实际结果 | 问题 |
|---|---|---|
| μ=.4、σ=e^20、float32、u=.01/.25/.5/.75/.99 | 五个动作均≈.4；logp≈-2.498；评分密度积分≈.164456 | 正确极限应为近似 Uniform[-1,1]；CDF 消差使采样塌缩，Z 下限使密度失真 |
| 同上 float64 | 动作近似分位点，但评分密度积分仍≈.164456 | 只改 dtype 不消除业务式 Z 截断 |
| μ=.99999999、σ=1e-8，float32/64 | 所有动作被裁成≈.999999；logp 均值约-5115/-4883 | 固定 inward clip 大于真实分布尺度；有限值并不代表合法重参数化 |
| μ=0、σ=1e4、float32 | 密度积分≈1.00009598，中位动作附近存在偏差 | 已能观察到数值误差，需精度/容差契约 |

**SAC 适配方向（尚未生产实现）：** 利用 μ∈[-1,1] 的条件，用非负 erf-sum 算 Z；直接在 erf 空间插值后 erfinv，敏感计算用受控高精度，不把小而合法的 Z 提高到1e-8。保护应针对浮点端点/下溢，并记录触发率；不能把固定大裁剪当作理论密度的一部分却继续按原 TN 评分。无效动作/非有限参数 fail loud；不通过缩窄八格的业务 σ 域掩盖无界格问题。

候选 float64 erf 内核在大 σ 均匀极限的动作误差≤2.22e-16，6 个代表点通过 μ/logσ gradcheck（A4.10）。**不覆盖 u 极靠0/1、所有参数、最终 float32 量化或 GPU**；浮点无法分辨的极窄边界分布必须显式暴露数值支持范围，不能承诺任意 σ 的无限精度。

N-SAC-01 完成门槛：独立积分归一、采样分位点/矩、固定噪声梯度、尾部概率、action/support 与密度一致性、kernel/导出一致性全部通过；GPU/float32 的实际路径也需测。阶段二 actor 验收前必须落实，后续无界格开放前补齐其极端范围测试。

**首个 actor 选择 S01**：single/shared/bounded，沿用 σ_min=.05、σ_max=2、σ_init=e^-1；这仍是标准 SAC 的合法随机策略族。避免把极端无界参数作为首条闭环的前置负担，但 bounded 也必须通过端点测试。S00 等七格并未删除或偷偷加相同界值，按全八格路线扩齐。

### A4.6 两条路线与正则记账

S 路线 `regularizer_mode=shannon` 为基线；U 路线包含两个独立配置（不是同时开启）：

| 模式 | 状态正则奖励 Bθ(s) | actor 中该项 | target 中该项 | 系数控制 |
|---|---|---|---|---|
| shannon | α H(πθ) | 样本估计 `+α logπ` | 样本估计 `-α logπ_next` | 固定 α 或 A3 自动温度 |
| u_bonus | λ Uθ(s) | `-λ Uθ(s)` | `+λ Uθ(s')` | 首轮固定 λ；自动模式按 U_target 单独验证 |
| u_floor | `-λ relu(f-Uθ(s))²` | `+λ relu(f-Uθ(s))²` | `-λ relu(f-Uθ(s'))²` | 固定 λ；不套用标准 α loss |

共同公式：`Q^π_B = E[r_t + Σ_{k≥1} γ^k(r_{t+k}+Bπ(s_{t+k}))]`（到真终止为止）；actor 优化 `-E_a Q_B - Bπ(s)`。U 每状态只算一次；target 里的 U 来自当前 π 而非历史行为策略；整个 target detach，actor 里的 U 不能 detach。

A3 常量权重合成证明继续适用：共同 γ/m、Σw=1 时公共未来 B 只计一次。动态 actor 门控仍是 A3 的局部扩展，不因换正则获得全局标量目标保证。多通道 regularizer counting 的三模式数值对照已通过。

- U-floor 不是仅给 actor 添加惩罚而保留原 Shannon target；后者是不同的混合/辅助方案，本轮默认拒绝。S/U 都不默认叠加第二种正则。
- U-floor 是软惩罚，不保证所有状态/关节 U≥f；B≤0 还可能影响对存活时长的偏好，需与任务 reward 分开记录。U-bonus 的持续覆盖奖励也会改变 Q 尺度，不能仅按原始奖励界判发散。
- U 不保证联合动作多样性、每个关节的探索下限或窄分布的恢复梯度；此前同边际不同联合熵、σ缩小使 U 梯度衰减的结论纳入故障 fixture。
- **保留原 U 定义，不未经试验统一它们**：S 格 native=peak，M 格 native=L2。run 必须把 `uncertainty_kind` 解析成明确值；不能在换族时把相同 f/λ 称为相同正则强度。首轮算法 A/B 固定同一 S01 策略，比较 S、U-bonus、U-floor。固定系数允许 λ=0 的显式消融；使用 logλ 自动模式时必须 λ_init>0，不能对0取对数。
- 若以后跨 single/mixture 做严格 U 正则对照，可另设 common-L2（single=K1）模式，但这是显式新度量，不替换 PPO 原定义，本次不宣布它为默认。
- 当前 U 函数并不普遍满足 Shannon 的凹性性质，不直接继承 Boltzmann/KL 形式的最优策略或通用收敛保证；枚举梯度正确也不等于该正则优于熵。

### A4.7 自动系数、可达范围与标定

**Shannon：** 用动作联合熵（21维求和，nats）；不能将维均值与总和混用。`η=log α`，`Lη=η·sg(H_est-H_target)`，H_est 与 actor 采用相同状态、概率权重；mixture 用 `Σp_k[-logπ(a_k)]`，不是 categorical entropy 加各分量熵（分量重叠时会过计）。方向及更新时钟沿用 A3。

第一条机械闭环先使用显式固定 α；auto 模式要求显式 H_target，不接受 None→-D 的隐式回退。`H_target=0 nats/整向量`可作为后续预登记试验候选（下面见证值之间可通过连续调 σ 达到），不是已验证收敛默认。具体训练系数/目标在正式 run 配置中固定，不能训练后用保留评估挑选。

**U-bonus 自动模式候选：** `η_U=log λ`，`Lη_U=η_U·sg(U-U_target)`；低于目标时 λ 增大。U_target 是明确 U_kind 的无量纲数值，不是 nats，更不能用 -21。该控制针对 replay 状态上的经验均值，不保证逐状态/逐关节下限；本次仅核验调整方向，首轮 U 路线仍用固定 λ。U-floor 的 f 是逐状态软阈值，不与平均 U_target 混称。

代表性 TN 校准（float64，21维重复同一分布；μ=1 为边界极限）：

| μ | σ | H/维 | H/21维 | U_peak |
|---|---|---|---|---|
| 0 | .05 | -1.576794 | -33.112669 | .062666 |
| 1 | .05 | -2.269941 | -47.668759 | .031333 |
| 0 | e^-1 | .385218 | 8.089587 | .458043 |
| 0 | 2 | .692464 | 14.541743 | .959850 |
| 1 | 2 | .682786 | 14.338505 | .855624 |

这些是**见证点，不是任意状态/mixture 的已认证包络**。理论全动作立方体最大熵21ln2≈14.556；σ上界2的族不等于真正均匀分布，U=1 也未必可达。配置先拒绝超过理论界的目标，再按族/状态验证实用可达范围；饱和率、∂logσ/∂v、温度漂移必须诊断，不用 clamp λ/α 掩盖不可达目标。bounded sigmoid v 饱和时梯度也可接近0，不能声称有界 σ 消除所有塌缩风险。

单分量可用 TN 闭式熵作 oracle：`H_d=log(σ√(2πe)Z)+(aφ(a)-bφ(b))/(2Z)`；大 σ 下有相消，需稳定实现/独立积分，不直接替代首版 MC 路径。mixture Shannon 无通用简单闭式，采用枚举+条件采样估计并用低维积分验证。

### A4.8 标准化旋钮：作用域、默认语义与时钟

名称是接口草案；数值变化必须校验、记录 requested/effective 值，所有调度状态需可恢复。

| 层 | 旋钮 | 单位/范围及默认 | 生效位置/时钟 |
|---|---|---|---|
| 采集 β | behavior_explore=e | [-1,1]，默认0 | 只在下一采集 round 的行为分布生效；调度以已执行 env steps 为横轴，记录实际 round 边界 |
| 采集 β | random_start_transitions | 非负整数，默认0；是否开启需显式配置 | 初始 Uniform[-1,1]^D 行为，不是 e=1；计有效 agent transitions，具体预算接 W5 |
| π 正则 | regularizer_mode | shannon/u_bonus/u_floor；默认shannon | run 创建时固定；中途改模式不是 exact resume |
| π 正则 | alpha_mode / α_init | fixed/auto；机械基线fixed，α>0 显式给定 | actor/target 同一 α_used；auto optimizer 按 temperature tick 更新 |
| π 正则 | H_target | 显式 joint nats，可负；auto 必填 | 不消费 β 的 log_prob，也不由 uncertainty 换算 |
| π 正则 | U_kind / λ / U_target / f | U_kind 明确；固定λ≥0；f∈[0,1]且需可达检查；auto-U 必填 U_target | U路线专用；默认无跨模式调度，λ=0 明确标记无正则消融 |
| 优化 | actor/critic/temperature LR | 正实数，各自独立 | 各 optimizer 成功 step；调度器状态入 checkpoint |
| 优化 | actor_update_interval | 正整数，默认每critic tick一次 | target 不随 actor 延迟而改变软更新时钟 |
| 优化 | τ | (0,1]，显式配置 | 每成功 critic optimizer tick，沿用 A3 |
| 优化 | batch_size / UTD / round cap | 正整数/非负比率/明确上限 | UTD=critic ticks/有效新agent transitions；余数、warmup及限额行为在W5冻结 |
| 优化 | grad_clip / reward_scale | 裁剪阈值正值或显式关闭；reward_scale默认1 | 记录裁剪前后梯度；reward_scale 是目标单位变化，不是探索旋钮 |

e 的映射保留八格设计：unbounded `σβ=σπ·3^e`，bounded `σβ=f(v+κe)`；κ只在初始化点匹配局部斜率，不承诺 e=1 时 σ 恰三倍。实测初始化 e=-1/0/+1：unbounded≈.122626/.367879/1.103638，bounded≈.131499/.367879/.943633。π 的 U 在三种 e 下不变。mixture logits/μ不随 e 改变，也不承诺联合熵或边际 U 单调增大。

reference/delta σ-floor、mixture 权重温度、独立额外动作噪声暂不支持，传入对应配置应报错；不把 PPO 复杂探索协议带入 SAC。评估不用上述 e，不加随机动作噪声；另行做随机策略评估时明确标记，不混入已冻结的确定性验收流。

### A4.9 最小新增诊断与恢复字段（交 W5/W6）

- 分布：σ及有效σ分位数/最小最大、均值距边界、log_Z、尾部/保护触发率、bounded sigmoid 饱和率与灵敏度；异常概率/密度与 sampling mismatch 不以 finite 检查替代。
- mixture：p_k、categorical entropy、有效分量数、分量 overlap、per-component E[Q]/logπ、**Q-only logits 梯度**，不能只凭 U/熵的梯度非零判定分量会学任务。
- 正则：H_joint 与 U_kind/U_mean/U_dim_min 分开；U-floor 的 active_state_fraction、gap、实际 penalty；raw reward、target continuation regularizer、actor regularizer、α/λ_used 与 after。
- 采集：β/e、策略版本、随机启动阶段、采集时钟；训练：π、objective_mode（A3门控模式）、regularizer_mode（A4正则模式）分别标识，二者不是同一个字段。
- 可重放截面保存每个分量的噪声/权重/动作/联合logp/twin选择及参数版本，不能只存最终混合动作。日志诊断不得消费训练 generator；新 kernel/export 改变采样位流时更新版本，不宣称与旧 PPO 导出 bit-identical。

### A4.10 本次实际验证记录

统一环境：CPU、`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、1线程；公式/梯度用float64，stress另用float32。没有GPU训练或MuJoCo任务训练。

1. **既有八格测试：288 passed / 3 skipped**，24.84s；跳过项均为 CUDA act/sample 迁移测试（因本次禁用CUDA）。只有 pynvml deprecation warning。命令为 `python3 -B -m pytest -q -p no:cacheprovider` 加 policies 下这8个文件：
   `test_truncated_normal.py`、`test_bounded_std_truncated_normal.py`、`test_state_truncated_normal.py`、`test_state_bounded_std_truncated_normal.py`、`test_mixture_truncated_normal.py`、`test_shared_mixture_truncated_normal.py`、`test_shared_mixture_bounded_std_truncated_normal.py`、`test_state_mixture_bounded_std_truncated_normal.py`。它们不是 SAC 更新测试。
2. **8格固定噪声方向导数**：seed404，B=3、obs_dim=2、act_dim=2、hidden=4，obs=((.2,-.4),(.7,.1),(-.3,.8))；u为[.13,.87]等距点；mixture K=3、非对称logits/均值；toy `Q=-||a-(.3,-.2)||²+.15a0a1`，α=.07，中心差分ε=1e-5。single调用现有注入u路径，mixture用内存枚举候选+现有参数映射/密度公式。全参数随机单位方向误差最大5.64e-12（门槛2e-7）。这不是所有参数点全梯度穷尽检查。
3. **旧mixture路径的负对照**：四个mixture格中 Q-only 的 logits head bias 梯度均严格为0；枚举总loss对应梯度范数分别约 .162471/.124170/.192532/.107205。是否正确另由下一条独立积分确认，非只测非零。
4. **独立积分与全坐标梯度**：1D、K=3，logits=(-.4,.2,.6)、μ=(-.55,.15,.65)、σ=(.12,.28,.08)，Q=-(a-.2)²+.1a；512点Gauss-Legendre分别在u空间和动作密度空间积分。检查9个坐标（logits/raw μ/logσ）的中心差分ε=1e-5：

| 目标 | 候选期望与密度积分差 | 梯度与密度积分导数最大差 | 全9坐标有限差分最大差 |
|---|---|---|---|
| Q-only | 3.293e-9 | 1.107e-7 | 2.044e-11 |
| Shannon α=.07 | 1.211e-9 | 5.516e-8 | 2.045e-11 |
| U-bonus λ=.2 | 3.293e-9 | 1.107e-7 | 2.646e-11 |
| U-floor λ=.7/f=.6 | 3.293e-9 | 1.107e-7 | 2.491e-11 |

   门槛分别为梯度参考5e-5、有限差分1e-7，全部通过；U项使用现有解析L2。有限阶积分是高精度参考，不是无限精度证明；多维/极端分量尚需永久测试。
5. **数值stress**：A4.5所列异常均已复现；不是只读注释推断。候选erf float64在 `(μ,σ)=(0,.05),(.999,.05),(-.999,2),(.2,1e-4),(.4,1e4),(.4,e^20)` 的固定u=(.05,.2,.5,.8,.95)、`mean[a²+.03logp]` 通过 gradcheck，eps=1e-6/atol=2e-6/rtol=2e-4。大σ零梯度的绝对容差不能证明其微小相对误差。
6. **其他公式核验**：相同分量拆分后的L2 U不变；三正则模式在A3常量权重下只计一次B；U_target=.4时U=.2/.6的log系数梯度为-.2/+.2；行为e改变σβ但π的U不变。下面最小probe另以logits=(-18,0)检查低概率分量：p≈1.523e-8，Q-only logits梯度≈8.961e-9，与ε=1e-3的差分绝对误差3.727e-15，符合p_k(f_k-Ef)解析式；不覆盖float32 softmax下溢。此前调研中的同分布peak/L2差异、同边际不同联合熵和窄σ梯度衰减保留为实现期回归项。

下面为枚举核心的可重现最小 probe（项目目录，现有torch；只证明所示点的Q logits项与有限差分，不替代上表完整积分）：

```python
import math
import torch
torch.set_default_dtype(torch.float64)
u = torch.linspace(.05, .95, 19)
mu = torch.tensor([-.5, .5])
sigma = torch.tensor([.15, .2])
lo = torch.erf((-1-mu)/(math.sqrt(2)*sigma))
hi = torch.erf((1-mu)/(math.sqrt(2)*sigma))
a = mu[:,None] + math.sqrt(2)*sigma[:,None]*torch.erfinv(
    (1-u)*lo[:,None] + u*hi[:,None])
component_loss = ((a-.3)**2).mean(-1)
z = torch.tensor([-.2, .4], requires_grad=True)
loss = (z.softmax(0)*component_loss).sum()
grad = torch.autograd.grad(loss, z)[0]
expected = z.softmax(0)*(component_loss-loss.detach())
assert torch.allclose(grad, expected)
def loss_at(v):
    return (v.softmax(0)*component_loss).sum()
fd = torch.stack([(loss_at(z.detach()+1e-5*d)-loss_at(z.detach()-1e-5*d))/2e-5
                  for d in torch.eye(2)])
assert torch.allclose(grad, fd, atol=1e-9, rtol=1e-7)
assert grad.norm() > 0
print('enumerated Q-only logits gradient verified')
```

### A4.11 实现期验证矩阵与替代路线准入

| 层级 | 必须验证 | 通过标准/失败处理 |
|---|---|---|
| 分布 oracle | TN归一、尾部、真实动作支持；mixture全联合密度；大σ/窄边界 | 独立积分与分位点/矩匹配；新保护触发可解释，N-SAC-01不通过则不进训练 |
| 梯度 oracle | μ/σ/logits；Q、Shannon、U各项；重叠/分离/低概率分量 | float64中心差分/独立积分；低概率用概率加权绝对容差且报告相对误差，不因梯度小直接跳过 |
| 结构退化 | K1、重复分量、置换、shared/state拷权重对拍 | density/期望一致；native peak与L2 U不要求相等；梯度/随机数语义明确 |
| RNG/export | actor/target/β/eval流分离；CPU/GPU各自重放；导出参数/密度一致 | 同设备同内核逐位重放；跨设备只按登记容差，不许用PPO对象过桥 |
| SAC更新 | 8格均能更新任务均值/σ；mixture额外验证Q-only logits | clip/temperature/target隔离；训练不能只靠正则驱动logits |
| 正则模式 | S/U-bonus/U-floor三模式的actor/target/系数方向与模式拒绝 | 无隐式叠加；切模式/度量版本不得冒充exact resume |
| 短训 | 每格S基线固定toy训练；U路线先固定S01比较 | 独立评估回报改善且数值/梯度诊断合理，不只loss有限；正式步数/seed在W7预登记 |
| 真实任务 | 默认S01、S路线两任务各42/43/44；结构替代优先M11两任务至少seed42 | 沿用A2预算/阈值；其余格完成接口/梯度/短训，不虚称全八格已收敛 |
| 替代路线训练 | 同一S01上的U-bonus/U-floor与S对照，先toy后真实任务 | 数据预算、任务事实、评估集一致；各模式调参预算单列，不用holdout选模式或替代失败seed |

替代路线已获授权作为研究路径，但本次不预设它优于S或替换默认验收。其额外真实训练资源不从A2单run预算推断为无限授权：W7/正式实验前登记模式×seed矩阵及总资源上限。首轮U路线用固定系数，自动U及Shannon+U混合目标不阻塞标准SAC闭环。

### A4.12 出口与后续任务

- [x] 八格能力矩阵、SAC自有actor契约和阶段二首个策略明确。
- [x] mixture枚举方案及替代score估计边界明确；任务logits梯度经负对照、全坐标差分及独立积分核验。
- [x] Shannon与uncertainty替代路线的目标、target、系数及命名分开；不偷换U定义。
- [x] 探索/正则/优化三层旋钮及RNG/export/调度约束明确。
- [x] 记录了实际测试及数值失败，不以已有PPO测试替代SAC可行性结论。

**W4设计出口完成；实现门槛尚未完成。** N-SAC-01稳定内核由阶段二落实；全八格SAC永久测试/导出/GPU/短训由总体阶段四完成。下一工作包W5整合数据、实验、时钟与完整恢复契约，必须带入 expectation weights、noise、regularizer_mode/U_kind、φ_pre/post、采集调度实际生效时点。没有启动生产改写或大规模训练。

## A5：数据、采集、时钟与恢复契约（W5，2026-10-08）

**范围与状态：** 本节是阶段一的契约设计，不是生产实现。基于当前工作树复核了 `train.py → ExperimentSAC → Job/rollouter → EpisodeRunner/EnvRuntime/Recorder → build_slices → TaggedReplay → trainer → eval/checkpoint` 的真实链路，并执行了一个低成本 replay 覆盖探针；未改训练代码、未启动训练。用户在 W5 中确认：容量与 UTD 以**有效 agent transition**计；样本来源/行为元数据**强制保存**；首版**显式拒绝** n-step/PER/relabel/stratified retention；完整恢复默认**持久化 replay**。`phi_pre` 由本轮设计选择显式动作前采集；resume 的覆盖语义按用户要求参照 PPO 的“恢复完整状态，但当前白名单优化配置可生效并记录”。

### A5.1 决策登记

| 编号 | 决策 | 适用边界 |
|---|---|---|
| SAC-R1-D16 | replay 容量、warmup、UTD 分母统一按**已准入的有效 agent transition**计数 | 一条 `(agent, s,a,r,s')` 计 1；双 agent 同帧最多 2。`env_step` 仍单独记录，不用于 replay 容量 |
| SAC-R1-D17 | 每条样本强制携带稳定 `sample_id`、确定性 `source_key`、行为策略版本/fingerprint、实际探索参数、schema/语义版本 | 训练可以不消费这些字段，但不得缺省；诊断和恢复依赖它们 |
| SAC-R1-D18 | 首版 replay 只支持 1-step、FIFO、uniform sampling；n-step>1、PER、relabel、stratified retention 传入即拒绝 | **用户已确认**；不为旧接口保留“看似可用”的旋钮 |
| SAC-R1-D19 | 完整恢复默认把 replay 内容、游标、身份、采样 RNG 与训练状态一起持久化 | **用户已确认**；可用 rolling latest bundle 控制磁盘，但不得让 `--resume-from` 静默降级成 warm-start |
| SAC-R1-D20 | `phi_pre` 由 SAC 自有采集 runner 在调用 `runtime.step()` 前，通过 accessor 显式计算并记录 | 用户授权决定；不从归一化 obs 猜下标，不把 `phi_post` 当作 pre-action 事实；移位构造仅可作一致性校验 |
| SAC-R1-D21 | SAC vendor 自有 collection/job/runner/recorder 层 | 遵循 D-ROLLOUT-b；不引入 `SamplingContext`、ratio、GAE、reference/delta σ-floor |
| SAC-R1-D22 | 时钟分为 env action step、executed env frame、agent transition、collection round、critic/actor/temperature/target tick、eval/export/checkpoint tick | 所有计数器进入 checkpoint；UTD 使用 fractional credit 且每轮上限之外的部分显式丢弃 |
| SAC-R1-D23 | checkpoint 分为 `manifest + trainer state + replay snapshot + RNG/counters + experiment state` 的 bundle，并原子落盘 | model-only `.pt` 只能是 warm-start，不是完整恢复 |
| SAC-R1-D24 | resume 覆盖语义参照 PPO：恢复全部训练状态，白名单优化/调度参数按当前配置在下一 tick 生效并记录；另保留严格 config-lock 模式 | 实验名、schema、通道语义、policy 架构等身份字段不允许覆盖；这与 bitwise exact 不同，见 A5.8 |
| SAC-R1-D25 | `action` 字段定义为物理环境实际接收/执行的动作 | 若未来 pre-action 插件改写动作，policy 原始输出另存 `policy_action`，不得混用 |
| SAC-R1-D26 | collection 返回 `CollectedEpisode = Episode + job/collection provenance` | `Episode` 本身的 episode_index 只是 recorder 内局部序号，不能当全局身份 |

### A5.2 W5.0：现状链路复核

**真实链路（代码事实）：**

```text
train.py
  ├─ 顶层 eager import PPO/SAC registry
  └─ --algo sac → get_sac_experiment → sac.loop.train_sac
       ├─ experiment.build_actor / MultiHeadQCritic / TaggedReplay
       ├─ ParallelRollouter
       └─ per collection_round:
            actor.to_blueprint(stochastic=True)
            experiment.build_jobs(policy_bp, rollout_seed, n_episodes)
            rollouter.collect(jobs) → List[Episode]（与 jobs 同序）
            experiment.build_slices(episodes) → List[TrajectorySlice]
            replay.add_slices(slices)
            replay.sample_nstep + sac_update_v2（按新增 transition 数驱动）
            可选 deterministic eval / relabel / best export / video
            checkpoint（模型+actor optimizer+α，无 replay/RNG）
```

| 发现 | 证据/性质 | W5 处理 |
|---|---|---|
| `sac/experiment.py` 直接 import `ppo.TrainablePolicy`，并有用 `Job = Tuple[...]` 遮蔽 rollout `Job` dataclass 的幽灵别名 | A1 已确认；本 W5 复看仍存在 | A5.9 定义 SAC 自有 actor 与 `SACJob`，实现期删除该别名 |
| `train.py` 顶层 eager 加载两个 registry；SAC 路径会加载 PPO | A1 已确认 | W5 契约保留 C1/C2/C3 为测试门槛 |
| `Episode` 有 `base_seed/episode_index/blueprint_hash/boundary/observer_outputs/final_observation`，但无 collection_round、job_index、policy version、behavior spec | `rollout/episode.py` dataclass 字段 | 不由 `Episode` 强行扩展；collection 返回包装对象携带 provenance |
| `EnvRuntime.step()` 在 `core.step()` 前捕获 observation；recorder 收到的是 pre-action `obs_t`，observer output 是 post-action `s_{t+1}` 事实 | `env_runtime.py::_invoke_recorders` / `step` | transition schema 显式区分 `obs_t`、`phi_pre`、`phi_post`、`next_obs` |
| `simulator.get_action()` 在 post-action recorder 中读取，因此 `Episode.actions` 是实际 staged/applied action，不一定是 policy 原样输出 | `env_runtime.py` recorder dispatch | A5 定义 `action=executed action`；若未来动作改写发生，另存 `policy_action` |
| `Episode.agent_frame_boundary` 已定义逐 agent 边界；`final_observation` 是整段 episode 末帧，不是早终止 agent 的真实后继 | A2 已冻结 | `next_obs` 必须在 slice 构造时逐帧显式生成，禁止一律使用 `final_observation` |
| `EpisodeRecorder` 会在 episode 结束要求每个 agent 有终止记录；`Episode.load()` 当前没有恢复 `episode_metrics` | `episode_recorder.py`、`episode.py` | SAC collection manifest 需保存 metrics；若沿用 Episode save/load，实现期补齐并回归测试 |
| `_stack_action_extras` 在 extras 缺失时可能整 agent 丢弃，`_stack_explore_factors` 对缺帧填 0 | `episode.py` stacking helpers | SAC recorder/validator 必须把必需 behavior/pre-action 字段缺失判为错误，不允许补 0 |
| `TaggedReplay` 是扁平环形数组；`traj_id/traj_step` 只是本地标签，`_traj_lengths` 覆盖后不清理，`buffer_stats()['n_trajectories']` 会累计历史轨迹数 | 低成本探针：capacity=5、插入 3×3 transitions 后 size=5 但 n_traj=3 | replay 改用 run-scoped `sample_id/source_key/slice_id`；slot/generation 仅作诊断，不作身份 |
| 现有 checkpoint 只含 actor/critic/actor optimizer/log_alpha/alpha optimizer/实验 state/env_step/grad_step；actor LR 被当前 `cp.learning_rate` 强制覆盖；无 critic optimizer、target、replay、RNG、round/eval 计数器 | `sac/loop.py::save/load_checkpoint_sac` | A5.8 定义 bundle manifest 与 PPO 风格 whitelist override，不得称为 exact resume |
| 现有 `rollout_seed = seed + round*episodes`、`eval_seed = seed + 100000 + round*97` 不保证训练/验证/保留流互斥 | `sac/loop.py` | A5.7/A5.10 改为显式 seed manifest + stream key 派生并做无重叠测试 |
| 现有 UTD `max(1,int(utd*added))` 丢失小数且 warmup 前无明确口径；一轮跨过多个 eval interval 也只评一次 | `sac/loop.py` | A5.6 冻结 fractional credit、warmup 分母和调度边界 |

**对 A1-F5 的修正：** 旧 `sample_nstep` 确实未校验后继 `traj_id`，`_traj_lengths` 也有过期条目；在严格顺序插入、整 slice 原子写入、容量按 FIFO 覆盖的现有调用下，环形下标多数情况下仍沿原轨迹尾段行进，但这只是调用不变量，不是 replay 自身保证。首版已裁决拒绝 n-step；未来若启用，必须先实现轨迹身份/覆盖校验，不能把现状当成已证明正确。

### A5.3 W5.1：实验侧与框架侧职责边界

| 责任 | ExperimentSAC / 实验文件 | SAC framework |
|---|---|---|
| 任务定义 | 实验名、环境蓝图、episode options、reward channels、终止原因映射、observer/fact 需求、评估指标 | 不解释 reward 的语义，不从 obs 下标猜测状态事实 |
| collection 规格 | `build_jobs` 产出 `SACJob`：policy/env blueprint、seed、options、逐 agent `BehaviorSpec`、需要记录的 pre-action fact key | `SACRollouter` 执行 jobs、保证返回顺序、附加 provenance、传播 worker 异常 |
| episode→transition | `build_slices` 决定 agent 边界、reward/valid/terminated/bootstrap、actor gate 和 task facts；缺失必需字段 fail loud | `TransitionValidator` 做 shape/dtype/finiteness/schema/通道一致性检查；不“修好”实验数据 |
| replay 策略 | 声明 uniform FIFO 配置及容量；首版不提供采样分布扩展 | replay 负责插入、覆盖、sample_id、source_key、RNG、统计与持久化 |
| 目标函数/更新 | 选择 `objective_mode`、`regularizer_mode`、通道配置与 actor gate 语义 | trainer 执行 A3/A4 公式、梯度隔离、时钟和诊断 |
| 评估 | `on_eval` 解释 episodes、产出任务指标和 best/stop 请求 | eval 调度、导出、结果记录、保留集/验证流边界 |
| 调度语义 | 可声明静态 schedule config；实验自身 curriculum state 进入 `experiment.state()` | framework 统一计数、持久化 schedule state、记录 requested/effective 参数 |
| 恢复 | 提供 `state()/load_state()` 与语义版本 | checkpoint bundle 的组装、校验、原子写入与恢复模式 |
| debug | 声明任务特有诊断字段与单元 | debugkit/日志/RNG 隔离/单样本重放由框架实现 |

未声明的实验能力一律 unsupported：例如多 data source、opponent pool、buffer reset、relabel、异步采集都不在首版承诺内。框架不得为了满足旧接口而保留未实现字段。

### A5.4 W5.2：transition schema（`sac_transition_v1`）

以下均按单条 agent transition 定义；slice/batch 是相同字段沿 T/B 轴堆叠。`schema_version="sac_transition_v1"`。

| 字段 | shape/dtype | 必需性 | 时间语义 / 规则 |
|---|---|---|---|
| `schema_version` | scalar int/str | 必需 | 当前固定 `sac_transition_v1` |
| `sample_id` | scalar uint64 | replay 分配 | 由 replay 在准入时单调分配并持久化；实验侧不得伪造 |
| `source_key` | scalar string | 必需 | 由 `(run_id, collection_round, job_index, agent_id, frame_index, env_blueprint_hash, transition_schema_version)` 规范化生成；同一 run 内不得重复使用 |
| `obs` | `(obs_dim,) float32` | 必需 | `s_t`：policy 看到的 pre-action observation |
| `action` | `(act_dim,) float32` | 必需 | 实际送入 simulator/被执行的动作；范围 `[-1,1]` |
| `next_obs` | `(obs_dim,) float32` | 必需 | 该 transition 的真实后继：非末转移取 `obs[t+1]`；末转移取该 agent 边界后的真实观测 |
| `reward` | `(C,) float32` | 必需 | `(s_t,a_t)→s_{t+1}` 的逐通道 reward，不能缺失补零 |
| `channel_valid` | `(C,) bool` | 必需 | `1` 表示该通道数据可学习；缺 observer 不等于 `0`，而是准入失败 |
| `terminated` | scalar bool | 必需 | 任务真终止，决定不 bootstrap；timeout 不为 true |
| `truncated` | scalar bool | 必需 | agent trajectory 到边界但非真终止；timeout 为 true |
| `bootstrap` | scalar bool/float | 必需 | `0` iff `terminated`；timeout/truncation 为 `1` |
| `termination_reason` | scalar string | 必需；未终止可为 `""` | 首条提议原因；语义映射由实验负责 |
| `physics_delta` | scalar int32 | 必需 | 本帧实际执行物理子步数；准入要求 `>0` |
| `actor_gate` | `(C,) float32` | 必需 | 非负有限 `g_c(s_t)`；全零拒绝 |
| `actor_weight` | `(C,) float32` | 必需 | `g/Σg`，非负、有限、和为 1；供审计与 trainer 校验 |
| `sample_weight` | scalar float32 | 必需 | 默认 1；非负有限；不是 importance ratio |
| `task_facts` | dict[str, array] | 按实验声明 | `basic_balance` 必须含 `phi_pre`、`phi_post_reference`；standup 可不含 phi |
| `behavior` | struct | 必需 | `policy_version/policy_fingerprint/behavior_mode/requested_explore/effective_explore`；`behavior_log_prob` 可选，仅诊断 |
| `collection` | struct | 必需 | `run_id/collection_round/job_index/episode_seed/agent_id/frame_index/slice_index/inserted_env_step` |
| `versions` | struct | 必需 | `env_blueprint_hash/policy_arch/reward_semantics_version/objective_mode/regularizer_mode/transition_schema_version` |
| `reward_features` | dict[str, array] | 可选 | 首版只作 provenance/诊断；不启用 relabel |
| `policy_action` | `(act_dim,) float32` | 条件必需 | 若 action mapping/插件可能改写 policy 输出则必须记录；两目标任务当前与 `action` 相同 |

准入规则：

1. 所有必需字段缺省、shape 错误、dtype 不符、NaN/Inf、`actor_gate<0`、`Σg<=0` → `ValueError`，不得补零。
2. `physics_delta=0` 的帧不产生 transition；若该帧不是 episode 末尾退化帧而是中段空洞，直接报错。
3. `terminated` 与 `bootstrap` 互斥语义固定：`terminated=True → bootstrap=0`；`truncated/timeout=True → bootstrap=1`。
4. `next_obs` 必须对应该 agent 的真实后继状态；早终止 agent 不得使用整段 episode 的 `final_observation` 顶替。
5. `phi_pre` 是采集器在 `a_t` 执行前从 accessor 计算的状态事实；`phi_post_reference` 是 observer 对 `s_{t+1}` 的输出。二者字段名不得互换。
6. `actor_weight` 不进入 critic target；`channel_valid` 不表示 actor 偏好；`sample_weight` 不改变 Bellman 语义。

### A5.5 W5.3：replay 契约

**身份与插入：**

- `sample_id: uint64` 由 replay 在准入时从 `next_sample_id` 单调分配，在同一 run 生命周期内不复用；slot 覆盖、replay 持久化、重启都不改变它。
- `source_key` 是进入 replay 前的确定性来源键；active buffer 中重复 `source_key` → raise。对已覆盖样本，靠持久化的 `issued_slice_set`（完整 slice 级 manifest，而不是逐 transition 永久字符串集合）拒绝同一 slice/frame 被重新采集或重插。
- `slice_id` 由 `(run_id, collection_round, job_index, agent_id, slice_index)` 组成；每条 transition 记录 `frame_index`/`slice_step`。slice 级 issued set 随 run 保存，规模按 slice 数而非样本数增长。
- slot 下标只表示当前物理位置；另存 `slot_generation/write_epoch` 供 debug 判断引用是否已过期。
- slice 插入按 `(collection_round, job_index, agent_id, slice_step)` 的顺序原子执行；`slice_len > capacity` 直接失败，不静默保留尾部。

**容量与覆盖：**

- `capacity`、warmup、当前 size 都按已准入的 agent transition rows 计；配置必须满足 `warmup <= capacity`。
- FIFO 环形覆盖最旧 `sample_id`；不实现类别保留、优先级保留或跨版本 quarantine。
- `n_trajectories` 统计定义为当前 buffer 中仍至少有一条 transition 的活跃 `slice_id` 数，不再返回历史累计值。
- 样本 age 用显式计数定义：`transition_age = current_critic_tick - inserted_critic_tick`；另报 `env_step_age = current_env_step - inserted_env_step` 和按 `policy_version` 的分桶统计。

**采样与返回 batch：**

- 使用独立 `np.random.Generator`，seed 由 run seed manifest 的 `stream="replay"` 派生；禁止 `np.random` 全局状态。
- 首版 `batch_size` 行在一个 minibatch 内 **uniform without replacement**；`batch_size > live_size` 直接失败。
- batch 必须返回训练字段与 provenance：`obs/action/next_obs/reward/channel_valid/bootstrap/actor_gate/actor_weight/sample_weight/task_facts/sample_id/source_key/policy_version/inserted counters`。
- 配置中出现 `n_step>1`、PER、relabel、stratified retention、per-channel 异构 γ/m bootstrap → 在配置/准入边界 `ValueError`，不进入“接受但忽略”状态。

**持久化：**

- `replay.npz` 保存所有数值数组；`replay_meta.json` 保存 `sample_id/source_key/slice_id`、版本、counters、采样 RNG state、capacity、`next_sample_id`、active slice map、校验和。
- 加载后必须验证 `capacity`、字段 shape、`sample_id` 唯一递增、active source_key 无重复、游标与 size 一致；任一不符 fail loud。

### A5.6 W5.4：时钟与 UTD

| 时钟 | 定义 | 计数语义 |
|---|---|---|
| `physics_step` | episode 内实际执行物理子步 | 由 runtime/episode 记录；不跨 episode 累积 |
| `env_step` | `EnvRuntime.step()` 被调用的次数 | 跨 episode 累计；包含物理 delta=0 的退化帧，用于资源/评估调度 |
| `executed_env_step` | 物理 delta>0 的 env frame 数 | 诊断字段；不直接驱动 UTD |
| `agent_transition` | 一条通过准入并插入 replay 的 transition | replay/UTD 的单位；双 agent 一帧最多 2 |
| `collection_round` | 完成一次 job batch 的轮次 | 从 1 开始；checkpoint 恢复后从 `next_collection_round` 继续 |
| `critic_tick` | 一次成功 critic optimizer.step | UTD 分子；目标网络时钟也以此为基准 |
| `actor_tick` | 一次成功 actor optimizer.step | 默认每个 critic tick 一次，可由显式 interval 改变 |
| `temperature_tick` | 一次成功 α/λ optimizer.step | 固定系数模式不计；独立记录 |
| `target_tick` | 一次成功 target 软更新 | 与成功 critic tick 对齐，但单独持久化 |
| `eval_index` | 完成一次评估 | 只由评估边界递增；不能从日志行数推断 |
| `policy_version` | 一次行为策略导出/采集配置版本 | 采集 round 开始前生成；eval 导出不覆盖训练版本语义 |
| `checkpoint_index` | 完成一次 checkpoint bundle | 原子写入成功后递增 |

**轮内顺序：**

```text
1. 由 seed manifest 派生本 round 的 job seeds；导出当前行为策略并计算 fingerprint
2. 构造 SACJob 并执行 collection；worker 异常直接终止本轮
3. CollectedEpisode → experiment.build_slices → TransitionValidator
4. replay.add_slices；更新 env_step/agent_transition/insertion counters
5. warmup 判断与 UTD credit → critic/actor/temperature/target ticks
6. 到达 eval 边界则 deterministic eval（不进入 replay/UTD）
7. round 完整结束后写日志、导出、checkpoint
```

checkpoint 只承诺恢复**完整 round 边界**；worker 中途失败或训练中途 kill 不产生“半轮恢复”。

**UTD 记账：**

```text
eligible_new = 本轮新增且越过 warmup 阈值的 agent transitions
credit += utd_ratio * eligible_new          # warmup 前不累计
requested  = floor(credit)
actual     = min(requested, max_grad_steps_per_round)
dropped    = requested - actual             # round cap 之外丢弃，不滚存
credit    -= requested                      # 保留 <1 的小数预算
UTD_round  = actual / transitions_added
```

`requested/actual/dropped/credit` 均记录并持久化。`max_grad_steps_per_round` 是硬上限；被丢弃的 tick 明确报告为未满足训练预算，不在后续轮次偷偷追平。

**评估与预算：**

- 评估按 `next_eval_env_step` 边界调度；一轮跨过多个边界也只执行一次评估，`next_eval_env_step` 前进到当前 env_step 之后的下一个边界并记录被跨越的 index。
- `max_env_steps` 在完整 round 后检查；由于 episode 长度不可预知，允许 overshoot，但必须记录 overshoot 并按最后 checkpoint 验收，不在 episode 中途截断。
- `warmup`、eval episodes、video episodes 均不写入 replay；eval 用时和训练用时分别入账。

### A5.7 W5.5：配置、状态与版本分层

| 层 | 内容 | 变更规则 |
|---|---|---|
| `RunConfig` | experiment name、算法/schema/version、channels、objective/regularizer mode、policy arch、SACParams、CommonParamsSAC、env blueprint hash、seed policy、resume mode | run 创建时固定；canonical JSON 计算 `config_fingerprint` |
| `EffectiveConfig` | 当前实际生效的 LR、UTD、eval interval、explore schedule 等白名单可覆盖字段 | resume/热更新时记录 requested/effective/`effective_at_tick` |
| `ScheduleState` | 探索调度位置、curriculum/动态权重状态、LR schedule state | 必须 checkpoint；不能仅靠 env_step 重放 |
| `RuntimeCounters` | round、env/transition、四类 update tick、eval/export/checkpoint、`next_sample_id`、UTD credit | 由框架统一维护并持久化 |
| `ExperimentState` | `experiment.state()` 返回的任务级状态 | 语义归实验，但格式版本必须入 manifest |
| `RNGState` | collection root、replay、actor/target/temperature/diagnostic/eval streams、torch CPU/CUDA、policy私有 generator | 首选独立 generator；若某实现仍用全局 RNG，必须全量保存，否则不完整 |
| `DataManifest` | replay schema、capacity、source-key 状态、active slice map、replay RNG | 与 checkpoint 同生命周期校验 |

版本字段至少包括：`checkpoint_format_version`、`transition_schema_version`、`collection_contract_version`、`trainer_objective_version`、`policy_arch_version`、`regularizer_mode`、`objective_mode`、`env_blueprint_hash`、`reward_semantics_version`、`code_snapshot`。

`--set` 写入的是本次 run 的 `EffectiveConfig`，不是偷偷改实验代码语义；未列入白名单的 override 在完整恢复中拒绝。

### A5.8 W5.6：完整恢复、PPO 风格覆盖与 warm-start

**checkpoint bundle：**

```text
checkpoints/ckpt_<index>/
  manifest.json            # 版本、配置指纹、counters、artifact hashes、依赖文件列表
  trainer.pt               # actor、critic online/target、全部 optimizer、α/λ及optimizer、schedule state
  replay.npz               # replay 数据数组
  replay_meta.json         # source/sample identity、游标、active slices、checksums
  runtime_state.pt         # RNG、counters、collection/eval/export/checkpoint 状态
  experiment_state.json    # experiment.state() + semantics version
```

写入必须先写临时目录、校验 manifest，再原子 rename；失败/半成品目录不可作为 resume 输入。为控制磁盘，可以实现 `checkpoints/latest/` rolling bundle：每次 checkpoint 替换 latest，历史 `checkpoint_s*.pt` 可以只是不可完整恢复的 archival model artifact，但 manifest 必须明确标记 `resume_kind=warm_start_only`。

**恢复模式：**

| 模式 | 加载内容 | 配置语义 | 承诺 |
|---|---|---|---|
| `resume`（默认完整恢复） | trainer + replay + RNG + counters + experiment state | **参照 PPO**：白名单优化/调度字段按当前 `EffectiveConfig` 在下一 tick 生效；每次覆盖记录 event | 若当前 config 与 checkpoint config 相同且环境/内核一致，则为 state-exact continuation |
| `resume --config-lock` | 同上 | checkpoint config 全部冻结；任何 CLI/当前配置差异均拒绝 | 在相同代码、设备、库版本和确定性条件下追求 bitwise continuation |
| `warm_start` | 默认只加载 actor/critic 权重；可显式选择 optimizer/α | 新 run 的 config 全部生效 | 不恢复 replay/RNG/counters；新 `run_id`，manifest 记录 parent checkpoint |
| `reset_update` | 等同于 warm_start 的参数化入口 | 计数器清零 | 不得放在 `resume` 名下暗示完整续训 |

**PPO 对齐点与有意差异：**

- 对齐：PPO 恢复模型+optimizer+experiment state+RNG/loop state，同时让当前 config 的 actor/critic LR 在 resume 后生效；SAC 也允许白名单 LR/UTD/调度类字段生效并记录。
- 不复制：PPO 在 experiment name 改变时会打印后重置 experiment state；SAC 的完整 `resume` 对实验名/schema/通道/policy 架构不匹配直接拒绝，只有 `warm_start` 可跨实验加载权重。
- 不复制：PPO 没有 replay；SAC 的 `resume` 默认必须有完整 replay artifact，否则只能 warm_start。
- 不复制：旧 SAC optimizer mismatch 只打印后继续；SAC 完整恢复中 optimizer/state 形态不符直接失败。

**兼容性矩阵（resume 时）：**

| 检查项 | `resume` | `resume --config-lock` | `warm_start` |
|---|---|---|---|
| experiment/schema/objective/channel/policy arch | 必须一致 | 必须一致 | 只需可安全加载权重；不一致字段显式记录 |
| replay artifact | 必需 | 必需 | 不使用 |
| RNG/counters | 必需 | 必需 | 重新初始化 |
| LR/UTD/eval/schedule whitelist | 可按当前 config 覆盖并记录 | 拒绝覆盖 | 新 config |
| device/code/kernel | 记录并校验可用性；不一致时给兼容性错误 | 要求一致，否则拒绝 | 允许但标记非续训 |
| artifact 缺失/hash/schema 不符 | fail loud | fail loud | 若被指定字段缺失则 fail loud |

### A5.9 W5.7：SAC 自有 collection/job 边界

阶段二按 D-ROLLOUT-b vendor 到 `baseline/framework/sac/collection/`（模块名可微调）：

```text
sac/collection/
  job.py        # SACJob / SACBehaviorSpec / SACFactSpec
  episode.py    # 中立 Episode 数据拷贝，可加入 pre_action_facts / metrics 持久化
  recorder.py   # SACEpisodeRecorder，记录 behavior extras、pre-action facts、metrics
  runner.py     # 从 EpisodeRunner 拷贝修改：调用 pre-action fact provider，再 step
  rollouter.py  # 从 ParallelRollouter 拷贝裁剪：同序返回 CollectedEpisode，CPU 优先
```

**`SACJob` 草案字段：**

- `job_index`、`collection_round`、`policy_a_bp`、`policy_b_bp`、`env_bp`、`episode_seed`、`episode_options`；
- 每 agent 一个 `SACBehaviorSpec`：`explore_factor`、是否 random-start、`policy_mode=stochastic/deterministic`、诊断开关；
- `fact_specs`：实验声明的 pre-action facts（例如 `phi_pre`）及 post-action observer key；
- 不含 PPO 的 `SamplingSpec/SamplingContext/reference/delta_factor/ratio` 字段。

**`CollectedEpisode` 草案字段：** `episode`、`job_index`、`collection_round`、`episode_seed`、`agent_ids`、`policy_version`、`policy_fingerprint`、`behavior_specs`、`env_blueprint_hash`、`worker_id`、`wall_time`、`episode_metrics`。collection 返回顺序必须等于 jobs 顺序，不能只靠 episode_index 局部值拼接身份。

**关键行为约束：**

1. self-play 下两个 agent 即使使用同一权重，也必须拥有独立 behavior RNG 流；不得共享一个会互相推进的私有 generator。
2. `SACRunner` 在 `runtime.step()` 之前调用声明的 `PreActionFactProvider(accessor, agent_id)`；结果随帧进入 `pre_action_facts`，缺失/非有限即失败。
3. `basic_balance` 的 `phi_pre` 由 root height/uprightness 计算，与 `HeightPhiObserver` 的公式一致；`phi_post` 仍来自 post-action observer。fixture 必须额外验证 `phi_pre[t] == phi_post[t-1]`（t>0）以及 `phi_pre[0]` 与 reset 后实际状态一致。
4. worker 中任一 job 失败 → 终止该 collect round，不返回部分 episodes；partial data 不入 replay。
5. eval 使用 deterministic behavior spec，不消费训练 explore factor，也不写 replay。
6. 首版 CPU collector；GPU inference server、remote policy、async collection均为后续显式工作包。

### A5.10 W5.8：契约验证设计

| Fixture | 输入/构造 | 预期 |
|---|---|---|
| FX-1 双方 timeout | 合成 Episode，双 agent 到 T=200 timeout | boundary=200；terminated=False、truncated=True、bootstrap=1；末转移 `next_obs=final_observation` |
| FX-2 单方 imbalance | A 在 frame80 真终止、B timeout | A 的末转移 `next_obs=obs[80]`；terminated=True/bootstrap=0；B 正常 bootstrap |
| FX-3 中段终止/退化帧 | 帧内物理 delta>0 后终止，尾部 delta=0 | 终止帧计入；尾部退化帧不产生 transition |
| FX-4 缺必需字段 | 缺 reward observer、phi、action_extras/pre_action_fact | build/validate raise，不补零、不丢轨迹 |
| FX-5 长度/类型错 | observer 长度不等于 num_frames、gate 非有限 | ValueError |
| FX-6 PPO 对拍 | 同一条录制 basic_balance episode 分别过 PPO/SAC 语义转换 | reward、boundary、done、post-action reference 权重一致；SAC 的 pre-action gate 单独断言 |
| FX-7 replay wraparound | capacity 小于多 slice 总量并覆盖 | `sample_id` 唯一不复用；覆盖样本不再返回；active slice/trajectory 统计正确；slot 不是身份 |
| FX-8 source conflict | 重复插入同一 `source_key`，或同一 key 数据不同 | 均 raise；覆盖历史后也拒绝同 run 重插 |
| FX-9 恢复缺件 | 分别删 replay.npz、replay_meta、RNG、optimizer、manifest 字段 | `resume` fail loud；`warm_start` 只在声明字段足够时成功 |
| FX-10 config override | resume 时只改白名单 LR/UTD；再改实验名/schema/通道 | 白名单生效并记录 override event；身份字段变更拒绝完整 resume |
| FX-11 UTD | utd=0.5、新增 3 transitions 连续多轮、round cap、跨 warmup | requested/actual/dropped/credit 与冻结公式逐项一致 |
| FX-12 seed manifest | 训练/eval/holdout 全计划 seeds | 无重叠；同一 `(stream,round,job)` 恢复后不重复采集 |
| FX-13 phi_pre 对拍 | 真实/合成 balance episode | `phi_pre` 与同一 pre-action 状态一致；与 shifted `phi_post` 的一致性仅作回归校验 |
| FX-14 import independence | `import baseline.framework.sac`、SAC 单测、SAC smoke | `sys.modules` 无 `baseline.framework.ppo*`；挪走 `ppo/` 后仍通过 |

这些是阶段二必须变成永久测试的设计；当前只完成设计和一个旧 replay 探针，不代表生产实现已经满足。

### A5.11 阶段二实现拆分（由 A5 派生）

| 包 | 内容 | 出口 |
|---|---|---|
| P2-COLL-1 | vendor SAC collection/job/recorder/runner；CollectedEpisode provenance；pre-action fact provider | FX-4/6/13 的采集侧通过，无 PPO import |
| P2-DATA-1 | `sac_transition_v1`、slice schema、validator、两个实验的 build_slices | FX-1～6 通过 |
| P2-REPLAY-1 | SoA replay、sample_id/source_key、FIFO、uniform no-replacement、RNG、stats | FX-7/8 通过 |
| P2-CKPT-1 | checkpoint bundle、replay 持久化、runtime/RNG/counters/manifest | FX-9/10/12 通过 |
| P2-LOOP-1 | 新 train loop 时钟、UTD credit、eval/export/checkpoint 边界、seed manifest | FX-11/12 通过 |
| P2-TRAIN-1 | 按 A3/A4 接入新 batch、actor contract 和 S01 actor；不在本包启用八格全量 | 数学/梯度永久测试通过后再短训 |
| P2-IND-0 | package/lazy registry/train.py SAC 路径改造 | C1–C3 通过；PPO 路径不回归 |

顺序建议：P2-COLL-1 与 P2-DATA-1 先行，P2-REPLAY-1/CKPT-1 再接入 P2-LOOP-1；P2-TRAIN-1 最后接真实 batch。任何一项没通过对应 fixture，不进入真实任务训练。

### A5.12 W5.9：出口、限制与仍需注意项

- [x] 真实调用链与现状缺口已复核。
- [x] 实验/框架职责边界已冻结。
- [x] transition schema、时间对齐和准入规则已冻结。
- [x] replay 身份、覆盖、采样、持久化契约已冻结。
- [x] 时钟、UTD、eval/checkpoint 边界已冻结。
- [x] config/state/version 分层已冻结。
- [x] 完整恢复、PPO 风格白名单覆盖、strict config-lock、warm-start 的边界已冻结。
- [x] SAC 自有 collection/job/runner 边界已冻结。
- [x] W5 fixture 与阶段二拆分已列出。

**仍需后续落实而非本 W5 已完成：**

1. replay bundle 的实际序列化格式与磁盘成本需在阶段二 benchmark；契约要求完整恢复，但没有宣称已找到最优存储实现。
2. `basic_balance` 的 `phi_pre` provider 需要在 SAC runner 中实现；移位校验只是回归，不是数据来源。
3. `strict config-lock` 只保证状态与配置锁；bitwise 还依赖相同代码、设备、库版本和确定性内核，不能跨 GPU/CUDA 版本自动承诺。
4. `CollectedEpisode` 与 `SACJob` 的最终字段名可在实现期微调，但 provenance、pre-action facts、行为参数和版本不可缺。
5. 历史 `TrajectorySlice`、旧 `Job` tuple alias、旧 checkpoint、旧 replay 都不是兼容基线；阶段二实现应按新 schema 重写，而不是在原接口上打补丁。

## A6：诊断、捕获与溯源契约（W6，2026-10-08）

**范围与状态：** 本节是阶段一的诊断/捕获契约，不是生产实现。已对照 `ppo/dumpkit` 的能力分层与当前 SAC loop/trainer/replay；未移植 PPO 代码、未启动训练。PPO dumpkit 的可借鉴点是「run → dump → episode/trajectory/timeline」数据阶梯、懒加载访问层、CLI/HTTP 共用分析实现；SAC 的截面语义必须围绕 collection round、replay minibatch 与 critic tick 重新设计。

### A6.1 决策登记

| 编号 | 决策 | 适用边界 |
|---|---|---|
| SAC-R1-D27 | SAC 诊断因果链固定为 `CollectedEpisode → TransitionSlice → replay admission/overwrite → sampled minibatch → target → critic → actor → regularizer/target update → export/eval` | viewer 页面、CLI 和 dump schema 都按此链组织；不套用 PPO GAE/ratio/clip 页面 |
| SAC-R1-D28 | 运行指标 canonical 存储为 `metrics/events.jsonl`，事件类型至少区分 `round/tick/eval/export/checkpoint/debug/config` | `train.log` 可保留紧凑 `__RAW_STATS__` 镜像，但不是唯一数据源；每条事件带完整时钟与 schema version |
| SAC-R1-D29 | 捕获分 L0–L3：常态标量、结构摘要、预约可重算截面、重型分析 | L0/L1 默认可长期开启；L2 按需/预约；L3 仅 dump-only |
| SAC-R1-D30 | 最小可重算 dump 主键是 `critic_tick`，collection round 是上下文键 | 目录建议 `dumps/critic_c{:08d}/`；round-only 采集 dump 与 L3 tick-range 另立类型，不把一整轮更新压成一个截面 |
| SAC-R1-D31 | 常态 tick 信息进入有界内存 ring，不逐 tick 全量写盘 | 初始 `debug_recent_ticks=4096`；round 事件写聚合/分布，异常或 dump 时可转储 ring |
| SAC-R1-D32 | L2 dump 默认保存 actor/critic online/target、actor/critic/regularizer optimizer、α/λ、batch、显式采样噪声/RNG、EffectiveConfig 与更新前后摘要 | 没有 optimizer 的截面只能解释 forward/loss，不承诺复算 optimizer step |
| SAC-R1-D33 | dump 保留策略为 latest-N + 显式 pin + 总字节上限 | 初始 `keep_last=8`、`max_total_bytes=64GiB`；超限删除最早未 pin dump 并记录 debug event |
| SAC-R1-D34 | 样本溯源使用 `sample_id/source_key/slice_id`，dump 中 batch 行自包含 | slot/index 只作现场诊断；replay 覆盖后仍能用 dump 数据解释该样本，但完整 episode/video 只在对应 artifact 存在时可用 |
| SAC-R1-D35 | 诊断必须使用独立 `debug` RNG stream；重型分析不得改训练参数、optimizer、replay、schedule | 逐样本梯度在 clone/隔离会话中计算，或严格保存恢复 `.grad`；诊断采样不消耗训练 RNG |
| SAC-R1-D36 | schema/训练必需数据缺失 fail loud；可选重型诊断失败默认隔离并写 `failed.json` 与 debug event | `debug_strict=true` 时升级失败；任何路径不得补零伪造缺失数据 |
| SAC-R1-D37 | SAC debug 拥有自己的 catalog/access/capture/analysis/CLI 层 | 不 import `ppo.dumpkit`、PPOBuffer、Trajectory、UpdateStats；CLI/HTTP/viewer 共用同一 SAC 分析函数 |

### A6.2 PPO 能力对照与 SAC 差异

| PPO dumpkit 能力 | SAC 采用方式 | 不可照搬点 |
|---|---|---|
| run summary / metrics / catalog | 采用；指标按 `metrics/events.jsonl` 与 SAC catalog | PPO x 轴是 update；SAC 必须同时支持 collection_round、critic_tick、env_step、agent_transition |
| sentinel/scheduled dump | 采用请求模型 | PPO dump=一个 on-policy update；SAC 最小可重算单位是一个 critic tick，collection round 只提供来源上下文 |
| episodes/trajectories/frame 下钻 | 采用分层思路 | SAC replay 样本可能来自旧 round；batch 样本不能假设属于当前 episodes |
| gradsig / per-sample gradient | 作为 L3 候选能力 | 梯度定义为 SAC actor/critic/regularizer 各项，不含 GAE/ratio/floor surrogate |
| render/delta/rollout 派生工件 | 保留为后续工具 | 物理重演需要足够 state/seed/blueprint；符号 transition 数据不等于可复现 MuJoCo 轨迹 |
| CLI/HTTP/UI 同源 | 强制采用 | 只能共享 `sac.debugkit` 的分析函数，不能反向依赖 PPO 实现 |

### A6.3 指标事件与时钟契约

每条 metric event 使用 `sac_metrics_v1`：

```json
{
  "schema_version": "sac_metrics_v1",
  "event_type": "round|tick|eval|export|checkpoint|debug|config",
  "run_id": "...",
  "collection_round": 42,
  "env_step": 123456,
  "executed_env_step": 123450,
  "agent_transition": 240000,
  "critic_tick": 98765,
  "actor_tick": 98765,
  "temperature_tick": 98765,
  "target_tick": 98765,
  "eval_index": 20,
  "checkpoint_index": 12,
  "policy_version": "r00042:abcd1234",
  "metrics": {},
  "meta": {}
}
```

不要求每个字段在每个事件中非空，但主键必须明确：`round` 事件必须有 `collection_round`；`tick` 事件必须有 `critic_tick`；`eval` 必须有 `eval_index`。UI 可按任一时钟切换 x 轴，但每个指标 catalog entry 必须声明主时钟和单位。

**命名空间：**

| 前缀 | 内容 |
|---|---|
| `run.*` | run 配置、版本、resume/override manifest |
| `collect.*` | jobs、episodes、frames、pre-action facts、episode timing、worker failure |
| `data.*` | admitted/rejected transitions、boundary 类型、terminated/truncated/degenerate 计数 |
| `replay.*` | size/capacity、inserted/overwritten、active slices、age、source/policy-version 分布 |
| `batch.*` | batch size、sample age、source round/job/policy version、behavior params、sample weight |
| `target.*` | reward sum、bootstrap、entropy/U 正则项、Q twin/min、target/TD 分布 |
| `critic.<channel>.*` | 各通道 Q1/Q2、pred/target/TD/loss/grad/clip |
| `actor.*` | actor loss、Q term、entropy/U term、gate/contribution、action/σ/logit 分布、grad |
| `regularizer.*` | α/λ、target entropy/U、floor mask、penalty、clamp、optimizer state摘要 |
| `update.*` | requested/actual/dropped UTD ticks、LR、隔离关系、耗时 |
| `eval.*` | deterministic eval 指标 |
| `export.*` | policy artifact/fingerprint/version |
| `checkpoint.*` | bundle、耗时、大小、校验 |
| `debug.*` | dump 请求/完成/失败、诊断耗时、ring/disk 使用 |
| `config.*` | effective/requested override event |

通道名、agent 名等进入扁平 metric key 前必须先 sanitize（`.`/空白/控制字符替换为 `_`）；catalog 保存原始名到显示名的映射，避免同名冲突。

### A6.4 L0–L3 捕获分层

| 层 | 默认状态 | 内容 | 用途 |
|---|---|---|---|
| L0 | 常开 | round/eval/checkpoint/config/debug 标量事件；每 tick 的小标量进入 ring | 长跑趋势、异常定位入口 |
| L1 | 常开，轻量 | 每 round 的 replay/batch/target/actor/channel 分布摘要、provenance histogram、UTD 记账 | 判断旧数据、通道压制、α/Q 异常 |
| L2 | sentinel/CLI/schedule 触发 | 单个 critic tick 的完整 batch + pre/post trainer state + optimizer + RNG/noise + config | 离线逻辑重算 target/loss/update |
| L3 | 显式请求 | tick range、per-sample gradients、channel contribution、mixture 分解、action Jacobian、replay snapshot/index、core-state replay | 深度诊断；默认可关闭 |

L1 不保存整条 transition；但保存足以解释结构问题的聚合量：活跃 slice 数、policy_version/round/age 分布、terminated/truncated 比例、per-channel valid/reward/actor_gate 分布、batch 来源分布、target/TD 分位数、实际 UTD。

### A6.5 Dump 目录与 `sac_dump_v1`

推荐目录形态：

```text
dumps/
  critic_c00098765/                    # L2 最小可重算截面
    request.json
    manifest.json
    events.jsonl                       # 本 dump 相关 debug/config/tick 摘要
    collection.json                    # round/job/policy/env/provenance manifest
    collection_episodes.npz            # 当前 round 可选 episode 数据；缺失时 manifest 标注
    replay_index.npz                   # active sample_id/source_key/slot/age 索引，不是完整 replay
    batch.npz                          # 本 tick sampled rows + sample_id/source_key + schema 字段
    target.npz                         # reward/bootstrap/noise/next action/Q/logp/U/target/TD
    actor_update.npz                   # new action/logp/U/gate/channel Q/contribution/loss terms
    regularizer.npz                    # α/λ/target/error/floor/clamp
    trainer_pre.pt                     # actor/critic online/target + optimizers + α/λ + schedule
    trainer_post.pt
    runtime_state.pt                   # replay/sampling/train/debug RNG 与 counters 快照
    analysis.json                      # capture-time summary / checks
    failed.json                        # 可选；只在捕获失败路径存在
  round_r00042/                        # L1+ collection-side dump，不承诺 tick 重算
  range_r00042_c00012000_c00012127/    # L3，可有界 tick range
```

`manifest.json` 必须包含：`dump_schema_version=sac_dump_v1`、target kind、clocks、run_id、experiment/schema/config/code fingerprints、capture level、evidence level、artifact 列表及 hash、缺失项、请求来源/hypothesis、保留策略标记。

### A6.6 可重算证据等级

| 等级 | 必需工件 | 承诺 |
|---|---|---|
| `explain` | manifest、batch/index、已保存中间量 | 只解释已记录值，不重新执行 forward/update |
| `recompute` | explain + trainer_pre + batch + 显式 action noise/RNG + config | 重新计算 target、critic loss、actor loss、regularizer loss；以声明容差比较 |
| `apply_update` | recompute + optimizer state + update RNG | 重放 optimizer.step/target update；同代码/设备/库/确定性条件下可追求 bitwise |
| `physical_replay` | collection episode + env blueprint + seed + 必要 core_state | 只在此类工件存在时承诺；缺件显示不可用 |

默认 L2 目标是 `apply_update` 级别的数据完备性；是否达到 bitwise 仍受代码、设备、库版本和确定性内核约束。未预约的历史 tick 只能给 `explain`，不能事后伪造成可重算。

### A6.7 样本溯源与回放边界

- `batch.npz` 每行必须含 `sample_id/source_key/slice_id/collection_round/job_index/agent_id/frame_index/policy_version/inserted counters`。
- `source_key` 是规范化身份，不通过 obs 内容反推 frame；`flat:<idx>` 只允许用于无法映射的诊断占位，并在 manifest 计数。
- 当前 round 的 collection dump 可提供 episode 级页面；历史 replay 样本若对应 episode artifact 已不存在，只能显示 transition/batch/provenance，不得伪造视频或完整轨迹。
- 物理重演要求保存或能重建 `(env_blueprint, episode_seed, episode_options, plugin state, action sequence)`；若有中途 mutator/随机插件，必须保存相应 RNG/core_state，否则仅做符号级回放。
- 视频渲染是从 blueprint+policy artifact 再跑环境得到的派生工件；manifest 必须记录它可能与原 MuJoCo 轨迹存在数值漂移，不能把渲染帧当作 bitwise 证明。

### A6.8 诊断隔离与失败边界

1. `debug` RNG stream 由 run seed manifest 独立派生；诊断采样、样本选择、dropout/shuffle 均不得推进 replay/training/eval RNG。
2. L0/L1 字段缺失或类型/schema 错误属于数据契约失败，训练 fail loud。
3. L2/L3 是可选重型诊断：捕获失败写 `failed.json`、`debug.capture_status=error` 并继续训练；`debug_strict=true` 时终止训练。已写半成品先落临时目录，不进入正式 dump 索引。
4. 重型梯度/参数 delta 使用 clone 或严格恢复的隔离会话；诊断结束后训练模型参数、optimizer state、`.grad`、replay cursor、schedule state 必须与诊断前一致。
5. 诊断自身成本进入 `debug.time_s/debug.memory_bytes/debug.disk_bytes`；超过配置预算可拒绝新的 L3 请求，但不能覆盖已有 pinned dump。
6. viewer/CLI 的只读分析不得写训练目录，除明确派生工件目录（render/delta/replay）且需要独立 job 状态。

### A6.9 CLI/API/viewer 数据面

阶段二先交付 CLI 与结构化 JSON，阶段五补完整 viewer；但访问层 schema 从首版冻结：

```text
sac/debugkit/
  metric_catalog.py   # 指标语义、分区、单位、主时钟
  events.py           # metrics/events.jsonl reader/writer
  access.py           # SacDumpDataset：row spaces + lazy npz/pt/json access
  capture.py          # L2/L3 capture orchestration
  analysis.py         # inspect/batch/target/sample/timeline/recompute functions
  requests.py         # dump request/schedule schema
  viewer/             # 阶段五 HTTP/UI；调用同一 analysis/access
sac/debug.py          # runs/summary/metrics/dump/inspect/samples/trace/query/...
```

离线 CLI 子命令至少规划为：`runs`、`summary`、`metrics`、`catalog`、`dump`、`inspect`、`batch`、`target`、`trace`、`timeline`、`recompute`、`query`、`render`、`viewer`。`query` 与 HTTP endpoint 共用同一 dispatch；输出 JSON-safe。

### A6.10 故障注入与验证矩阵

| 编号 | 注入/场景 | 预期观测与定位 |
|---|---|---|
| DX-1 | bootstrap/terminated 错、timeout 标成真终止 | `target.bootstrap`/`data.terminated` 异常；batch trace 指到 source frame |
| DX-2 | 缺 reward observer、phi_pre 或 behavior extra | collection/validation fail loud；事件记录缺失字段，不进入 replay |
| DX-3 | 重复 source_key 或同 key 不同内容 | replay admission 拒绝；`debug` event 含冲突双方身份 |
| DX-4 | replay 覆盖后查询旧样本 | dump/batch 仍可解释样本；live replay 显示 overwritten，episode 页按 artifact 可用性降级 |
| DX-5 | behavior policy 过旧/陈旧数据过多 | `replay.age_*`、`batch.policy_version`、source-round histogram 可见 |
| DX-6 | 采样 RNG 恢复 | checkpoint 后同一 cursor 产生同一 batch `sample_id` 序列 |
| DX-7 | α/λ 失控或 fixed/auto 配置错 | `regularizer.*` 与 effective config 可区分 requested/effective/clamped |
| DX-8 | Q/TD 异常或 reward_scale 误读 | target 分解显示 reward、bootstrap、entropy/U 项，不能直接看混合 loss |
| DX-9 | actor_gate 全零/通道压制 | `actor.gate/contribution` 与 `critic.<ch>` 分开，零 gate 不等于 critic 不学 |
| DX-10 | mixture logits 无梯度 | L3 显示 component weights/logits/任务梯度路径；能区分采样无梯度与枚举梯度 |
| DX-11 | U-bonus/U-floor 接错位置 | target/actor/regularizer 三处字段一致性检查失败 |
| DX-12 | optimizer/critic target 缺失 | dump/recompute/apply_update 等级降级并明确原因；checkpoint resume fail loud |
| DX-13 | 诊断消耗训练 RNG | 诊断前后 replay sample 序列与 policy noise 不变，否则测试失败 |
| DX-14 | dump 重算不一致 | `recompute` 输出 stored/recomputed/abs_diff/tolerance，超限列字段 |
| DX-15 | CLI 与 HTTP 结果不一致 | 同一输入 dump 的 JSON canonical 输出必须一致 |
| DX-16 | import independence | `baseline.framework.sac.debugkit` 不加载 `baseline.framework.ppo.*` |

### A6.11 阶段二实现拆分（由 A6 派生）

| 包 | 内容 | 出口 |
|---|---|---|
| P2-DBG-1 | `metrics/events.jsonl`、metric catalog、双时钟/多时钟事件写入 | DX-15/16 的基础数据面可用 |
| P2-DBG-2 | tick ring、round L0/L1 聚合、debug RNG/资源计数 | 常态训练开销可测，诊断不改变状态 |
| P2-DBG-3 | dump request/schedule、L2 critic-tick capture、manifest/retention | 指定 tick 可产生完整 `sac_dump_v1` |
| P2-DBG-4 | access/analysis/CLI；sample trace 与 target/loss recompute | DX-4/5/6/12/14/15 通过 |
| P2-DBG-5 | viewer/API、round/episode/replay/target/actor/regularizer 页面 | 阶段五完整交互能力，不反向改 schema |
| P2-DBG-6 | fault injection harness | DX-1～14 自动化回归 |

顺序建议：P2-DBG-1/2 与 collection/data 包并行；P2-DBG-3 等 trainer/replay 契约落地后接入；P2-DBG-4 必须在首个真实训练验收前可用，否则无法解释失败训练。

### A6.12 出口状态与限制

- [x] PPO debug 能力已按架构层对照，未把 PPO 指标语义迁入 SAC。
- [x] SAC 诊断因果链、指标事件、时钟和命名空间已冻结。
- [x] L0–L3 分层、dump identity、目录 schema、保留策略已冻结。
- [x] explain/recompute/apply_update/physical_replay 四级证据已冻结。
- [x] 样本溯源、诊断 RNG、失败隔离、CLI/HTTP 同源与独立性边界已冻结。
- [x] 故障注入矩阵与阶段二 debug 工作包已列出。

**本 W6 仍未完成实现：** 尚无 `sac/debugkit`、metric events、dump capture、access/analysis 或 viewer；默认数值（tick ring 4096、keep 8、64GiB）是实现起点而非性能实测结论；`physical_replay` 只在相应 state 工件存在时成立。

## A7：风险、验证矩阵与阶段二实施 DAG（W7，2026-10-08）

**范围与状态：** 本节是阶段一的验证收口，不是实现测试报告。已把 A1–A6 的决策、fixture 和阶段二包统一到同一索引；未改生产代码、未启动训练。W3/W4 的数学探针只作为可行性证据，不替代阶段二永久测试。

### A7.1 ID 规范化

- 决策编号：`SAC-R1-D01`～`SAC-R1-D37` 连续且无重复；每条按 `approved / delegated / proposed / design` 标注证据强度，不能混写。
- Fixture 重名修正：A2 中的 `FX-1`～`FX-6` canonical 为 `FX-A2-01`～`FX-A2-06`；A5 中的 `FX-1`～`FX-14` canonical 为 `FX-A5-01`～`FX-A5-14`；A6 的 `DX-1`～`DX-16` canonical 为 `DX-01`～`DX-16`。
- 阶段二包 canonical：`P2-IND-0`、`P2-COLL-1`、`P2-DATA-1`、`P2-REPLAY-1`、`P2-DBG-1`、`P2-TRAIN-1`、`P2-LOOP-1`、`P2-CKPT-1`、`P2-DBG-2`、`P2-DBG-3`、`P2-DBG-4`、`P2-ENV-1`。A5 旧写法 `P2-IND-1` 统一为 `P2-IND-0`。
- 证据状态：`design`（仅有契约）、`probed`（阶段一低成本验证）、`implemented-test`（永久测试）、`integration-passed`、`training-passed`。当前没有 `training-passed` 项。

### A7.2 风险矩阵 schema 与验证层级

每条风险按以下字段登记：`risk_id, decision_refs, contract, failure_mode, blast_radius, likelihood, detectability, severity, validation_level, test_id, input_or_injection, expected, tolerance, evidence_artifact, owner_phase, gate, status`。

- **severity**：`blocker` = 目标错误/恢复错误/独立性破坏；`major` = 可能导致错误训练或诊断失效但有检测路径；`minor` = 可用性/成本问题。
- **validation_level**：`static`、`unit`、`property`、`integration`、`smoke`、`short-train`、`acceptance`。
- **gate**：`pre-P2`、`P2-gate`、`pre-P3`、`pre-P4`、`pre-P5`、`pre-P6`、`optional`。
- blocker 不允许以训练曲线、单个 smoke 或历史 PPO 测试作为唯一证据；必须有一个可独立证伪的输入/预期。

### A7.3 风险矩阵

| risk_id | refs | 失败模式 | severity | 验证/gate | canonical tests / fixtures | status |
|---|---|---|---|---|---|---|
| R-MATH-01 | D01–D03,D12 | Bellman target 公式错、熵/U 重复计入或漏计、twin 选择语义错 | blocker | unit+property；P2-gate | T-MATH-01/02/04 | probed（W3） |
| R-MATH-02 | D05,D08 | actor gate、channel_valid、bootstrap、sample_weight 混用；actor/critic 梯度泄漏 | blocker | unit+integration；P2-gate | T-MATH-03、DX-09 | design |
| R-MATH-03 | D07,D12 | 不支持组合被静默接受，或 U 模式只改 actor 不改 target/系数 | blocker | static+unit；P2/pre-P4 | T-MATH-04、DX-11 | design |
| R-POL-01 | D09,D11,D14 | SAC actor API、logπ、可微采样或 TN 数值内核不满足 | blocker | unit+property；P2-gate | T-POL-01/02 | probed（W4 风险已定位） |
| R-POL-02 | D10,D15 | mixture logits 得不到任务梯度，或分量语义被误改 | major | unit+property；pre-P4 | T-POL-03、DX-10 | probed（方案已验证，生产未实现） |
| R-POL-03 | D13,D25 | β/π/eval 分布混淆，或 `action` 与实际执行动作不一致 | blocker | unit+integration；P2-gate | T-POL-04、T-DATA-05 | design |
| R-DATA-01 | D16,D26,A2 | transition 边界、`next_obs`、slice 身份或 provenance 错 | blocker | unit+property+integration；P2-gate | T-DATA-01/02、FX-A5-01～03 | design |
| R-DATA-02 | D05,D20,A2 | timeout/terminated/bootstrap 错，或 `phi_pre` 来源错 | blocker | unit+integration；P2/pre-P3 | FX-A5-01～06、FX-A5-13、FX-A2-01～06 | design |
| R-DATA-03 | D17,D36 | 缺 reward/fact/extra 被补零、截断或静默丢弃 | blocker | unit+static；P2-gate | FX-A5-04/05、DX-02 | design |
| R-RPL-01 | D16,D18 | replay 容量按错单位、采样非 uniform、批内重复或不支持功能被接受 | blocker | unit+property；P2-gate | T-RPL-01/03/05、FX-A5-07 | design |
| R-RPL-02 | D17,D19,D34 | `sample_id/source_key` 不稳定、覆盖后身份混淆、持久化丢身份 | blocker | unit+property；P2-gate | FX-A5-07/08、DX-03/04 | design |
| R-RPL-03 | D19,D23 | replay 恢复不完整却被当成完整 resume | blocker | integration；P2-gate | FX-A5-09、T-CKPT-01 | design |
| R-COL-01 | D21,D26 | collection 仍依赖 PPO、job 顺序/provenance 丢失、worker partial data 入 replay | blocker | static+integration；P2-gate | FX-A5-14、T-COL-01/02 | design |
| R-COL-02 | D17,D20,D25 | 行为参数、policy fingerprint、pre-action fact 或 executed action 记录错 | blocker | unit+integration；P2/pre-P3 | T-COL-03、FX-A5-13、T-DATA-05 | design |
| R-CLK-01 | D22 | UTD 小数预算丢失、counters 错位、eval/checkpoint 边界不可恢复 | blocker | unit+property；P2-gate | FX-A5-11/12、T-CLK-01/02 | design |
| R-CLK-02 | D22,A2 | eval 写 replay、消耗训练 RNG 或使用 stochastic spec | major | integration；pre-P3 | T-CLK-03、DX-13 | design |
| R-CKPT-01 | D19,D23,D24 | artifact 缺失仍静默 warm-start、optimizer/target/RNG 恢复错 | blocker | integration+property；P2-gate | FX-A5-09/10/12、T-CKPT-01～04 | design |
| R-CKPT-02 | D24 | 非白名单配置覆盖导致“恢复”语义改变 | blocker | static+integration；P2-gate | FX-A5-10 | design |
| R-DBG-01 | D28–D31 | 无 canonical events、时钟混乱或常态诊断成本不可控 | major | unit+integration；P2/pre-P5 | T-DBG-01/02、T-PERF-01 | design |
| R-DBG-02 | D30,D32,D34 | dump 无法重算，或 batch 样本不能追到 source | blocker | integration；pre-P3 前至少有 recompute 基础能力，P5 完整 | T-DBG-03/04、DX-04/14 | design |
| R-DBG-03 | D35,D36 | 诊断污染训练 RNG/状态，或可选诊断失败造成静默数据缺口 | blocker | unit+integration；P2-gate | DX-13、T-DBG-05 | design |
| R-DBG-04 | D37 | CLI/HTTP/viewer 各自解析导致结果不一致 | major | integration；pre-P5 | DX-15/16、T-DBG-06 | design |
| R-TASK-01 | A2,D04 | standup/basic_balance 的奖励、边界、评估指标与冻结语义不一致 | blocker | integration+smoke；pre-P3 | FX-A2-01～06、T-TASK-01 | design |
| R-TASK-02 | A2 | 评估 seeds、预算、持续达标或最终 checkpoint 口径被修改 | blocker | acceptance；pre-P6 | T-TASK-02 | design |
| R-TASK-03 | A3.10 | self-play 非平稳与 replay 陈旧数据导致目标不可解释 | major | debug+short-train；pre-P6 | DX-05、T-TASK-03 | design |
| R-IND-01 | D21,D37,A1 | SAC 直接或传递 import PPO、registry/train.py 静默耦合 | blocker | static+integration；P2-gate | FX-A5-14、DX-16、T-IND-01 | design |
| R-INT-01 | A5,A6 | fake env/真实 env smoke 前未覆盖端到端路径 | blocker | integration+smoke；P2-gate | T-INT-01/02 | design |
| R-PERF-01 | D31,D33,A5 | replay/checkpoint/dump 磁盘内存不可控，诊断拖垮训练 | major | property+short-train；P2/pre-P6 | T-PERF-01/02 | design |

### A7.4 测试注册表

| test_id | 输入/注入 | 预期与容差 | 证据 artifact | phase |
|---|---|---|---|---|
| T-MATH-01 | 单通道、固定 obs/action/reward/Q/α | target 与手算一致；CPU float64 `atol<=1e-10`，torch float32 `atol<=2e-5` | pytest + `target.npz` | P2-TRAIN-1 |
| T-MATH-02 | 两通道、常量非负权重、相同 γ/bootstrap | 合并 target 等于标量 SAC；同 tolerance | pytest | P2-TRAIN-1 |
| T-MATH-03 | actor 参数梯度目标、独立 critic、target net | critic optimizer 不改 actor；actor optimizer 不改 critic；target 只在 target_tick 变 | pytest state digest | P2-TRAIN-1 |
| T-MATH-04 | `shannon/u_bonus/u_floor` 三种 mode 的合成 batch | actor、target、系数状态一致；互斥配置非法即报错 | pytest + metrics event | P2-TRAIN-1，pre-P4 扩展 |
| T-POL-01 | S01 actor contract fixture | `sample_action` 返回可微 action/logπ；`deterministic_action`、blueprint、私有 RNG 满足契约 | pytest + export artifact | P2-TRAIN-1 |
| T-POL-02 | 极端 σ、边界均值、batch 样本 | log_prob 有限、密度积分误差在声明容差、无静默塌缩；固定噪声有限差分通过 | pytest | P2-TRAIN-1 |
| T-POL-03 | mixture toy Q、K=1/K>1 | 枚举梯度与独立积分/FD 一致；logits 梯度非零且方向正确；K=1 退化一致 | pytest | pre-P4 |
| T-POL-04 | 相同 obs 下 stochastic behavior、training sample、deterministic eval | 三者显式分离；behavior spec 不改训练 π | pytest/integration | P2-COLL-1 |
| T-DATA-01 | FX-A5-01/02/03 合成 episode | boundary、next_obs、terminated/truncated/bootstrap exact | pytest | P2-DATA-1 |
| T-DATA-02 | FX-A2-01～06 + A5 等价构造 | PPO 语义对拍；SAC pre-action gate 独立断言 | pytest + fixture episode | P2-DATA-1/pre-P3 |
| T-DATA-03 | FX-A5-04/05 缺字段/长度错 | 全部 fail loud，不补零 | pytest | P2-DATA-1 |
| T-DATA-04 | FX-A5-13 phi_pre | 与 pre-action accessor 同值；shifted `phi_post` 只作回归 | pytest | P2-COLL-1 |
| T-DATA-05 | action mapping/插件改写模拟 | `action=executed`，原始输出进入 `policy_action`；无二义性 | pytest | P2-COLL-1 |
| T-RPL-01 | 双 agent 合成 slices，容量以 transition 断言 | 双活帧=2、单活帧=1；capacity/warmup/UTD 分母 exact | pytest | P2-REPLAY-1 |
| T-RPL-02 | FX-A5-07 wraparound | `sample_id` 单调不复用；覆盖样本不可采样；active slice 统计 exact | pytest | P2-REPLAY-1 |
| T-RPL-03 | FX-A5-08 source conflict | active duplicate/issued slice 重插/同 key 数据冲突均 raise | pytest | P2-REPLAY-1 |
| T-RPL-04 | 固定 seed 采样、save/load RNG | batch 内无重复 sample_id；恢复后序列一致 | pytest | P2-REPLAY-1/P2-CKPT-1 |
| T-RPL-05 | n_step/PER/relabel/stratified 配置 | 配置或准入边界显式拒绝 | pytest | P2-REPLAY-1 |
| T-COL-01 | jobs 乱序返回/重复 episode_index | `CollectedEpisode` 与 job 顺序、round/job/agent 身份一致 | pytest | P2-COLL-1 |
| T-COL-02 | worker 任一 job 抛错 | 本轮无 partial data；错误传播；不污染 replay/RNG | pytest/integration | P2-COLL-1 |
| T-COL-03 | behavior spec/explore metadata | 每个样本记录 requested/effective params、policy fingerprint | pytest | P2-COLL-1 |
| T-CLK-01 | FX-A5-11 UTD 序列 | requested/actual/dropped/credit exact；round cap 明示丢弃 | pytest | P2-LOOP-1 |
| T-CLK-02 | resume 前后 counters/边界 | next round/tick/eval/checkpoint 不重复、不回退 | pytest/integration | P2-CKPT-1/P2-LOOP-1 |
| T-CLK-03 | eval round | eval 不写 replay、不消耗 train/replay RNG、不产生 UTD | integration | P2-LOOP-1 |
| T-CKPT-01 | FX-A5-09 删 artifact/字段 | resume fail loud；warm_start 只在声明字段足够时成功 | pytest | P2-CKPT-1 |
| T-CKPT-02 | FX-A5-10 白名单/身份字段 override | 白名单生效并记录；身份字段变更拒绝 resume | pytest | P2-CKPT-1 |
| T-CKPT-03 | FX-A5-12 seed manifest | train/eval/holdout/debug streams 无重叠；恢复后不重复采集 | pytest | P2-CKPT-1 |
| T-CKPT-04 | checkpoint 中途失败/半成品目录 | 原子性成立；半成品不可 resume | pytest | P2-CKPT-1 |
| T-DBG-01 | 合成 round/tick/eval/config event | `sac_metrics_v1` schema、主键、时钟、单位校验 | pytest + events.jsonl | P2-DBG-1 |
| T-DBG-02 | tick ring 容量与 dump 触发 | 最近 N tick 可查，不写爆磁盘；round 聚合口径一致 | pytest | P2-DBG-2 |
| T-DBG-03 | 预约 critic_tick dump | `sac_dump_v1` 必需 artifact、hash、manifest、identity 完整 | pytest + dump dir | P2-DBG-3 |
| T-DBG-04 | 对已捕获 L2 dump 重算 | stored vs recomputed target/loss 在声明容差内；缺失字段显示不可用 | pytest + analysis.json | P2-DBG-4 |
| T-DBG-05 | 诊断前后 RNG/state digest | replay sample 序列、模型、optimizer、schedule 不变 | pytest | P2-DBG-2/4 |
| T-DBG-06 | 同一 dump 走 CLI/HTTP | canonical JSON 输出一致；无 `baseline.framework.ppo.*` import | pytest/integration | P2-DBG-4/pre-P5 |
| T-IND-01 | `import baseline.framework.sac`、单测、smoke | `sys.modules` 无 PPO；临时挪走 `ppo/` 后 SAC 仍通过 | pytest/integration | P2-IND-0 |
| T-INT-01 | fake/small env + SAC collector | episode→slice→replay→tick→events 端到端通过 | integration test | P2-ENV-1 |
| T-INT-02 | 真实 humanoid21 少量 episode/update | 真实 smoke 通过；只证明链路，不证明收敛 | smoke log + events | P2-ENV-1 |
| T-TASK-01 | 录制 episode 对拍与 on_eval | 指标/边界同 A2；SAC 差异字段显式存在 | pytest + eval fixture | P2-DATA-1/pre-P3 |
| T-TASK-02 | 阶段六验收 run manifest | seeds、阈值、连续窗口、预算、最终 checkpoint 符合 A2 | run manifest/eval log | pre-P6 |
| T-TASK-03 | 行为版本老化注入 | replay/batch age 与 opponent/policy version 分布可见 | metrics/dump | pre-P6 |
| T-PERF-01 | 长 round + dump 开关 | L0/L1 成本、disk、内存有界并记录 | metrics event | P2-DBG-2 |
| T-PERF-02 | replay/checkpoint 序列化规模 | bundle 大小、IO 时间、内存峰值满足实测预算 | benchmark artifact | P2-CKPT-1 |

### A7.5 阶段二 DAG 与 gate

```text
G2.0 P2-IND-0 ── independence scaffold / lazy registry / CLI gate
  ├─ P2-COLL-1 ── collection contract / provenance / pre-action facts
  │    └─ P2-DATA-1 ── transition schema + validators
  │         └─ P2-REPLAY-1 ── identity/FIFO/uniform/RNG/persist
  │              ├─ P2-CKPT-1 ── bundle/manifest/resume
  │              └─ P2-LOOP-1 ── clocks/UTD/eval/checkpoint loop
  ├─ P2-DBG-1 ── metrics/events/catalog
  │    └─ P2-DBG-2 ── tick ring + L0/L1
  └─ P2-TRAIN-1 ── S01 actor + trainer math
       └─ P2-LOOP-1
            ├─ P2-DBG-3 ── L2 dump capture
            │    └─ P2-DBG-4 ── access/analysis/recompute/CLI
            └─ P2-ENV-1 ── fake env integration → real env smoke
```

**阶段二 gate：**

- **G2.0**：SAC import 不加载 PPO；registry/train.py 不静默吞 SAC 不支持参数。
- **G2.1**：collection/data 全部 FX-A5-01～06/13 和 T-COL-* 通过。
- **G2.2**：replay/clock/checkpoint 的 FX-A5-07～12 与 T-RPL/T-CLK/T-CKPT 通过。
- **G2.3**：S01 actor 与标准 Shannon SAC 的 MATH/POL 永久测试通过。
- **G2.4**：fake env 集成通过，metrics event 和 tick ring 可观测。
- **G2.5**：真实 env smoke 通过；此时仍不声称任务收敛。
- **G2.6**：L2 dump 可对指定 critic tick 做 recompute 级核对；未通过前不得进入长训验收。

### A7.6 未决项分类与重开条件

**不阻塞阶段二，但必须在对应阶段前解决：**

- 八格全量适配、mixture 生产实现、U 路线训练效果：pre-P4。
- 完整 viewer/HTTP 页面与所有 DX 故障注入自动化：pre-P5。
- 多 seed 收敛、预算内效率、self-play 陈旧数据影响：pre-P6。
- `physical_replay` 的完整 core-state 方案：仅当阶段五需要物理重演时解决。

**仍属设计假设/需在实现中验证：**

- `debug_recent_ticks=4096`、dump keep 8、64GiB 是初始容量建议，不是性能实测结论。
- `replay.npz + replay_meta.json` 是契约参考实现；序列化格式可微调，但字段、hash、identity 和原子性不可降。
- `P2-*` 模块名可微调；schema 字段、测试语义和 gate 不可降。
- bitwise resume/重算只在 config-lock、同代码、同设备、同库版本和确定性内核下追求。
- 现有旧 SAC tests 16/16 与 PPO policy tests 288/3 skipped 只能作为资产健康度，不计作新契约通过。

**首版明确不承诺：**

- n-step>1、PER、relabel、stratified retention、异步采集、GPU inference server、opponent pool。
- 无保存 core-state 的逐帧物理重演。
- 跨设备/CUDA/库版本的 bitwise continuation。
- 用 Shannon/U 之外的新正则未经裁决直接混入。

### A7.7 W7 低成本核查结果

- 决策编号连续性核查：`SAC-R1-D01`～`D37` 无缺口/重复。
- Fixture 命名冲突已用 `FX-A2-*`、`FX-A5-*` canonical 解决。
- 阶段二包编号已统一；`P2-IND-0` 为入口包。
- 未执行训练；未把旧测试或历史 run 作为新契约的通过证据。

### A7.8 出口状态

- [x] A1–A6 决策、fixture 和阶段二包已建立唯一索引。
- [x] 风险矩阵 schema、严重度、验证层级和 gate 已冻结。
- [x] 十个风险域均有可证伪测试或明确阻塞说明。
- [x] 阶段二依赖 DAG 和 G2.0–G2.6 门槛已冻结。
- [x] 未决项、暂缓功能和重开条件已分类。

**W7 不声称：**生产实现、永久测试、真实 env smoke、训练收敛或 viewer 已完成。进入 W8 时，需要用户对 A1–A7、阶段二 DAG、资源默认与未决项做最终审阅。

## A8：阶段一收口与用户批准（W8，2026-10-08）

**用户批准记录：** `W8评审：A1–A7批准，阶段二DAG和默认资源口径批准，首版排除项批准，进入W8汇总。`

**收口结论：** A1–A7 的设计契约、A7 的阶段二 DAG、默认资源/存储口径及首版排除项均已获用户批准；阶段一以「设计契约完成」口径收口，阶段二获准从 `P2-IND-0` 开始。此批准不是实现正确性、测试通过或训练收敛证明。

### A8.1 历史决策逐项判定

历史区实际包含编号 `N1–N9`，另有两个未编号条目（package structure、first-iteration scope）；计划文本中的 `N1–N11` 表述已修正。判定只决定本轮是否继承其结论，不改写历史记录。

| 历史条目 | 原决定摘要 | W8 判定 | 本轮替代/依据 |
|---|---|---|---|
| N1 memory/replay | in-memory replay；checkpoint 不持久化 replay；resume 重新 warmup；逐 transition 存 obs/action/next_obs/reward/done/actor_weight 等 | **整体拒绝，字段部分采用** | 完整 resume 默认持久化 replay、游标、身份与 RNG（D19/D23）；显式 transition 字段被 `sac_transition_v1` 吸收，但容量单位与身份语义由 D16/D17 重定义；旧 `traj_id/traj_step` 不作稳定身份 |
| N2 relabel | 全量扫描 replay，用 reward_features 重算 reward/actor_weight | **首版拒绝，列为后续可选增强** | 首版显式拒绝 relabel（D18）；若未来启用，需要版本化 reward features、权限边界、验证与重开裁决 |
| N3 trunk grouping | 按 `trunk_group` 共享 Q trunk，默认按 γ 自动分组 | **首版拒绝** | 初版为每通道两套独立 Q 与 target，不共享 trunk、不自动按 γ 分组（D01/D06）；共享 trunk 只能作为后续消融 |
| N4 async collection | Phase 1 同步采集，replay 写接口保留线程安全 | **同步口径采用，线程安全不继承** | 首版仍同步；异步采集在 A7 首版排除项内。新 replay 不依赖旧锁语义 |
| N5 σ/α | 宽 `log_std=(-10,2)`，探索完全由 α 控制，`target_entropy=-action_dim` | **部分采用，关键默认值拒绝** | α/自动温度作为 SAC 原生控制保留，但首版 S01 使用 bounded σ，熵目标必须按可达范围标定；`-action_dim` 不自动继承；uncertainty 路线独立定义 |
| N7 gradient normalization | action-gradient normalization 是主 actor loss，naive weighted Q 为 fallback | **首版拒绝** | D06 排除梯度尺度归一化；D08 要求区分动作梯度、参数梯度、冲突与真实位移，旧 `grad_share` 不作份额证据 |
| N6 shared batch / multi-head Q | 所有通道共享 batch；多通道走共享 trunk multi-head；`aw=0` 帧不贡献该通道 critic loss | **部分采用，错误语义拒绝** | 共享 batch 与 per-channel sampling 暂缓被采用；共享 trunk 不继承；D05 明确 actor gate 不关闭 critic，`aw=0` 不屏蔽该通道 critic 学习 |
| 未编号 package structure | 旧 `sac/` 目录与 `experiments_sac` 结构 | **仅作历史结构参考，整体由 A7 DAG 取代** | 实现边界以 `P2-*` 包为准；旧模块名不自动保留，新增 `sac` 自有 collection/debug/policy 层 |
| 未编号 MVP scope | n-step、multi-head Q、grad norm、relabel/replay_plan 接口、真实训练 run 属 MVP | **整体被本轮范围替代** | 首版只支持 1-step/FIFO/uniform；n-step/relabel/grad norm 移除；真实 env smoke 在 G2.5，长训在阶段六，不属于阶段二入口门槛 |
| N8 stability tuning | 调整 target_entropy、alpha 边界、reward_scale、critic_lr 来压住 collapse/divergence | **不作为默认决策，保留为风险证据** | 观测到的问题进入风险矩阵；新基线 `reward_scale=1`、显式 α/λ 状态与 fail-loud 边界；任何 clamp/scale 都需新消融 |
| N9 scale/budget | `max_env_steps=10M`、`utd=0.25`、1M replay、96 workers、按 env step 调度 | **被 A2/A5 验收与计数契约替代** | 正式预算取 A2.7；UTD 分母为有效 agent transition 并使用 fractional credit；历史“比 PPO 省样本”判断不作承诺 |

### A8.2 已批准的阶段一契约边界

- **独立性**：SAC 拥有自有 collection、transition、replay、trainer、debugkit；复制适配允许，PPO 算法语义和类型不可进入 SAC 数据路径。
- **算法基线**：1-step、双 Q、当前策略采样、固定或自动 α、无共享 trunk、无梯度归一化；`u_bonus/u_floor` 为 Shannon 基线之外的独立替代路线。
- **首个策略**：S01 单分量、shared σ、bounded σ；mixture 分量枚举方案已证明可行，但生产化属于后续阶段。
- **数据/恢复**：`sac_transition_v1`、有效 agent transition 计数、完整 replay checkpoint、PPO 风格白名单 override、可选 `config-lock`。
- **诊断**：`metrics/events.jsonl`、L0–L3、以 `critic_tick` 为最小可重算截面、独立 debug RNG、CLI/HTTP/viewer 共用分析实现。
- **验收**：两任务 seeds、阈值、连续窗口、最终 checkpoint、预算与失败处理按 A2.7 冻结。

### A8.3 阶段二入口条件与执行顺序

1. `P2-IND-0` 先行：先证明 SAC import、registry 和 CLI 路径不带入 PPO，再开始实现其他包。
2. `P2-COLL-1`/`P2-DATA-1` 与 `P2-DBG-1` 可按 A7 DAG 并行开发，但只能 mock A5/A6 schema，不能反向修改契约。
3. replay/checkpoint/clock 的 blocker 测试必须在声称闭环可用前通过；smoke 不能代替这些永久测试。
4. `P2-TRAIN-1` 只承诺 S01 actor 与标准 Shannon SAC 数学；不提前承诺八格全量或 U 路线训练效果。
5. `G2.5` 的真实 env smoke 只证明链路可用；进入长训/阶段六前必须通过 `G2.6` 的 L2 dump recompute 能力。

### A8.4 重开与变更条件

- 修改 A2–A7 的 schema、数学、验收、阶段二 DAG、默认资源口径或首版排除项，必须新增 `SAC-R1-*` 决策并更新 A7 风险矩阵。
- 启用 n-step、PER、relabel、stratified retention、异步采集、opponent pool 或 GPU inference server，必须先补目标、偏差、持久化与诊断设计，不得由旧实现接口隐式启用。
- 若阶段二测试推翻当前数学/数据假设，回到对应 A 节修正，不用放宽 gate 或调参绕过。
- 旧 PPO/旧 SAC 测试和历史训练日志只作为代码健康度或风险线索，不作为新契约的通过证据。

### A8.5 阶段一出口状态

- [x] A1–A7 已由用户批准。
- [x] 历史 N1–N9 与未编号历史条目已逐项判定。
- [x] 阶段二 DAG、默认资源口径和首版排除项已由用户批准。
- [x] 阶段一完成门槛全部满足，阶段二获准从 `P2-IND-0` 开始。

**阶段一最终不声称：**生产实现完成、永久测试存在、真实 env smoke 通过、任何任务收敛或 debug viewer 已交付。

# 阶段二执行计划：最小可信 SAC 闭环（2026-10-08）

**状态：** 阶段二已实现并通过收口验证；`G2.0`～`G2.6` 均已满足。计划依据 A1–A8，尤其 A5/A6/A7 的契约、风险矩阵与 DAG。本阶段仍不声称任务收敛、八格策略全量、U 路线训练效果、viewer/HTTP 或长训验收。

## P2.0 阶段目标与非目标

**目标：** 建成一个可验证、可恢复、可观测的最小 SAC 闭环：独立 import、SAC collection/data、FIFO replay、checkpoint bundle、S01 actor、标准 Shannon SAC trainer、时钟/UTD loop、基础 metrics、fake env 集成、真实 env smoke，以及可对指定 `critic_tick` 做 recompute 的 L2 dump。

**本阶段不承诺：**

- 两个真实任务收敛；`G2.5` 只要求链路 smoke。
- 八格策略全量生产化、mixture actor 生产实现或 `u_bonus/u_floor` 训练验证。
- 完整 viewer/HTTP 页面；阶段二只要求 CLI/analysis 所需的数据面与 recompute 能力。
- n-step、PER、relabel、stratified retention、异步采集、opponent pool、GPU inference server。
- bitwise resume；只在同代码、同设备、同库版本和确定性内核下追求。

## P2.1 执行顺序

```text
S2-W0 P2-IND-0
  → S2-W1 P2-COLL-1 + P2-DATA-1
  → S2-W2 P2-DBG-1（可与 S2-W1 并行设计/开发）
  → S2-W3 P2-REPLAY-1
  → S2-W4 P2-CKPT-1
  → S2-W5 P2-TRAIN-1
  → S2-W6 P2-LOOP-1 + P2-DBG-2
  → S2-W7 P2-ENV-1：fake env → 真实 env smoke
  → S2-W8 P2-DBG-3 + P2-DBG-4
  → G2.6 阶段二收口评审
```

实际开发可按依赖局部调整，但不得跳过 gate；`P2-DBG-1/2` 可以与 collection/data 并行推进，`P2-TRAIN-1` 不应等 collection 全部完成才开始单测，但接入真实 batch 前必须依赖 `P2-DATA-1`/`P2-REPLAY-1`。

## P2.2 工作包计划

| wave | 包 | 实施内容 | 必须落地的永久测试/证据 | 出口 |
|---|---|---|---|---|
| S2-W0 | `P2-IND-0` | 清理 SAC import 边界；`baseline/framework/__init__.py` 与 `train.py` registry 改 lazy；SAC 不支持参数显式报错；建立 SAC 自有命名空间入口 | `T-IND-01`；`--list-experiments`；PPO 基本路径回归 | `G2.0` |
| S2-W1 | `P2-COLL-1` | vendor/copy-adapt collection：SACJob、BehaviorSpec、CollectedEpisode、runner/recorder、provenance、pre-action fact provider、worker failure atomicity | `T-COL-01/02/03`、`T-DATA-04/05`；collection provenance manifest | `G2.1` 的一部分 |
| S2-W1 | `P2-DATA-1` | `sac_transition_v1`、TransitionSlice/batch schema、validator、边界构造；standup/basic_balance 的 `build_slices` 与任务事实；`phi_pre/phi_post` 分离 | `T-DATA-01～03`、`T-DATA-02` 中可在本阶段完成的部分、`FX-A5-01～06` | `G2.1` |
| S2-W2 | `P2-DBG-1` | `metrics/events.jsonl` writer、metric catalog、schema version、统一时钟字段、config/debug event | `T-DBG-01`；最小 run 目录可产生合法 events | 支撑 `G2.4` |
| S2-W3 | `P2-REPLAY-1` | SoA replay、有效 transition 容量、FIFO overwrite、uniform sampling、batch 内不重复、sample_id/source_key/slice_id、采样 RNG、统计与持久化格式 | `T-RPL-01～05`、`FX-A5-07/08` | `G2.2` 的一部分 |
| S2-W4 | `P2-CKPT-1` | checkpoint bundle、manifest、trainer/replay/RNG/counter/experiment state、原子写入、warm-start 与 config-lock 边界 | `T-CKPT-01～04`、`FX-A5-09/10/12`、`T-PERF-02` | `G2.2` |
| S2-W5 | `P2-TRAIN-1` | S01 actor copy-adapt、稳定 TN 数值内核、SAC actor contract、双 Q/target、Bellman target、actor loss、固定/自动 α、梯度隔离 | `T-MATH-01～03`、`T-MATH-04` 的 Shannon/固定与自动 α 部分、`T-POL-01/02`；U 路线部分推迟到 pre-P4 | `G2.3` |
| S2-W6 | `P2-LOOP-1` | collection round、env/transition/tick counters、fractional UTD credit、round cap、eval/export/checkpoint 边界、seed manifest、schedule 状态 | `T-CLK-01～03`、`FX-A5-11/12` | `G2.2`/`G2.4` |
| S2-W6 | `P2-DBG-2` | tick ring、L0/L1 聚合、debug RNG、资源计数、异常事件 | `T-DBG-02/05`、`T-PERF-01` | `G2.4` |
| S2-W7 | `P2-ENV-1` | fake/small env episode→slice→replay→tick→events 集成；再接 Humanoid21 最小 smoke | `T-INT-01/02`；run manifest + events | `G2.4`/`G2.5` |
| S2-W8 | `P2-DBG-3` | dump request/schedule、`critic_tick` L2 capture、`sac_dump_v1`、manifest/hash/retention | `T-DBG-03` | `G2.6` 的一部分 |
| S2-W8 | `P2-DBG-4` | dump access、analysis、sample trace、target/loss recompute、最小 CLI 输出 | `T-DBG-04/05/06` 中 CLI/recompute 部分；`DX-04/12/13/14` 的核心路径 | `G2.6` |
| 后续 | `P2-DBG-5/6` | viewer/HTTP、完整 fault-injection harness | 完整 `DX-*` 回归 | 不属本阶段出口，最迟 pre-P5 |

## P2.3 Gate 收口标准

- **G2.0 独立性**：`import baseline.framework.sac`、SAC registry、SAC smoke 路径不加载 PPO；SAC 不支持的 CLI 参数显式报错；PPO 原路径无回归。
- **G2.1 数据面**：`FX-A5-01～06/13`、`T-COL-*`、核心 `T-DATA-*` 通过；缺字段、非法边界和 `phi_pre` 来源错误全部 fail loud。
- **G2.2 状态面**：`T-RPL/T-CLK/T-CKPT` 通过；replay 持久化、采样 RNG、UTD credit、白名单 override 与 config-lock 可测试。
- **G2.3 数学面**：`T-MATH/T-POL` 通过；actor/critic/temperature/target 梯度隔离和状态更新方向可证伪。
- **G2.4 最小闭环**：fake env 集成跑通，`metrics/events.jsonl` 与 tick ring 能解释一轮训练和若干 critic tick。
- **G2.5 真实 env smoke**：Humanoid21 最少轮数运行通过；只证明采集、训练、评估、导出、checkpoint 链路，不证明任务学习。
- **G2.6 可解释闭环**：指定 `critic_tick` 可生成 `sac_dump_v1`，并达到 `recompute` 级证据；未通过前不得开始长训或阶段六验收。

## P2.4 实施规则

1. **先测试契约，再接实现**：每个包先落地对应永久测试或最小失败测试，再实现生产路径；旧 SAC/PPO 测试只作回归，不计作新契约通过。
2. **不改契约迁就实现**：字段、时钟、身份、fail-loud 边界与 A5/A6 不一致时，回到 DECISIONS 新增裁决。
3. **独立命名空间**：新增代码进入 `baseline/framework/sac/` 或 `baseline/experiments_sac/`；允许复制后适配，不允许 import PPO 算法内部类型。
4. **中立共享层最小改动**：只触碰 A8 授权的 `baseline/framework/__init__.py`、`train.py` registry/CLI 边界；不改 PPO 内部和 `envs/` 语义。
5. **证据优先**：每个 gate 提交时附测试命令、结果、run/dump artifact 路径；没有 artifact 的项目不得标成完成。
6. **资源默认可观测**：replay、checkpoint、tick ring、dump 写入必须记录大小/耗时；若默认值导致资源问题，先提交实测证据再调整默认。
7. **不启动长训**：阶段二只允许 fake/small env 集成和 `G2.5` 的真实 env smoke；任何收敛判断都推迟到阶段六。

## P2.5 阶段二完成定义

阶段二完成要求同时满足：

- `G2.0`～`G2.6` 全部通过，并附测试/artifact 证据；
- 新 SAC 路径可独立 import、采集、入池、训练、评估、导出、checkpoint、resume；
- L2 dump 能对一个指定 `critic_tick` 重算 target/loss；
- PPO 路径无回归；
- 所有已知限制在文档中明确，不存在「配置被接受但功能未实现」。

## P2.6 实施进度记录

- **2026-10-08 `P2-IND-0` / `G2.0` 完成：**
  - `baseline/framework/__init__.py` 改为 lazy export；`train.py` 改为按 `--algo` lazy registry，并让 `--list-experiments --algo sac` 不加载 PPO。
  - SAC 新增自有 `actor.py`、`collection.py`、`observer_utils.py` 边界；`experiments_sac` 不再 import `baseline.framework.ppo` 或 `baseline.framework.rollout`。
  - `sac.loop` 在 `P2-COLL-1` 前显式抛出 `SACRollouterNotImplemented`，不再通过 legacy `ParallelRollouter` 进入 PPO 依赖。
  - 证据：`pytest baseline/framework/sac/tests -q` → 22 passed；`pytest baseline/framework/ppo/tests/test_param_overrides.py baseline/framework/ppo/algos/test_advantages.py -q` → 21 passed；`train.py --algo ppo --list-experiments` 与不带 `--algo` 的联合列表正常。
  - 边界：SAC smoke 仍止于 `P2-COLL-1` 未实现；这满足 G2.0，不表示训练闭环可运行。
- **2026-10-08 `P2-COLL-1` / `P2-DATA-1` 完成（G2.1 数据面实现）：**
  - SAC 新增自有 collection 层：`SACJob`、`SACBehaviorSpec`、`SACFactSpec`、`CollectedEpisode`、`SACEpisodeRecorder`、`SACEpisodeRunner`、`SACParallelRollouter`；不再依赖 `baseline.framework.rollout`，worker/job 失败会终止整轮且不返回部分数据。
  - `SACJob` 携带 `run_id/collection_round/job_index/episode_seed/episode_options`、双 agent behavior spec、pre-action fact specs 与 policy/env blueprint；`CollectedEpisode` 记录 job key、policy fingerprint、behavior、worker/wall time。
  - SAC runner 在每次 `runtime.step()` 前调用声明的 pre-action provider；`policy_action` 独立进入 `action_extras`。`sac_balance` 注册 `phi_pre`，provider 直接从 accessor 按 `uprightness * height / standing_height` 计算。
  - 新增 `sac_transition_v1`：`SACTransitionSlice`、strict validator、`build_agent_transition_slice()`。字段包含 `obs/actions/next_obs/rewards/channel_valid/terminated/truncated/bootstrap/termination_reason/physics_delta/actor_gate/actor_weight/sample_weight/task_facts/reward_features/policy_action/source_keys/behavior/collection/versions`。
  - `basic_balance` 和 `standup` 的 SAC 语义分别落到 `exp_sac_balance.py` 与新增 `exp_sac_standup.py`；balance 的 `r_cross` gate 使用 `phi_pre²`，`phi_post` 作为 `phi_post_reference`。
  - 证据：`pytest baseline/framework/sac/tests/test_collection_data.py -q` → 9 passed；`pytest baseline/framework/sac/tests -q` → 30 passed；SAC/PPO CLI listing 正常；`git diff --check` 通过。
  - 边界：SAC smoke 现在止于 `P2-TRAIN-1` 的 SAC actor 未实现，而不是 collection 未实现；replay 仍待 `P2-REPLAY-1` 接管 `SACTransitionSlice`。
- **2026-10-08 `P2-DBG-1` 完成（metrics/data-plane 基础）：**
  - 新增 `SACClockState`，统一 `collection_round/env_step/agent_transition/critic_tick/actor_tick/temperature_tick/target_tick/eval_tick/export_tick/checkpoint_tick`。
  - 新增 `sac_metrics_v1`：`MetricEvent`、`MetricCatalog`、`SACMetricsWriter`、`load_events`；canonical 路径为 `<run_dir>/metrics/events.jsonl`。
  - 事件类型限定为 `round/tick/eval/export/checkpoint/debug/config`；metric namespace 按事件校验，显式拒绝 `advantage/ratio/clip/gae/ppo` 命名空间和非有限 metric 值。
  - 证据：`pytest baseline/framework/sac/tests/test_metrics.py -q` → 6 passed。
- **2026-10-08 `P2-REPLAY-1` 完成：**
  - 新增 `SACReplayBuffer`：SoA 存储、有效 transition 容量、FIFO overwrite、独立 `np.random.Generator`、batch 内唯一 `sample_id`、`source_key/slice_id/frame_index/metadata` 溯源、replay/RNG 持久化；`n_step/PER/relabel/stratified/freshness` 显式拒绝。
  - 证据：`pytest baseline/framework/sac/tests/test_replay.py -q` → 18 passed。
- **2026-10-08 `P2-CKPT-1` 完成：**
  - 新增 `sac_checkpoint_v1` bundle：manifest/hash、原子目录写入、trainer/replay/runtime/experiment/config 状态、full resume 与 model-only warm-start 分离、白名单 override 与 `config-lock`。
  - 证据：`pytest baseline/framework/sac/tests/test_checkpoint.py -q` → 16 passed。
- **2026-10-08 `P2-TRAIN-1` 完成：**
  - 新增 `S01Actor`（单分量、shared bounded σ、tanh Gaussian、可导出 runtime policy）与标准 Shannon SAC update：1-step Bellman target、双 Q、terminated/truncated/bootstrap 校验、actor entropy loss、auto-α、soft target update。
  - 证据：SAC suite 中 actor/trainer 永久测试通过；未知 `actor_arch`、非法 batch、`n_step>1`、非双 Q 均 fail loud。
- **2026-10-08 `P2-LOOP-1` / `P2-DBG-2` 完成：**
  - SAC loop 接通 collection→slice→replay→updates→eval/export/checkpoint；实现 fractional UTD credit、round cap dropped accounting、完整 clock 推进、tick ring、`metrics/events.jsonl`、checkpoint bundle resume。
  - 证据：`pytest baseline/framework/sac/tests/test_loop_diagnostics.py -q` 与集成测试通过。
- **2026-10-08 `P2-ENV-1` 完成：**
  - fake env 闭环覆盖 episode→slice→replay→trainer→metrics→checkpoint。
  - Humanoid21 smoke：`sac_balance` 与 `sac_standup` 均完成 collection、slice、replay、8 次 critic/actor/temperature/target tick、policy export 和 checkpoint；`sac_balance` 另验证 eval/new-best policy。真实完整 resume 从 `/tmp/sac_balance_smoke_phase2/checkpoints/checkpoint_s00000039` 恢复成功。
- **2026-10-08 `P2-DBG-3` / `P2-DBG-4` 完成：**
  - 新增 `sac_dump_v1` L2 critic-tick dump：request/clocks/batch/forward outputs/trainer pre-post/spec/analysis/manifest/hash/latest-N retention；`--dump-at` 对 SAC 表示 critic tick。
  - 新增 `baseline.framework.sac.debugkit` CLI：`summary`、`find-sample`、`recompute`；recompute 使用冻结 batch/action/log-prob 与 pre/post trainer state，复核 Bellman target、critic loss、actor loss、alpha loss。
  - 证据：真实 `/tmp/sac_balance_dump_smoke_phase2/debug_dumps/critic_tick_00000002` recompute passed，最大差 `1.91e-06`。
- **2026-10-08 `G2.6` 收口：**
  - 移除误导性旧契约：`TrajectorySlice`、`TaggedReplay`、未实现的 `ExperimentSAC.relabel()` 和 `SACActorNotImplementedError` 不再出现在新 SAC API；`request_relabel` 在 loop 中显式拒绝。
  - `data_sources()` 已由 loop 校验；v1 只允许 `self` source 和正 `sampling_share`，其他数据源 fail loud。
  - 收口证据：`pytest baseline/framework/sac/tests -q` → 54 passed；`pytest baseline/framework/ppo/tests/test_minimal_example.py baseline/framework/ppo/tests/test_param_overrides.py -q` → 20 passed；静态审计未发现 SAC/experiments_sac 对 `baseline.framework.ppo` 或 `baseline.framework.rollout` 的 import。

# 阶段三执行计划：多 critic 与两个目标实验（2026-10-08）

**状态：** 阶段三计划已定，尚未开始实现。阶段二完成的是最小可信闭环；阶段三必须把 A3 已批准的多通道 soft-Q 数学、两任务语义和持续诊断落实到生产路径。不得把阶段二 smoke 或当前 shared-trunk/raw-gate 实现误认为已经满足 A3。

## P3.0 阶段目标与非目标

**目标：** 将 SAC 从“可运行的最小闭环”升级为“多通道语义可信、两个任务可持续训练、每轮数据/通道贡献可追踪”的实现。

**必须覆盖：**

- A3 的 per-channel soft-Q target、共同 twin-pair 选择、归一化 actor gate、动作前/动作后门控时序；
- `sac_balance` / `sac_standup` 的任务事实、reward、agent boundary、termination/bootstrap 和 eval 语义；
- 多轮持续训练、resume、replay 来源/版本追踪；
- per-channel Q/target/TD/actor contribution 诊断与 recompute；
- 不支持组合继续 fail loud，不留“接受但不生效”的配置。

**不承诺：**

- 两个任务收敛；正式收敛仍在阶段六。
- 八格策略全量、mixture actor、U 路线训练验证；属于阶段四。
- 完整 viewer/HTTP；阶段三只要求机器可读的 per-channel debug 数据面。
- n-step、PER、relabel、stratified retention、异步采集、opponent pool、GPU inference server。

## P3.1 已发现的实现差距

1. **target twin 选择不符合 A3：** 当前按每个 channel 分别 `min(Q1,Q2)`；A3 要求先用归一化权重合成 `F_j=Σw_c Q_c,j`，再选择共同 `j_next`。
2. **actor twin 选择不符合 A3：** 当前 actor 对每个 channel 分别 `min(Q1,Q2)` 后加权；A3 要求对合成 `F_j` 选择共同 `j_actor`。
3. **next-state gate 缺失：** balance 的实际 `w_pre` 来自 `phi_pre`，target 下一状态权重应使用 `phi_post` 对应的下一时刻 gate；当前 batch 没有独立的 next-state gate 字段。
4. **shared trunk 与 A3 默认不符：** 当前 `MultiHeadQCritic` 默认/实验使用 shared trunk；A3 首版默认是每通道独立 twin Q。阶段三应切回独立 critic，shared trunk 只能作为后续显式消融。
5. **actor 有效行过滤不完整：** A3 要求 actor/alpha 只使用所有 cohort 通道均有效的行；当前 actor loss 未显式按全 `channel_valid` 过滤。
6. **时钟边界不足：** 某通道空分母、actor/alpha 空分母、target update 是否跳过，当前没有独立 clock/test。
7. **诊断不足：** 当前 dump 能重算旧实现，但缺少 pair-index 选择率、next gate、per-channel actor contribution、replay age/version 分桶。

## P3.2 执行顺序

```text
S3-W0 P3-AUDIT-0
  → S3-W1 P3-DATA-2
  → S3-W2 P3-MATH-1
  → S3-W3 P3-DIAG-2
  → S3-W4 P3-EXP-1
  → S3-W5 P3-RUN-1
  → G3.6 阶段三收口评审
```

`P3-DATA-2` 与 `P3-MATH-1` 有强依赖：先固定 batch/schema 字段，再改 target/actor 公式。诊断随 trainer 同步改，不允许最后再补。

## P3.3 工作包计划

| wave | 包 | 实施内容 | 必须落地的永久测试/证据 | 出口 |
|---|---|---|---|---|
| S3-W0 | `P3-AUDIT-0` | 对当前 trainer/replay/transition/experiments 与 A3 逐项对拍；登记所有偏差；决定是否以 `sac_transition_v2` 承载 next gate | 差异矩阵；新 `SAC-R1-*` 记录；不静默改 schema | `G3.0` |
| S3-W1 | `P3-DATA-2` | 扩展 transition/replay batch：`actor_gate/actor_weight` 与 `actor_gate_next/actor_weight_next`（或等价的显式 next-gate 字段）；balance 用 `phi_pre/phi_post` 构造；standup 使用常量 next gate；checkpoint/replay persistence 更新 | schema validator、版本、shape/finite、terminal boundary、source identity 测试 | `G3.1` |
| S3-W2 | `P3-MATH-1` | 实现 A3 soft-Q：归一化 `w`、共同 `j_next`、per-channel target、共同 `j_actor`、actor 全有效行过滤、per-channel critic mask；同 γ cohort 校验；独立 twin Q；shared trunk 默认拒绝 | A3 M01～M16 对应永久测试、零 w/缺 v/空 actor/空 channel/不同 γ/负 gate 反例 | `G3.2` |
| S3-W3 | `P3-DIAG-2` | 扩展 trainer stats/capture/metrics/dump：j_next/j_actor 选择率、w/next_w 分布、per-channel target/TD/Q、actor contribution 向量范数与 cosine、alpha used/after、replay age/version | `sac_dump_v2` 或向后兼容 dump；recompute 通过；样本可追回 source/frame | `G3.3` |
| S3-W4 | `P3-EXP-1` | 两实验对拍：observer/fact/reward/boundary/eval/state/version metadata；修正 `data_sources`、objective_mode、policy/reward versions | 同批 episode FX 对拍；缺 observer/fact fail loud；eval/resume 测试 | `G3.4` |
| S3-W5 | `P3-RUN-1` | 两实验多 round 持续短训：replay 跨轮保留、checkpoint mid-run resume、metrics/dump 可用、sample/source 可解释 | Humanoid21 连续多 round smoke + resume；不判定学习效果 | `G3.5` |
| S3-W6 | closeout | 审计 PPO 独立性、旧接口残留、配置接受边界、测试证据 | `G3.0`～`G3.5` 全通过；PPO 快速回归 | `G3.6` |

## P3.4 Gate 收口标准

- **G3.0 契约审计：** 所有与 A3/A5/A6 的实现差距已登记；next-gate/schema 版本处理有明确决策。
- **G3.1 数据契约：** 当前/下一时刻 gate 均可追踪、可持久化、可校验；balance 的 `phi_pre/phi_post` 时序不能错位。
- **G3.2 数学契约：** A3 公式测试全通过；负 gate、全零 gate、不同 γ、非法 `channel_valid` 组合显式拒绝；`actor_weight=0` 不冻结 critic。
- **G3.3 诊断契约：** 指定 critic tick dump 可按新公式重算；能回答每个 channel 的 target/TD/actor 贡献来自哪里。
- **G3.4 任务契约：** 两实验同批 episode 的任务事实、reward、终止边界、eval 指标与版本元数据通过测试。
- **G3.5 持续运行：** 两实验可多 round 训练、恢复、导出并保留样本来源；不以 loss 下降冒充任务成功。
- **G3.6 收口：** PPO 无回归，SAC 独立性保持，所有已知限制写明。

## P3.5 默认技术口径

- **schema：** 若新增 next-gate 必需字段，使用显式新版本（建议 `sac_transition_v2`/`sac_replay_v2`/`sac_dump_v2`），不把 v1 字段重新解释。
- **critic：** A3 首版使用每通道独立 twin Q；`trunk_group` 若表示参数共享应拒绝，除非未来另立共享-trunk 消融决策。
- **gate：** `actor_gate` 存原始非负 `g(s)`；`actor_weight` 存归一化 `w(s)`；`actor_gate_next/actor_weight_next` 存 target 所需的 `w(s')`。
- **actor 有效行：** 仅所有 cohort 通道 `channel_valid=True` 的样本进入 actor/alpha loss；无有效行则跳过对应 optimizer/tick 并记录。
- **诊断：** 捕获必须保存 pair 选择、当前/下一 gate、per-channel Q/target/TD 和足以重算 loss 的冻结采样输出。

## P3-AUDIT-0 执行结果与新增裁决（2026-10-08）

**状态：** `P3-AUDIT-0` 已完成代码事实核对；本审计不改生产代码。以下差异将进入后续工作包，而不是作为不可执行的注意事项。

| 对象 | 当前事实 | A3/A5 契约 | P3 处理 |
|---|---|---|---|
| `actor_gate/actor_weight` | `sac_transition_v1` 已保存 `g` 与归一化 `w`，并验证二者一致 | `g≥0, Σg>0, w=g/Σg` | 已满足；继续使用现有字段 |
| critic target pair | 每通道独立 `min(Q1,Q2)` | 先用 `w(s')` 合成 `F_j`，共同选择 `j_next` | `P3-MATH-1` 修正 |
| actor pair | 每通道独立 min 后加权 | 用 `w(s)` 合成 `F_j`，共同选择 `j_actor` | `P3-MATH-1` 修正 |
| next-state gate | transition/replay/batch/dump 均无 `w(s')` | `basic_balance` target 分支使用 `phi_post` 对应 gate | `P3-DATA-2` 新增显式字段 |
| critic grouping | `MultiHeadQCritic` 按 `trunk_group` 或 gamma 聚合，两实验配置为 `shared` | 首版默认每通道独立 twin Q | `P3-MATH-1` 默认按 channel 分组；共享 trunk fail loud |
| `channel_valid` | critic 按通道过滤，但任一通道空 batch 直接失败；actor/alpha 不按“全通道有效”过滤 | 某通道空分母跳过其 optimizer/target；actor/alpha 只用全有效行；全批无 actor 有效行则 fail loud | `P3-MATH-1` 修正 |
| gamma cohort | 只校验 `n_step=1,twin=2`；未校验共同 γ | 首版同 cohort γ/m 一致 | `P3-MATH-1` 添加配置校验 |
| replay age/version | `buffer_stats` 只有 sample-age 均值和 gate/reward 均值 | 需要按 critic/env age 与 policy/version 分桶 | `P3-DIAG-2` 补齐 |
| dump recompute | `sac_dump_v1` 重算旧的 per-channel-min 公式 | 必须能重算 A3 的 joint pair/weight 公式 | `P3-DIAG-2` 升级 dump/recompute |

**新增决策：**

| 编号 | 决策 | 理由与适用边界 |
|---|---|---|
| SAC-R1-D38 | next-state actor gate 通过显式 schema 字段进入 batch；建议命名为 `actor_gate_next/actor_weight_next`，schema 升级为 `sac_transition_v2` / `sac_replay_v2` / `sac_dump_v2` | `basic_balance` 的 `phi_post` 是 `s'` 的 pre-next-action gate 事实；不能从当前 `phi_pre` 或 reward feature 隐式推导。v1 数据缺字段，不允许静默当作新语义解释 |
| SAC-R1-D39 | v2 checkpoint 的 full resume 不兼容 v1 replay；v1 checkpoint 只能走显式 model-only warm start | 防止旧数据缺 next gate 后被错当 v2 训练；模型参数语义仍可在显式 warm-start 中复用 |
| SAC-R1-D40 | 阶段三首版 critic 分组固定为每通道独立 group；任何把多个 channel 放进同一 critic group 的配置均拒绝 | 这是 A3 已批准的默认实现边界；shared trunk 只能作为后续显式消融，不作为阶段三默认实现 |

`G3.0` 通过标准：以上矩阵已登记，且没有“配置接受但语义未实现”的已知路径。当前仍有 `baseline/framework/train.py` 和 `baseline/framework/code_snapshot.py` 的用户侧未提交改动；P3 审计不触碰它们。

- **`P3-DATA-2` 完成（G3.1 数据面升级）：** `SACTransitionSlice` 升级为 `sac_transition_v2`，新增必需 `actor_gate_next/actor_weight_next` 并校验 shape、非负、行和与归一化；`build_agent_transition_slice()` 要求实验显式提供 next gate。`sac_balance` 使用 `phi_pre` 生成当前 gate、使用 `phi_post` 生成 next gate；`sac_standup` 使用常量 next gate。`SACReplayBuffer` 升级为 `sac_replay_v2` 并在 admission/sample/get_by_sample_ids/persistence 中保留两个字段；v1 replay/checkpoint 在 full resume 中 fail loud。trainer batch 校验已拒绝缺 next-gate 字段的 batch。证据：`test_collection_data + test_replay + test_checkpoint` → 25 passed；完整 SAC suite → 54 passed。
- **`P3-MATH-1` 完成（G3.2 数学面修正）：** critic target 改为用 `actor_weight_next` 先合成 `F_j` 再共同选择 `j_next`；actor loss 改为用 `actor_weight` 合成 `F_j` 并共同选择 `j_actor`。actor/alpha 只使用所有通道 `channel_valid=True` 的行；每通道 critic 独立 mask/step/target update，空分母跳过该通道。同 cohort γ 不一致、shared critic group、n-step/ensemble 扩展显式拒绝。`actor_weight=0` 不冻结对应 critic 的永久测试已落地。`MultiHeadQCritic` 现在默认每 channel 一个独立 group。
- **`P3-DIAG-2` 完成基础版（G3.3）：** `sac_dump_v2` 捕获并重算 A3 公式所需的 batch gate、pair index、per-channel Q/target/TD、actor/alpha 有效行；`find-sample` 输出当前/下一 gate；tick metrics 增加 pair 选择率、actor 有效行、每通道 critic 更新与 gate/next-gate 均值；replay stats 增加 collection round、policy fingerprint、reward/objective version 分桶。证据：完整 SAC suite → 57 passed，包含指定 critic tick dump 的新公式 recompute。

### P3.7 实验语义对拍（P3-EXP-1）结论

对拍 `experiments_ppo/exp_basic_balance.py`、`exp_standup.py` 与 SAC 对应实现，逐字段核对 reward、gate、boundary、eval 与版本语义：

- **一致：** balance `r_fall=0.01·φ_post`、双 agent 独立 `agent_frame_boundary`、imbalance → terminated/`bootstrap=0`、timeout → truncated/`bootstrap=1`、eval `survival_rate` 口径；standup `r_potential=0.01·φ`、无早终止语义、eval `max_pot/final_pot/max_stage/success` 口径、`_AGENT_OBS` 映射与 blueprint observer key 一一对应。
- **有意差异（沿用已批准语义）：**
  - PPO 对缺失的 `r_cross`/`potential` observer 静默填 0；SAC 按 fail-loud 契约 `KeyError`，不复制该回退。
  - PPO balance 的 r_cross actor weight 用 post-step `φ`；SAC 当前 gate 用 `phi_pre`（即 `w(s_t)`），next gate 用 `phi_post`（即 `w(s')`），这是 D38/A3 的正确时序，不是回归。
- **修复缺口：**
  - `SacStandup._ep_final_pots` 此前无消费方；新增 `ExperimentSAC.post_round_metrics(episodes)` 可选 hook，loop 将其结果以 `task.*` namespace 写入 round 事件（`task.online_success`、`task.final_potential_mean`），对齐 PPO `post_update` 的 per-update 在线指标。
  - `SacStandup.on_eval` 补齐 `h_torso` 对应的 `max_h` 指标，与 PPO eval 字段对齐。
- **新增永久测试：** balance 缺 `cross_support`/`height_phi` observer 与缺 `phi_pre` fail loud；standup dense-potential slice 语义（reward=0.01φ、常量 gate、timeout→truncated+bootstrap）、缺 `potential` observer fail loud、`post_round_metrics` 消费一次即清空。
- **证据：** 完整 SAC suite → 61 passed。

### P3.8 持续训练运行（P3-RUN-1）与 G3.6 收口

真实 Humanoid21 多 round 运行（均为 `--set` 缩小规模，非 smoke 路径）：

- **`sac_balance`：** 17 rounds，`env_step=1579`，`critic_tick=510`；replay capacity=120 下累计插入 2732、`overwritten=2612`（FIFO wraparound 真实发生）；每 round 独立 policy export（`policy_exports/rNNNNN`）；2 次 eval（`eval_interval=600` env_step 调度正确）；3 个 checkpoint（round1 + 两次 eval）。
- **full resume：** 从 `checkpoint_s00000616` 恢复，clocks 完整还原（`collection_round=7, critic_tick=210, replay restored`），继续 round 8–17 到 `env_step=1580, critic_tick=510`；白名单外配置变更（`max_env_steps`）被 config-lock 正确拒绝。
- **`sac_standup`：** 5 rounds，`env_step=1200`，`critic_tick=150`；eval 输出 `max_pot/final_pot/max_stage/max_h/success`；round 事件含 `task.online_success`/`task.final_potential_mean`；replay `overwritten=1800`。
- **来源追踪：** round 事件 `replay_stats` 含 `collection_round_counts`、per-channel `gate_mean/weight_mean/reward_mean/reward_std`、`reward_semantics_counts`、`objective_mode_counts`、`next_sample_id`、`overwritten`。
- **sac_dump_v2 真实环境验证：** `critic_tick_00000005` dump → `debugkit recompute` `passed=true`，`max_abs_diff=7.63e-06`；`target_pair1_frac/actor_pair1_frac/per-channel Q/loss/TD/gate(next)` 全部对账通过。
- **回归：** 完整 SAC suite 61 passed；PPO 快速回归（minimal_example/param_overrides/post_update_artifacts）38 passed；`baseline/framework/sac` 与 `baseline/experiments_sac` 内无 `baseline.framework.ppo`/`baseline.framework.rollout` 运行期 import；`transition.py` 过期 `v1` docstring 已修正为 `v2`（checkpoint/metrics/trainer/collection schema 未变，保持 v1 是正确版本语义）。

**阶段三收口判定：通过。** 两实验在 v2 多通道契约下持续训练、评估、导出、checkpoint、full resume；batch→dump→recompute 链路可按 A3 公式逐通道对账；样本来源与版本可追踪。不声称任务收敛（本次运行均为极小预算），不声称八格策略或完整 debug UI 完成。

## P3.6 完成定义

阶段三完成要求：

- 两目标实验在新多通道契约下持续运行；✅
- 每个合法 transition 的 source、版本、gate、reward、bootstrap 可解释；✅
- critic/actor/alpha/target 的更新语义与 A3 一致；✅
- 指定 critic tick 可重算并展示 per-channel 贡献；✅
- 无 PPO import/runtime 依赖；✅
- 不声称任务收敛、八格策略完成或完整 debug UI 完成。

## P4.0 阶段四执行计划（八格策略与标准化探索/优化控制）

**目标：** 按 A4 把 2×2×2 八格 TN 策略体系、枚举 mixture 梯度、三层旋钮（采集 β / π 正则 / 优化）和 S/U 两路线落到 SAC 自有命名空间。阶段四完成的是**接口 + 数值 + 梯度 + 短训验证**，不是全八格 × 两任务 × 多 seed 收敛（属阶段六）。

### P4.1 现状审计结论（先行事实，非计划假设）

- 当前 `S01Actor` 是**未截断 tanh-Normal**（`mean + std·ε` 后 `tanh`），`log_std` 只做 clamp；与 A4.2 的 TN bounded sigmoid-σ 语义不同，且不具备 A4.3 的 `expectation_samples`/`integration_weights`、`sample_behavior(e)`、`uncertainty(kind)`、按流分离的 RNG state/export 元数据。阶段四必须升级 actor 契约并迁到 TN 内核，不是在现类上旁挂 7 个变体。
- PPO 侧八格实现（`baseline/framework/ppo/policies/*_mlp.py`）只能**复制后适配**：不 import 原类、不继承 `TrainablePolicy`/`SamplingContext`/ratio 语义（D09）。
- 旧 single TN 内核的 `Z=Φ(b)-Φ(a)` + 固定 clamp 在 A4.5 已被实测证伪（大 σ 塌缩、窄边界失真）；SAC 必须用新的 erf-space 内核（D11 / N-SAC-01）。
- trainer 当前 `sample_action` 路径要改为消费 `expectation_samples(obs, noise)`（`[B,K,M,D]` 动作 + 联合 log_prob + 可微 `integration_weights`），不按八个类名写分支（A4.3）。

### P4.2 工作包顺序

```text
S4-W0 P4-AUDIT-0：SAC actor 契约与 A4.3 逐项对拍，登记差距；冻结八格 arch 命名与 --set 接入面
  → S4-W1 P4-KERNEL-1：N-SAC-01 erf-space TN 数值内核（Z/采样/密度/逆 CDF/保护计数/fail loud）
  → S4-W2 P4-ACTOR-1：A4.3 actor 契约升级 + S01 迁移到 TN bounded sigmoid-σ
  → S4-W3 P4-SINGLE-1：S00/S10/S11 单分量格（shared/state × bounded/unbounded）
  → S4-W4 P4-MIX-1：M00/M01/M10/M11 枚举 mixture（联合 logp、integration_weights、整向量分量采样）
  → S4-W5 P4-TRAIN-2：trainer 消费 expectation_samples；target/actor 逐候选 twin 枚举；分量级诊断
  → S4-W6 P4-REG-1：regularizer_mode=shannon/u_bonus/u_floor 三模式、Bθ 记账、λ/f/U_kind 旋钮
  → S4-W7 P4-KNOB-1：采集 e/random_start、优化旋钮校验、requested/effective 记录、调度状态恢复
  → S4-W8 P4-EXP-2：两实验 actor_arch 八格可换；每格 toy 短训 + smoke 验收
  → G4.x 阶段四收口评审
```

### P4.3 Gate 与验收映射

- **G4.0 契约审计（P4-AUDIT-0）：** actor 契约差距矩阵登记完毕；arch 命名（如 `s00/s01/s10/s11/m00/m01/m10/m11`）与实验接入面冻结；不改动训练实现。
- **G4.1 数值内核（P4-KERNEL-1）：** N-SAC-01 门槛逐项落地为永久测试：独立积分归一、采样分位点/矩、固定噪声梯度（μ/logσ/v/logits 路径）、尾部概率、action/support 与密度一致、float32/GPU 实际路径、保护触发率可诊断。
- **G4.2 actor 契约（P4-ACTOR-1）：** `expectation_samples` 权重沿 K/M 求和为 1；`sample_behavior` 的 e 映射（unbounded `3^e`、bounded `v+κe`）生效值可观测；`deterministic_action` 按 A4.2 约定；RNG state 可保存/恢复且各流分离；导出元数据 strict 校验。
- **G4.3 单分量格（P4-SINGLE-1）：** 四格分布/梯度/导出/RNG 重放测试；shared↔state 拷权重对拍密度一致。
- **G4.4 mixture 格（P4-MIX-1）：** 枚举积分对 toy 目标的 Q-only logits 梯度与独立积分/有限差分一致（含低概率分量）；K=1 退化与置换/重复分量结构测试；collection 端整向量 categorical 采样。
- **G4.5 trainer 集成（P4-TRAIN-2）：** actor/target 逐候选 `a_k` 先 min 后按 `p_k` 期望（A4.4 次序不可交换）；`sac_dump` 扩展保存每分量噪声/权重/twin 选择，recompute 对账通过。
- **G4.6 正则路线（P4-REG-1）：** 三模式 actor/target/系数公式逐项对拍；无隐式叠加；切模式/换 U_kind 不构成 exact resume；λ=0 显式消融可用；auto-λ 仅验证调整方向。
- **G4.7 旋钮面（P4-KNOB-1）：** A4.8 旋钮表逐项实现或显式拒绝；requested/effective 入 metrics 与 checkpoint。
- **G4.8 实验接入（P4-EXP-2）：** 两实验 `--set actor_arch=...` 换八格零 trainer 改动；每格完成 toy 短训（回报改善而非仅 loss 有限）+ Humanoid21 smoke。
- **G4.9 收口：** SAC suite + PPO 回归通过；SAC 内无 PPO policy import；八格逐项证据登记。

### P4.4 明确不做（阶段四边界）

- 不做 score-function/straight-through/Gumbel 的 mixture 梯度替代路径（A4.4 保留为大 K 候选）；
- 不做 reference/delta σ-floor、mixture 权重温度、独立额外动作噪声（传入即报错）；
- 不承诺 U 路线优于 Shannon、不默认叠加两种正则、不在阶段四做 U 路线正式训练验收（仅 toy + 单格对照探针）；
- 不做全八格收敛矩阵（阶段六）；不引入 PER/n-step/relabel。

### P4.5 `P4-AUDIT-0` 契约审计结果（G4.0）

逐项核对当前 SAC actor 面与 A4.3，登记差距如下：

| A4.3 契约 | 当前实现 | 差距 |
|---|---|---|
| `distribution(obs)` 返回 π 参数 | `forward()` 返回 `(mean, std)`，std 是共享标量 clamp | 缺分布对象；σ 参数化不是 A4.2 的 `(D,)` sigmoid-v TN bounded 语义 |
| `expectation_samples(obs, noise)` → `[B,K,M,D]` + 联合 logp + 可微 `integration_weights` | 只有 `sample_action(obs)` 返回 `[B,D]` + 标量 logp | trainer 的 target/actor 两处调用需迁移；K=1 退化仍需按统一结构返回 |
| `sample_behavior(obs, behavior_spec, rng)` | `SACBehaviorSpec.explore_factor` 已随 job 传递但 runtime policy 未消费 | e→σ 映射（unbounded `3^e`、bounded `v+κe`）未实现；生效值未记录 |
| `deterministic_action` | 存在，返回 `tanh(mean)` | TN 格需返回未截断中心 μ（mixture 取最大 p_k 分量），语义不同需重定义 |
| `uncertainty(obs, kind)` | 不存在 | peak/L2 度量、`uncertainty_kind` 解析、逐维 U 均需新增 |
| RNG 流分离与状态保存 | 单个 CPU `torch.Generator`，`state_payload` 未存 RNG 状态 | actor/target/β/eval 分流与 checkpoint RNG 恢复缺失 |
| 导出 strict 校验 | `S01RuntimePolicy` 校验 `policy_arch` | 缺 K、σ 参数化/界值、U_kind、内核版本、确定性语义元数据 |
| 分布本体 | 未截断 tanh-Normal | 不满足八格 TN 定义；`evaluate_actions` 的熵代理是 Normal 熵非 TN 熵 |

**PPO 复制适配边界：** 源面共 8 个 `*_mlp.py`（~3400 行）+ 10 个 `_export_template_*.py`（~4500 行）。复制范围限于：网络布局、σ/μ/logits 参数化与初始化、σ_min/σ_max/κ 常数、K 默认值、导出元数据字段名；不复制 SamplingContext/ratio/GAE 相关接口、`torch.multinomial→gather→Q` 训练路径（A4.4 已证其 logits 梯度缺失）、以及与 PPO collector 共享 RNG 的语义。

### P4.6 冻结决策

| 编号 | 决策 | 理由与边界 |
|---|---|---|
| SAC-R1-D41 | 八格 arch 命名固定为 `s00/s01/s10/s11/m00/m01/m10/m11`（第一位 0=shared σ / 1=state σ；第二位 0=unbounded / 1=bounded；`s`=single / `m`=mixture）；`actor_arch` 只允许这些值 | 与 A4.2 表格一一对应；`--set actor_arch=...` 是实验侧唯一接入口 |
| SAC-R1-D42 | 阶段四把现有 `S01Actor`（tanh-Normal）标记为 `actor_arch="legacy_tanh"` 保留兼容旧 checkpoint，新增八格走 TN 内核；八格默认不提供 legacy 语义 | 旧 checkpoint 可 warm-start 加载；新 run 默认 `s01`，避免把未截断分布冒充 TN |
| SAC-R1-D43 | `expectation_samples` 返回结构作为 trainer 唯一消费面：`actions[B,K,M,D]`、`log_prob[B,K,M]`、`integration_weights[B,K,M]`（可微，沿 K/M 和为 1）、`component_ids` 或等价诊断 | K=M=1 时必须逐位退化为单采样路径；trainer 不出现按 arch 名分支 |
| SAC-R1-D44 | `uncertainty(obs, kind)` 的 kind 只允许 `"peak"`（S 格 native）与 `"l2"`（M 格 native）；跨格调用另一度量允许但必须在导出/日志标记 `uncertainty_kind` | 沿用 A4.6「不统一两种 U」；相同 λ/f 跨度量不视为等强度 |
| SAC-R1-D45 | behavior spec 的 `explore_factor` 保留字段名，语义即 A4.8 的 `e∈[-1,1]`；`mode="deterministic"` 时 `e` 必须为 0 否则拒绝 | 复用已持久化的 provenance 字段，不引入第二套采集参数 |

`G4.0` 通过标准：以上矩阵与命名/接入面冻结完成；无「配置接受但语义未实现」的 arch 值进入注册表。

### P4.7 `P4-KERNEL-1` / `P4-ACTOR-1` 执行记录（2026-10-13）

**P4-KERNEL-1（`5f6f3e53`）：** 新增 `tn_kernel.py` — erf-space TN 内核（support `(-1,1)`，`μ∈[-1,1]`、`σ>0`）：erf 空间非负差分求 Z、erf 空间插值 + `erfinv` 逆 CDF 采样、`log_prob`/`cdf`/解析熵/解析均值方差；`TNKernelStats` 保护计数；敏感段内部 float64，输出 dtype 跟随输入；大 σ 均匀极限下解析均值/方差走级数回退（消除 `1−1` 相消）。永久测试 10 项：积分归一（含 σ=e²⁰ 均匀极限、μ≈±1、σ=1e-4）、采样矩、固定噪声 gradcheck、cdf/logp 自洽、端点保护、非有限输入 fail loud、float32/64/CUDA。

**P4-ACTOR-1（本次提交）：** 新增 `tn_actor.py` — `TNActor` 单一实现按 `arch` 轴覆盖八格（D41 命名），当前仅启用 `s00/s01/s10/s11`，`m*` 构造即报 `P4-MIX-1`：

- 网络：shared-σ 单分量走 `net`（Linear→Tanh→Linear→Tanh→Linear D 维 raw mean）；state-σ 与 mixture 走 trunk+head 布局（state 单分量 head=`[raw_mean|σ_ctrl]`，σ 半零权重+偏置 init 使 σ(obs)≡init_std）；`μ=tanh(raw_mean)` 是 TN 中心。
- σ 参数化：bounded → `σ=exp(r_min+Δr·sigmoid(v))`，`v_init=logit(p0)≈0.1644`（沿用 PPO 几何常数 σ_min=0.05/σ_max=2.0/init_std=e⁻¹）；unbounded → `σ=exp(clamp(logσ,±20))`。
- e 映射（D45）：unbounded `logσ + e·ln3`（≡σ·3^e）；bounded `v + α·e`，`α=ln3/(Δr·p0(1−p0))` 标定为 init 点一阶匹配 `3^e` 斜率；`e∉[-1,1]` 拒绝，deterministic 且 e≠0 拒绝。
- A4.3 契约：`distribution` / `expectation_samples`（`[B,K,M,D]` + 联合 mixture logp + 可微 `integration_weights=p_k/M`，K=1 退化为 1/M）/ `sample_behavior`（返回 `component_id`/`explore_factor`/`log_prob`/`sigma_eff_mean` extras）/ `deterministic_action`（单格=μ）/ `uncertainty("peak"|"l2")`（l2 用 TN 两两重叠积分闭式）/ 私有 CPU `torch.Generator` 可保存恢复 / `TNRuntimePolicy` 导出 strict 校验 `policy_arch=tn_*` + `kernel_version=tn_kernel_v1`。
- 兼容：`sample_action` shim（K=M=1 squeeze）保持 trainer 接口不变，`P4-TRAIN-2` 再迁移；`actor_arch="legacy_tanh"` 仍走 `S01Actor` 读旧 checkpoint（D42）。
- 新增 `actor_n_components` 实验属性（m* 格启用前无消费方）。
- 测试 46 项：分布形状/界、init σ=e⁻¹、expectation 契约（weights 和=1、梯度）、采样-评分一致、e 映射生效值（σ_eff_mean extra）、deterministic=μ、RNG 重放、导出 round-trip + e 注入、unknown/mixture arch 拒绝、payload round-trip、`build_actor` 分发。
- 证据：`pytest baseline/framework/sac/tests` → **117 passed**；`test_independence` 断言更新为 `policy_arch="tn_s01"`（默认 s01 已是 TN 语义）。
- 遗留：`loop.py` 目前 `to_blueprint(stochastic=True)` 未传 e —— e 的调度/生效值记录属 P4-KNOB-1；`uncertainty` 尚无消费方，P4-REG-1 接入；debugkit `_build_actor` 仍只认 `S01Actor`，P4-TRAIN-2/dump 迁移时一并处理。

### P4.8 `P4-SINGLE-1` ~ `P4-EXP-2` 执行记录（2026-10-13）

**P4-SINGLE-1（`9874cfde`）：** `trainer_state_dict` 增加 `actor_rng` 可选字段并在 load 恢复（A4.3 流可恢复）；每格（s00/s10/s11）经真实 `sac_update` + fake-env 闭环验证（trainer 只走 `sample_action` shim，不感知 arch）。测试扩至 153 项。

**P4-MIX-1 / P4-TRAIN-2 / P4-REG-1（`bf913eef`）：**
- m00/m01/m10/m11 启用：`expectation_samples` 返回 `[B,K,M,D]` 候选 + 联合 mixture logp + 可微 `integration_weights`（softmax(logits)/M）；整向量 categorical 采样；mixture L2 不确定度用 TN 两两重叠积分闭式。
- trainer 迁移到 `expectation_samples` 单入口：target 在 `next_obs` 枚举候选后**逐候选**做联合 twin-min 再按权重求期望（A4.4 次序不可交换，有反例测试）；actor 同理；`Σw·logp` 作为期望熵估计供 alpha；trainer 自有 `expectation_rng` 随状态保存/恢复；`legacy_tanh` 无该接口 → 训练 fail loud（仅 warm-start 加载）。
- `sac_dump_v2` 保存冻结 `u_next`/`u_actor`，debugkit recompute 用冻结噪声精确复算（实测 `max_abs_diff=0.0`）。
- `regularizer_mode ∈ {shannon,u_bonus,u_floor}` + `reg_lambda/u_floor/u_kind`：u 模式 target 加 `B(s')=λU` 或 `−λ·relu(f−U)²`（no_grad），actor 加 `−B(s)`（可微）；u 模式仅固定 λ，传 `alpha_optimizer` 拒绝；`u_floor` 必须显式 `f`；λ=0 的 u_bonus ≡ α=0 的 shannon（测试对拍）；模式/kind 进 config fingerprint（切模式非 exact resume）。
- G4.4 证据：`test_mixture_q_only_logits_gradient`（logits 梯度经 enumeration 权重路径）+ `test_mixture_low_prob_component_still_gets_gradient`（低概率分量仍获非零信号）。

**P4-KNOB-1（`b8d2b5cb`）：** 实验新增 `behavior_explore`（e∈[-1,1]）与 `random_start_transitions`；loop 每轮导出行为 bp 携带 `explore_factor`，随机启动期改用 `RandomCombatPolicy(scale=1.0)`（Uniform[-1,1]^D）；`SACBehaviorSpec` 从 bp config 读回实际生效值（requested=effective）；round metrics 记 `behavior.explore_factor`/`behavior.random_start`；deterministic+e≠0 拒绝。

**P4-EXP-2（`8dc502d9`）：** `_ENABLED_ARCHS` 放开全部八格；`distribution()` 返回原始 σ 控制量 `ctrl`，`_distribution_with_e` 直接在 ctrl 上加 e 偏移（避免 σ→ctrl 反演在饱和区失真）。实验构造测试扩为 八格 × 两实验。真实环境验收：
- `sac_standup`：s00–m11 全部完成 3 轮 smoke（每轮 60 env_step / 60 critic ticks），metrics 全有限；
- `sac_balance`：s11 + m11 smoke 通过（含双通道 + 早终止边界）；
- m11 checkpoint 完整 resume（clocks/replay/RNG 恢复）；
- m01 + `expectation_samples=4` 的 `sac_dump_v2` recompute `passed=true, max_abs_diff=0.0`；
- 导出 bp 自包含（`TNRuntimePolicy` + model.pt，`explore_factor` 写入 config）。

### P4.9 G4.x 收口审计（2026-10-13）

| Gate | 结论 | 证据 |
|---|---|---|
| G4.0 契约审计 | 通过 | P4.5 差距矩阵 + D41–D45 命名冻结（`5ddc22f5`） |
| G4.1 数值内核 | 通过 | `tn_kernel.py` + 10 项永久测试（积分归一/矩/gradcheck/保护/CUDA）（`5f6f3e53`） |
| G4.2 actor 契约 | 通过 | A4.3 全契约实现 + 46→117 项测试（`4c2957cd`） |
| G4.3 单分量格 | 通过 | 四格 trainer/update/RNG-resume 测试（`9874cfde`） |
| G4.4 mixture 格 | 通过 | 枚举积分 + logits 梯度 + 低概率分量测试（`bf913eef`） |
| G4.5 trainer 集成 | 通过 | 逐候选 twin-min + 次序反例测试 + dump recompute `max_abs_diff=0.0`（`bf913eef`, `8dc502d9`） |
| G4.6 正则路线 | 通过 | 三模式逐项对拍 + λ=0 消融等价性 + fingerprint 锁（`bf913eef`） |
| G4.7 旋钮面 | 通过 | e/random_start 生效值入 bp config + SACBehaviorSpec + metrics（`b8d2b5cb`） |
| G4.8 实验接入 | 通过 | `--set actor_arch=` 八格 × 两实验；真实 smoke 八格 + resume + dump recompute（`8dc502d9`） |

**回归：** SAC suite **164 passed**；PPO suite **232 passed**。静态审计：`framework/sac` 与 `experiments_sac` 无 PPO/rollout 运行期 import。

**不声称：** 八格收敛（阶段六）；U 路线优于 Shannon（仅 toy 对拍）；`legacy_tanh` 参与 expectation 训练（只允许 warm-start）；per-arch 长训稳定性。

---

## P5.0 阶段五执行计划（SAC 原生完整 debug 系统 · 第一版）

### P5.1 现状审计结论（先行事实）

阶段二~四已交付的数据基础（阶段五在其上建分析面，不改训练语义）：

- `sac_dump_v2`：`batch.pt`（含 `sample_ids`/`source_keys`/gate/valid）、`trainer_pre/post.pt`、`forward.pt`（含冻结 `u_next`/`u_actor`/`target_pair_index`/`actor_pair_index`）、`spec.json`、`analysis.json`、`manifest.json`（sha256）；`recompute` 精确复算。
- `source_key` 编码 `run/rNNNNNN/jNNNNNN/agent/eSEED/fNNNNNN/hHASH/vSCHEMA`，可反解 episode/agent/frame；`sample_id` 单调稳定。
- `events.jsonl`：config/export/round/tick/checkpoint/eval 事件；metric_catalog 校验命名。
- CLI 现仅有 `summary`/`find-sample`/`recompute`；dump 只有 `--dump-at` 预约，无按需触发；无样本级排序/trace、无 replay 人口学查询、无 HTTP/viewer。

PPO 侧对标深度（不照搬 GAE/ratio/clip）：`debug.py` CLI 11 个子命令、`dump_analysis.py` 样本级分析、`frame_access.py` 帧回溯、`viewer/server.py` HTTP UI、`dump_request.json` 按需捕获。

### P5.2 工作包顺序

```text
S5-W0 P5-AUDIT-0   现有 debug 面 × 分析链逐环差距矩阵；sac_dump_v3 增补字段冻结；页面/API 清单冻结
→ S5-W1 P5-DUMP-1  dump 增补：逐样本 TD/Q_c/target/actor 候选贡献/混合权重明细（若 forward 已有则只落盘），RECORD 说明
→ S5-W2 P5-CLI-1   debugkit CLI 扩展：catalog/inspect/samples(--sort)/trace(--sample-id|--source-key)/timeline/query
→ S5-W3 P5-REPLAY-1 replay 人口学：年龄/来源 round/policy·reward version/复用/overwrite 分布，入 metrics + checkpoint replay.pt 离线查询
→ S5-W4 P5-ONDEMAND-1 dump_request.json 哨兵按需捕获（复制适配 PPO 机制，SAC 语义=critic_tick）
→ S5-W5 P5-API-1   单一分析模块 + 只读 HTTP JSON server：/api/runs、/api/run、/api/metrics、/api/dumps、/api/dump/*
→ S5-W6 P5-FAULT-1 五类注入故障可被定位的永久测试：终止掩码错/旧数据过多/alpha 失控/通道压制/Q 异常
→ S5-W7 P5-INV-1   不变量：capture 开关不改 RNG/更新结果（确定性对拍）；常规 vs 重型诊断成本记录
→ G5.x 收口：SAC+PPO 回归、证据登记、PLAN 标记
```

### P5.3 Gate 与验收映射（第一版口径）

- **G5.0 审计：** 分析链每一环列出"已可查询 / 缺字段 / 缺工具"；v3 dump 增补清单与 API 清单冻结后才动实现。
- **G5.2 CLI：** `trace --sample-id` 能输出 run→round→job→agent→episode_seed→frame 完整链 + 该样本的 TD/Q/target/贡献；`samples` 可按 td_abs/Q 排序；任意 `events.jsonl` 指标可 `query`。
- **G5.3 replay：** 对给定 checkpoint 能回答"当前 buffer 中样本来自哪些 round/策略版本、平均年龄、每样本被采样次数"；正常训练 round metrics 有 replay 人口学摘要。
- **G5.4 按需：** 训练进行中写 `dump_request.json` → 下一 critic tick 捕获含 hypothesis 的 dump；与 `--dump-at` 共用同一 capture 路径。
- **G5.5 API：** HTTP server 与 CLI 调同一分析函数；只读、可指向 run 目录或 runs 根；离线 dump 目录可直接查询（不需训练在线）。
- **G5.6 故障：** 每类注入故障至少一条测试断言"debug 面能看到并指到正确环节"。
- **G5.7 不变量：** 同 seed 同配置 capture on/off 两次短训，关键指标序列逐位相等；dump 写盘耗时与体积记录在案。

### P5.4 第一版明确不做

- HTML/交互式 viewer 前端（先 CLI + JSON API，viewer 视使用情况再立第二版工作包）；
- 像素级帧渲染（transition 不存 core_state，重渲染需重仿真；第一版 `trace` 输出观测/动作/奖励事实而非 PNG）；
- 训练中在线 HTTP attach（server 只读已落盘工件）；
- eval 视频索引、render/delta/rollout 子命令（PPO 有；SAC 版待 frame 渲染能力就位后再议）；
- 跨 run 自动对比/统计显著性（先做"多 run 指标并列查询"，显著性判断留给使用者）。

### P5.5 冻结决策（本计划登记）

- **D50：** SAC debug 第一版的"截面"单位 = critic tick（与 `sac_dump` 一致）；按需触发的最小粒度同为 critic tick，不引入 env_step/round 级新截面。
- **D51：** 分析实现唯一化原则——`sac_debug/analysis.py`（拟）是 CLI 与 HTTP server 的唯一计算层；CLI 不得内联分析逻辑，server 不得绕过该层直接读 .pt。
- **D52：** replay 槽位不作身份（PLAN 已规定）；一切样本级查询以 `sample_id`/`source_key` 为键，槽位 index 仅作为物理定位手段出现在 trace 输出中。
- **D53：** 帧回溯第一版语义 = source_key 反解 + 该帧 transition/reward/task_facts 重放展示；不承诺像素渲染。

### P5.6 `P5-AUDIT-0` 审计结果（G5.0，2026-10-13）

#### 分析链逐环现状

| 链环 | 字段现状 | 查询现状 | 缺口 |
|---|---|---|---|
| episode→transition | `source_key` 含 run/round/job/agent/seed/frame；`task_facts`/`reward_features`/`policy_action`/`behavior`/`collection`/`versions` 随 slice 与 replay metadata 持久化 | 无 | `trace` 子命令：source_key 反解 + 逐字段展示 |
| replay 写入/保留/抽样 | `buffer_stats()` 已有 age/round/policy_fingerprint/reward_semantics/objective_mode 分桶 + overwrite 计数 | round metrics 已带 `collection_round_counts` | 缺**逐样本 draw_count**（复用次数）；缺行为 provenance（e/random_start）分桶；checkpoint `replay.pt` 无离线查询命令 |
| Bellman target | `forward.pt` 存 `targets`/channel、`u_next`/`next_*` 候选、`target_pair_index`、`reg_next` | recompute 精确复算 | 缺逐候选 `F_j` 网格与逐通道候选 Q（无法解释 pair 为何选 j）；缺逐样本 `td` |
| 各通道 Q | `q1_pred`/`q2_pred` per-channel 已存 | `samples` 无 | 需落盘即有的 per-channel 网格 |
| actor Q/熵梯度 | `new_*` 候选、`actor_pair_index`、`weighted_q`、`actor_integration_weights` 已存；`reg_actor` 条件存 | 无 | 逐样本 actor 项分解（α·logp vs −F）可从已有字段推导，不新增存储 |
| α/target 更新 | `alpha_pre`、stats 含 alpha/temperature/target_tick | tick metrics 有 | 无（series 查询即可覆盖） |
| 策略变化 | `trainer_pre/post.pt` 全量参数 | 无 | 分析层算 param-delta 摘要（L2/max|Δ|），不必另存 |
| 真实评估 | eval/export/checkpoint 事件在 events.jsonl | 无 series 查询 | `series` 子命令按 metric key 抽时间序列 |
| run 总览/多 run | config.json + events.jsonl | 无 | `runs`/`run` 摘要 + HTTP `/api/runs` |

#### `sac_dump_v3` 增补字段冻结

在 `sac_dump_v2` 基础上**只加不改**：

1. `forward.pt`：`next_cand_f` `[B,K,M]`（逐候选 twin-min 后 F_j）、`actor_cand_f` `[B,K,M]`、`next_cand_q`/`actor_cand_q`（dict: channel→`[B,K,M,2]` 双塔）、`td_q1`/`td_q2`（dict: channel→`[B]`）；
2. `analysis.json`：新增 `replay_stats` 快照（dump 时刻 buffer_stats）与 `consistency`（`terminated∧bootstrap>0` 等行级违例计数）；
3. `request.json`/`manifest.json`：`schema_version="sac_dump_v3"`；
4. `load_dump`/分析层接受 v2 与 v3，v3-only 字段对 v2 dump 报告 `unavailable`（D-fail-loud，不静默回退）。

#### CLI 子命令清单冻结（`python -m baseline.framework.sac.debugkit`）

已存在：`summary`、`find-sample`、`recompute`。
新增：`catalog [--prefix]`、`inspect`、`samples --sort td_abs|q_mean|logp [--channel] [--limit]`、`trace --sample-id|--source-key`、`series <run_dir> --metric`、`runs <runs_root>`、`replay <run_dir|replay.pt>`、`query <artifact> <json-path>`、`serve <runs_root|run_dir> --port`。

#### HTTP endpoint 清单冻结（与 CLI 共用分析层，D51）

`/api/runs`、`/api/run?name=`、`/api/run/metrics?key=&event=`、`/api/run/dumps?name=`、`/api/run/replay?name=`、`/api/dump/inspect?path=`、`/api/dump/samples?path=&sort=&channel=&limit=`、`/api/dump/trace?path=&sample_id=|source_key=`、`/api/catalog`、`/api/dump/recompute?path=`（同步调用，第一版不做异步 job）。

### P5.7 执行记录与 G5.x 收口（2026-10-13）

**P5-DUMP-1（`58f4cca4`）：** `sac_dump_v3`——forward.pt 新增 `next_cand_f`/`actor_cand_f`（逐候选 twin-min 后 F，`[B,K,M]`）、`next_cand_q`/`actor_cand_q`（channel→`[B,K,M,2]` 双塔）、`td_q1`/`td_q2`（channel→`[B]`）；analysis.json 新增 `replay_stats` 快照与 `consistency`（`terminated∧bootstrap>0`、`truncated∧bootstrap=0` 行级违例）；`load_dump` 兼容 v2/v3，v3-only 字段报 `unavailable`。真实 m11 dump 验证落盘。

**P5-CLI-1（`73d0ce0a`）：** 新增 `analysis.py` 为唯一计算层；debugkit CLI 扩展 `catalog/inspect/samples/trace/series/runs/run/replay/query`；`parse_source_key` 反解 run→round→job→agent→seed→frame→schema；`param_delta` 由 pre/post state 计算；`save_run_config_sac` 补 `knobs`（actor_arch 等入 config.json——此前 run 级分析无法识别架构，属审计新发现缺口）。

**P5-REPLAY-1（`2a515d0c`）：** replay 增 `draw_counts`（逐样本复用计数，随 state_dict 持久化，旧 checkpoint 缺失时置零——state 增量字段，不 bump schema）；`buffer_stats` 增 `sample_age_quantiles`/`draw_count_*`/`explore_factor_counts`/`random_start_rows`；`replay` CLI 支持 run_dir/checkpoint/replay.pt 三级离线查询。

**P5-ONDEMAND-1（`724e204e`）：** loop 每 critic tick 前轮询 `dump_request.json`（`{"hypothesis","critic_tick?"}`），原子消费并入调度表，与 `--dump-at` 共用 capture 路径；CLI `dump <run_dir>` 写哨兵；per-tick hypothesis 映射。

**P5-API-1（`ec14e19e`）：** `debugserver.py` 只读 HTTP server（stdlib ThreadingHTTPServer，无新依赖），10 个端点全部直连 `analysis` 层；`debugkit serve` 子命令；命名避开根 `.gitignore` 的 `debug_*` 规则。

**P5-FAULT-1 + P5-INV-1（`a1a78054`）：** `test_debug_faults.py`——五类注入故障各有定位断言（终止掩码→`consistency.violations`；旧数据→版本/round 分桶；α失控→`temperature.alpha` 序列单调发散；通道压制→`actor_weight=0` 时 critic 仍更新且 trace 可见；Q 爆炸→`targets_absmax`/`td_abs_max`）。不变量：capture on/off 两次 fake 训练 tick/round metrics 与最终 actor 权重**逐位相等**；成本经 `debug.capture_s`/`debug.bytes` 记录。

| Gate | 结论 | 证据 |
|---|---|---|
| G5.0 审计 | 通过 | P5.6 逐环矩阵 + v3 字段/API 冻结 |
| G5.2 CLI | 通过 | trace 输出完整溯源链 + 逐样本分解；samples 排序；series/query 覆盖 |
| G5.3 replay | 通过 | draw_count/年龄分位/版本与行为分桶，run_dir→replay.pt 三级查询 |
| G5.4 按需 | 通过 | `dump_request.json` 实测捕获 critic_tick_2（测试） |
| G5.5 API | 通过 | 测试内 urllib 打 8 类端点均返回 200/404 正确语义 |
| G5.6 故障 | 通过 | 5/5 注入故障定位测试 |
| G5.7 不变量 | 通过 | on/off 对拍逐位一致 + 成本指标 |

**回归：** SAC suite **181 passed**；PPO suite **232 passed**；静态独立性不变。

**第一版不声称：** HTML viewer、像素级帧渲染、在线 attach、跨 run 统计显著性——均已在 P5.4 冻结为后置范围。

---

## P6.0 阶段六执行计划（诊断驱动两任务稳定收敛）

### P6.1 先行事实（核对结果，非假设）

- 验收口径数值（阶段一已冻结）：standup `success` = `final_pot ≥ 0.9` 的 agent 比例；balance 以 `survival_rate` 为主；候选门槛 ≥90%，不得事后降低。
- 实验默认预算：`sac_balance` `max_env_steps=10M`、`episodes_per_update=256`、`utd_ratio=0.25`、`eval_interval=100K`、`eval_episodes=32`、`rollout_workers=96`；`sac_standup` `2M`/`64`/`0.5`/`20K`/`16`/半数 CPU。
- eval seed 已与训练隔离：`eval_seed = seed + 100_000 + round*97`；最终验收将再加一段完全不重叠的 held-out eval。
- 已具备的能力底座：export/eval/checkpoint/resume、`dump_request.json` 按需截面、`metric_series`/`runs`/`trace` 查询、replay 人口学、五类故障定位测试、divergence guard（Q 爆炸自动 checkpoint + 终止）。
- PPO 参考：basic_balance 约 27M env_step 首次 survival_rate=1.0；SAC 预算按更高样本效率设定但未经实测——阶段六即为实测。
- 用户裁决（本次计划）：**替代策略 = `m11`**（mixture + state-σ + bounded，与默认 `s01` 双轴差异）；**执行顺序 = 先单 seed 探路，趋势确认后铺 3 seed**。

### P6.2 验收协议冻结（先于任何训练）

- **达标判据：** 任务验收指标（standup=`eval.success_rate`，balance=`eval.survival_rate`）在**连续 ≥3 次 eval 事件**中 ≥0.9，且末次为该 run 的最终 checkpoint 策略；单次冲高不算。
- **训练 seed 集：** `s01` 主格 `{42, 1337, 2024}`；`m11` 替代格先 `{42}`，达标证据充分后补 `{1337}`。
- **最终验收 eval：** 训练结束后用 held-out seeds（`seed+900_000+i`，64 episodes/任务）对最终导出策略独立评估；与训练内 eval 无重叠。
- **失败预算纪律：** pathfinder run 若在给定 env_step 节点前无上升趋势（判据见下），停止并进入诊断，不烧满预算；任何"调参复跑"必须是单变量改动并记录假设。
- **每 run 记录项：** env_step@首次≥0.9、env_step@持续达标起点、wall-clock、critic/actor/alpha 终态、replay 终态人口学、dump 列表（固定 tick + 按需）、失败原因（若未达标）。
- **run 记账：** 每条 run 固定 `--run-name` 含 `{task}_{arch}_s{seed}`，配置经 `--set` 全部显式化并留存在 config.json `knobs`。

### P6.3 工作包顺序

```text
S6-W0 P6-PROTO-1   上述验收协议登记；写 P6 诊断 playbook（异常→debug 命令对照表）
→ S6-W1 P6-PREP-1  启动就绪：预算/并行度估算、磁盘与 keep_last 检查、真实 env 下
                   dump_request 热路径演练、导出 bp 被 round_runner 加载复核
→ S6-W2 P6-PATH-1  pathfinder：s01 × {balance, standup} × seed42，后台长训，
                   里程碑检查点审视（见 P6.4），不收敛则走 P6-DIAG
→ S6-W3 P6-SEEDS-1 s01 补齐 seed 1337/2024 ×两任务
→ S6-W4 P6-ALT-1   m11 × 两任务 seed42（达标则补 1337）
→ S6-W5 P6-FINAL-1 held-out eval + 策略加载/部署复核 + 视频抽检
→ S6-W6 P6-REC-1   全部 run 证据汇总入 DECISIONS；失败如实记录
→ G6.x 收口
```

### P6.4 里程碑与停止判据（pathfinder）

- `standup`（预算 2M）：若在 `env_step=500K` 时 `success_rate` 仍 <0.3 → 停训入诊断；
- `balance`（预算 10M）：若在 `env_step=2M` 时 `survival_rate` 仍 <0.5 → 停训入诊断；
- 触发 divergence guard / alpha 撞界且 entropy 崩溃 / 某通道长期零更新 → 立即停训入诊断；
- 诊断输出必须落到具体链环（mask/旧数据/通道/α/Q），不允许"换超参再跑"式的无证据迭代。

### P6.4a S6-W0/W1 执行记录（P6-PROTO-1 / P6-PREP-1）

- `P6-PROTO-1`（`ec76d5b1`）：新增 `baseline/framework/sac/DEBUG_PLAYBOOK.md`——症状→debugkit 命令→预期证据对照表（9 节）+ `debug_notes.md` 证据记录格式。
- `P6-PREP-1` 探针实测（`_probe_s01_standup_v3`、`_probe_s01_balance_v1`，均 1 round）：
  - **发现并修复两处 fp32 边界缺陷**（真实启动障碍，commit `8810accf`、`a2b0f240`）：
    1. `sample_from_uniform` 的 fp64 端点 clamp `1−eps64` 转 fp32 舍入为恰好 ±1.0 → `log_prob` 域检查崩溃；修：输出 dtype 空间二次 clamp + `dtype_endpoint_clamp` 计数。
    2. `mu=tanh(mean_head)` fp32 饱和为恰好 ±1.0 → `_check_params` 崩溃；修：`distribution()` 输出端按 dtype eps clamp。
  - 吞吐实测（与两条在跑的 PPO 训练共享 192 核）：standup round1 `12800 env / 465s ≈ 27.6 env/s`（rollout 200s + train 264s）；balance round1 `6794 env / 211s ≈ 32 env/s`（早期 episode 均长 26.5，随策略变好会变长，round 时长非线性上升）。
  - 量级估算：standup 2M ≈ 20h/run；balance 10M 预计数天/run（episode 变长 + `max_grad_steps_per_round=2000` 封顶后 train 段稳定）。
  - checkpoint 每 eval round 一个、无上限保留：balance 全程约 100 个 × ~1GB replay → 单 run ~100GB，磁盘（729G 可用）需在长训中监控，必要时人工裁剪中间 checkpoint。
  - 导出 bp 复核通过：`TNRuntimePolicy` blueprint 可由 `PolicyBlueprint.load` 独立加载并推理。
  - 资源注记：本机已有用户 PPO run ×2（standup / balance_step_mbs）在跑，SAC pathfinder 与其共享 CPU；GPU 分配 standup→cuda:0、balance→cuda:1。

### P6.5 明确不做（阶段六边界）

- 不达标时**不降门槛、不改任务语义**；产出失败诊断报告本身就是合规结果；
- 不引入未经消融的新机制（要引入须回到阶段四式验证链）；
- 不做 PPO 联合对比重训（PPO 参考值取自既有记录，不重新训练 PPO）；
- 不承诺壁钟时长；GPU 资源排队/失败恢复按 checkpoint resume 处理。

---

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
