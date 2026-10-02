# CombatBench 批量框架设计：E0 契约提案与原则总纲

## E0 提案状态与阅读方式

版本：**Draft 0.1 / 2026-10-01 / 待用户审阅**。源码审阅基线：`b44278dc`。实施顺序见 [ROADMAP 的 E0–E8](ROADMAP.md)；本文回答接口、状态和执行语义应如何设计，不代替路线图。

**本版没有实现新接口，没有执行可行性探针或训练验证，也没有冻结 E0。** 下文类型名、签名、目录均为提案，不能当作已有 API。标记含义：

- **[F] 已确认范围**：用户已选定，提案不得越界。
- **[P] 推荐契约**：本版提出的规范性选择，审阅后才能冻结；未单独标记的设计条款均属此类。
- **[Hn] 待验证假设**：设计依赖尚待实验确认的能力，集中列于 D17；不能用注释或成功案例代替验证。

建议先读 D0–D3 看整体，再读 D6–D11 看关键语义，最后读 D16–D18 看适用范围、风险和实施拆分。后半部分保留第一轮原则总纲作为历史背景。

## D0. 本轮边界与设计立场

### D0.1 已确认范围 [F]

1. 面向大规模训练采样，不要求任意负载下胜过 CPU；旧 M7 性能正式验收、旧 M8 独立 Agent 迁移验收暂停，不补记为通过。
2. 对训练侧继续提供 **`collect(jobs) -> List[Episode]`**。Job 同序、Episode 字段与数据语义不变；不改 Experiment/Trajectory/GAE/PPOBuffer 的公开数据接口。
3. **同一训练的单机 1–8 卡 rollout + 单 learner**。不做 DDP、多机、异步陈旧策略训练或 Tensor-native learner。
4. 仿真器与框架在设计和代码上真正分离；CPU 模型与任务仍为语义参考，CPU 实验迁移、验证和来源追踪继续建设。
5. 本次先给尽量完整的契约提案；允许保留未经验证的点，先供审阅，不因尚有假设就提前缩减目标。

### D0.2 本版推荐的八个核心选择 [P]

- **一组设备契约 + 显式 host 导出**，不继续维护两组职责重叠的 NumPy/device 主接口。
- **后端拥有物理状态，runtime 拥有生命周期，collector 拥有记录，coordinator 拥有分片**；禁止借私有字段绕过边界。
- **reset 不隐含在 step 中**；结束、记录、导出、释放和重置是不同操作。
- **mask-first、固定容量、按行隔离**；动态长度留在计数和有效位中，原生路径不依赖逐步 host 分支。
- **插件按静态声明装配，但以普通设备函数实现**；不先造通用 DSL、编译器或任意 Python 自动转换器。
- **一个语义模型、多个执行器**：eager 是可读的基准执行器，graph/fused 是可选优化，不能定义不同任务。
- **采样内部 tensor-first，对外 Episode-first**；不是借工程化之名扩展训练侧范围。
- **第一版同步定长波次、允许行内提前结束、不立即补位**；先让生命周期和多卡收集可解释，再评估补位优化。

## D1. 术语、维度与不可破坏的不变量

| 名称 | 含义 |
|---|---|
| world / env | 一场共享物理场景；首轮包含 robot_a、robot_b，不能把两个机器人拆到不同 GPU |
| job | 上层要求采集的一场 episode；模型、策略、seed、options 和 SamplingSpec 均由 Job 指定 |
| slot / row | 某 worker 上的物理存储行；只是位置，不是 episode 或随机身份 |
| wave | 一批槽位承载的 jobs；首版每有效 slot 一场 episode，完成后封存，不在波内补新 job |
| chunk | 一波中可选的执行/传输分块；不结束 episode，不产生假的 timeout，不更新策略 |
| collect | 一次完整采样请求；可能包含多个同构组、多个 worker 和多波 |
| agent transition | 一名 agent 的一条训练候选记录；是否用于训练仍由现有实验决定 |
| physics step / action step | 原始积分步 / 包含 S 个物理子步的控制步；二者不能混计 |

记号：`B` 为本 worker 的分配容量，`A=2` 为首轮任务 agent 数，`S` 为每 action 的物理子步数，`T` 为波次记录容量，`C` 为每 world 接触容量，`R` 为静态终止原因数。核心记录按 `[T,B,...]`，运行状态按 `[B,...]`；agent 轴可用 `[B,A,...]` 或按 agent 命名的静态映射，异维 observation 不强制 padding 到同一维。

**不变量：**

- 一个成功 collect 对每个输入 job 恰好产出一个完整 Episode；无漏、重、跨 episode 拼接、隐式截短或混版本。
- 物理 world 隔离；同 world 两 agent 的物理互动不被拆开。单 agent 终止不自动停止整个 world。
- 输入 job 序号与返回列表一一对应；padding、失败行和初始化用的辅助仿真不产生训练样本。
- 采样使用冻结的策略集合；observer/奖励输出与记录的采样时刻显式对应。
- reset 前保留最后后继观测；Episode 的 final_observation 继续采用现有整场结束语义，不擅自改为各 agent 结束时的不同快照。
- 不改模型、物理步数、奖励、初始化分布、采样数学来满足容量或性能约束。
- “同后端、同随机输入”与“不同 batch/卡数时浮点逐位一致”是两种要求；只把前者作为调度契约。

## D2. 层次、依赖方向与拟议代码归属

```text
既有 Experiment / Job / EnvBlueprint / SamplingSpec
                  |
          MigrationResolver
                  |
      ResolvedTaskPlan + DeploymentPlan
                  |
         BatchRolloutCoordinator
          /         |         \
     GPU worker 0   ...    GPU worker 7
            |
       WaveCollector ---- PolicyExecutor
            |
       BatchRuntime ---- Plugin / Observer / ObservationProgram
            |
       PhysicsBackend + ModelBinding / ControlProgram
            |
       mujoco-warp（首个真实后端）
            |
设备记录 -> host EpisodeExporter -> IPC -> 原序 List[Episode]
            |
既有 build_trajectories / PPOBuffer / GAE / 单 learner
```

图中下方导出是 collector 的数据出口，不是物理后端的职责。

| 拟议归属（并非已有目录承诺） | 内容与允许依赖 |
|---|---|
| `envs/batchframework` 核心 | contracts、state、runtime、execution；依赖 torch 和自身契约，不依赖具体任务或 baseline/PPO |
| `envs/backends/warp_batch` 后端包 | Warp/MuJoCo 驱动、物理视图、容量与完整状态；只依赖核心契约，不依赖 runtime/plugin/collector |
| `envs/humanoid21` 的 batch 绑定 | 模型/meta、PD、动作映射、观测和物理能力适配；连接模型与后端，但不管理 PPO |
| `baseline/humanoid21` 的设备任务单元 | 奖励、终止等原生派生逻辑，引用权威参数；不直接访问 Warp 私有数据 |
| `baseline/framework/rollout` 的 batch 适配 | Job/Episode、策略适配器、coordinator、worker、导出与来源记录；这里可以依赖既有训练策略实现 |

不把上述每个名词机械地变成一个大类；第一版只需要少数 protocol、结构化数据和组合对象。目录最终按依赖测试决定。`DeviceRollouter` 可暂作为既有入口的 facade，不能继续承担后端选择、任务特判、策略重建和多卡调度的全部责任。

**真正解耦的检查：** 可以不创建 runtime 而单独推进后端；可以只接 FakeBackend 测 runtime；换受支持策略不改 runtime；添加任务绑定不改 coordinator；单卡与多卡复用相同的 worker 内采样实现。

## D3. 配置解析与执行计划

### D3.1 分开两种配置

`ResolvedTaskPlan` 表达“执行什么任务”：源 blueprint、模型摘要、agent/obs/action schema、物理参数、生命周期配置、插件与 observer 顺序、options schema、采样能力、记录 schema 和语义 profile。

`DeploymentPlan` 表达“怎样部署”：后端/精度、设备列表、每卡 B、chunk 大小、eager/graph、显存/host 内存预算、进程传输、调试级别。改变 B/设备数不修改任务 hash，也不偷偷改变 jobs/update。

独立记录 `source_hash`、`task_plan_hash`、`execution_plan_hash`、`policy_set_hash` 与验证 manifest。任务语义相同不表示数值执行版本相同；旧 `Episode.blueprint_hash` 保留原逻辑 blueprint 含义。

### D3.2 拟议入口

```python
resolve_task(env_blueprint, sampling_requirements, registry) -> ResolutionReport
plan_collect(jobs, deployment, policy_snapshots) -> CollectPlan
instantiate_worker(task_plan, placement, resource_budget) -> WorkerSession
```

第一阶段解析在 host 完成；可检查源码/config/schema，不为探测“能不能运行”而先启动一轮 rollout。资源分配和后端编译前再次校验实际能力。

### D3.3 每项配置都必须得到处置

- 实例身份用 blueprint 位置/显式名称，不用 Python 对象地址或仅类名；同类两个插件具有独立状态和随机域。
- 参数分类为结构参数、可按行变化的 episode 参数、策略采样参数、纯诊断参数。只有显式声明为按行参数的值可在同一 batch 中不同。
- 每个输入键必须映射、显式确认为无语义影响，或拒绝；不能通过 `cfg.get` 默认吞掉未知参数。
- 未注册 callable、任意宿主副作用、私有 sim 访问不能假定可迁移。host 兼容必须显式获准，并标记失去哪些执行保证。
- “已注册”“可执行”“通过哪层验证”分开表示。`native` 不等于已经验证；`stale` 不等于 unsupported。
- 解析报告列出能力缺口、字段/配置路径、来源版本和建议转换单元；缺失必需能力时不得先运行再静默降级。
- 执行中禁止 attach/detach 插件、替换 observer 或修改结构参数。此类变更只能在无未完成 job 的边界重新解析并重建。

建议同构分组键包括：模型/物理配置、schema、插件/observer 布局及顺序、控制时序、记录布局、策略实现与冻结权重组合、stochastic 模式。首版不同策略组合分组执行，不实现任意 per-row 网络路由。

## D4. 物理后端契约

### D4.1 后端只负责物理，不负责 episode

后端拥有模型实例、积分/求解状态、接触缓冲、容量状态、设备资源和后端原生的恢复快照。它不知道 Job、Episode、奖励、策略版本或 plugin pool。

模型绑定提供静态 `ModelDescriptor`：模型 hash、body/joint/actuator 命名与索引、单位/坐标系、dt、控制 schema、可用视图及能力。Humanoid21 的归一化与 PD 作为 `ControlProgram`，在每个物理子步根据当前状态计算控制；既不移到 policy 内，也不因批量化只计算一次。

```python
class PhysicsBackend:
    def describe(self) -> BackendDescriptor: ...
    def initialize(self, mask, initial_state, scratch_policy) -> None: ...
    def read(self, fields) -> PhysicsView: ...
    def apply_patch(self, mask, patch, refresh_policy) -> None: ...
    def advance(self, mask, control, wrench) -> None: ...
    def capture(self, mask, destination, level) -> None: ...
    def restore(self, mask, snapshot) -> None: ...
    def status(self) -> DeviceStatus: ...
    def close(self) -> None: ...
```

这些是语义原语，不强制每调用一次就启动一个 kernel。`advance` 的基准语义是一个物理子步；优化执行器可以融合控制与多个子步，但必须保留相同反馈时刻、mask、外力寿命与终止边界。

### D4.2 mask 与写入

- `mask` 是设备上的固定形状 `[B] bool`，不要求把活跃行 `nonzero` 成可变长度列表。输入张量具有静态容量，只有被 mask 选中的行生效。
- masked advance 不改变未选中行会影响未来推进的状态，包括积分状态、warm-start、控制和挂起外力；允许重建不影响后续物理的派生缓存，但须遵守视图时效。[H1]
- 不允许仅恢复 qpos/qvel 就声称完整冻结行；不能让已结束行继续产生无效接触或 NaN，污染其他行的容量与错误状态。
- `initialize` 表示构造全新物理状态；`apply_patch` 表示有声明的字段修改；`restore` 表示同后端完整状态恢复。三者不得用同一个隐式“清零所有状态”方法替代。
- `scratch_policy`、`refresh_policy` 是有限枚举，不是任意回调：例如新场景清求解器状态、保留未修改状态、刷新运动学/接触。具体映射须与 CPU 对应操作核对。[H2]
- 同一阶段的覆盖写保持稳定调用顺序；声明的累加力可合并。不能为了 kernel 融合将“最后写入生效”变为求和，或重排有依赖的状态投影。

### D4.3 视图、坐标与接触

公共视图应提供任务实际需要的语义字段；低级 integration state 可以作为显式可选能力，不把 MuJoCo 的全部内部数组提升为通用契约。

| 字段族 | 推荐契约 |
|---|---|
| body pose | position `[B,Nbody,3]`，米、世界系；orientation 明确 quaternion `[w,x,y,z]` 或矩阵，不混用 |
| body motion | linear/angular velocity 分字段，声明参考点、世界/局部系、m/s 与 rad/s；不直接给一个语义不明的 cvel |
| joints/control | shape、单位、限制与控制语义来自 ModelDescriptor；normalized action 属模型绑定，不属通用物理单位 |
| contacts | 固定容量记录 + valid mask/count/overflow；geom/body 身份、位置、frame、力方向与作用对象明确，槽顺序不承诺稳定 |
| native integration | 后端命名空间内的 qpos/qvel 等；用途和跨后端可移植范围显式标记 |

接触默认向插件提供标准化记录或聚合查询，而不是 `efc_address/efc_force` 私有布局。flat/padded 是存储选择；建议支持固定容量 flat view + world_id 和语义聚合，必要时派生 padded view，避免强制每步复制全部接触。capacity 的 per-world/total 含义由 descriptor 明确，真实溢出必须报错，不能截断后仍提交数据。

每个视图字段带静态 `sample_phase` 与有效性规则：`post_integrate` 的姿态、`last_solver` 的接触力不必对应同一求值时刻。不能为了“全部最新”偷偷额外 forward，改变 CPU 原有的观测/接触时序；需要统一时刻时显式申请对应能力并验证。[H2]

`PhysicsView` 是借用视图，不是快照。任何推进、写入、reset 或 close 都可能使旧借用失效；要跨操作保留值必须复制到调用方拥有的缓冲。正常路径使用同一设备 stream 建立顺序，不用全设备 synchronize 保证偶然正确。

### D4.4 快照等级

- `semantic`：可移植的姿态/速度等，用于跨后端对照，不承诺精确续跑。
- `integration`：后端完整的未来推进状态，包括控制、warm-start、时间及必要 solver 状态；用于同版本恢复。[H3]
- `trace`：用于实际记录渲染/分析的选定字段与已计算 observer 输出，不把 CPU 重演计算的奖励当成原始数据。

快照携带 schema、模型、后端版本、精度、字段完备度。后端不能提供某等级就拒绝该请求，不能返回少几个字段的“完整”快照。

## D5. 状态、张量与所有权

### D5.1 唯一所有者

| 状态 | 所有者 | 生命周期 |
|---|---|---|
| 求解器/物理缓冲 | backend | worker 内实例；reset 修改选中行，close 释放 |
| 任务/model 常量 | task binding | plan 生命周期；只读，可按设备缓存 |
| world/agent 生命周期与逻辑时钟 | runtime | slot 承载的 episode；不放进 simulator |
| 插件/observer 状态 | runtime StateStore | 声明的 scope；unit 不私藏未登记的可变训练状态 |
| policy 权重 | PolicySnapshot/worker cache | 按内容 hash 冻结，有界缓存 |
| policy 隐状态/随机流 | PolicyExecutor 的声明状态 | per-agent/per-episode；不能与共享权重混同 |
| action/obs/observer 本步输出 | runtime/executor 的 IO 缓冲 | 借用至下一次推进；recorder 要保留则复制 |
| 轨迹、终止事件、final_obs | collector RecordStore | 波次/记录块；不能随 runtime reset 清空 |
| host Episode 数组 | exporter/结果持有者 | 返回后独立有效；不得引用即将复用的共享内存 |
| 分片、inflight、完成集合 | coordinator | collect 事务；不以 GPU slot 作为永久身份 |

### D5.2 schema 与 state 声明

```python
StateField(name, shape, dtype, scope, initial_value, checkpointed)
OutputField(path, shape, dtype, export_kind, validity)
UnitSpec(instance_id, hooks, reads, writes, states, outputs, rng_domains)
```

- scope 至少区分 `STEP`、`EPISODE`、`COLLECT`、`WORKER`；可影响任务的跨 episode 状态须绑定逻辑身份，不能因换 worker 改变含义。
- partial reset 按声明的 initial_value 恢复选中行，支持非零、NaN sentinel 和设备初始化函数；不能一律清零。
- 持久计数器必须声明；诊断计数可 worker-local，影响奖励/随机分布的计数不可以。
- 热路径输出是固定键的 tensor tree；标量、向量、矩阵、nested observer 都由 schema 描述。`None` 用 presence mask 表达；动态事件用固定容量和计数。字符串用预登记枚举，边界还原。
- `export_kind` 保留旧 Episode 的标量 list、ndarray、嵌套映射等表现，不强迫所有 observer 叶子都变成一维 scalar。
- 未声明字段、shape/dtype 不符、缺必需输出均报错；不再像当前 recorder 那样遇到不合规格叶子就跳过。

### D5.3 读写权限的实际保证

只读 facade 限制可调用操作，但普通 torch.Tensor 没有不可绕过的只读安全边界。提案不声称能防御恶意插件。[H4]

推荐：原生插件视为受信任扩展；正常路径借用零拷贝视图，禁止原地修改；debug/契约测试用字段写入校验、借用 epoch、受保护快照对比或隔离副本检测违规。写能力采用有阶段 token 的窄接口，缓存的 mutator 在 hook 结束后失效。每种写操作按字段/阶段授权，不只给一个“可写全部物理”的布尔值。

## D6. Runtime 与插件公开契约

### D6.1 Runtime 的最小操作

```python
class BatchRuntime:
    def reset_rows(self, mask, episode_inputs) -> ResetResult: ...
    def observations(self) -> ObservationView: ...
    def step(self, action_batch) -> StepResult: ...
    def abandon(self, mask, reason) -> EndResult: ...
    def snapshot(self, mask, destination) -> None: ...
    def close(self) -> None: ...
```

运行时结果可复用预分配存储；返回对象只是设备视图，不隐含 host 等待。`StepResult` 至少含 pre/post obs、实际执行动作、observer 输出、有效帧 mask、完成 action mask、实际子步数、agent/world 结束标记和终止事件切片。策略采样 extras 由 collector 配对，不传入物理后端。

`step` **不会 auto-reset**。结束行保持封存，collector 完成记录后才能请求下一次 reset。直接交互调用显式 reset；训练模式下对仍运行的行直接 reset 应报错，必须先 abandon，避免丢弃未完成 episode。

### D6.2 插件与 observer

保留熟悉的生命周期名作为语义层：`on_pre_episode`、`on_pre_action_step`、`on_pre_phy_step`、`on_post_phy_step`、`on_post_action_step`、`on_post_episode`。不把所有子步插件强行压缩成 pre/post_batch_step。

| hook | 物理写 | 其他允许操作 |
|---|---|---|
| pre_episode | 声明的初态/控制写 | 初始化自身状态，读 options，提出初始化结果/终止 |
| pre_action | 声明的动作/控制写 | 自身状态、事件、终止请求 |
| pre_phy / post_phy | 声明的力或状态投影 | 精确子步反馈、自身状态、终止请求 |
| post_action | 否 | 自身状态、命名指标/事件、终止请求 |
| post_episode | 否 | 自身状态和终局指标；不得改写已记录的历史帧 |

Observer 不得改物理、别的 unit 状态或提出终止，但可更新自己声明的内部状态与输出。ObservationProgram 专门生产 policy-ready observation，与 reward/debug observer 分开：reset 后刷新初始 observation 不等于凭空执行一次 reward step。

unit 只收到声明能力的 `HookContext`：只读任务输入、物理/指标视图、自有 StateHandle、RngView、EventWriter，以及获准的 Mutator。它拿不到 backend/runtime 的完整实例，不允许递归调用 runtime.step/reset。

### D6.3 调度与黑板

- 每个 phase 有一个装配时固定的 schedule；迁移时先保留 CPU 的 priority 降序和相同 priority 的注册顺序。
- observer dispatcher 是 schedule 中的显式位置，不规定永远第一。高优先级计分单元可以先写 metrics，observer 随后读取。
- 新单元可声明 reads/writes/after/before；依赖只用于验证/明确排序，不以推断出的拓扑顺序偷偷改写 CPU 执行顺序。循环、缺生产者或同字段无明确顺序的冲突写在装配时失败。
- 同一 hook 内一个插件提出终止后，仍执行该 phase 中其后的单元；在 phase 屏障消费终止，与 CPU 在 `_PluginManager.invoke` 之后检查的边界对齐。
- metrics 是有 schema 的命名数据，events 是有 schema 的追加流；“post_action 不许物理写”不等于禁止更新计分黑板。
- 原生 unit 不得在热路径 `.cpu()`、`.item()`、读取 CUDA bool、执行任意 I/O 或分配无界动态状态。是否 graph-safe 是另一项能力，不与 native 混为一谈。

## D7. 生命周期、时间与终止契约

### D7.1 每行状态机

```text
FREE -> INITIALIZING -> RUNNING -> ENDED -> FREE
           |              |
           +-----------> FAILED
```

`ENDED` 保留末态直到显式释放/reset。`FAILED` 是执行错误，不是一次任务失败或零奖励 episode。coordinator 收到执行错误应使整个 collect 失败。

至少分开四种 mask：`slot_valid`（不是 padding）、`world_running`（仍推进物理）、`agent_done`（训练/任务终止状态）、`policy_eval_mask`（本步是否调用该 agent 策略）。不能用一个 active_mask 同时代表这四件事。

### D7.2 正常一个 action step 的时序

以零基数组下标表示观测：记录帧 t 为 `obs_t, action_t, observer_after_t`；T 帧后的 final_obs 为 `obs_T`。这是记号澄清，与旧注释的一基 `obs_{T+1}` 是同一个物理时刻。

1. collector/PolicyExecutor 读 runtime 的 `obs_t`，冻结本次采样值和 extras。
2. runtime 接受 action，执行 pre_action schedule，处理该 phase 的终止请求。
3. 对尚运行的行，重复 S 个子步：pre_phy schedule → 终止屏障 → ControlProgram 根据当前物理状态算控制 → backend.advance → 物理计数增加 → post_phy schedule → 终止屏障。
4. 完成全部 S 子步的行增加 `episode_step`；执行 post_action schedule（包含相应位置的 observer 刷新及 timeout）。
5. 消费终止；新结束 world 执行一次 post_episode schedule。捕获用于 bootstrap 的后继 observation，禁止 reset。
6. collector 消费 StepResult，记录帧、observer、终止和 final_obs。若 post_episode 会改 observer 输出，兼容导出遵循 CPU recorder 在 `_core.step` 返回后读取的实际结果，不擅自改成 end-hook 前快照。
7. 仍运行的行进入下一 action；结束行不再采样、积分或生成新帧。波末导出后才释放槽位。

每个 phase 的 observer 输出带 freshness 描述；读取旧值若是 CPU 原有时序则明确保留，若调用方要求本 phase 新值而尚未计算则报错，不自动重跑有状态 observer。

### D7.3 终止历史，而非一个 reason

原因目录在 plan 中冻结，保留原字符串（包括 `imbalance_robot_a` 等自定义原因），设备端用整数编码。对每行、agent、原因保存首次出现的逻辑时钟和确定性顺序；同原因重复请求不重复导出，不同原因保留。

- env 级请求广播到全部 agent；`world_done = all(agent_done)`，除非任务显式定义了额外 world 结束规则并能映射 CPU。
- 请求顺序是 phase、子步、unit 调度序、unit 内声明的请求序；不依赖原子写的完成顺序。
- `records[0]` 保持首次有效原因，不引入“KO 总是压过 timeout”等新优先级；同一步多原因按 CPU 顺序处理。
- 新原因即使在该 agent 先前已终止后出现，仍按 CPU 的去重历史规则记录；不得只保存“第一次终止”后丢掉其他原因。
- post_termination_action 默认继续 `policy`；可明确选择 `hold`，但不能静默变成零动作或停止该机器人的物理互动。是否训练这些后续帧仍由现有实验截取。

### D7.4 子步内结束与异常边界

分别维护 `action_call_index`、完成的 `episode_step`、实际 `physics_step`、`substep_index`、`record_frame_index`。不要假定它们永远可互相换算。

CPU 当前实现可能在 pre_action/子步中结束：`episode_step` 未增加、post_action 插件未执行，但外层 recorder 仍收到一次帧调用。这可能使“终止步数”和“记录帧数”不同。推荐分两层处理：

- runtime 表达实际过程，StepResult 标记推进子步数、是否完成 action、输出 freshness，不伪造一个完整 S 子步。
- EpisodeExporter 通过明确的 CPU compatibility profile 映射原因步数、帧数和 observer 值。此 profile 尚待边界样例核对。[H5]

第一版正式支持 `POST_ACTION_END`；`INTRA_ACTION_END` 和 reset 立即结束在契约中有表达，但在导出/训练兼容验证前必须标为 pending 并拒绝生产使用，不能因为 runtime 有终止 mask 就宣布支持。zero-frame episode 也不能靠补一帧零数据过关。

`abandon` 用于显式退出/重置诊断，不作为成功 collect 的完整 Episode；采样预算耗尽、取消或设备错误不能冒充任务 timeout。

## D8. Reset 与初始化程序

### D8.1 Reset 输入

`EpisodeInputs` 含 mask、Job 身份、base_seed、派生随机域、经过 schema 校验的 episode_options 和必要静态索引。options 缺省在解析时明确填充或保留“未指定”语义，不以 padding 行的值当作真实 job 的默认值。

顺序：验证目标行可重置 → 清/初始化 episode scope 状态 → 发布 seed/options → backend.initialize 基础物理态 → 按计划执行初始化与 pre_episode 单元 → 刷新初始 policy observation → PolicyExecutor 重置对应行状态 → 标记 RUNNING。初始化中改态后的可见性由显式 refresh 约定保证，不用无条件额外 reward refresh。

跨 episode 状态不随 reset 清零；若它影响任务，需要定义逻辑身份与迁移/恢复规则。首版不支持跨 job 顺序耦合的任务状态，除非执行计划显式串行该链；不能让 worker 复用顺序改变任务。

### D8.2 随机摔倒这样的“带仿真的初始化”

不让插件拿到真实 backend 私有对象递归步进。推荐引入受限 `InitializationProgram`：可使用公开物理能力及由 worker 提供的辅助物理实例，按设备 mask 推进并捕获首个满足条件的状态。[H7]

- 第一版优先独立 scratch instance，保证部分 reset 不扰动运行中的行；scratch 资源计入预算。
- 全量 reset 时复用当前实例作为优化选项，但两种实现必须有相同语义，不能由插件自行窥探/恢复私有状态。
- 初始化子步、随机抽样、首次命中、超预算的任务定义都记录；不计入 episode 的正式 action/physics 时钟。
- 对 standup 保留 CPU 的首次命中条件、非目标机器人保护、速度与状态写回规则；不替换为有限初态库，不擅自清零速度。
- `max_init_steps` 若是源任务参数，按源定义处理；资源 watchdog 是执行错误，不能用它产生“初始化成功但其实未完成”的 Episode。

## D9. 随机性、策略与动作语义

### D9.1 随机身份与工作分片分离

`JobRef=(collect_id,input_index)` 用于路由、排序和去重；`SampleKey` 用于随机性。**不能为了保证 transport ID 唯一，就把 rank/slot/完成顺序混进随机种子。**

推荐默认 SampleKey 由 Job.seed 派生。相同 Job.seed、任务与采样配置重复提交，保留同随机输入语义；若上层要独立样本，应提供不同 seed，而不是 collector 偷加扰动。不同 episode/agent/unit/用途域/逻辑步/抽样槽通过命名域和 counter 区分。collect_id 不作为默认随机盐，重试不改变样本身份。

保留 CPU `SeedSequence` 派生树的解释及版本，必要时在波次开始的 host 阶段派生小规模 seed 元数据再上传。设备随机算法可不同，但必须声明，不能伪称与 NumPy/Torch 全局 generator 逐抽样一致。

目标是在给定输入下，调换 job 行号、分到不同 worker、增加 padding 不改变对应原始随机数。策略内部可变次数采样需有明确的 draw/attempt 编号，不能靠 batch 内共享 generator 消耗顺序实现。[H6]

### D9.2 PolicyExecutor

```python
class PolicyExecutor:
    def describe(self) -> PolicyCapabilities: ...
    def bind(self, frozen_snapshot, sampling_plan) -> None: ...
    def reset_rows(self, mask, sample_keys) -> None: ...
    def act(self, observations, mask, step_context) -> PolicyOutput: ...
    def close(self) -> None: ...
```

`PolicyOutput` 包括采样动作、extras、存在位及策略/采样版本引用。权重可在两 agent 间共享，隐状态和随机流不可因共享同一 module 而串流。首轮支持已适配的 feed-forward 策略；有状态策略必须声明状态与 reset 语义，不能假装 stateless。

- adapter 位于 rollout/策略集成层，复用既有分布和确定性动作数学；核心不直接 import TruncatedNormalPolicy。
- policy snapshot 按源码/架构/权重内容 hash 固定，不以可变文件路径作为版本保证；旧导出格式明确适配或拒绝。
- 一个 collect 可使用不同对手或 reference 历史策略，但整个引用集合冻结。首版按组合分组，不要求所有 jobs 使用一个 policy。
- `stochastic=False` 遵循 CPU 的 deterministic 调用及 extras 行为，不强行创建 log_prob=0，不评估本应未被消费的 SamplingSpec。
- reference、delta、callable explore_factor 的支持分别声明；有设备实现才能 native 执行，不让旧 `_check_spec` 白名单永久充当框架能力边界。
- 随机输入注入若需扩展底层策略采样入口，只能作为不改变既有调用数学的适配；不复制/另调 PPO。这一实现方式仍待验证。[H6]

### D9.3 三种动作不能混淆

记录内部区分 `sampled_action`（策略随机变量）、`command_action`（模型映射/插件后的命令）、`applied_control`（实际执行器输入）。单位、裁剪和变换都属于相应 schema。

CPU Episode recorder 当前读取 simulator.get_action；设备导出必须遵循该语义，同时不能让记录动作与保存 log-prob 对不上。对有非恒等动作变换的实验，需要声明哪一个动作是训练变量、如何重放对应分布并通过既有训练消费验证。[H8]

第一版默认只有“导出动作与采样动作一致”的 stochastic PPO 任务可通过；不一致时若未存在经验证的适配就拒绝迁移，不把 clipped Gaussian 当截断正态，不把执行器控制当 policy action。debug trace 可以同时保存三种动作，不改变 Episode 的既有训练字段。

## D10. 设备记录与 Episode 导出

### D10.1 RecordStore

预分配字段包括：按 schema 的 pre_obs、导出动作、policy extras/ctx、observer tensor tree、frame_valid、每行长度、终止原因首次记录、final_obs、episode metrics。需要逐 agent 的终止时刻快照时可作为诊断字段保存，但不替代 Episode 的整场 final_obs。

缓冲形状固定为 `[T,B,...]` 或分块同构布局。recorder 不是任意低 priority 插件，而是 runtime StepResult 的消费者；其状态不会被 partial reset 清理，也不能被任务插件直接改写。

- frame_valid 表示该 world 本次 step 调用应产生 CPU 兼容记录，不代表这名 agent 的该帧一定训练有效。
- 保留终止后仍参与世界的 agent 数据；不在 collector 内执行实验特有的 trajectory 截断或奖励组合。
- 一个记录块结束只触发封存/传输；未结束 episode 在下一块继续，策略版本不变，计数不重置。
- 首轮对每个任务要求有可证明的最大 episode 帧容量，或已实现的有界分块策略。不能以资源 T 上限代替源任务 timeout。

### D10.2 Exporter

```python
export_episodes(sealed_records, job_refs, source_profile) -> list[Episode]
```

采用批量 D2H、按行视图/复制和直接构造堆叠字段，避免每帧 dict 中转；用现有 `Episode.from_buffer_frames` 作为同数据参考测试，而非热路径必经的组装方式。

| Episode 字段 | 导出规则 |
|---|---|
| base_seed / episode_options | 原 Job 值及原 options 结构，不用 worker 内部编码覆盖 |
| episode_index | 保留现设备 collector 的 collect 内索引约定；不能把它当跨 run 唯一 ID，内部唯一身份另存 |
| blueprint_hash | 原逻辑环境 hash；实际后端与解析计划另记 provenance |
| num_frames / observations / actions | 只含有效记录帧，时间轴完整，首帧是初始 observation |
| extras / explore_factors / sampling_contexts | 缺省、None、静态/逐帧值与 CPU 表现一致；不补不存在的采样字段 |
| observer_outputs | 保留嵌套结构、scalar-list/array 形态与字段含义，不只支持 `(B,)` 叶子 |
| termination records | 原因字符串、首次出现 episode_step、稳定顺序，保留同 agent 的多个不同原因 |
| final_observation | 该 world 最终后继态，reset 前；缺失是错误，不像旧兼容代码一样回退为空映射 |
| episode_metrics | 源任务终局指标完整保留；后端来源用保留命名空间或外部 manifest，避免碰撞 |

早停 agent 的 bootstrap/trajectory 边界继续由现有 experiment 消费。若某个新迁移案例需要 Episode 目前无法表达的逐 agent timeout 后继态，先报告为接口约束，不在本轮偷偷修改 final_observation 含义或训练接口。

Episode format v3 当前没有保存 episode_metrics。建议本轮用采样/debug manifest sidecar 持久化 provenance 和缺失诊断字段，不暗中修改 v3 格式；单独导出 Episode 时明确提示它不是完整 provenance 包。[H9]

### D10.3 规模与内存

估算必须覆盖：simulator + solver/contact scratch + 初始化副本 + 策略缓存 + runtime/plugin 状态 + `[T,B]` 记录 + staging + 已导出的 host Episodes + IPC + learner 既有 buffer。不能只报 simulator 的几 GB 显存。

全量 `List[Episode]` 仍要求 host 持有整个 collect 的结果。wave/chunk 可降低设备峰值，无法消除此公共接口带来的最终 host 内存需求；超预算应在计划阶段拒绝。异步传输需要明确 CUDA event 与 host 缓冲所有权，不能在拷贝完成前复用设备页，也不能返回引用随后被覆盖的共享内存。[H10]

## D11. Eager、设备原生与 Graph 执行

推荐只有两个语义等价的执行器：`eager` 和可选 `captured`，两者都消费同一个静态 plan；HOST compat 是能力组合/模式，不是另一套偷偷改语义的快实现。

| 边界 | 允许的 CPU 工作 |
|---|---|
| plan/build | 配置解析、源码/权重校验、模型编译、内存规划 |
| wave 开始 | Job/options/seed 上传，策略绑定，必要初始化控制 |
| 原生 action/substep 循环 | Python 可以提交 kernel；不得下载 tensor 值决定正常任务分支 |
| chunk/wave 边界 | 检查错误、收集已封存记录、进度与取消；不更新策略或伪造终止 |
| collect 结束 | Episode 构造、排序、验证、诊断落盘、交给单 learner |

reset 内复杂初始化也需要逐渐设备化；如果依赖有限 host 检查，必须在计划中声明它属于初始化边界，不把整个执行标为全程无同步。

第一版不要求捕获完整长 rollout；可从单个 action 或固定长度 chunk 起。graph key 至少包括 backend/model/schema、B、子步 schedule、策略结构、记录布局和执行器版本；权重更新、地址变化与 shape 变化如何影响捕获要明确处理。[H11]

graph-safe 不能仅靠插件声明：`nonzero`、变长布尔索引、隐藏分配、随机数推进以及 mujoco-warp 内部路径都需要验证。未能捕获时，只有用户/部署允许 eager 才能显式选择 eager；不能自动切 HOST_SLOW 后继续标 native。

## D12. 多卡 Coordinator / Worker 协议

### D12.1 对外接口保持小

```python
class BatchRollouter:
    def collect(self, jobs) -> list[Episode]: ...
    def close(self) -> None: ...
```

部署与资源策略在构造时提供；诊断可经独立 report/metrics 暴露，不改变返回类型。首版 collect 不可重入；训练与 eval 串行或使用独立实例，不共享可变 runtime。

### D12.2 一次 collect 的事务

1. coordinator 固定输入顺序，为所有 job 分配 JobRef，解析任务并冻结全部策略快照；验证数据/资源预算。
2. 按同构键分组，按确定规则分到 worker，再切 wave；padding 没有真实 JobRef。
3. 每卡一个独立进程，先确定设备，再初始化 CUDA/Warp/Torch；不 fork 已初始化 GPU 上下文。[H12]
4. worker 收到 plan/version/sample keys，准备本地资源，执行采样；所有动作决策在本地 GPU，不每步向 coordinator 索取动作。
5. wave/chunk 完成后传输结构化 host 记录或 Episode；传输对象包含 collect_id、plan/policy hash、JobRef 列表、状态与完整性元数据，不发送活 GPU tensor 跨进程引用。
6. coordinator 检查每个 JobRef 恰好一次、字段合法、版本一致；全部成功后才按输入顺序返回。任何失败都不提交“剩余成功样本”。
7. learner 开始既有 update。下次 collect 使用新的冻结版本；历史策略引用仍按 Job 指定，不被“同步最新权重”覆盖。

### D12.3 调度与资源政策

- 默认所有 worker 同步 on-policy；首版静态分片，不要求动态 work stealing。完成快的卡等待不是错误，后续再用测量决定是否补充调度策略。
- world 不跨 GPU；不同组可以分不同 worker 或串行多波，不要求所有卡同时运行同一模型。
- learner 卡可与 rollout 卡重叠；采样和更新分阶段，但内存预算包含 learner 常驻资源。必要的资源释放只能在无借用缓冲时进行。
- 改每卡 B/卡数不改变输入 Job 列表；固定 global jobs 的规模测试与固定 local B 的吞吐测试分别报告。
- worker cache 按 plan/policy 内容 hash 命中，必须有显存上限和安全淘汰点；不能每 update 永久缓存一个新策略。
- host pinned memory/共享内存是传输实现选项，不是语义要求；首版可先用简单正确的 CPU 序列化，再基于测量优化。

## D13. 失败、取消、恢复与调试

### D13.1 错误模型

区分 `ResolutionError`、`CapacityError`、`NumericalError`、`BackendError`、`ContractError`、`WorkerLost`、`Cancelled`。失败信息至少含 collect/job/worker/device、plan/policy 版本、phase/逻辑时钟、错误类别及可用重放输入。

设备计算将错误写入固定状态缓冲；在明确边界检查。出现可能导致非法内存访问的容量溢出必须由 kernel 安全防护或后端立即失败，不能先越界再指望波末发现。污染 CUDA context 后 worker 退出，不尝试在同进程静默继续。

任一卡失败使 collect 整体失败；取消也是显式失败，不把半个 Episode 回传为成功。首版不自动重试：由上层决定是否从同一 checkpoint/输入重试，并保留失败记录。副作用日志按 collect/job 身份可去重，但不承诺 GPU 内核“恰好执行一次”，只承诺成功结果提交不重复。

### D13.2 恢复边界

本轮只承诺 update 边界恢复。优先把正式 sampler 设计为 **同一冻结输入集合的可重复调用**：episode 状态结束即释放，随机状态由输入派生，worker cache 不影响任务。这样恢复不依赖上一次 worker 的本地行号或启动顺序。

若确需影响语义的跨 collect 插件/策略状态，必须提供版本化 state_dict 并通过现有运行 checkpoint 附属机制原子关联；在集成验证前列为 pending，不能只保存权重却声称可恢复。[H13]

完整物理 snapshot 服务于定位、回放与未来扩展，不等于本轮承诺任意 rollout 中途无损恢复。恢复时模型/源码/plan 不兼容必须拒绝或显式创建新的实验分支，不能继续沿用旧验证标签。

### D13.3 调试不篡改数据来源

- DebugSpec 在波次边界指定 job/agent/时间窗口/字段，估算存储后分配；不用常态保存全部环境完整物理状态。
- trace 包含真实观测、动作、observer/终止与所选物理快照；CPU renderer 可画姿态，数值面板仍读原始记录。
- 实际记录回放、同后端重跑、CPU 交叉评估是不同操作，分别标记。
- Debug 不消耗训练随机域、不改任务时序；有扰动的 profiling 模式单独声明，不用其 timing 冒充正常吞吐。
- 显式 close 按 collector 借用/拷贝完成、runtime/unit 清理、backend 释放、worker join 的顺序处理；支持重复 close，报出资源清理失败，不后台遗留失联采样进程。

## D14. CPU 实验迁移与支持状态

迁移不是重新定义实验，也不是把任意 Python 插件自动 GPU 化。流程：

```text
冻结 CPU 输入与依赖
 -> 提取实际能力需求/配置处置表
 -> 选择已验证模型、物理、插件、observer、policy 绑定
 -> 为缺失单元补原生实现或显式兼容模式
 -> 生成解析计划 + manifest
 -> 相同输入与生命周期对照
 -> 固定策略、Episode/训练消费检查
 -> 按需要短训练/更高层验证
```

每个单元与整个计划都报告三组独立状态：

| 维度 | 例子 |
|---|---|
| 执行方式 | device-eager / device-captured / host-compat / unsupported |
| 能力完整度 | required-satisfied / pending / rejected |
| 验证证据 | not-run / pass / fail / stale + 实际验证层级、输入版本 |

来源变更须按显式依赖使相关证据失效；无法可靠追踪动态依赖的单元应保守扩大依赖集或拒绝“自动判为仍有效”。新 plan 没有证据可以用于标记清楚的开发探针，不能拿旧 plan 的 pass 自动背书。

迁移时可提取共享纯数值规则和常量，但不移动/改写 CPU 规则而不建立新参考；既有 CPU 主路径在每个重构工作包仍可单独运行。HOST 适配不代表“先用慢版撑住后默认算已迁移”。

## D15. 两个具体场景检验设计

### D15.1 standup_floor04

- 512/4096 等数量由上层 Job 列表决定；模型绑定提供 96 维 observation、21 维 normalized action、每 action 25 子步和每子步 PD。
- InitializationProgram 生成真实随机摔倒态；两 observer 保持四阶段 potential 规则；runtime 的 timeout 仍为 200 action steps。
- 单 worker 或多 worker 只改变放置；每个 slot 一次 episode，结束后不再执行一轮摔倒 reset。
- 输出完整 Episode，由既有 Standup.build_trajectories 计算 `0.01 × potential` 等训练通道；collector 不理解 potential 的含义。

### D15.2 basic_balance（建议第二个真实迁移案例，尚未验证）

该现有实验及 `basic_balance_v2_phi_dual_env.yaml` 包含逐 agent imbalance 终止、多个 observer 和两个奖励通道，适合暴露 standup 没覆盖的生命周期与数据结构问题。

示意：A 在 action 17 结束，B 在 43 结束；world 继续到 43。默认 A 的策略与物理互动继续，记录整个 world 的 43 帧；A/B 原因历史保留各自首次步数。实验现有 trajectory builder 再按原因切片，collector 不自作主张只导出 A 的 17 帧或删除其后续物理作用。

此例仍需核对 CPU observer/接触与终止逻辑，不能仅看 YAML 就标 native。另用小型受控任务覆盖子步状态反馈、同一步多个原因、非零状态初值与主动 abandon，不强求一个复杂真实实验承担所有边界测试。

## D16. 首轮能力矩阵提案

| 能力 | 契约目标 / 首轮处理 |
|---|---|
| 完整 world、双 agent、不同初态/options | 必需；schema 化按行输入，不写死 simulator 构造参数 |
| 固定上限 episode、同步 wave、padding | 必需；超资源预算不改变任务 timeout |
| post-action per-agent 早停 | 必需；完整 world 记录与 trajectory 截取分离 |
| partial reset、显式结束/abandon | runtime 必需；collector 首版不波内补位 |
| 子步外力 schedule、状态反馈/投影 | 契约必需；至少受控案例验证，不能把 feedback 当 schedule |
| 子步内终止、reset 立即终止 | 能表达；CPU Episode profile 核对前 pending [H5] |
| 标量/向量/nested observer、多通道 | 必需；奖励通道组合仍由原实验处理 |
| 截断正态、不同对手、deterministic eval | 必需；先按同构策略组合分组 |
| reference/delta/设备采样调度 | 设计扩展点；逐能力迁移验收，未实现明确拒绝，不承诺全部 callable 自动转换 |
| recurrent policy | 显式状态接口预留，首轮默认 pending，不改现有 trainer 来迁就 |
| 同机 1/2/8 卡采样 | 必需；同一 collect、同策略快照集合、同序完整结果 |
| eager 无逐步 host 数据分支 | 原生路径目标；后端实际限制必须可见 [H1/H6/H7] |
| CUDA Graph | 可选执行器；不能成为表达全部任务的前提 [H11] |
| 任意 model/agent 数、跨 job 状态链 | 非首轮普适承诺；按 schema/能力拒绝或显式另案 |
| 多卡 learner、多机、异步滞后策略 | 不在本轮 [F] |

“必需”表示新框架工程验收目标，不表示当前实现已经具备。若关键后端能力不成立，应报告设计分叉，不静默删除验收项。

## D17. 待验证假设与失败时的处理

| 编号 | 未确认的点 | 后续验证方式 | 不成立时的处理 |
|---|---|---|---|
| H1 | Warp masked advance/reset 能保持未选中行的完整未来物理状态，且不让无效行污染容量 | warm-start/外力/接触下逐行对照，多次 reset/推进 | 后端增加经验证的隔离机制；否则早停等能力保持 pending，不能只恢复 qpos/qvel |
| H2 | 统一视图的采样时刻与 patch/forward 语义能与 CPU 对齐 | 单步、子步、状态写入后读、接触力时序 fixtures | 将差异留在显式 compatibility profile；无法保持的配置拒绝 |
| H3 | 完整 solver/integration snapshot 的字段集合足以同版本恢复 | capture→推进→restore→重放，检查隐含状态 | 降低所支持的 snapshot 等级，不称精确恢复 |

> **H1–H3 已于 E1-W0/W2 落地结论（probe_isolation.py 实测，详见
> E1_PLAN.md 附录）**：H1 的原语假设不成立——warp 1.12.1 无
> masked advance（`mjw.step` 无 mask 参数），契约改为
> `advance()` 全行推进 + `capture(mask)`/`restore(mask)` 原语，
> ENDED 行冻结由 runtime 以 write-back 组合（冻结行逐位稳定、
> 不污染他行）；H2 确认 `mjw.step` 内部刷新 derived，写后须显式
> forward——已编码为 `RefreshPolicy`；H3 按预案降级——integration
> 快照为近似恢复（~1e-7/10 步，warp 同输入本就非逐位确定），
> 契约不承诺逐位续跑。
| H4 | 低成本 debug 检查足以发现常见只读/越权写错误 | 故意越权与缓存 mutator 负例，成本测量 | 加强隔离或限定受信任扩展；不宣称 tensor 安全沙箱 |
| H5 | CPU 子步中止/零帧/终止后 observer 的 Episode 表达可无歧义兼容 | 精确 hook/计数/recorder 对照及现有 trajectory 消费 | 生产迁移拒绝相应 profile；必要时另行讨论 CPU 契约，不能批量侧自改 |
| H6 | per-job 设备 RNG 可与现策略分布数学及 graph 重放共存 | 重排/分卡/padding 原始随机输入对照、log-prob 与分布检查 | 使用显式能力较弱的开发模式；不伪称分片稳定，原生目标仍待解决 |

> **H4–H6 部分落地（E2，2026-10-02）**：
> - **H4**：装配期校验已实现（declared_reads/writes、per_hook_mutator
>   动词收窄、HOST_SLOW 拒绝、observer ctx 无 mutator）；hook 前后
>   快照比对式 debug 写检查未做——留到有真实越权案例时。
> - **H5**：子步内终止已可表达且屏障在子步粒度生效（封存于提议
>   子步、`episode_step` 不 +1）；但"子步内终止的 Episode 导出与
>   trajectory 消费逐字段核对"仍未完成——collector 当前路径下
>   终止只发生在 post_action 屏障（无子步插件时），INTRA_ACTION_END
>   profile 保持 pending。
> - **H6**：`RngView.unit_seed` 行重排不变性有测试；graph 重放与
>   log-prob 对照未验（E4/E7 继续）。
| H7 | 有界 scratch 初始化可忠实实现随机摔倒及部分 reset | 首次命中/速度/非目标保护与分布对照，资源测量 | 降低并发/分波，显式边界同步；不替换初态分布 |
| H8 | 动作修改后的记录与现有 PPO 训练变量兼容 | 对照 sampled/command/get_action/log_prob 与重算 | 非恒等变换任务拒绝，除非另有经验证的适配 |
| H9 | 不改 Episode v3 仍可完整对接 provenance/debug | sidecar 保存/加载、dump/viewer 引用检查 | 明确功能缺口；若要改格式另行提案，不偷加不持久字段 |
| H10 | 全量 Episode/IPC 的 host 峰值可由预算控制 | 大规模记录导出与内存生命周期测试 | 缩小显式采样配置或优化导出；不自动改变训练接口/丢样本 |
| H11 | 当前 Warp/PyTorch 组合可安全捕获目标 action/chunk | eager/captured 同输入对照、随机推进/地址/分配审计 | 保持 eager 原生执行，graph 不阻塞语义能力 |
| H12 | 1 进程 1 卡的库初始化/stream/共享 learner 资源行为可靠 | 非默认 GPU、1/2/8 卡启动/关闭与故障注入 | 修复资源隔离与进程协议；不改为多个独立训练冒充多卡采样 |
| H13 | 必要持久 sampler 状态可与 update checkpoint 原子关联 | kill/resume、版本变更、拓扑变更负例 | 首版仅接纳输入决定的无跨 collect 语义状态任务 |

以上是后续验证任务，不是本次已经得到的结果。每条验证失败都必须保留证据，并回到相应契约决定是否修订；不能通过改变 CPU oracle 或降低任务能力悄悄通过。

## D18. 审阅重点与 E1 实施拆分

### D18.1 建议先审阅的设计选择

1. 后端只提供物理原语，PD/动作/观测属于 model binding；runtime 不再依赖具体 sim 私有字段。
2. reset 显式、step 不 auto-reset；结束行封存，首版 wave 不补位。
3. 插件仍使用熟悉的生命周期，但需声明状态/读写/输出/随机域，并保留 CPU 调度顺序。
4. 用设备视图 + mask + 固定容量表达动态行为；eager 先完整，graph 是可选优化。
5. Episode 与单 learner 不变，接受 host 导出/全量结果内存是本轮边界，而非假装消除了它。
6. 多卡共享同一 collect 的冻结策略集合，静态分片起步；任一卡失败导致整体 collect 不提交。
7. 超出已验证语义 profile 的任务提前拒绝，尤其子步中止、动作修改和跨 job 状态，不能为“通用”牺牲可解释性。

这些均为提案供审阅，不要求用户现在逐项选择技术细节。若整体方向认可，再核验高风险假设并冻结具体签名与首轮支持矩阵。

### D18.2 E1 的建议拆分（审阅通过后才执行）

- **E1-a / 依赖切面**：从 MJX 父类提取共享模型/归一化/状态映射；保留旧 facade 和已有测试，不同时更改物理或任务公式。
- **E1-b / 设备原语**：引入 BackendDescriptor、device/stream 和 masked 物理/视图契约；先验证 H1/H2，再扩展相应生产能力。
- **E1-c / 状态归属**：将 episode/IO/plugin/RNG 分配从 simulator 迁到 runtime；保持旧 collector 输出，建立隔离与生命周期测试。
- **E1-d / 任务绑定**：把 Humanoid 维度、PD、观测与 backend 标准视图连接；消除插件/observer 对 `_model/_robots/_torch_views` 私有访问。
- **E1-e / 兼容出口**：旧 NumPy accessor 改为显式 snapshot/export 适配；固定公共入口与弃用提示，不未经批准删除历史 JAX 资产。

先拆所有权和依赖，再改调度与采集；不要把 E1、E2、E3 合并为一次不可审计的大重写。现有物理/任务 fixtures、FakeBackend 生命周期、Episode 管线与 log-prob 对照作为后续实施的回归资产，本次未重跑。

---

## 历史原则总纲（2026-09-29）

以下保留第一轮原则与设计背景，不是当前完成能力清单，也不是新 API。主路径权威、原生执行、不静默改变任务、分层验证等原则继续适用；第二轮范围以 ROADMAP 为准，具体新契约以上述 **未冻结草案** 为审阅对象。

## 1. 目标与定位

当前 CPU MuJoCo 路径是主路径，也是任务语义的权威来源。MJX 是可选的加速路径，不取代主路径，不为了加速反向限制主路径的实验表达能力。

我们要提供四项能力：

1. **对等的仿真实现**：在明确的支持范围内，与主路径 simulator 的功能和数据语义基本一致，尽量只保留不可避免的数值计算差异。
2. **以 Rollout 为边界的另一套执行框架**：拥有适合批量设备执行的原生运行时和插件体系，不要求复刻主路径的内部执行方式。
3. **与原插件体系的兼容接口或工具**：支持有边界的旧代码执行、原生代码转换和两种实现之间的对照验证。
4. **面向 AI 的转换与验证机制**：让低成本 Agent 在清晰约束、模板和工具帮助下，把主路径实验转换为加速路径实验，降低用户心智负担，并提供可检查的正确性证据。

总体原则：

> 不保证任意实验都能被自动加速；保证受支持实验有低心智负担的转换流程和明确的验证证据，不受支持的情况能够被可靠识别，而不是静默转换出一个不同的任务。

不追求所有代码完全复用。追求的是：**一份权威任务语义、一套主要训练与分析体系、两种采样执行实现，以及受验证约束的派生代码。**

## 2. 主路径与加速路径的关系

### 2.1 主路径是权威，加速路径单向派生

- 新实验优先在主路径实现和验证。
- 模型、观测、奖励、初始化、终止和课程等任务规则，优先在主路径变更。
- 加速路径根据主路径版本进行转换、更新和验证。
- 加速路径维护执行差异，不独立改变任务目标。
- 主路径更新后，相关转换结果应被标记为需要重新验证，不能默认仍然有效。
- 若发现主路径缺陷，单独修复并建立新的参考版本；不能只在加速路径中修正，再宣称两者等价。

“单向派生”指语义来源和维护关系，不强制所有文件都由工具自动生成，也不禁止提取两条路径共用的数值函数。

### 2.2 加速不是降低语义标准的理由

以下变化都不是普通的数值误差：

- 替换模型、减少接触、删除对手、改变摩擦或控制参数；
- 修改 observation 字段、顺序、单位、坐标系或归一化；
- 改变初始化分布、奖励公式、阶段门槛或终止规则；
- 减少物理子步、把逐物理步 hook 改成每 action step 执行一次；
- 把 timeout 当作真正 termination，或者用 reset 后的观测做 bootstrap；
- 为提高 GPU 利用率，未经声明就改变 PPO 每次更新的数据量。

这些改变可能构成有价值的新实验，但必须另行声明和验证，不能作为原实验的透明加速实现。

## 3. 对等性的范围与层级

### 3.1 仿真对等的范围

首期对齐对象是 CombatBench 的 `Humanoid21Simulator` 及目标实验实际使用的能力，不是 MuJoCo 的全部功能集合。

应对齐：

- 模型、物理参数、关节和执行器映射；
- 动作含义、裁剪、PD 控制、控制频率和物理时间步；
- core/derived/sensor 等数据的字段语义、坐标系、单位和采样时刻；
- 接触归属、有效性、力的表示与聚合语义；
- 状态写入后的派生数据刷新、缓存失效和可见性；
- 外力施加、累积或覆盖、持续与清除语义；
- reset、种子与 options 的传播，以及需要清除的仿真和插件状态。

批量维度、接触 padding 和内部存储允许不同，但适配后的语义必须一致；容量不足或溢出必须显式报告，不能静默丢接触。

MJX 支持范围受版本和实现限制。若主路径使用不受支持的功能，应选择合适后端、显式限定支持范围，或拒绝转换。不得偷偷替换成近似功能。

### 3.2 不追求 Bit Identical，不等于接受逻辑漂移

| 层级 | 要求 |
|---|---|
| 接口与离散语义 | 字段含义、时序、分类、终止规则等严格对齐 |
| 相同输入上的数值逻辑 | 观测、奖励、插件状态转移在约定容差内一致 |
| 物理单步与短程推进 | 分字段测量误差，解释接触和求解差异 |
| 长期轨迹 | 不要求逐帧或逐位一致，关注分布和任务行为 |
| 学习结果 | 用多 seed、固定评估协议和样本预算验证接近 |
| 实际收益 | 同资源口径下测量端到端速度和达到目标质量的耗时 |

不同后端可能不止存在浮点舍入差异，还可能有碰撞检测、接触生成或求解算法的差别。“最好只有数值差异”是工程目标，不是预先成立的事实。

相同 seed 也不保证 NumPy、Torch 和 JAX 产生相同随机序列。逻辑对照测试应能注入相同初态、动作或随机输入；训练验收则验证随机性语义和分布，不伪称逐样本复现。

## 4. 架构切面：统一 Rollout 契约，两套执行引擎

```text
主路径实验与任务定义：权威来源
                │
         AI 转换、适配与验证
                │
共同上层：实验接口、训练算法、评估口径、数据分析
                │
        统一 Rollout 输入输出契约
          ┌─────┴─────┐
          │           │
       主路径       加速路径
     EnvRuntime   BatchRuntime
     原插件体系    原生设备插件
     CPU MuJoCo   MJX simulator
          │           │
          └─────┬─────┘
                │
      Episode → Trajectory → PPO
                │
       checkpoint / 日志 / dump
```

### 4.1 训练层的主要边界：Job → Episode

保留现有 Job/Episode/Trajectory 契约作为首期接入点，允许必要的后端选择、能力检查和来源元数据扩展。

Rollout 契约必须覆盖：

- Job 的策略、任务配置、seed、episode options 和 SamplingSpec；
- 输入 Jobs 与输出 Episodes 的对应关系；
- `obs_t → action_t → 后继状态的 observer/reward` 的时间关系；
- agent、episode、trajectory 边界和有效帧数量；
- final observation、termination/truncation 及 per-agent 终止记录；
- 行为策略版本、SamplingContext 和训练侧 log-prob 重算语义；
- observer 字段、指标、任务配置与实际后端的来源信息。

这个边界是数据和语义契约，不要求设备热路径每步构造 Python 对象。内部可以保存紧凑设备数组，采样结束后批量导出 Episodes。只有 profiling 证明该边界成为瓶颈时，才进一步设计 tensor-native 通路。

### 4.2 复用与派生范围

| 部分 | 目标 |
|---|---|
| 实验超参数、奖励通道、课程和评估定义 | 一个权威来源，后端无关部分尽量直接复用 |
| `build_trajectories`、`on_eval`、`post_update` | 同契约 Episodes 下复用 |
| Actor/Critic、策略分布、GAE、PPO 更新 | 首期复用现有 PyTorch 实现 |
| 模型 XML、机器人元信息、控制和任务参数 | 共享源，后端分别加载或编译 |
| runtime、runner、simulator、数据采集 | 允许两套执行实现 |
| 高频插件、reset、接触处理 | 原生适配或转换，避免复制后独立演化任务规则 |
| 日志、dump 数据契约、主要分析工具 | 共享，必要时扩展后端元数据 |
| 实际轨迹捕获与仿真重演 | 后端适配 |

不强制“所有实验永远只能有一个文件”。理想情况直接复用实验类；必要时允许生成薄适配类或执行侧代码，但不能把整套 PPO、指标和任务参数复制成另一个独立维护的实验家族。

### 4.3 配置与能力解析

现有 blueprint 引用具体 Python 类，不会自动成为设备端实现。需要显式映射或任务解析层：

- 参数和任务组合保留权威来源；
- 已支持的 simulator/plugin 映射到对应后端实现；
- 任意 Python callable、私有运行时访问等能力必须单独审计；
- 未支持配置启动时失败，不能静默忽略；
- 同时记录逻辑任务配置和实际解析后的执行配置，避免相同 blueprint hash 掩盖后端差异。

首期可以只映射目标实验所需能力，不提前建设任意插件的自动编译器。

## 5. 加速路径原生插件体系

原生体系应围绕批量设备执行设计，而非模仿 Python 对象接口的外形。建议采用：

- 显式插件状态输入输出，隔离静态配置与动态数组；
- 固定形状或有明确容量的数组，使用 mask 表达有效元素；
- 明确的随机状态和拆分规则，避免环境间意外共享随机流；
- 明确的 episode/action/physics 生命周期与执行顺序；
- 可编译的高频逻辑，宿主侧承担低频编排和 I/O；
- 区分只读 observer 与允许修改物理状态的插件，保留权限边界；
- 对部分 reset、插件状态清理和终止后的有效性作明确约定。

逐物理步 hook 必须在对应物理步执行。事后 history 能补充观察，不能替代中途施力、状态投影和终止。

“不暴露所有 JAX 细节”是合理的易用性目标；“所有接口必须返回 NumPy、每次读写都经过 CPU”不作为最终高速架构的硬约束。

## 6. 三种兼容能力

| 能力 | 用途 | 边界 |
|---|---|---|
| 执行兼容 | 通过宿主侧适配器调用旧插件 | 适合低频功能、调试和过渡，不保证快 |
| 转换兼容 | 把旧插件转换为加速路径原生插件 | 依赖转换规则、模板和支持范围 |
| 验证兼容 | 相同输入下对照输出和状态变化 | 以主路径为参考，不依赖长期轨迹逐位一致 |

转换兼容和验证兼容比“任意旧插件原样执行”更重要。

插件应明确报告：原生支持、兼容执行但可能慢、需要转换、暂不支持。不得静默回退后仍把性能结果标为完整设备热路径。

纯数值 observer/reward 最适合提取共享函数；动态接触列表、Python set、数据依赖分支和对象原地修改通常需要改写。目标是共享公式、阈值和规则，不是强求每一行执行代码只出现一次。

## 7. 面向低成本 AI Agent 的转换机制

### 7.1 原则

将 Agent 的任务从“自由重写系统”约束为“按模板转换有限逻辑，调用现成工具获得验证证据”。可靠性主要来自清晰边界和独立验证，不来自提示词长度。

长期应提供四类资产：

1. **转换规范**：允许改动的范围、必须复用的接口、状态与 batch 转换、随机数、hook、接触、终止和错误处理规则。
2. **已验证模板**：无状态 observer、有状态 reward、reset、逐物理步施力、per-agent termination 和完整实验案例。
3. **验证工具**：标准化输入、差异报告、失败样例与重放入口。
4. **来源与失效检查**：主路径、模型、参数、后端和验证版本关联；源变化后提示重新验证。

这些是后续需要建设的交付物，不代表目前已有对应命令。

### 7.2 建议转换流程

1. 读取冻结的主路径实验、完整依赖链及支持矩阵。
2. 列出实际使用的 simulator、插件、observer、SamplingSpec 和特殊能力。
3. 复用已有原生实现；只对缺失的部分生成代码。
4. 从受验证模板转换，尽量提取或引用共享数值逻辑。
5. 运行接口检查、相同输入对照和状态生命周期测试。
6. 根据结构化失败报告局部修正，不自行改变任务或验收标准。
7. 通过后再做固定策略评估、PPO 接入和训练验证。
8. 记录转换来源、通过的验证层级、已知偏差和性能结果。

### 7.3 Agent 禁止事项与升级条件

禁止为跑通而填零、漏字段、跳过插件、改奖励、改初始化分布、减少物理子步或接触范围；禁止放宽测试阈值使自身实现通过。

遇到以下情况必须报告并升级处理：

- 主路径语义不明确或文档与代码冲突；
- 使用不受支持的物理功能、任意宿主副作用或私有内部状态；
- 无法解释的观测、奖励、接触或状态转移差异；
- 需要修改模型、物理参数、任务规则或验收阈值；
- 正确性通过但性能不达标。

低成本 Agent 不必独立解决所有情况。可靠识别超出支持范围，本身就是成功行为。

### 7.4 验证不能由转换代码自证

主要判定标准和参考样例应由框架预先提供，CPU 主路径是参考实现。转换 Agent 可以增加失败用例，但不能同时自由修改实现和验收标准。

失败报告应至少包含：测试层级、源/目标版本、seed 或输入标识、episode/agent/frame、差异字段、参考值、实际值、容差和重放方式。不能只输出“训练 reward 差不多”。

## 8. 分层验证与成功状态

### 8.1 验证层级

| 层级 | 验证内容 | 不能替代什么 |
|---|---|---|
| V0 接口与能力 | schema、shape、字段、配置、支持矩阵 | 不能证明数值逻辑正确 |
| V1 相同输入逻辑 | 观测、奖励、插件状态转移、终止、时序 | 不能证明物理引擎推进一致 |
| V2 物理与生命周期 | 单步/短程动力学、reset、外力、缓存、环境隔离 | 不能证明长期学习效果 |
| V3 固定策略交叉评估 | 不同成熟度策略在 CPU/MJX 上的行为与指标 | 不能替代从头训练 |
| V4 训练对照 | 多 seed、同预算的质量、学习曲线和样本效率 | 不能证明实际加速 |
| V5 性能 | 冷启动、稳态、端到端吞吐与 time-to-quality | 不能牺牲前面的语义标准 |

V1 应把“相同状态特征上的逻辑”与 V2 的“后端产生了什么物理状态”分开，避免把物理差异和转换错误混在一起。接触/阶段阈值附近的样例单独统计；非边界样例不应以数值差异为由接受逻辑分歧。

V2 对物理对照应使用相同完整输入状态，注意 warm-start、控制、外力等隐含状态。仅将 CPU 已算出的 derived/contact 数据导入 MJX 后读取，不能证明 MJX 求解结果正确。

### 8.2 成功状态不能合并

每个转换产物分别标记：

- **转换完成**：执行实现及适配代码齐备。
- **语义验证通过**：支持范围内的接口、逻辑、物理和固定策略验证完成。
- **训练效果验证通过**：在冻结协议下达到质量和样本效率要求。
- **加速验证通过**：最终实现达到约定的端到端性能目标。

smoke test 只能证明基本链路运行，不能代表后面三个状态。优化、模型或版本变化后，应重新运行受影响层级的验证。

### 8.3 训练与性能协议

- 历史 Run 是效果目标；同版本冻结的 CPU Run 是公平 A/B 对照。两者可能不是同一份代码。
- 训练按实际 transitions 和更新预算对齐，不只比较 update 编号。
- 固定训练/评估设置，使用独立留出评估集，避免挑最有利 checkpoint。
- 至少区分曾经成功、结束时成功和持续稳定，不能只看单个峰值指标。
- 报告多 seed 波动与置信区间；同一物理场景内两个 agent 有相关性，不能视为完全独立样本。
- 正式数值容差和非劣效门槛在基线阶段冻结，不能在看到 MJX 结果后临时放宽。
- 性能对照使用合理配置的 CPU 并行路径，不以单环境 CPU 作为唯一比较对象。
- 计入 reset、观测奖励、策略推理、传输、数据整理、PPO 和固定评估/导出成本。
- 编译与冷启动、稳态分别报告；GPU 计时必须正确同步，不能只计异步提交时间。
- 固定并报告 CPU/GPU 资源预算，避免靠额外资源冒充实现加速。
- 执行 batch size 与 PPO 更新样本量分开管理；后者变化属于训练配置变化。

## 9. Debug、记录与重演

### 9.1 优先保留数据分析资产

通过统一 Episode、trajectory、buffer、update 和日志/dump 契约，复用现有指标查询、曲线、GAE/advantage、confidence、KL、minibatch timeline、梯度诊断和 viewer。

首期保留 PyTorch trainer，避免为了物理加速同时重写优化器和训练侧 Debug。

基于已保存 observations 对不同代策略求动作的 policy-drift 分析，不需要重新推进环境，应继续复用。

### 9.2 区分三种不同能力

1. **查看真实采样轨迹**：按需捕获选定 episodes 的物理状态和 observer 输出，渲染保存的状态。
2. **同后端重新运行**：调用相应 CPU/MJX runner，保存必要的初态、随机状态、版本和执行配置；仅靠 seed 不承诺精确重演。
3. **跨后端评估策略**：把 MJX 策略放回 CPU 环境评估，这是能力验证，不是原 MJX 训练轨迹的重演。

CPU MuJoCo renderer 可以用于渲染保存的 MJX 姿态，但显示的接触力、奖励等数值应来自原始记录，不能用 CPU 重算值冒充原采样数据。

物理轨迹捕获、重演入口和 provenance 需要后端适配；不承诺全部 Debug 代码零修改。捕获可按 dump 请求进行，不要求日常训练保存所有环境的完整物理历史。

## 10. 首个纵向验收：standup_floor04

目标实验：`baseline/experiments_ppo/exp_standup_floor04.py`。它继承 Standup，使用固定 200 action steps、双机器人采样、单奖励通道，适合作为首个完整案例。

当前任务语义的关键点，实施前仍需按冻结版本核对：

- 每 action step 25 个物理子步，物理 dt 为 0.002 秒。
- 每 update 512 场双机器人 episodes，对应 1024 条 trajectories、204,800 agent transitions；不能改成互不相关的单机器人环境。
- 奖励为 `0.01 × potential`，gamma=0.99、GAE lambda=0.95。
- 200 步 timeout 保留 bootstrap；最后观测必须在 reset 前获取。
- 使用截断正态策略，不能替换为普通高斯采样后裁剪；行为分布必须与 PPO 重算 log-prob 一致。
- uncertainty floor=0.4、coef=1.0 是现有训练损失机制，不是把策略标准差直接截到 0.4。
- RandomFallenStatePlugin 用内部仿真生成初态；当前双方为目标时，任一目标高度低于阈值即停止，不额外等待双方静止或清零速度。
- 四阶段奖励依赖接触力、承重比例和其他接触 body 的去重数量，不能用接触点数量替代。
- `eval.success` 是曾达到 potential≥0.9；`exp.online_success` 是随机训练采样的末帧达到阈值，两者不是同一指标。

初期可用 CPU 生成的相同初态隔离后续任务差异，再迁移设备端 reset。固定状态集是诊断工具，不能悄悄替代正式训练的随机初态分布。

已查到的历史参考候选：

| Run | updates | 首次 eval.success≥0.9 | 最后 10 次 eval.success 均值 | 最后 10 次 final_pot 均值 |
|---|---:|---:|---:|---:|
| train_standup_floor04_ppo_20260916_103522 | 1500 | 385 | 0.9984 | 0.9986 |
| train_standup_floor04_ppo_20260920_164819 | 1500 | 380 | 0.9976 | 0.9981 |

这些是 2026-09-29 从历史日志提取的结果，不是重新评估结论；两个 Run 都是 seed 42，不能作为多 seed 证据。主参考 Run 尚待确认，其 snapshot 与当前 PPO/策略实现已有变化。

之前讨论的质量差距不超过 2 个百分点、final_pot 差距不超过 0.02、样本量不超过 CPU 的 1.25 倍，以及端到端 2× 加速，均只是验收协议的候选值，不是已批准的硬门槛或性能承诺。应在基线阶段依据统计波动、资源预算和实际用途冻结。

## 11. 分阶段实施大纲

| 阶段 | 主要产物 | 放行条件 |
|---|---|---|
| P0 基线与协议 | 冻结版本、参考 Run、能力清单、测试输入、评估与性能协议 | CPU 参考可复评，语义与验收口径明确 |
| P1 仿真可行性 | 完整真实模型上的 MJX 后端原型、误差与吞吐报告 | 功能可支持、误差可解释、存在值得集成的性能空间 |
| P2 最小原生任务 | standup 所需 runtime、插件、观测、奖励和 reset | V0–V3 对照通过，未通过项显式列出 |
| P3 Rollout 接入 | collector 选择、批量策略采样、Episode 导出和 Debug 适配 | 数据时序、样本数、log-prob、bootstrap、smoke/resume 正确 |
| P4 训练验收 | 同协议 CPU/MJX 训练与交叉评估 | V4 质量与样本效率门槛通过 |
| P5 性能优化 | 基于 profiling 的设备常驻、传输和编排优化 | 最终版本重新通过效果验证并满足 V5 |
| P6 AI 迁移产品化与推广 | 转换指南、模板、工具、支持矩阵和失效检查 | 低成本 Agent 可按流程迁移新实验，失败可诊断、可升级 |

指南、模板和测试工具从 P0/P2 起随实现积累，不等到 P6 才开始。P6 的重点是让其他 Agent 独立使用，验证流程本身的可操作性。

P4+P5 是首个总体交付：原实验的能力接近，并且训练确实更快。P6 再用其他实验检验复用与转换效率，不将“能跑 standup”等同于“支持全部 framework”。

性能应早期检查。如果完整模型或完整 collector 没有加速空间，先解决后端和执行方式，不继续堆通用框架。MJX-JAX 与 MJX-Warp 都可作为候选，但需分别验证功能、接触访问、精度、依赖和性能，固定版本后再实施。

## 12. 当前状态与首轮注意事项

现有 batchframework 是探索性原型，不是已经认证的对等实现。当前已观察到：

- CPU simulator 加载 `battle_circular_v2.xml`，MJX 原型加载 `battle_v1.xml`。
- 两边虽然都输出 96 维观测，具体字段、坐标处理和速度变换已发生漂移。
- MJX 原型的状态写入刷新、外力持续语义和部分 derived 字段尚未与 CPU 对齐。
- 原型采用 NumPy 宿主接口和部分 Python 接触循环，尚未证明端到端加速。
- 现有 MJX 测试未覆盖完整目标任务与训练数据链路，不能以旧测试注释代替验证。

这些是启动阶段的审计输入，不是沿用原型设计的约束。本轮总纲形成时，仅完成源码、历史日志及官方资料调研，未运行新的 GPU 数值对照、吞吐基准或训练。

## 13. 实施入口与参考

相对项目根目录：

- `envs/framework/backend.py`、`envs/framework/env_runtime.py`：主路径 simulator 契约与执行时序。
- `envs/humanoid21/simulator.py`、`envs/humanoid21/meta.py`：当前仿真、观测、控制和模型元信息。
- `envs/humanoid21/disturbance_plugins.py`：包括随机摔倒初始化在内的主路径插件。
- `baseline/humanoid21/rewards/standing_balance_4stage.py`：首个任务的 potential 定义。
- `baseline/humanoid21/blueprints/standup_4stage_dense_v2_env.yaml`：首个任务组合。
- `baseline/framework/rollout/job.py`、`episode.py`：Rollout 数据边界。
- `baseline/framework/ppo/loop.py`、`trainer.py`：训练接入与 PPO 数据消费。
- `baseline/framework/ppo/dumpkit/CONTEXT.md`：现有 Debug 能力地图。
- `tests/test_mjx_validation.py`：已有物理对照测试的起点。

外部能力说明以所选版本为准：

- [MuJoCo MJX 官方文档](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

未来在本文之上细化接口设计、转换规范和验证工具；不要将本文中的候选设计或待验证假设当作已经实现的能力。
