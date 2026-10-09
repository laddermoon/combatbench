# BATCHFRAMEWORK_AUDIT — 批量框架系统审计（2026-10-01）

> 类型：记录

**范围**：`envs/batchframework/` 全量，以 `envs/framework/`（CPU 生产框架）为参照。
**四个维度**：A1 功能一致性与成熟度差距；A2 设计文档完善性/自洽性；
A3 AI 迁移指导齐全性与流程标准化；A4 旧插件封装/转换工具。
**方法**：逐 API/逐 hook 对照源码与文档；结论标注证据文件与行号。

---

## A1 功能一致性审计（CPU framework ↔ batchframework）

### A1.1 Hook 生命周期对照

| Hook | CPU `BasePlugin` | `BaseDevicePlugin` | 判定 |
|---|---|---|---|
| on_attach / on_detach | ✓ | ✓ | 平齐 |
| on_pre_episode | ✓ | ✓ | 平齐（设备侧 reset 语义见 RngView/reset 编排） |
| **on_envs_reset** | — | ✓ | 设备超集（部分 reset 波内补齐） |
| on_pre_action_step | ✓ | ✓ | 平齐 |
| **on_pre/post_batch_step** | — | ✓ | 设备超集（物理块边界） |
| on_pre_phy_step | ✓ | ✓ | 平齐语义，设备侧 eager-only（图化路径不接） |
| on_post_phy_step | ✓ | ✓ | 同上 |
| on_post_action_step | ✓ | ✓ | 平齐（每进入 step 恰一次，终止帧契约已对齐） |
| on_post_episode | ✓ | ✓ | 平齐 |
| **set_episode_seeds** | `set_episode_seed`（标量） | ✓（(B,) 张量） | 批量化平齐 |
| **export_episode_metrics** | — | ✓ | 设备超集（声明式指标导出） |
| **declare_state / declared_reads/writes / per_hook_mutator / rng_salt** | — | ✓ | 设备超集（契约声明层，CPU 无对应物） |

**结论**：hook 覆盖是**严格超集**。设备侧独有的批量语义（部分 reset、块边界、
种子广播、声明式契约）都是 CPU 不需要或没有的。无缺失 hook。

### A1.2 ctx 数据面对照（差距主战场）

| CPU `SimContext` 能力 | 设备侧对应 | 状态 |
|---|---|---|
| `accessor.get_core_state/get_derived_state` | `ctx.sim.*` 类型化张量视图（qpos/xpos/contacts_flat/...） | **形态不同**——读能力覆盖，API 形状不可直移（G3） |
| `accessor.get_static_data` | binding `task_tables` | 平齐（声明即冻结） |
| `accessor.get_sensor_data` | — | **缺口 G3**：设备侧无 sensor 抽象（mjw 模型未用 sensor 通道） |
| `accessor.get_broadcastview_image` | — | **缺口 G5**：无渲染（设计内；record→CPU replay 替代） |
| `mutator.set_action` | `mutator.set_action` | 平齐 |
| `mutator.apply_external_force` | `mutator.add_ext_force` + `upload_force_schedule` | **设备超集**（schedule 是图化前置条件） |
| `mutator.set_core_state` | `mutator.set_integration_rows` | 平齐（批量化） |
| `mutator.reset` | `mutator.reset_rows` | 设备超集（部分行） |
| **`ctx.metrics` 共享黑板** | — | **缺口 G1（高）**：无跨单元数据通道 |
| **`ctx.events` 事件日志** | — | **缺口 G2（高）**：无事件机制（且 CPU 刚升级 EventJournal） |
| `ctx.episode_options` | — | **缺口 G4**：options 只到 `sim.reset`，hook 内不可见 |
| `ctx.episode_step / physics_step` | `ep.episode_steps / physics_steps` (B,) | 平齐（张量化） |
| `ctx.request_termination(reason, agent)` | `ctx.request_termination(env_ids, reason, agents)` | **设备超集**（批量 + reason code + 历史归档） |
| `ctx.agent_terminated` | `ep.agent_done` | 平齐 |
| per-plugin RNG | `ctx.rng`（RngView, salt 分配, job-keyed splitmix） | **设备更严**（CPU 是 ad-hoc `np.random`，见 SEED.md §3） |
| per-plugin 私有状态 | `ctx.pstate`（declare_state 声明式） | 平齐偏优（CPU 是插件实例属性，partial reset 语义弱） |

### A1.3 结构性差距详述

**G1 — 无共享 `metrics` 黑板（高优先级）**

CPU 侧 `ctx.metrics` 是插件↔observer 的实时数据通道：
`CombatScoringPlugin` 写 `damage_taken_*`，`rewards/damage.py` 等
rewarder 同帧读取。活跃消费者：`envs/humanoid21/plugins.py`、
`baseline/humanoid21/rewards/damage.py`、`fight/damage.py`、
`random_fall.py`、`relative_impulse_plugin.py`、`standing_triggered_force.py`。

设备侧 `DeviceCtx` 只有 per-plugin `pstate`（私有）与 episode 记账字段
（term_history 等，语义固定）——**没有通用键值黑板**。
后果：任何依赖 plugin→observer 数据流的实验**结构性不可迁移**。
已迁移的 standup/basic_balance 恰好不走此通道，所以未暴露。

候选补法（记录备 E9 决策）：
- `state.shared` 声明式共享池（declare_state 的跨单元变体，读声明
  纳入 declared_reads 审计）；
- 或定义 metrics 为 episode 记账面的扩展（(B,) 标量张量池 +
  export_episode_metrics 同通道导出）。

**G2 — 无 `events` 通道（高优先级，且 CPU 契约刚演进）**

CPU `ctx.events` 刚被升级为 append-only `EventJournal`（epoch 游标差分、
框架统一 reset——用户 2026-10-01 在途工作）。设备侧**完全没有事件
机制**：plugin 无法发射"瞬时事件"（命中、犯规、状态切换），observer
也无法订阅。

现有近似物：`term_history`（终止 reason 归档）是事件的一个特例；
设备若要通用事件，需要 padded 事件池（(B, K, payload) + count + epoch），
K 上界声明制——与 term_history 同构，机制可复用。

**G3 — accessor 形态差异 + sensor 抽象缺失（中）**

CPU 插件读 `ctx.accessor.get_derived_state()["contacts"]` 这类 dict；
设备插件读 `ctx.sim.contacts_flat` 等类型化命名空间。**读能力基本覆盖，
但代码不可直移**——这决定了 A4 的"包装"只能走 per-env 回放视图，
不能原地运行。`get_sensor_data` 无对应（mjw sensor 通道未接线）。

**G4 — `ctx.episode_options` hook 内不可见（中低）**

CPU 插件在任意 hook 读 `ctx.episode_options`（课程扰动参数等）。
设备侧 options 经白名单广播进 `sim.reset(options)` 后**对插件不可见**。
`binding_registry.episode_options_keys` 已管入口校验，但 ctx 无字段。
补法：`reset_env_ids` 场景随 ctx 发布 options 张量切片即可，量小。

**G5 — 无渲染/录制管线（中低，设计内）**

无 broadcastview/VideoRecorder 对应物。替代路径已存在：debug_capture
npz 快照 → CPU replay → CPU VideoRecorder。文档已声明，判定"设计内缺口"。

**G6 — 采样 spec 子集（中低，显式拒绝已注册）**

`required_ctx_fields`：callable explore_factor / reference_action /
delta_factor 显式拒绝；executor 仅 `TruncatedNormalExecutor` 一族
（register_executor 扩展点已开）。注册表如实标 UNSUPPORTED——
非暗坑，但限制策略多样性实验迁移。

**G7 — 无设备侧评测 runner（低）**

CPU 有 RoundRunner/MatchRunner/EpisodeRunner CLI 评测面。设备侧只有
训练 collect + `probe_standup_xeval` 科研探针——无通用"两 policy
对打评分"的设备 runner。可走 device collect → Episode → CPU 评分，
语义等价但多一次导出。

### A1.4 成熟度对比

| 维度 | CPU framework | batchframework | 评估 |
|---|---|---|---|
| 生产使用 | 全部训练实验的主路径 | 2 个实验已迁移（E5），PPO smoke 通过 | 设备侧生产证据尚浅（正确，属早期） |
| 测试规模 | `envs/framework/tests/` 25 文件 / 232 项 + humanoid21 47 | `tests/` 中设备相关 ~12 文件（含 wave/lifecycle/multi/migration/validation） | 设备侧契约测试密度不差，但 CPU 侧有多年边界用例积累 |
| 契约测试 | 语义测试为主 | golden/生命周期/RNG/故障注入/容量矩阵 | 设备侧契约**更硬**（E2/E6 建的） |
| 规格文档 | DESIGN/RESET/SEED/README/CONTEXT 五件 | 无 DESIGN/SEED/RESET 对应物（见 A2） | 设备侧文档**薄一档** |
| 边界用例 | 终止/超时/重置/seed 链路多年打磨 | 终止帧契约刚对齐（2026-10），partial reset 是新语义 | 设备侧新语义更先进但历练少 |
| 性能 | ~472K env-sub/s（CPU 池） | 582K/单卡，2.14M/8卡 | 设备侧已反超 |

**总评**：设备框架在**契约严格性**（声明式 reads/writes、RNG 纪律、
终止语义、provenance）上已超过 CPU 参照；在**插件生态数据面**
（metrics/events/options 通道）和**文档规格件**上有实质缺口。
它不是"还没追平"——是追平的部分和超车的部分不均匀。

---

## A2 设计文档审计

### A2.1 文档清单对照

| CPU `envs/framework/` | 设备侧对应 | 状态 |
|---|---|---|
| `DESIGN.md`（分层架构、接口表、设计模式） | — | **缺失**：无单一架构规格件；设计散在 E1–E8 计划 + 模块 docstring |
| `README.md`（用法文档） | `README.md`（E8 新写：入口+边界+扩展） | ✓ 已有 |
| `RESET.md`（reset 传导契约） | — | **缺失**：设备 reset 语义（波/部分 reset/on_envs_reset/种子广播）只在 docstring + E2 计划 |
| `SEED.md`（seed 派生树） | — | **缺失**：设备 RNG 设计（job-keyed splitmix、salt、RngView）其实**更好**，但只有 `device_state.py` docstring 记述 |
| `CONTEXT.md`（AI memo） | — | **缺失**（mono 约定件） |
| —（CPU 无） | `PUBLIC_INTERFACE.md`、`E8_SUPPORT_MATRIX.md`、`LIFECYCLE_TRACE.md`（CPU 侧语义源）、`TERMINAL_FRAME_PLAN.md` | 设备独有，质量好 |

### A2.2 自洽性抽查

抽查 `PUBLIC_INTERFACE.md` / `README.md` / `E8_SUPPORT_MATRIX.md` /
模块 docstring 之间的一致性（E8 刚做过收口，基线高）：

- **一致**：host_compat 休眠状态（PUBLIC_INTERFACE §4 ↔ E8 矩阵 ↔
  host_compat 文件头三处表述一致）；图化资格（README 边界 ↔ E7_RESULTS ↔
  device_examples 警告）；休眠模块表（batch_plugin/batch_context/mjx_*）。
- **注意**：`bootstrip.md`/`discuss.md`/`M*.md` 是历史快照——文件头
  大多有时期标记，读者须自觉区分；建议在 README 链接区加"历史档案"
  分组标签。
- **发现一处漂移**：E8 README"运行验证"列的测试文件名与当前 `tests/`
  实际文件存在出入（早期会话提到的 `test_device_api/contract/export/
  debug_*` 等文件名在当前树中不存在——已合并/更名）。文档引用测试
  名时应以 `ls tests/` 现状为准。**建议审计收口时统一核对一遍**。

### A2.3 结论

**不缺执行记录，缺规格件**。E1–E8 + M0–M6 共 20+ 份过程文档证明
"怎么走过来的"；但没有一份"现在是什么"的架构规格——新会话/新 AI
进入项目要从十几份 PLAN 里逆向提取现状。建议补：

1. `DESIGN.md`（架构分层 + 核心对象图 + 数据平面三命名空间 + 生命周期）
2. `SEED.md` + `RESET.md` 对应件（可合并为一篇 `SEMANTICS.md`）
3. `CONTEXT.md`（mono 约定 AI memo，<150 行）

---

## A3 AI 迁移指导审计

**问题**：一个 AI agent 拿着 CPU 实验 X，能不能靠现有文档把它迁到
batchframework？

### A3.1 现有机制（齐）

- `capability_registry`：unit → 支持状态查询，`check_support(units)`
  批量判定；14 条注册项 + 显式 UNSUPPORTED/PENDING。
- `migration_audit.py` + `find_manifest_for`：迁移记分卡 + 证据链。
- 验证阶梯已定义：`unit_replay`（CPU fixture 回放逐帧等价）→
  `e2e_collect`（device collect 结构校验）→ `train_smoke`（PPO 消费）。
- `device_examples.py`：三个可抄模板（balance/dual-imbalance/cross-support）
  + SubstepProbePlugin。
- PUBLIC_INTERFACE §6.3 有装配片段；README 有扩展指南骨架。

### A3.2 缺失（关键）

**没有一份 MIGRATION_GUIDE**。迁移所需的全部知识存在，但分散在
≥6 个文件里且顺序不明。AI 迁移者需要自己悟出这个流程：

```
1. check_support() 判定可行性与缺口          ← 文档在 registry docstring
2. 声明 device blueprint（binding 注册）     ← 散在 E5_PLAN/binding_registry
3. 逐个 CPU 插件 → BaseDevicePlugin 改写     ← 只有 example 没有 rules
   - dict accessor → 类型化 sim 命名空间
   - ctx.metrics/events → 无通道（G1/G2 阻塞点！）
   - per-env 对象 → (B,) 张量 + env_ids
   - np.random → ctx.rng / declare_state
   - 终止语义 → request_termination(env_ids, reason, agents)
4. observer → output_schema 声明 + get_output
5. episode_options → binding whitelist keys
6. unit_replay → e2e_collect → train_smoke 验证阶梯
7. 证据写 manifest（schema：level/passed/input_hash/detail/pass_ts）← 我刚踩过这个坑！
```

**标准化程度评估**：流程本身**可标准化**（阶梯和判定都已机制化），
缺的是 (a) 单一入口文档，(b) 每步的输入/产出/验证命令模板，
(c) 陷阱清单——特别是：
- `torch.nonzero`/动态形状进不了图路径；
- host sync 触发点（bool/.item()/打印）；
- metrics/events 通道缺失（迁移前先判定实验是否依赖）；
- manifest 证据 schema 必填字段（本次 E8 实际踩坑）。

**结论**：机制 A-，指导 D+。补 `MIGRATION_GUIDE.md` 即可闭合——
这是本轮审计最高 ROI 的文档件。

---

## A4 旧插件封装/转换工具审计

**问题**：CPU `BasePlugin`/`BaseObserverPlugin` 写的插件，有没有
到 batchframework 的封装或转换路径？

### A4.1 现状

`host_compat.py` 提供**两种适配器，机制完整**：

| 适配器 | 包装对象 | plane | 语义 | 状态 |
|---|---|---|---|---|
| `HostBatchCompatAdapter` | 旧 numpy `BaseBatchPlugin`（batch_plugin.py 契约原型） | HOST | per-hook 缓存快照 + SyncStats | 休眠，无用户无测试 |
| `LegacyPluginAdapter` | CPU 单 env `BasePlugin` | HOST_SLOW | 逐 env SimContext 视图循环，语义精确 | 休眠，无用户无测试 |
| `LegacyObserverAdapter` | CPU `BaseObserverPlugin` | HOST_SLOW | 同上 | 休眠，无用户无测试 |

- `LegacyPluginAdapter` 默认被 runtime 拒绝（`allow_host_slow=True` 放行）；
  **覆写子步 hook 的旧插件 attach 时直接拒绝**（诚实，不静默降级）。
- `BatchRuntime`/`Binding` 支持声明 HOST plane——机制层面接得进去。

### A4.2 差距

1. **零测试**：三个适配器均无测试——"存在"≠"可用"，是否 bit 正确未验证。
2. **零生产用户**：registry 14 条无一条走适配器路径。
3. **无转换工具**：没有 codegen/scaffolder 把 `BasePlugin` 骨架翻成
   `BaseDevicePlugin`——因为 G1/G2/G3 表明这不是机械翻译，是重写。
   适配器解决的是"先跑起来对照"，不是"迁移快车道"。
4. **正确用法未文档化**：何时该用 LegacyPluginAdapter（调试/对照
   golden）vs 直接重写——没有指导。

### A4.3 结论

**封装层存在且设计诚实（语义精确、显式 opt-in、拒绝不可忠实场景），
但处于休眠+未验证态**。两条可行路：

- **A：扶正对照路径**——给 `LegacyPluginAdapter` 补契约测试
  （同一插件 CPU 跑 vs 经适配器跑，断言逐帧一致），把它定义成
  "迁移时生成 golden 参考的工具"，这正是它语义精确的用途；
- **B：维持休眠 + 文档化决策**——明确"迁移=重写，适配器仅供调试
  对照"，写进 MIGRATION_GUIDE。

两者不互斥；B 是底线，A 看迁移工作量是否真需要对照。

**处置记录（2026-10，E9 收口执行）**：采 B 路线——正式休眠。理由：
迁移正道（MIGRATION_GUIDE 的 Layer-A 同输入对照）不依赖本模块，
GPU-only 依赖使扶正成本与当前无用户现状不匹配。同时做了一处
**诚实修复**：`_drain_env_ctx` 的 `events.clear()` 在 EventJournal
append-only 契约下已实际损坏（AttributeError），改为 (epoch, len)
游标差分消费——保持"休眠但代码不失真"的底线。

---

## 综合差距清单（按优先级）

| # | 差距 | 严重度 | 维度 | 建议处置 |
|---|---|---|---|---|
| G1 | 无共享 metrics 黑板 | ~~高~~ **已闭合（E9）** | A1 | `ctx.metrics` 共享张量池落地（BLACKBOARD_DESIGN §1 + test_blackboard） |
| G2 | 无 events 通道 | ~~高~~ **已闭合（E9）** | A1 | `ctx.events` padded journal 落地（BLACKBOARD_DESIGN §2） |
| — | 无 MIGRATION_GUIDE | ~~高~~ **已闭合（E9）** | A3 | MIGRATION_GUIDE.md 已交付 |
| G3 | accessor 形态差异 + 无 sensor 抽象 | 中 | A1 | 已文档化映射表（MIGRATION_GUIDE §1）；sensor 抽象按需 |
| G4 | episode_options hook 内不可见 | ~~中低~~ **已闭合（E9）** | A1 | `ctx.episode_options` 行快照已发布 |
| G5 | 无渲染管线 | 中低 | A1 | 设计内，文档已声明 |
| G6 | 采样 spec 子集 | 中低 | A1 | 显式拒绝已注册，按需扩展 executor |
| G7 | 无设备评测 runner | 低 | A1 | 按需；CPU 评分路径可用 |
| — | DESIGN/SEED/RESET/CONTEXT 规格件缺失 | 中 | A2 | 补 BATCH_DESIGN + SEMANTICS + CONTEXT |
| — | host_compat 适配器零测试 | 中 | A4 | 扶正为 golden 对照工具（补测试）或文档化休眠 |
| — | README 测试引用与 tests/ 现状漂移 | 低 | A2 | 统一核对文档引用 |

## 审计后建议的工作序列（供决策）

1. **MIGRATION_GUIDE.md**——含 ctx 映射表 + 陷阱清单 + 验证阶梯模板
   （A3 缺口 + A1/G3/G4 文档化 + A4 决策记录，一份文档吃三个维度）；
2. **规格件**——BATCH_DESIGN.md + SEMANTICS.md（seed/reset/生命周期）+ CONTEXT.md；
3. **G1/G2 设计**——共享黑板与事件池的设备形态设计（这是真编码工作，
   阻塞 scoring/damage 类实验迁移）；
4. **host_compat 处置**——补契约测试扶正或正式文档化休眠。
