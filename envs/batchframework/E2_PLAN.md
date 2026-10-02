# E2 计划：生命周期、插件、随机与终止契约

**状态**：提案待审（2026-02-20）
**上游**：[E0 契约提案 discuss.md](./discuss.md) D5/D6/D7/D8/D9 | [E1_PLAN.md](./E1_PLAN.md)（已完成）
**ROADMAP 对应**：E2 —— "补全生命周期、插件与随机契约"

## 背景与定位

E1 完成后，代码依赖方向已干净（backend ← runtime ← collector），但**语义层仍是 E0 之前的原型状态**：

| 契约要求（discuss.md） | 当前实现 | 差距 |
|---|---|---|
| D7.1 四种 mask 分离（`slot_valid`/`world_running`/`agent_done`/`policy_eval_mask`） | 单一 `active_mask` | 语义压缩，无法表达"行仍跑物理但 agent 已终止" |
| D7.1 `ENDED` 封存至显式释放；`step` **不 auto-reset** | `step()` 末尾内嵌 `dev_reset_rows` | ENDED 行靠即时 reset 回收，无封存态；无 `FAILED` |
| D7.3 终止历史（per-agent、per-reason、首次时钟、确定性顺序） | `agent_term_reason (B,2)` 单值覆盖写 | 多原因/重复终止/顺序全部丢失 |
| D7.4 计数器分离（`action_call_index`/`episode_step`/`physics_step`/`substep_index`/`record_frame_index`） | 仅 `episode_steps` + `rng.step_counter` | 子步内终止的边界样例无法表达 |
| D6.2 hook 含 `on_pre_phy_step`/`on_post_phy_step` | 仅 pre/post_batch_step（action 粒度） | 无原生逐子步反馈位 |
| D6.2 unit 声明读写/输出 schema/随机域，装配期校验 | 仅 `plane`/`priority`/`require_mutator` | 无声明式能力面，无装配校验 |
| D6.3 写能力按字段/阶段授权，hook 结束失效 | `DeviceMutator` 按 hook 授予/撤销 ✓，但授权粒度是"全 mutator" | 字段级授权缺失（H4 debug 校验未做） |
| D6.3 observer 不改物理/不提终止 | 约定俗成，无机制 | 无隔离保证 |
| D9.1 runtime 分配随机域/salt | `seed_offsets`+`step_counter` 裸字段，插件各自哈希 | 无 `RngView`，分片重排稳定性未验证（E4 前置） |
| D13 `abandon`、FAILED 传播 | 无 | 无 |

## 关键设计判断（本计划的前提）

**J1 — sealed-ENDED 是"更便宜且更正确"，不是负担。**
源码核查（`device_rollouter.py:440-441`）：`t_use = term_step` 使 terminated 行 reset 后产出的数据**在 assemble 时被整体截断丢弃**。即当前 mid-wave auto-reset 是纯粹的算力浪费 + njmax 占用，Episode 输出上没有任何收益。改为"terminated 行封存至波末"后：
- Episode 输出**严格等价**（截断点之前的数据完全一致）；
- ENDED 行经 write-back 冻结（W0 已验证原语），物理状态不再漂移 → 单行失稳不再能撑爆 `nconmax/njmax` 杀掉整个 wave（u55 类崩溃的结构性解法）；
- `step()` 语义变干净：不带隐藏副作用，collector 显式决定何时 reset。

**J2 — 终止走"pending 请求 → phase 屏障消费"，不直写 episode 标志。**
当前 `ctx.request_termination` 直接写 `agent_terminated`。契约要求：同一 phase 内提出终止后**后续单元仍执行**，在 phase 屏障统一按确定性顺序消费（对齐 CPU `_PluginManager.invoke` 之后检查的边界），且要支撑多原因历史。改为 pending-request 模型后这两点自然成立。

**J3 — `INTRA_ACTION_END` 表达但不生产启用。**
契约 D7.4：第一版正式支持 `POST_ACTION_END`；子步内终止在数据模型中可表达（`substep_index`/`physics_steps`），但导出/训练兼容验证（H5）完成前标 `pending`，collector 默认 profile 不启用。防止"runtime 有 mask 就宣布支持"。

**J4 — 子步级 hook 先建槽位与约束，首个原生消费者就是已有能力。**
`upload_force_schedule`（外力表，已存在）就是最自然的 per-substep 消费者：把"上传一张 (B,S,nbody,6) 表"升级为"post_phy hook 经声明缓冲逐子步读"。同时为 DEVICE 插件开放真 per-substep 回调槽位（kernel 级插件未来用），HOST 插件被约束在 action-step 粒度，HOST_SLOW 在 production profile 下拒绝。

## 工作包拆分

```
W0 CPU 语义核对 → W1 数据模型（mask/计数器/终止历史）→ W2 sealed-ENDED+显式reset
→ W3 子步 hook 与终止屏障 → W4 声明式插件契约 → W5 随机服务 → W6 契约测试矩阵+回归
```

### E2-W0：CPU 生命周期语义核对（纯调研，产出文档）✅ 已完成

产出：**[LIFECYCLE_TRACE.md](./LIFECYCLE_TRACE.md)**——CPU 权威时序（5 终止屏障）、proposals append-only + recorder 帧扫描去重的真实两段式模型、子步内终止的精确后果表、reset 链、observer 刷新时机、seed 派生、post_termination_action。

**核对结论：D7 契约无需修订**——草案与实测逐项吻合（含"observer 输出可能是 post_episode 刷新值"的预判）。两条实现级精度要求转入 W1/W2：

1. `request_termination` 必须**立即**写 `agent_done`（CPU 中同 phase 后续插件可见 `agent_terminated`），屏障只归档历史与判定 env 结束，不得延迟提出效果；
2. 终止帧/records 的扫描必须在 `on_post_episode` schedule **之后**（CPU 序：插件 post_episode → recorder 帧 → recorder post_episode）——当前 `_WaveRecorder` 次序相反，现无实质差异（standup rewarder post_episode 为 no-op）但属结构性漂移隐患；
3. 新发现设备 bug 级差距：`_WaveRecorder._seen_term` 会丢弃"agent 已终止后新提出的不同 reason"，CPU 语义是每 reason 首次都记。

### E2-W1：EpisodeNamespace 数据模型重构

`device_state.py` + `device_runtime.py`：

- `active_mask` 拆分为：`slot_valid`（静态，非 padding）、`world_running`（仍在推进物理）、`agent_done (B,2)`（训练终止态，=现 `agent_terminated` 重命名对齐）、`policy_eval_mask`（本步是否调用策略——ended/sealed 行=False）。
- 新增计数器：`physics_steps`（累计物理子步数）、`substep_index`（当前 action step 内子步位置）、`action_call_index`（episode 内第几次 step 调用，区别于完成的 `episode_steps`）。`record_frame_index` 归 collector/recorder 侧所有。
- `agent_term_reason` 升级为**终止历史**：定长 `term_history (B, 2, K, 2)` int32（K≈8，[code, logical_step] 对）+ `term_history_len (B,2)`；首次有效原因语义由 collector 导出时取 `history[...,0]`，同 code 去重、异 code 保留、按提出顺序排列。
- `FAILED` 行状态：`world_failed` mask + `fail_reason` code；FAILED ≠ agent 失败，是执行错误（njmax 溢出/数值非有限）→ 传播为 collect 失败（本阶段建状态位与检测挂点，完整故障管理归 E6）。

### E2-W2：sealed-ENDED、显式 reset、abandon

`device_runtime.py` + `device_rollouter.py`：

- `step()` **移除内嵌 reset**。ENDED 行处理：`backend.capture(mask) → step 末尾 restore(mask)`（action-step 粒度一次，W0 已验证原语与成本 ~1%）；ended 行 `policy_eval_mask=False`（策略不采样）、不累计 episode_step、不产出新帧。
- `reset_rows(mask, episode_inputs)` 显式接口：校验目标行可 reset（ENDED/FREE），初始化 episode scope → 发 seed/options → backend.initialize → pre_episode schedule → 刷新初始 obs → 标 RUNNING。运行中行 reset → 报错，须先 `abandon(mask, reason)`。
- `abandon(mask, reason)`：显式终止+封存，导出侧标为 abandon，不冒充 timeout。
- wave collector 语义切换：ended 行封存至波末（输出等价已证，J1）；波间全量 reset 不变。`env_term_step`/`term_records` 记录逻辑不变。
- FakeBackend 同步实现 capture/restore（E1 契约方法已有声明，补齐语义）。

### E2-W3：子步级 hook 与终止屏障

`device_runtime.py` + `device_plugin.py`：

- schedule 扩展为每个 phase 的固定 unit 序列；新增 `on_pre_phy_step`/`on_post_phy_step` 槽位，在 `physical_step` 的子步循环内调用 DEVICE 插件（无 host 同步约束）。
- **终止屏障**：`request_termination` 改为写 pending 请求队列（env_ids/code/agents/source_unit/phase/substep）；每个 phase 末尾的屏障按提出顺序消费到 `term_history` 并更新 mask。phase 内后续单元照常执行。
- HOST plane：action-step 粒度 lazy 物化（每 hook 至多一次 host 传输，计费到 timing）；HOST_SLOW 在 `production` profile 下装配拒绝。
- `upload_force_schedule` 消费语义文档化：物理块内逐子步索引，是 per-substep 反馈的当前唯一数据平面形式。

### E2-W4：声明式插件契约与装配校验

`device_plugin.py` + `device_runtime.py`：

- `DeviceUnitSpec` 扩展：`declared_reads`/`declared_writes`（字段级，对 sim namespace 白名单校验）、`output_schema`（metrics/events 的键与形状）、`rng_domain`（W5 用）、`per_hook_mutator`（哪个 hook 授哪个写动词）。
- 装配期校验：声明字段不存在→报错；依赖 `reads`/`after`/`before` 冲突或循环→报错；observer 声明物理写→拒绝注册。
- `DeviceCtx` → `HookContext` 收窄：unit 拿不到 runtime/backend 实例（`mutator._sim` 从 facade 改为后端窄接口）；mutator 按字段授权粒度收窄（如 `set_action`/`add_ext_force`/`reset_rows` 分列权限位）。
- debug 写检查（H4）：debug profile 下对声明只读字段做 hook 前后快照对比，违规报错。不做运行时防御（契约明确不防恶意插件）。

### E2-W5：随机服务

`device_state.py` + `device_plugin.py`：

- runtime 在装配时为每个声明 `rng_domain` 的 unit 分配 salt（注册表 → 确定性 salt，记录进 provenance）。
- `RngView`（挂在 ctx）：`uniform(shape, lo, hi)`/`normal(...)`/`randint(...)`——内部 `hash(seed_offsets, step_counter, salt, agent/stream)` 派生，与 fallen 插件现有 splitmix64 约定兼容；插件不再自己拼盐。
- 分片重排稳定性：种子绑定 **job identity**（base_seed + job_idx），不绑定 GPU slot/行号——E4 多卡前置。测试：同一 job 在两个不同 B/行号下产出相同初始化序列。

### E2-W6：契约测试矩阵与回归

`tests/test_physics_contract.py`（扩展，FakeBackend+warp 参数化）+ `tests/test_device_runtime.py`：

| 用例 | 验证点 |
|---|---|
| 正常终止→显式 reset→active | 状态机转换正确性 |
| 终止后**不** reset 继续 step | ENDED 封存：冻结行多步不变，active 行轨迹与无 ended 基线一致（W0 探针结论的回归化） |
| reset 请求无终止（主动重置） | options/seed 发布、pre_episode、初始 obs 刷新 |
| 部分 reset 隔离 | 未选行逐位不变（含 plugin pool/RNG） |
| 重复终止/多原因同步 | history 去重、顺序、records[0] 首次语义 |
| ended 行 policy_eval/帧产出 | 不采样、不增 step、无新帧 |
| 空 mask / padding 行 | 无操作无错误 |
| 非零状态初值 | reset 后状态 = 初始化程序输出，非隐含零 |
| RNG 分片重排 | job 绑定的种子跨 slot/B 一致 |
| save/restore 往返 | 状态复原继续 step |
| FAILED 传播 | 模拟溢出 → collect 显式失败而非静默 |
| HOST plane 物化 | lazy 传输计数 ≤1/hook |
| 装配校验 | 非法声明/循环依赖/observer 越权 → 装配报错 |

收尾回归：device_runtime/device_standup/warp_runtime/warp_validation/device_rollouter/physics_contract 全套 + cuda:1 冒烟 + 2-update 训练冒烟。discuss.md H4/H5/H6 行更新。

## 风险与降级路径

| 风险 | 概率 | 降级 |
|---|---|---|
| W0 核对发现 CPU 时序与 D7 草案实质冲突（如 recorder 读观测时机在 post_episode 前） | 中 | 先修订契约再实施；冲突项记入 discuss.md |
| sealed-ENDED write-back 在 ended 行占多数时反而变慢（每步 masked 拷贝） | 低 | ended 行 >50% 时切换策略：ended 行照旧跑但不消费数据（退化为现状），由 profile 选择 |
| 终止历史定长数组对超 K 原因的截断 | 低 | K=8 足够 CPU 现状；溢出写 overflow 位并在导出时报 warning |
| per-substep DEVICE hook 引 python 级循环开销 | 中 | W3 只承诺"槽位存在+无 host 同步约束"；standup 无子步插件，实际开销为零——性能归 E7 |
| 字段级 mutator 授权对现有插件改动面大 | 中 | 先按动词分组授权（set_action/ext_force/reset/force_schedule 四位），字段级留到 E5 迁移时再收紧 |

## 放行条件

1. W0 语义核对文档落盘，D7 草案与 CPU 实测的差异全部记录并裁决；
2. 上表测试矩阵全绿（FakeBackend 全用例 + warp 关键路径子集）；
3. sealed-ENDED 下 wave collector 输出 Episode 与 auto-reset 版在相同种子下**逐字段一致**（等价证明的测试化，不留口头结论）；
4. `step()` 无隐藏副作用：源码内无 `dev_reset_rows` 调用（依赖方向测试检查）；
5. 训练冒烟（device collector, 2 updates）正常；
6. cuda:1 in-process 冒烟仍绿。

**明确不在本阶段**：CUDA Graph、多卡（E4）、HOST_SLOW 的完整物化实现（只立 profile 拒绝位）、新任务迁移（E5）、完整故障管理（E6）。
