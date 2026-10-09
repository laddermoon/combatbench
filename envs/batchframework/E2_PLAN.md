# E2 计划：生命周期、插件、随机与终止契约

> 类型：记录

**状态**：W0–W6 完成（2026-10-02）；放行条件逐项核对见文末（项 3 为源码论证 + 端到端契约测试，非双跑 A/B）
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

**J2 — 终止"提出即生效、即归档"，屏障只判 env 结束。**（实现修正）
原计划写"pending 请求 → 屏障归档"。实现中发现 pending 单 code 槽位会丢同 phase 多 reason（后者覆盖前者），且与 CPU 语义核对后确认：CPU 的 `agent_terminated` 本就**立即**置位、recorder 逐帧扫描 proposals——"归档时机"是记录端行为而非提出端约束。改为：`request_termination` 立即写 `agent_done`/`term_pending`（观测位）并把 reason **立即去重归档**进 `term_history`（记录时刻 = 当前 `episode_step`，与 CPU 提议时刻一致）；phase 屏障只做 env 级 ENDED 判定 + 封存 + post_episode 调度。语义等价且更精确：`term_history` 天然承载多 reason 保序。

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

### E2-W1：EpisodeNamespace 数据模型重构 ✅ 已落地

实际形态（`device_state.py`）：四 mask 全量分离（`slot_valid`/`world_running`/`agent_done`/`policy_eval_mask`），`active_mask`/`agent_terminated` 保留为兼容别名；计数器 `episode_steps`/`action_call_index`/`physics_steps`/`substep_index`/`time` 齐备；终止历史 `term_history (B,2,K,2) i32` 空槽填 -1（避免 code 0 碰撞）；`world_failed`/`fail_reason` 承载 FAILED。`reason_registry: Dict[str,int]` 挂在 namespace 上（python 级装配级注册表），自定义 reason 按首见顺序确定性分配 ≥6 的 code。

`device_state.py` + `device_runtime.py`：

- `active_mask` 拆分为：`slot_valid`（静态，非 padding）、`world_running`（仍在推进物理）、`agent_done (B,2)`（训练终止态，=现 `agent_terminated` 重命名对齐）、`policy_eval_mask`（本步是否调用策略——ended/sealed 行=False）。
- 新增计数器：`physics_steps`（累计物理子步数）、`substep_index`（当前 action step 内子步位置）、`action_call_index`（episode 内第几次 step 调用，区别于完成的 `episode_steps`）。`record_frame_index` 归 collector/recorder 侧所有。
- `agent_term_reason` 升级为**终止历史**：定长 `term_history (B, 2, K, 2)` int32（K≈8，[code, logical_step] 对）+ `term_history_len (B,2)`；首次有效原因语义由 collector 导出时取 `history[...,0]`，同 code 去重、异 code 保留、按提出顺序排列。
- `FAILED` 行状态：`world_failed` mask + `fail_reason` code；FAILED ≠ agent 失败，是执行错误（njmax 溢出/数值非有限）→ 传播为 collect 失败（本阶段建状态位与检测挂点，完整故障管理归 E6）。

### E2-W2：sealed-ENDED、显式 reset、abandon ✅ 已落地

实际形态（`device_runtime.py`）：`step()` 无内嵌 reset；屏障内 `capture(newly)` 快照 → `_freeze_ended_rows()` 每 action step 末 `restore` 写回（ended 行物理被推进但每步末回滚封存态——W0 探针结论的工程化）。`reset_rows(ids, seeds)` 只对 ENDED/FAILED 开放（RUNNING → `ContractError`）；`abandon(ids)` 记 "abandoned" 提议 + 封存；`mark_failed` 走 `fail_reason` 显式通道。rollouter：`_run_wave` 全 ENDED 早退（`any_running()`）、波末 `failed_mask` 健康检查使 collect 显式失败、`_WaveRecorder` 改消费权威 `term_history`（修掉 `_seen_term` 丢迟发 reason 的 bug）、末帧 observer 输出按 CPU 序在 post_episode 刷新后覆写。

### E2-W3：子步级 hook 与终止屏障 ✅ 已落地

实际形态：插件覆写 `on_pre_phy_step`/`on_post_phy_step`（默认 no-op，覆写检测进 `_substep_units`）→ runtime 经 `physical_step(n, pre_step=, post_step=)` 回调驱动，避免拆块（pending wrench 只在块首子步消费）。子步内 `_post` 后跑 `_consume_terminations()` 屏障：子步内终止即刻封存，`episode_steps` 按块结束时仍 RUNNING 的行记（CPU：子步内终止该 step 不 +1）。无子步插件时仍走整块 `physical_step(n)` 快路径。HOST plane lazy 物化未做（现无 HOST 插件，留 E5/E6）。

### E2-W4：声明式插件契约与装配校验 ✅ 已落地（字段级收窄留 E5）

实际形态：`BaseDevicePlugin` 声明面 = `plane`/`priority`/`require_mutator` + `declared_reads`（对 backend.describe 白名单校验）/`declared_writes`（mutator 动词集）/`per_hook_mutator`（按 hook 收窄动词）/`rng_salt`。装配校验：未知动词/读字段报错、writes↔require_mutator 一致性、HOST_SLOW 默认拒绝、salt 冲突拒绝、**插件名唯一**（state pool 键）、`export_episode_metrics` 跨插件键冲突显式拒绝。observer 隔离经 dispatcher ctx（mutator=None，测试覆盖）。**未做**：debug 写检查（H4）与 `reads/after/before` 依赖排序校验——现有插件均无依赖声明，推迟到有真实需求时；字段级 mutator 授权按动词分组已实现，字段级留 E5 迁移收紧。

### E2-W5：随机服务 ✅ 已落地

实际形态：`RngNamespace = seed_offsets(B,) i64 + step_counter()`；`RngView.unit_seed(env_ids, counter)` = `seed_offsets[ids] + counter·MULT + salt`（与 fallen 原公式逐项一致——序列逐位不变）。runtime attach 时按声明 `rng_salt` 分配 view 挂 `ctx.rng`，salt 注册表防撞。fallen 插件迁移：`_draw_actions` 改走 `ctx.rng`（env_ids 混合项留在插件侧——属分布设计而非随机服务）。分片重排不变性有测试（`test_rng_row_shuffle_invariance`）。

### E2-W6：契约测试矩阵与回归 ✅ 进行中

新增 `tests/test_device_lifecycle_contract.py`（17 项，FakeBackend 可解释状态机）：

| 用例 | 状态 |
|---|---|
| 多 reason 保序/同 reason 去重/已终止后新 reason 仍记 | ✅ |
| 自定义 reason 字符串经 registry 往返 | ✅ |
| ENDED 行冻结不漂移（多步 qpos 逐位不变）+ 停在终止步 | ✅ |
| RUNNING 行 reset_rows 拒绝（ContractError）/ ENDED 可复用 | ✅ |
| abandon→abandoned 记录→reset 复用 | ✅ |
| mark_failed：FAILED≠ENDED、fail_reason、波末可检出 | ✅ |
| padding 行（slot_valid=False）不推进不结束 | ✅ |
| 空 ids 操作无退化 | ✅ |
| policy_eval_mask：policy/hold 两模式 | ✅ |
| 子步内终止屏障：当子步封存、episode_step 不 +1 | ✅ |
| RNG：重排不变性 / counter 换序列 / salt 冲突拒绝 | ✅ |
| 插件名唯一 / metric 键冲突 / observer 无 mutator | ✅ |
| 部分 reset 隔离（plugin pool/episode 簿记） | ✅（test_device_runtime） |

**未覆盖（如实记欠账）**：HOST plane lazy 物化计数（无 HOST 插件存在）、save/restore 整态往返（契约测试在 test_physics_contract）、warp 参数化契约子集（capture/restore 在 warp_runtime 测试覆盖）、非零初值专项检查。

收尾回归：device_runtime/device_lifecycle_contract/device_standup/warp_runtime/device_rollouter/physics_contract/dependency_direction 全套（67 项）+ device collector 训练冒烟。

## 风险与降级路径

| 风险 | 概率 | 降级 |
|---|---|---|
| W0 核对发现 CPU 时序与 D7 草案实质冲突（如 recorder 读观测时机在 post_episode 前） | 中 | 先修订契约再实施；冲突项记入 discuss.md |
| sealed-ENDED write-back 在 ended 行占多数时反而变慢（每步 masked 拷贝） | 低 | ended 行 >50% 时切换策略：ended 行照旧跑但不消费数据（退化为现状），由 profile 选择 |
| 终止历史定长数组对超 K 原因的截断 | 低 | K=8 足够 CPU 现状；溢出写 overflow 位并在导出时报 warning |
| per-substep DEVICE hook 引 python 级循环开销 | 中 | W3 只承诺"槽位存在+无 host 同步约束"；standup 无子步插件，实际开销为零——性能归 E7 |
| 字段级 mutator 授权对现有插件改动面大 | 中 | 先按动词分组授权（set_action/ext_force/reset/force_schedule 四位），字段级留到 E5 迁移时再收紧 |

## 放行条件（逐项核对结果）

1. ✅ W0 语义核对文档落盘（LIFECYCLE_TRACE.md），D7 草案与 CPU 实测核对无冲突，契约未修订；
2. ✅ 契约矩阵 17 项全绿（`test_device_lifecycle_contract.py`，FakeBackend）+ warp 关键路径（warp_runtime/rollouter 测试覆盖 capture/restore/导出链）；
3. ⚠️ **以源码论证代替 A/B 回归**：sealed-ENDED 的等价性证据是 `t_use=term_step` 截断使 mid-wave reset 后数据从未被消费（`device_rollouter._assemble_episode`）——新实现下 ended 行不再产生数据，导出源同一截断点。auto-reset 路径已删除无法同树 A/B；`test_collect_episode_contract`（真 warp 端到端）验证了导出契约完整性。残留风险：若未来 collector 改为消费终止后数据（如 post-termination 观测），等价性论证需重审。
4. ✅ `BatchRuntime.step()` 内无 reset 调用；reset 仅存在于 `reset()`/`reset_rows()`/`_consume_terminations` 的封存写回；
5. ✅ 训练冒烟（device collector standup，2 updates）正常，terms={timeout:16}；
6. ✅ cuda:1 in-process 冒烟（E1 验证，本阶段未回退）。

**明确不在本阶段**：CUDA Graph、多卡（E4）、HOST_SLOW 的完整物化实现（只立 profile 拒绝位）、新任务迁移（E5）、完整故障管理（E6）。
