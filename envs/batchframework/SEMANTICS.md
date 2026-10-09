# SEMANTICS — 设备批量框架语义规格（seed / reset / 生命周期）

> 类型：契约

CPU 侧 `envs/framework/SEED.md`、`RESET.md`、`LIFECYCLE_TRACE.md` 的
设备对应件。本文是**规范**（spec）——实现以此为准；`LIFECYCLE_TRACE.md`
仍是 CPU 侧的语义来源文档。

---

## 1. 随机性（SEED 对应）

### 1.1 原则

1. **job-keyed**：一切随机性绑定 `Job.seed`（base_seed）——**不绑**
   GPU slot / 行号 / worker。同一 job 在 1 卡第 3 行与 8 卡分片后的
   任意位置抽到**相同**序列（E4 分片不变性）。
2. **无全局状态**：不设 generator；每次 draw =
   `f(seed_offset, counter, salt)` 的纯函数。

### 1.2 原语（`device_state.py`）

```
seed_offsets (B,) i64   行装载 job 的 base_seed（reset 时按行写入）
step_counter () i64     全局 action-step 计数（reset 时清零）
splitmix64(x)           64-bit 纯函数混合（逐位可复现）
_SEED_COUNTER_MULT      Knuth LCG 乘子（counter→seed 偏移系数）

RngView(unit).unit_seed(env_ids, counter)
    = seed_offsets[ids] + counter * MULT + salt        → (M,) i64
rng_uniform(base_seeds, n_cols)
    → (M, n_cols) f32 ∈ [0,1)  —— splitmix 网格展开的外生流
```

### 1.3 派生规则

- 每个声明 `rng_salt` 的单元在 attach 时得到独立 `RngView`
  （salt 冲突即拒——`_rng_salts` 防撞）。
- 单元自选 counter 语义（reset 次数/step_counter/子步索引……），
  但**必须确定性**——同一 (seed, counter, salt) 永远同值。
- `set_episode_seeds(seeds)`：runtime 在每行 episode 开始时把
  seed_offsets 广播给各插件（CPU `set_episode_seed` 的批量化）。
- **禁止** `np.random` / `torch.Generator` / `random` 自备——那是
  不可复现的孤岛（CPU SEED.md 同规则，设备侧由机制强制）。

---

## 2. Reset（RESET 对应）

### 2.1 两级入口

| 入口 | 范围 | 使用方 |
|---|---|---|
| `rt.reset(seeds, options)` | 全量 B 行 | collector 波首装配 |
| `rt.reset_rows(env_ids, seeds, options)` | 仅 ENDED/FAILED 行 | collector 波内补齐 |
| `mutator.reset_rows(env_ids)` | 插件请求的行复位 | 插件（如课程复位） |

### 2.2 全量 reset 传导顺序

```
rt.reset(seeds, options)
 1. sim.reset(seeds_np, options)          # 后端写初态 + options 消费
 2. st.reset_episode_rows(all)            # 簿记清零（计数/终止/pending）
 3. seed_offsets ← seeds；step_counter ← 0
 4. 各插件 set_episode_seeds(seed_offsets)
 5. on_pre_episode(writable, reset_env_ids=None=全量)
```

### 2.3 部分 reset 传导顺序（reset_rows）

**前置契约**：只允许 ENDED/FAILED 行——对 RUNNING 行调用抛
`ContractError`（丢弃未完成 episode 是错误；须先 `abandon`）。

```
reset_rows(ids, seeds, options)
 1. sim.dev_reset_rows(ids)        # 后端写初态 + derived 刷新
 2. st.reset_plugin_rows(ids)      # declare_state 张量对应行清零
 3. st.reset_shared_rows(ids)      # 共享黑板行清零（E9）
 4. st.reset_events_rows(ids)      # 事件 journal：count=0 + epoch++（E9）
 5. st.reset_episode_rows(ids)     # 簿记复位
 6. _publish_episode_options       # ctx.episode_options 行快照（若给）
 7. seed_offsets[ids] ← seeds（若给）
 8. on_envs_reset(writable, reset_env_ids=ids)   # 行级钩子
 9. on_pre_episode(writable, reset_env_ids=ids)  # 与 CPU 同名语义
```

`on_envs_reset` 与 `on_pre_episode` 的区分：前者是**行级**事件
（部分 reset 独有），后者与 CPU 同名同语义（episode 开始——全量
时 reset_env_ids=None，部分时为被复位行）。插件应按
`ctx.reset_env_ids` 决定作用范围：None=全量，否则仅这些行。

### 2.4 options

`binding.episode_options_keys` 白名单键 → collector 把 per-env 值
广播为张量 → `sim.reset(options)`。同时 runtime 把白名单键物化为
`state.episode_options`（`{key: (B,) 张量}` 行快照），hook 内经
`ctx.episode_options` 只读可见（E9 G4 已闭合——与 CPU 差异：张量化
全体行视图，行差异在值上；CPU 是 per-env dict）。

### 2.5 reset_request（插件侧）

插件置 `ep.reset_request[ids]`（直写兼容口）→ 屏障判为
reset-while-active → 全员记 `abandoned` 提议（等价 CPU 语义），
ENDed 后由 collector reset_rows 复用行。

---

## 3. 生命周期（LIFECYCLE 对应）

### 3.1 step() 权威时序（`device_runtime.py` 模块 docstring 为源）

```
clear_step_flags；policy_eval_mask 刷新（running × post_term_action）
[dev_set_action]                       ← runtime 写
on_pre_action_step   (rw: io.action)          【屏障：提出即生效】
on_pre_batch_step    (rw: pending force / force schedule)
for s in substep:（仅当有插件覆写子步 hook 才逐子步驱动）
    on_pre_phy_step → physical_step(1) → on_post_phy_step 【每子步屏障】
否则 sim.physical_step(n_substeps) 整块推进（可图化）
on_post_batch_step   (rw: 受限投影)
episode_steps/physics_steps/... += 1（仅 RUNNING 行；步尾无条件）
obs 构建 → io.obs
on_post_action_step  (ro；可提终止/写 reward)
── 终止屏障 ── pending→归档；新 ENDED 封存（capture+每步 restore 冻结）
on_post_episode      (ro; terminated_env_ids=本步新 ENDED 行)
```

step() **不做 reset**。ENDED 行封存保持到 collector 显式
`reset_rows`。

### 3.2 计数器（终止帧契约，2026-10 修订后）

| 字段 | 语义 | 时机 |
|---|---|---|
| `episode_steps` (B,) i64 | 进入 `step()` 的无条件调用计数——含物理零推进帧 | **步尾**自增（步内读到上一步的值，对齐 CPU `ctx.episode_step`） |
| `action_call_index` (B,) i64 | 本步 1-based 帧序号 | **步首**即增——`request_termination`/`seal_rows` 在步内任意点读到的边界值（含端点） |
| `physics_steps` (B,) i64 | 实际执行的物理子步累计 | 逐子步按 running mask 记 |
| `substep_index` (B,) i32 | 当前子步位置 | 仅子步循环内有效 |
| `time` (B,) f32 | 累计物理秒 | 逐子步 |

两字段恒相等但时序语义不同（`episode_steps` 是 CPU parity 位，
`action_call_index` 是归档边界源）——合并留待后续清理。

### 3.3 终止两级模型

```
request_termination(env_ids, reason, agents)        —— 提出
  立即：agent_done=True（同 phase 后续插件立即可见）
        term_pending/code 标记
        term_history 归档 (code, action_call_index)
        ——每 (agent,reason) 首次出现记一次；同 reason 去重、
          异 reason 保序、超 K 置 overflow、已终止后新 reason 仍记
        reason 字符串 → reason_registry 确定性 code（≥6 首见序）

_consume_terminations（phase 屏障）                  —— 判定
  快路径：单 bool(any(reset_request|terminated_flag|agent_done.all)
          & running_valid) 无提议零额外 sync（E7-W1 融合）
  慢路径：reset_request→abandoned 提议；ENDED=agent_done.all(-1)
          → capture 快照封存 + 调度 on_post_episode + 冻结写回
```

屏障位点：pre-action 后（跳过整步）/ 子步循环内（子步 hook 模式
每子步后）/ action-step 尾。

### 3.4 状态机

```
RUNNING (world_running=1)
  → ENDED   (world_running=0；封存快照+逐帧 restore 冻结；
             待 collector reset_rows)
  → FAILED  (world_failed=1 + fail_reason(FAIL_CODES)；执行错误)
padding 行：slot_valid=False（不占状态机）
```

ENDED 冻结是**机制**不是约定：warp 无 masked-step，封存行每步
restore 快照——插件即使误写也被覆盖（见 physics.py 契约头注）。

### 3.5 帧边界与导出

- 终止帧 = 有效 transition（obs/action/物理后状态/observer 输出齐全），
  即使终止发生在物理子步中途——`action_call_index` 保证含端点边界。
- **退化帧**（进入 step 但零物理推进）：保留于缓冲供 provenance，
  经 `agent_frame_boundary` 排除出训练轨迹。
- `abandon(env_ids, reason)`：显式终止（诊断/预算），记 reason
  不冒充 timeout；封后走正常 reset_rows 复用。

### 3.6 多卡语义

job 按 identity 分片（非位置）——同一 job 在任意 GPU/行号得到相同
seed_offsets/相同随机序列；worker 故障使该分片 FAILED，聚合层如实
报告（`collect_report`），不静默补齐。

---

## 4. 与 CPU 语义的已知差异（显式登记，非 bug）

| 项 | CPU | 设备 | 理由 |
|---|---|---|---|
| 数值 | 确定性（同码同输入同轨迹） | run-to-run 有噪声（contact 原子序） | bit-identical 非目标；回放验证用容差 |
| metrics 通道 | `ctx.metrics` dict 任意对象 | `ctx.metrics` 声明式 (B,*shape) 张量池（E9 已闭合；结构化数据须拆键） | BLACKBOARD_DESIGN §1 |
| events 通道 | `ctx.events` append-only EventJournal（任意对象） | `ctx.events` padded journal（定长数值记录 [code,agent,value,aux] + epoch 游标）（E9 已闭合） | BLACKBOARD_DESIGN §2 |
| episode_options hook 可见 | ctx.episode_options per-env dict | `ctx.episode_options` {key:(B,)张量} 行快照（E9 已闭合） | §2.4 |
| 部分 reset | 无（env 级） | 行级 reset_rows/on_envs_reset | 批量必需 |
| RNG | set_episode_seed + 自管 | job-keyed 派生（更严） | 分片不变性要求 |
| 子步 hook | 每子步 Python 回调 | 存在但禁图化；schedule 上传为替代 | 融合步进不可逐子步回 host |
