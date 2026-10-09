# E6：调试、恢复、容量与故障管理 — 实施计划

> 类型：记录

> ROADMAP 原文：
> - 按需捕获指定 job/agent/frame 的真实设备状态与观测奖励；provenance 含逻辑配置、实际后端、模型/代码版本、策略版本和采样身份。
> - 区分实际轨迹回放、同后端重跑、CPU 交叉评估；复用现有 dump/viewer/指标，不以 CPU 重算结果覆盖设备原始记录。
> - update 边界恢复涵盖协调状态、随机身份、必要持久插件/策略状态和计划版本；恢复应校验拓扑/能力变更，不静默沿用过期验证。
> - 覆盖容量超限、非有限状态、设备错误、队列拥塞、worker 退出、资源释放和重复 collect 的内存稳定性。分配预算纳入录制缓冲与 host Episode，不只报 simulator 显存。
>
> **放行：** 可定位并重放一个失败样例；能按既定协议恢复；多次采样无无界缓存增长；异常不被当成空 Episode、零奖励或可忽略的单卡缺失。

## 定位

E1–E5 建完机制与迁移流程，但当前系统对"出问题时怎么办"只有骨架：FAILED 行会让 collect 显式失败（不当成空 Episode），worker 崩溃 all-or-nothing，队列有背压——这些已在。E6 补的是**可定位、可重放、可恢复、可预算**四块。前置审计（W0）确认的三个真实缺口：

| 缺口 | 证据 |
|---|---|
| **策略采样 RNG 非 job-keyed** | CPU `episode_runner` 每 episode `policy.reset(seed)` 重播种；设备 `PolicyExecutor.act()` 从不重播种 `policy._gen` → 动作噪声是**全局调用序列依赖**：同一 job 单独重跑 vs 在批中第 N 位，噪声不同；resume 后不可复现 |
| **插件 RNG 计数器是行历史** | `DeviceFallenResetPlugin._count` 按 env-row 递增跨 collect 存活 → 同一 job 落到不同行/不同 collect 顺序 → 不同 draw |
| **恢复面只覆盖 learner** | checkpoint v2 有 rng_state + loop_state + resume_ctx 校验，但 collector 拓扑（devices/B）、executor 版本、capability manifest 新鲜度不在恢复校验内 |

已覆盖无需重做的：FAILED→显式抛错、worker 退出 all-or-nothing、close 幂等、cmd_q maxsize=1 背压、`capture(mask, INTEGRATION)`/`restore`/`host_snapshot` 快照原语、`record_store_bytes` 记账。

## 设计判断

- **J1 — 随机身份 = f(job.seed, episode_step, agent_salt)，不依赖行/序列。** 策略噪声改为**注入式**：`sample_action(obs, ctx, u=None)`，`u` 由 executor 用 splitmix64(seed_offsets, step, salt) 逐行生成。u 注入只替换 `torch.rand(gen)` 那一行，截断正态数学不变。CPU 路径默认 `u=None` 走 `_gen`——**不改 CPU 行为**。插件 `_count` 同理改为 job 键（episode 序数随 Job 走），而非行历史。这是"单独重跑一个 job 得到相同外生随机流"的前提，也是恢复等价的前提。
- **J2 — 捕获 = 快照原语 + RecordStore 行切片 + sidecar manifest。** `debug_capture` 不做新记录器：指定 (collect 内 job_index | env_row, frame 集合) → 在该步 `backend.capture(mask)` 拿 INTEGRATION 快照 + 从 RecordStore 切该步 obs/act/observer 叶 + `manifest.json` 落 provenance（job_ref、seed_offsets、policy_version hash、bp_hash、binding、device、collect_id、warp/torch/cuda 版本、git commit、replay_kind 标签）。H9：**sidecar，不动 Episode v3**。
- **J3 — 回放三模式显式分文件分标签。** `debug_replay` 工具三个子模式：`recorded`（回放已存帧，经现有 viewer/recorder 路径）、`rerun`（同后端重采同一 job，依赖 J1 才有意义；与捕获数据做契约级对照，物理非逐位如实标注）、`cpu-eval`（同一 Job 走 CPU round_runner，标 `cpu_cross_eval`，输出独立文件，**绝不覆写设备记录**）。manifest 记 mode 与输入 hash。
- **J4 — 恢复校验 = 拓扑 + 版本 + manifest 新鲜度。** resume_ctx 扩展 collector 维度：devices 列表、batch_size、executor.version、capability manifest 的 `validate_freshness`；不匹配 → 显式拒绝或要求显式 `--force-revalidate`，不静默沿用。W1 后 collector RNG 已是 job 纯函数，恢复只需保证**同一 job 序列**（loop 端已有）——这是 J1 的复利。
- **J5 — 故障矩阵行为表格化后逐项测试。** 容量溢出（nacon > cap → FAILED 行 + reason，不静默截断）、非有限态（qpos/obs NaN → mark_failed + collect 抛诊断）、worker 退出/队列拥塞（已有，补断言）、重复 collect 内存（循环 N 次断言 RSS/GPU 无界增长阙值）、预算记账（collect_report 增 host_episode_bytes + 波容量行）。

## 工作包拆分

### W0 — 现状审计（调研，不改语义）

枚举并落表（写入本文件附录或直接列 commit message）：①全部失败路径的当前去向；②RNG 身份缺口清单（上表两例 + 逐个插件 `rng_salt` 声明核对）；③resume_ctx 现有校验项 vs 缺失项；④contacts 溢出当前行为（`contacts_vec` 是否截断）。产出审计表 + 冻结 W1–W5 目标清单。

### W1 — 随机身份 job-keyed

- `TruncatedNormalPolicy.sample_action(..., u=None)`：u 为 `(B, action_dim)` uniform 噪声，None 时维持 `_gen` 路径；u 注入路径 clamp 到 `(eps, 1-eps)` 防 icdf 边界。
- `TruncatedNormalExecutor`：act() 从 `RngView.unit_seed(env_rows, step_counter)` + agent/policy salt 生成 u 注入。需要把当前 running 行的 seed_offsets + episode_step 传入 executor（ctx 已有 step——接线确认）。
- `DeviceFallenResetPlugin._count` → job 键 episode 序数语义（与 CPU 共享公式 splitmix64(seed, episode_ordinal, salt)；单 job 单 episode 下序数=0，多 reset 场景由审计确认再定）。
- 契约测试：**同一 job 单独重跑 → 外生流逐位一致**（u 序列、reset draw 序列；物理轨迹允许漂移，断言注入流与 seed_offsets 派生量）。

### W2 — 按需捕获

- `debug_capture.py`：`CaptureRequest(job_refs, frames, level=INTEGRATION)`，接入 `DeviceRollouter.collect(..., capture=)` 或 env 触发；产物 `capture_<job>_<frame>.npz`（快照 + 帧切片）+ `manifest.json`（J2 provenance 全项）。
- FakeBackend 上契约测试捕获路径；真 warp 上对一个真 job 捕获并校验快照可 `restore`。

### W3 — 回放三模式

- `debug_replay.py`：`recorded` / `rerun` / `cpu-eval` 子命令；每种模式 manifest 落 `replay_kind` + 输入 hash；rerun 与捕获的对照报告标注"外生流一致/物理漂移量"；cpu-eval 输出写 `cpu_eval_*` 前缀，永不覆盖设备原始记录。
- 复用现有 `ReplaySimulator`/recorder_viewer/`--dump-at` 惯例，不新造查看器。

### W4 — update 边界恢复

- `resume_ctx` 扩展：`collector`、`collector_devices`、batch B、executor.version、manifest freshness；漂移 → 拒绝或显式强制再验证。
- 测试：checkpoint → resume，断言同一 job 序列得到相同外生随机流（W1 后可测）；拓扑变更（1 卡 ckpt → 2 卡 resume）显式报错。

### W5 — 故障矩阵与容量预算

- 表格化故障→行为→断言（溢出/NaN/worker/拥塞/重复 collect 内存）；缺口的补实现（如 contacts 溢出当前若静默截断则改 FAILED）。
- collect_report 增 `host_episode_bytes` 估算 + 各容量行；内存稳定性测试：循环 collect 断言 RSS/GPU 增长有界。

### W6 — 回归 + 文档

全量 tests/ 回归 + device 冒烟；E6_PLAN 执行结果、ROADMAP 状态、discuss 假设表（H9/H12 相关）更新。

## 放行条件核对

1. 可定位并重放一个失败样例 ← W2+W3（捕获→rerun→对照报告）；
2. 能按既定协议恢复 ← W1+W4（job-keyed RNG + resume 校验）；
3. 多次采样无无界缓存增长 ← W5 内存稳定性测试；
4. 异常不被当成空 Episode/零奖励/可忽略单卡缺失 ← 已有语义 + W5 矩阵逐项断言。

---

## 执行结果（2026-10-03）

全部工作包完成，放行条件逐项核对：

| 条件 | 结果 |
|---|---|
| 1. 可定位并重放失败样例 | ✅ `debug_capture`（job_ref×frame → INTEGRATION 快照+帧切片 npz + provenance manifest + jobs.pkl）+ `debug_replay` 三模式；真机验证 rerun 复现逐 agent 终止记录、obs_0 差 ~1e-7 |
| 2. 按既定协议恢复 | ✅ W1 job-keyed 外生流 + checkpoint `rollout_state` 拓扑校验（`_check_resume_rollout` 漂移即拒绝）+ 装配期 manifest 新鲜度（stale 单元拒绝执行） |
| 3. 无无界缓存增长 | ✅ `test_repeated_wave_store_bounded`：3 波 RecordStore 字节恒定；executor LRU 有界；report 增 `host_export_bytes` |
| 4. 异常不静默 | ✅ 既有 FAILED→raise 之外，新增 health scan：contacts per-world 饱和→FAILED(capacity)（修掉 `_contacts_padded` 静默丢弃漏洞面的热路径对应），NaN/Inf→FAILED(non_finite) |

**W1 的实质性修复**（超出原计划预期的问题）：
- `PolicyExecutor.act()` 从不调用 `policy.reset(seed)`——设备端动作噪声曾是**全局调用序列依赖**（同 job 单独重跑 vs 批中不同位置 → 不同噪声）。现在 `sample_action(u=)` 注入式噪声由 `splitmix64(seed_offsets, episode_step, agent_salt)` 逐行生成，job-keyed 逐位可复现。
- `DeviceFallenResetPlugin._count` 行历史计数 + `env_ids` 混入 → 同样改 job-keyed（CPU `RandomState(job_seed)` 每 episode 重建本就是 job-keyed 语义）。
- 契约测试 `test_job_keyed_action_noise`：同 seed 重跑逐位一致、改 seed 变、行位无关。

**验证**：tests/ 139 项绿（含真 warp 端到端、2 卡 multi、lifecycle/波契约）；device 冒烟 2 updates 正常；capture→recorded/rerun/cpu-eval 三模式真机跑通（cpu-eval 输出独立文件、轨迹差异如实记录 fp64-vs-fp32）。

**残余**：per-world 饱和判定用 `count >= cap` 保守策略（恰好 cap 也会报 FAILED）——诚实偏向误报而非静默；multi-worker 捕获 manifest 按 worker 分目录；HOST plane 物化与异步 D2H 仍归 E7。

提交：`c3950236`（W0–W1）→ `658a3a0c`（W2–W3）→ `47116c65`（W4–W5）→ 本次收尾。

**明确不在本阶段**：多机恢复、跨 collect 插件状态链持久化、CUDA Graph、异步 D2H、checkpoint 格式 v3、Episode v3 格式改动、自动重试故障 worker（保持 all-or-nothing 语义）。

---

## W0 审计结果（执行核对表）

### 失败路径现状

| 路径 | 现状 | 结论 |
|---|---|---|
| FAILED 行 | `device_rollouter.py:302` 收集 rows+reasons 显式抛错 | ✅ 不产空 Episode |
| mark_failed / FAIL_CODES | `{"capacity":0,"non_finite":1}` code 齐备 | ⚠️ **non_finite 无检测点**——无任何 isfinite 调用 |
| **contacts 溢出** | `_contacts_padded`：`slot >= cap: continue` **静默丢弃** | ❌ 违反"溢出必报"——warp 全局 capped，per-world 饱和是唯一可检信号 |
| worker 退出 | all-or-nothing `WorkerLost` | ✅ |
| 队列拥塞 | `cmd_q maxsize=1` 背压 | ✅ |
| 资源释放 | close 幂等 join→terminate→报告 | ✅ |
| 重复 collect | RecordStore 每波新建（torch 缓存复用），executor LRU 有界 | ⚠️ 需 W5 实测断言 |

### RNG 身份缺口

| 单元 | 现状 | 修复 |
|---|---|---|
| `TruncatedNormalPolicy._gen` | 设备路径**从不 reset**→噪声全局序列依赖 | W1 u 注入 |
| `DeviceFallenResetPlugin._count` | per-行跨 collect 递增 | W1 改 job 键（CPU `reset_count` 本身就是槽位历史语义，设备侧 job 键更严格） |
| `seed_offsets` | = job.seed，job-keyed | ✅ 已对 |

### resume_ctx 现状

携带 `prev_gvec`/`n_evals_done` + learner RNG（cuda device-count 变更仅告警跳过）。**collector 拓扑（devices/B）、executor 版本、manifest 新鲜度均不校验** → W4 目标。
