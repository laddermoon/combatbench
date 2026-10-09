# E8 支持矩阵结算

> 类型：产物

结算日期：E8-W0/W1（git `6d714d09` 计划之后）。
状态口径：原生通过 / 兼容通过 / 部分覆盖 / 拒绝 / 过期。
证据列一律指向可复现的测试名、manifest 条目或结果文档——
不接受"曾经跑通过"的无指针结论。

## A. blueprint 单元级（`capability_registry.REGISTRY`，共 11 条）

| 单元（blueprint cls） | 注册 | 结算 | 证据 |
|---|---|---|---|
| `device_runtime:DeviceTimeoutPlugin` | NATIVE | 原生通过 | `test_device_timeout_plugin`（runtime） |
| `disturbance_plugins:RandomFallenStatePlugin` → `DeviceFallenResetPlugin` | NATIVE | 原生通过 | manifest `standup_floor04`（e2e_collect + train_smoke）；E7 reset 分项计量 |
| `rewards.standing_balance_4stage` → `DeviceStandup4StageRewarder` ×2 | NATIVE | 原生通过 | `test_device_standup.py`（13 项同注入态对拍）；wave golden |
| `plugins.standup_termination:StandupTerminationPlugin` | UNSUPPORTED | **拒绝** | 不在目标蓝图；启用须显式原生转换（registry 注释即处置说明） |
| `plugins.imbalance_termination` → `DeviceDualImbalancePlugin` | NATIVE | 原生通过 | `test_device_balance.py` 13 项同注入态对拍 + e2e terms={imbalance_robot_a/b} + train_smoke |
| `rewards.cross_support` → `DeviceCrossSupportObserver` ×2 | NATIVE | 原生通过 | 同上 |
| `rewards.posture_reward` → `DevicePostureObserver` ×2 | NATIVE | 原生通过 | 同上 |
| `plugins.height_phi_observer` → `DeviceHeightPhiObserver` ×2 | NATIVE | 原生通过 | 同上 |
| `device_examples:SubstepProbePlugin` | NATIVE | 原生通过（限探针） | E7 矩阵 hooks-on 格；仅 benchmark 蓝图使用 |
| 未注册 cls | — | **拒绝**（启动即失败） | `lookup()` 默认 UNSUPPORTED；`test_unsupported_specs_rejected`、`test_host_slow_rejected_and_registry` |

**COMPAT/HOST_SLOW 机制**：`resolve_plugin` 保留适配器代码路径
（`HostBatchCompatAdapter`/`LegacyPluginAdapter`）但注册表**无一条
使用**——判定：机制保留但标注"无活跃消费"；W2 弃用判定中明确。

## B. 后端与运行时

| 能力 | 结算 | 证据 |
|---|---|---|
| WarpBackend standalone（不经 runtime） | 原生通过 | `test_warp_runtime.py`、`test_warp_validation.py`、`test_physics_contract.py` |
| FakeBackend runtime（无 GPU 全流程） | 原生通过 | `test_wave_contract.py`/`test_device_lifecycle_contract.py` 主体在其上运行 |
| 生命周期（含终止帧契约：`episode_step` 无条件计数、`on_post_action_step` 每步一次、退化帧排除） | 原生通过 | `test_device_lifecycle_contract.py`（22 项）、`test_audit_terminal_frame.py`、`test_degenerate_frame_excluded_from_boundary` |
| RNG（job-keyed 采样、盐冲突拒绝、行 shuffle 不变） | 原生通过 | `test_rng_*`（lifecycle 3 项）、`test_job_keyed_action_noise` |
| 记录/导出（wave store → Episode v3，含 `physics_steps`/`agent_frame_boundary`） | 原生通过 | `test_exporter_matches_golden_path`、wave golden 对照 |
| Debug 捕获 | 原生通过 | `test_debug_capture`（wave_contract） |
| Debug 回放（`recorded` 只读模式） | 原生通过 | `test_debug_replay_recorded`——capture→replay 往返逐字段核对 |
| Debug 回放（`rerun`/`cpu_eval` 模式） | 部分覆盖 | 需真实后端/CPU 池的 CLI 工具路径；矩阵标注"工具路径不承诺测试锁定" |
| Resume/拓扑校验 | 原生通过 | `test_resume_rollout_topology_check`（migration_manifest） |
| 故障矩阵（NaN/容量溢出/padding/早退） | 原生通过 | `test_health_scan_marks_nan_failed`、`test_health_scan_contact_overflow`、`test_slot_valid_padding_rows`、`test_early_exit_all_ended`、`test_mark_failed_is_explicit`、`test_repeated_wave_store_bounded` |
| CUDA Graph（物理 advance 无回调路径） | 原生通过 | E7_RESULTS §4–5；42→73 项 GPU 套件全绿；hooks-on 如实回退 eager |
| CUDA Graph（obs_build） | 原生通过 | wave golden 逐帧等价（dense 变体）；`CB_OBS_GRAPH=0` 可关 |

## C. Collector 与任务装配

| 能力 | 结算 | 证据 |
|---|---|---|
| 单卡 `DeviceRollouter.collect(jobs) -> List[Episode]` | 原生通过 | `test_device_rollouter.py`（契约/padding/logprob replay/unsupported 拒绝）；E7 1 卡矩阵 |
| 多卡 `MultiDeviceRollouter`（1–8 卡） | 原生通过 | `test_multi_rollouter.py`、`test_shard_plan.py`；E7 §2 规模表（1/2/8 卡） |
| CPU→设备迁移（basic_balance / standup_floor04） | 原生通过 | 两份 `migration_manifests/*.json`（unit_replay + e2e_collect + **train_smoke**）+ `test_migration_manifest.py` |
| 短程训练（device collector → PPO loop） | 原生通过（复验） | manifest evidence：E5 `devices=0,1 各2 updates` + **E8-W4 复验** `e8_smoke_device_ppo`（终止帧语义后，B=512 2 updates 端到端通过） |
| CPU `ParallelRollouter` 主路径 | 原生通过 | `envs/framework/tests/` + humanoid21 套件；414 项回归 |
| hooks-on（子步插件）路径 | 兼容通过 | E7 hooks-on 格（物理 eager 142K sub/s；obs 图仍生效）——可用但非图化，如实标注 |
| Episode/PPO 接口兼容 | 原生通过 | `test_ppo_pipeline_compat`、`test_logprob_replay_parity`、E5 manifest train_smoke |

## D. 旧路径与弃用候选（W2 判定项）

| 对象 | 现状 | 建议结算 |
|---|---|---|
| `WarpHumanoid21Simulator`（warp_simulator.py） | `binding_registry` 活跃装配点 + 测试/探针直接引用 | **保留**——内部装配点，标注"非公开入口"（W2 复核：非 facade） |
| `host_compat.py`（COMPAT/HOST_SLOW 适配器） | 无注册条目消费 | **休眠机制**：代码路径留存 + docstring 标注（已落地） |
| `batch_plugin.py`/`batch_context.py`（numpy 契约原型） | 零代码引用，仅 docstring 设计参照 | **休眠原型**：docstring 标注"勿在新代码引用"（已落地） |
| `coordinator.py` | `device_rollouter`/`multi_rollouter`/`worker` 活跃依赖 | **非弃用**——E4 分片协议，多卡内部模块（W0 误判已修正） |
| `test_m2_cross_backend_fixtures` stale | fixture 依赖哈希过旧 | **已重录**（`b1d4906f`，expected 逐件一致仅刷哈希）；**残余**：工作树在途 EventJournal 编辑（context/observer_plugin/observer_plugins/plugins）使依赖哈希漂移——该工作提交后重跑 `make-fixtures` 即绿 |
| `test_stage_seg_rewards.py` collection 错误 | 被测类已在源码中整体注释（旧框架退役） | **已结算**：`skipif` 保留文件（26 skipped），不阻塞收集；连带的 `curriculum/experiments/__init__` 死 import 已修 |

## E. 探针与工具（公开操作面）

| 工具 | 结算 | 用途 |
|---|---|---|
| `probe_e7_baseline.py` | 原生通过 | 计量/规模矩阵标准入口 |
| `probe_isolation.py`/`probe_worker_spawn.py`/`probe_standup_xeval.py`/`probe_e2e_warp.py`/`m6_*.py` | 部分覆盖 | 历史探针，随用随跑；不承诺入口稳定 |
| `validation*.py`（capture/replay harness） | 原生通过 | `test_batch_validation.py`/`test_mjx_validation.py`（含 1 既有 stale 待结算） |
| `migration_audit.py` | 原生通过 | `test_migration_manifest.py` |

## 汇总

| 状态 | 单元级 | 层间能力 | 说明 |
|---|---|---|---|
| 原生通过 | 9 组（10 条） | 15 项 | 全部有测试/证据指针 |
| 兼容通过 | 0 | 1 项（hooks-on） | 如实标注非图化 |
| 部分覆盖 | 0 | 2 项（debug_replay rerun/cpu_eval、历史探针） | rerun/cpu_eval 为工具路径；探针随用随跑 |
| 拒绝 | 2 类 | — | StandupTerminationPlugin + 一切未注册 cls |
| 过期/待判定 | — | 3 项（facade 标注 + host_compat + 旧原型） | W2 输出弃用表 |

**W1.5 补缺清单**（由本矩阵产出）：
1. ~~`debug_replay` 独立回放路径缺专项测试~~ → **已补**：
   `test_debug_replay_recorded`（capture→replay 往返，逐字段
   核对 npz 键集/dtype）；`rerun`/`cpu_eval` 两模式判"工具路径"，
   不承诺专项测试锁定；
2. CPU→设备迁移 manifest 证据新鲜度 → 见下方触发矩阵。

## F. 变更触发矩阵（E5 验收之后，ROADMAP「语义/生命周期/采样
变化才触发复验」逐条判定）

| E5 后变更 | 性质 | 触发判定 |
|---|---|---|
| 终止帧语义（`082187be`/`f342b68e`）：`episode_step` 无条件计数、post_action 每步一次、终止帧保留、`physics_steps`/`agent_frame_boundary`、实验消费侧改 boundary | **数据语义 + 生命周期** | **触发→已复验**：`e8_smoke_device_ppo`（basic_balance，device collector B=512 单卡 2 updates × 8 episodes，terms={imbalance_a/b}，trajs 正常进 PPO buffer）——证据已写回 `basic_balance.json` manifest |
| E7-W1 屏障融合 | 执行机制（语义不变，契约测试锁定） | 不触发 |
| E7-W2b/W2c CUDA Graph | 执行机制（eager/graph 等价性已证，golden 对照） | 不触发 |
| E8 本包（矩阵/文档/skipif/可选导入/fixture 重录） | 无行为变更 | 不触发 |

**结论**：唯一触发项（终止帧语义→device 数据契约）已复验：
`e8_smoke_device_ppo` 端到端通过（device collect → trajectories
→ PPO 2 updates）。其余变更均不触发复验。
