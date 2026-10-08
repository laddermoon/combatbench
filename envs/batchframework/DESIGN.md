# DESIGN — 设备批量框架架构规格

**状态**：E8 收官后的架构快照（2026-10）。过程史见 ROADMAP/E1–E8 与各
M 系文档；本文只描述"现在是什么"。术语与 `envs/framework/DESIGN.md`
同族（hook/屏障/黑板），差异处显式标注。

## 1. 分层

```
┌─────────────────────────────────────────────────────────────┐
│ 训练侧        baseline/framework/train.py --collector device │
│              DeviceRollouter / MultiDeviceRollouter          │
│              collect(jobs) -> List[Episode]（与 CPU 同契约） │
├─────────────────────────────────────────────────────────────┤
│ 装配层        capability_registry（cls→单元能力/factory）    │
│              binding_registry（sim cls→后端绑定+io_schema）   │
│              migration_manifest（证据→stale 判定→准入）       │
├─────────────────────────────────────────────────────────────┤
│ 运行时        BatchRuntime（hook 驱动/屏障/封存/计量）        │
│              DeviceCtx + DeviceMutator（按 hook 授予）        │
│              RecordStore（波缓冲→Episode 导出）               │
├─────────────────────────────────────────────────────────────┤
│ 数据平面      DeviceBatchState                               │
│              sim | episode | io | rng | plugin 五命名空间     │
├─────────────────────────────────────────────────────────────┤
│ 后端          WarpBackend（mjw 步进+图化）/ FakeBackend        │
│              （契约：build_sim_namespace + dev_* + physical_  │
│               step + capture/restore + task_tables）          │
└─────────────────────────────────────────────────────────────┘
```

单向依赖：上层经**注册表**解析下层，禁止反向 import
（`test_dependency_direction.py` 强制）。仿真器与框架在代码依赖和
测试上可独立（E8 完成标准之一）。

## 2. 核心对象

| 对象 | 文件 | 职责 |
|---|---|---|
| `DeviceRollouter` | device_rollouter.py | 单卡 collector：jobs→波调度→runtime→Episode；装配缓存、能力检查、manifest 准入、provenance |
| `MultiDeviceRollouter` | multi_rollouter.py | 同契约多卡：job 按 identity 分片到 per-GPU worker（spawn），聚合 Episode + collect_report |
| `BatchRuntime` | device_runtime.py | hook 驱动器：step 时序、mutator 授予/吊销、终止屏障、ENDED 封存、observer 调度、计量（hook_timing/sync_stats/seg_timing） |
| `DeviceBatchState` | device_state.py | 数据平面根：五命名空间张量视图；全部第一维 (B,) |
| `DeviceCtx` | device_plugin.py | hook 收到的上下文（state 视图 + mutator + pstate + rng + reset/terminated env_ids） |
| `BaseDevicePlugin` / `BaseDeviceObserver` | device_plugin.py | 单元基类 + 声明面（declared_reads/writes、per_hook_mutator、rng_salt、output_schema） |
| `RecordStore` | record_store.py | 波内帧缓冲→封存→`episode_exporter` 产出 Episode v3 |
| `PolicyExecutor` | policy_executor.py | 策略适配（kind→factory 注册；能力面声明采样 spec 支持性） |

## 3. 数据平面（DeviceBatchState 五命名空间）

| 命名空间 | 内容 | 写权限 |
|---|---|---|
| `sim` | 物理活跃视图：qpos/qvel/ctrl/xpos/xquat/xipos/xanchor/cvel/xfrc_applied/act_target/contacts_flat | **只读**——视图随物理步进原位更新；写一律走 mutator |
| `episode` | 簿记：状态机 mask（slot_valid/world_running/world_failed/agent_done/policy_eval_mask）、终止两级（pending→history）、计数器（episode_steps/action_call_index/physics_steps/substep_index/time）、reset_request、reason_registry | runtime 拥有；插件经 `request_termination` 间接写 |
| `io` | 本步 IO 缓冲：action_a/b、obs_a/b、reward、reward_channels | 插件可写（hook 限定的部分） |
| `rng` | seed_offsets(B,) + step_counter()——job-keyed 派生源 | runtime 发布；插件经 `ctx.rng` 读 |
| `plugin` | 各插件 `declare_state` 的持久张量池 | 所属插件写；partial reset 由 runtime 清零行 |

接触双形态：`contacts_flat`（后端原始 flat packed，内部消费）与
`refresh_padded()` 派生的 (B,cap) padded 视图（插件消费；`active =
dist<=0`；槽内顺序不稳定）。

## 4. 单元契约（插件面）

```python
class BaseDevicePlugin:
    # 声明面（装配期校验；默认空=不校验，向后兼容）
    name / priority / plane(=DEVICE) / require_mutator
    declared_reads        # sim namespace 字段名白名单
    declared_writes       # mutator 动词集
    per_hook_mutator      # {hook: frozenset(verbs)} 按 hook 收窄
    rng_salt              # 声明后 ctx.rng 分配 RngView
    # 生命周期
    declare_state(state)  # 声明 pstate 张量
    set_episode_seeds(seeds)      # (B,) i64 广播
    export_episode_metrics(state) # 波末指标导出
    on_pre_episode / on_envs_reset / on_pre_action_step /
    on_pre_batch_step / on_pre_phy_step / on_post_phy_step /
    on_post_batch_step / on_post_action_step / on_post_episode
    on_attach / on_detach
```

`ExecutionPlane`：DEVICE（原生，零 host 传输）/ HOST（每 hook 至多
一次物化）/ HOST_SLOW（逐 env 循环；`allow_host_slow=True` 才准入；
覆写子步 hook 的旧插件 attach 即拒）。

`DeviceMutator` 动词：`set_action` / `add_ext_force` /
`upload_force_schedule` / `reset_rows` / `set_integration_rows`——
按 `per_hook_mutator` 收窄，越权 `PermissionError`；hook 结束吊销。

## 5. 生命周期（要点；详规 SEMANTICS.md）

- `step()` = 一个 action step = `phy_substeps` 个物理子步（n=25
  humanoid21）。hook 时序见 `device_runtime.py` 模块 docstring。
- **计数语义**（终止帧契约）：`episode_steps` 步尾无条件自增=进入
  step() 次数；`action_call_index` 步首即增=本步 1-based 帧序号
  （终止归档/封存的边界值源）；`physics_steps` 实跑子步。
- **终止两级**：`request_termination` 提出即生效（agent_done 立
  写、reason 即归档 term_history）；env ENDED 由 phase 屏障判
  （`agent_done.all` 或 `reset_request`）。
- **ENDED 行封存**：capture 快照 + 每步 restore 冻结（warp 无
  masked-step）直至 collector 显式 `reset_rows`；对 RUNNING 行
  reset 是契约错误。
- step() 不做 reset；reset 语义（波/部分行/options/种子）见
  SEMANTICS.md §reset。

## 6. 观测与记录

- `obs_builder.build(state)` 每 action step 在 post_action 前刷新
  `io.obs_a/b`——固定形状张量，E7 后可入 torch CUDA Graph
  （dense 接触 + active mask；`CB_OBS_GRAPH=0` 关）。
- `RecordStore` 按波预分配帧缓冲；封存时写 (code,step) 边界；
  `episode_exporter` 产出 `Episode`（v3，含 `physics_steps`/
  `agent_frame_boundary`）——CPU ParallelRollouter 同款契约。
- 退化帧（零物理推进）保留于缓冲但经 `agent_frame_boundary` 排除出
  训练轨迹。

## 7. 装配与准入

```
env_bp.simulator.cls ──resolve_binding──> DeviceBinding
  ├ make_sim(B, device) → 后端对象
  ├ io_schema(sim) → (agent_ids, obs_dim, action_dim)
  └ episode_options_keys → options 白名单
env_bp.plugins[].cls ──resolve_plugin──> CapabilityEntry
  ├ NATIVE: factory(config, sim=) → BaseDevicePlugin
  ├ COMPAT/HOST_SLOW: host_compat 适配器（休眠路径）
  └ UNSUPPORTED/PENDING/未注册 → 启动即拒
env_bp.observer_plugins ──resolve_observer──> NATIVE only
migration manifest → validate_freshness → stale_units 拒跑
```

**未注册即拒、不支持即拒、证据过期即拒**——三个"不静默降级"
是装配层的共同原则。

## 8. 执行优化面（E7 落定）

- 无子步 hook 覆写时 `sim.physical_step(n)` 整块推进，后端可整段
  CUDA Graph 捕获（warp：`CB_WARP_GRAPH=0` 关，按 (n, sched) 惰性
  建图缓存；捕获失败回退 eager）。
- 子步 hook 插件存在 → 逐子步驱动（eager），物理图自动失效——
  如实记录，不静默近似。
- obs_builder 可独立图化（固定形状输出）。
- 计量常开：hook_timing / hook_plugin_timing / observer_timing /
  barrier_time / seg_timing / sync_stats——性能回归先查分解。

## 9. 休眠与遗产

`batch_plugin`/`batch_context`（M3 numpy 契约原型）、`mjx_runtime`/
`mjx_adapter`（被 warp 取代）、`host_compat`（双适配器，零用户零
测试）——保留供参考/调试，不是公开 API（PUBLIC_INTERFACE §4）。

## 10. 文件地图

| 文件 | 内容 |
|---|---|
| device_runtime.py | BatchRuntime + DeviceTimeoutPlugin |
| device_plugin.py | DeviceCtx/DeviceMutator/BaseDevicePlugin/Observer/调度器 |
| device_state.py | 五命名空间 + ExecutionPlane + RNG 原语 |
| device_rollouter.py / multi_rollouter.py / worker.py | collector 层 |
| record_store.py / episode_exporter.py | 波缓冲→Episode |
| warp_backend.py / warp_simulator.py / physics.py | 后端（warp/mjw） |
| fake_backend.py | 契约测试后端 |
| binding_registry.py / capability_registry.py | 装配注册表 |
| migration_manifest.py / migration_audit.py | 证据与审计 |
| device_balance.py / device_standup.py / device_obs.py / device_examples.py | 已迁移单元 + 模板 |
| host_compat.py | 旧插件适配器（休眠） |
| debug_capture.py / debug_replay.py | 快照/回放三模式 |
| validation.py / validation_mjx.py + validation_fixtures/ | 跨后端回放验证 |
| probe_e7_baseline.py | 性能计量探针 |
