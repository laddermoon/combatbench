# batchframework 公开接口与版本锚（E8-W2 固定）

> 类型：契约

本文件定义**对外承诺的接口面**。未列入的模块按"内部实现"对待：
可以改签名、拆并、删文件，不向外部兼容承诺。

## 1. 公开面

| 层 | 接口 | 稳定性 |
|---|---|---|
| collect 入口 | `DeviceRollouter(batch_size, device)` / `MultiDeviceRollouter(devices, batch_size_per_worker)`；`.collect(jobs) -> List[Episode]`；`last_collect_report` | 稳定 |
| 数据契约 | `baseline.framework.rollout.episode.Episode`（npz v3）、`Job`、`Trajectory`、`ChannelData`、`RewardChannel` | 稳定（训练框架共享契约） |
| 任务装配 | `EnvBlueprint` / `ParameterizedEnvBlueprint`（YAML，`BLUEPRINT_VERSION=1`）；`binding_registry`（任务绑定注册） | 稳定 |
| 能力结算 | `capability_registry.REGISTRY` / `resolve_plugin` / `resolve_observer` / `register`——blueprint 插件/observer → 设备实现的唯一裁决点；未注册即拒绝 | 稳定 |
| 迁移结算 | `migration_audit.py` + `migration_manifests/*.json`——CPU 蓝图 → 设备可行性与证据清单 | 稳定（工具契约，产物内部 schema） |
| Debug | `debug_capture.CaptureRequest` / `WaveDebugCapture`（收集期捕获）；`debug_replay` 三模式 | 稳定（`recorded` 只读模式有测试锁定；`rerun`/`cpu_eval` 为工具路径） |
| 计量探针 | `probe_e7_baseline.py`（分项计时/同步记账矩阵） | 稳定（E7 标准入口） |
| Runtime 契约 | `BatchRuntime` 生命周期（episode_step/physics_step 语义、终止帧、退化帧排除、hook 序列） | 稳定——语义变更走契约测试 |

## 2. 版本锚

| Schema | 版本 | 位置 | 兼容规则 |
|---|---|---|---|
| Episode npz | `EPISODE_FORMAT_VERSION = 3` | `baseline/framework/rollout/episode.py` | 严格等值校验；v3 可选键（`physics_steps` 等）缺失时按声明回退，不伪造 |
| Env blueprint | `BLUEPRINT_VERSION = 1` | `envs/framework/blueprint.py` | 版本不符启动即拒 |
| capture manifest | `kind="device_capture"` | `debug_capture.py` | 诊断产物，不承诺跨版本兼容 |
| migration manifest | 无版本 | `migration_manifests/*.json` | 可再生审计产物，以 `source_bp_hash` 判新鲜度 |
| RecordStore / wave 缓冲 | — | `record_store.py` | 纯内存结构，随代码版本 |

## 3. 内部实现（非公开，引用自由但无兼容承诺）

`device_runtime.py`/`device_plugin.py`/`device_state.py`/
`record_store.py`/`wave_runner.py`/`worker.py`/`coordinator.py`
（E4 分片协议，多卡内部使用）/`warp_backend.py`/
`warp_simulator.py`（`WarpHumanoid21Simulator`——binding 装配点，
测试/探针可直接用但非稳定入口）/`physics.py`/`device_obs.py`/
`device_balance.py`/`device_standup.py`/`policy_executor.py`/
`episode_exporter.py`/`fake_backend.py`/`binding_registry.py`
（注册表本身公开，条目属内部）/`validation*.py`（验收工具）/
各 `probe_*.py`/`m6_*.py` 历史探针。

## 4. 弃用/休眠清单

| 对象 | 判定 | 说明 |
|---|---|---|
| `batch_plugin.py` / `batch_context.py` | **休眠原型** | 零代码引用；作为 hook 语义契约的设计参照保留（`device_plugin.py` docstring 引用其顺序）——不删、不演进、勿在新代码引用 |
| `host_compat.py`（COMPAT/HOST_SLOW 适配器） | **休眠机制** | `resolve_plugin` 保留代码路径但 registry 无一条使用；保留以备显式兼容需求，勿默认走此路 |
| `WarpHumanoid21Simulator` | 内部装配点 | 非弃用——binding_registry 活跃依赖；标注"非公开入口" |
| 历史探针 `probe_isolation.py`/`probe_standup_xeval.py`/`probe_e2e_*.py`/`m6_*` | 工具路径 | 随用随跑，不承诺入口稳定 |

## 5. 变更规则

- 改公开面签名/语义 → 更新本文件 + 契约测试 + 支持矩阵对应行；
- 新增 blueprint 能力 → `capability_registry.register` 显式登记
  （默认拒绝语义不变）；
- Episode npz 结构变更 → 升 `EPISODE_FORMAT_VERSION` 并写明
  旧版本读取行为；
- 新增休眠/移除判定 → 先更新 §4 再动代码。
