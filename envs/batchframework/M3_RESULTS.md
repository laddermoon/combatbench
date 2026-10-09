# M3 结果：设备端批量运行时与插件体系

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

对应 [M3_PLAN.md](M3_PLAN.md) / [ROADMAP.md](ROADMAP.md) §6。已实现：设备数据平面契约、warp 后端绑定、原生插件基类 + BatchRuntime、HOST/HOST_SLOW 兼容层、能力注册表、FakeBatchBackend 生命周期测试。

## 1. 交付清单

| 文件 | 内容 |
|---|---|
| `device_state.py` | `DeviceBatchState` 契约：`sim/episode/io/plugin/rng` 五命名空间；`declare_state` 插件池；`ExecutionPlane`（DEVICE/HOST/HOST_SLOW）；**per-agent + per-env 两级终止簿记** |
| `warp_simulator.py` | `mjw.Data` → torch 零拷贝视图（`wp.to_torch`）；pending 外力迁设备；`physical_step` 在 `wp.ScopedStream(torch stream)` 上统一编排，全程零额外 host sync；dev mutator 五件套 |
| `device_obs.py` | 96 维观测的 torch 复刻（flat contacts + index_add 聚合足力）；与 host 路径对照 max diff ~5e-7（fp32 噪声级） |
| `device_plugin.py` | `BaseDevicePlugin` / `BaseDeviceObserver` / `DeviceCtx` / `DeviceMutator` / observer dispatcher |
| `device_runtime.py` | `BatchRuntime`：hook 调度、per-env 终止消费、部分 reset、`DeviceTimeoutPlugin` |
| `host_compat.py` | `HostBatchCompatAdapter`（HOST）+ `LegacyPluginAdapter`（HOST_SLOW）+ `LegacyObserverAdapter` + `SyncStats` + `_CachingSimProxy` |
| `capability_registry.py` | NATIVE/COMPAT/HOST_SLOW/PENDING/UNSUPPORTED 五态注册表；未注册类启动即失败 |
| `fake_backend.py` | 轻量契约后端（torch CPU），生命周期测试不依赖 warp；兼作新后端实现模板 |
| `device_examples.py` | 四个原生插件模板：stateless observer / stateful / 子步 schedule 施力 / reset 初始化 |
| `tests/test_device_runtime.py` | 7 个生命周期用例（FakeBackend，~3s） |
| `tests/test_warp_runtime.py` | warp 真实后端冒烟（GPU-gated） |

## 2. 关键语义决定

1. **契约张量类型是 torch.Tensor**——不向插件暴露 `wp.array` 或 warp flat-packed contact 布局；contact 的 padded 化留作后端绑定职责（当前 `contacts_flat` 保留 flat 视图 + cap 元数据，padded 派生按需实现）。
2. **逐子步修改 = schedule 上传** `(B, n_steps, nbody, 6)`，物理循环逐子步消费（实测：单发 pending 位移 0.0012 vs schedule 25 子步 0.0041，且逐子步位置经 `xfrc_log` 断言正确）。覆写 `on_pre_phy_step` 的 legacy 插件在 attach 时**直接拒绝**——不是降级到块边界。
3. **终止两级**：`agent_terminated (B,2)` 跨步持久，`terminated_flag (B,)` env 级；env 终止 = 显式 flag ∨ 全 agent 终止（对齐旧框架 `all_agents_terminated`）。
4. **stream 纪律**：`physical_step` 内 `wp.ScopedStream(wp.stream_from_torch())`——warp kernel 与 torch 组合操作同流有序；DEVICE 插件全程不得 host 传输（约定 + 性能审计，不在解释层拦截）。
5. **HOST 层如实计量**：episode 簿记拉取也计入 SyncStats；输出可报告 `mode=host_compat, syncs/step`。

## 3. 放行条件核对（ROADMAP §6）

| 条件 | 状态 |
|---|---|
| 轻量后端生命周期测试（不依赖真实后端编译） | ✅ FakeBatchBackend 7 用例全过 |
| hook 顺序、权限、reset 清理、env 隔离 | ✅ `test_hook_order_and_mutator_grant` / `test_partial_reset_isolation` |
| 逐物理步修改发生在正确子步（非事后 history） | ✅ `test_force_schedule_per_substep`（xfrc_log 逐子步断言） |
| per-agent 终止不误判整场 | ✅ `test_per_agent_termination_not_env_end` |
| 未识别插件启动即失败 | ✅ `test_host_slow_rejected_and_registry` |
| 模板 | ✅ `device_examples.py` 四形态 |

**遗留缺口（诚实标注）**：

- `contacts` 的插件视角 padded (B,cap) 视图尚未实现 scatter 填充（当前仅 flat + cap 元数据；standup 转换需要时在 M4 补 scatter kernel）。
- DEVICE 插件"零 sync"靠约定而非强制拦截（解释层无法廉价拦截 `tensor.cpu()`）；审计依赖 rollout 期 SyncStats + 计时。
- `LegacyPluginAdapter` 的 `set_episode_seed` 是逐 env 调用共享实例——依赖 per-env 独立 RNG 的旧插件**不能**经此路径忠实复用（docstring 已声明）。
- warp 后端 `dev_reset_rows` 的 per-env options（如逐 env initial_distance）未实现——M4 reset 语义验收时补。
- `keep_history` warp 端仍未实现（M3 不需要）。

## 4. 目标实验插件能力矩阵（standup_4stage_env.yaml）

| 蓝图类 | 状态 | 说明 |
|---|---|---|
| `RandomFallenStatePlugin` | PENDING | M4 原生转换（设备端 reset 姿态注入；逐 env 随机分布须保留语义） |
| `Standup4StageRewarder` | UNSUPPORTED→M4 | 原生 observer 转换目标 |
| `TimeoutPlugin`（框架内置） | NATIVE | `DeviceTimeoutPlugin` |
| VideoRecorder / 渲染 | HOST_SLOW | 低频 host 路径，不进训练 rollout |

## 5. 下一步（M4 输入）

- 以 `standup_4stage_env.yaml` 为首个真实转换对象：RandomFallenStatePlugin（native resetter）、Standup4StageRewarder（native observer）、timeout（native 已有）。
- 逐 env `initial_distance`/姿态 options 接入 `dev_reset_rows`。
- 接触依赖（足力/承重比例）若 rewarder 需要 padded 契约 → 补 contacts padded scatter。
