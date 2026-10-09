# M3 计划：设备端批量运行时、插件体系与兼容层

> 类型：历史 ｜ 取代者：E 系列路线（ROADMAP.md）——M 路线为历史协议

对应 [ROADMAP.md](ROADMAP.md) §6。M2 已确认 mujoco-warp 是唯一有加速潜力的后端（308K env-substeps/s @B=8192，fixture 在 FP32 容差下通过），本文定义 M3 的架构与执行顺序。前置文档：[discuss.md](discuss.md) §4.5/§6.3、`batch_plugin.py`/`batch_context.py`/`backend.py`（numpy 契约原型）。

## 1. 架构决策（已确认）

**用户确认的切法**：Batch 插件的接口数据定义在设备端；原生插件直接操作设备数据；旧插件通过显式转换层复用。

三层结构：

```
┌────────────────────────────────────────────────────────────┐
│ 插件层                                                      │
│   BaseDevicePlugin (DEVICE)   │   HostCompatAdapter (HOST)  │
│   读 torch 视图 / wp kernel   │   惰性 numpy 视图，每步缓存  │
├────────────────────────────────────────────────────────────┤
│ DeviceBatchState — 设备数据平面契约（backend 中立）          │
│   sim / episode / io / plugin_pool 四个命名空间，SoA 张量    │
├────────────────────────────────────────────────────────────┤
│ 后端绑定：warp_simulator.py（首实现）│ 未来可换其他后端       │
└────────────────────────────────────────────────────────────┘
```

关键决策：

1. **契约类型用 `torch.Tensor`，不暴露 `wp.array` / warp packed 布局**。策略是 PyTorch；wp↔torch 零拷贝（dlpack）。warp 的 contact flat-packed + worldid 布局由后端绑定层归一化为 padded (B, cap) 后再进契约——M2 的 `_contacts_padded` 已有此转换。
2. **保留 numpy 契约作为 HOST 平面**，不删除 `IBatchDataAccessor`——它就是转换层要喂给旧插件的东西。
3. **逐物理子步修改走 schedule 上传，不走逐子步 hook**。`xfrc_schedule: (B, n_steps, nbody, 6)` 一次上传、物理循环逐子步消费。语义诚实（真发生在对应子步），零中途同步。需要逐子步分支判断的插件只能写 wp kernel 或显式降级。
4. **执行平面显式声明**：每个插件声明 `plane ∈ {DEVICE, HOST, HOST_SLOW}`。HOST_SLOW（如逐子步 host 回调）只允许调试/离线路径，训练 rollout 检测到即拒绝启动。

## 2. 设备数据平面契约（`device_state.py`）

`DeviceBatchState` = 分组命名空间的 SoA 集合，全部第一维为 (B,)：

| 命名空间 | 字段 | 可写者 |
|---|---|---|
| `sim` | `qpos, qvel, act_target, xfrc_pending, contacts(padded), xpos/xquat/cvel/xipos 视图` | sim 后端；插件经 mutator 方法 |
| `episode` | `episode_steps(i32), active_mask(bool), terminated_flag, term_reason(i8), reset_request, time(f32)` | runtime |
| `io` | `actions(B,nu), obs_a/obs_b(B,96), reward(B,), reward_channels(B,C)` | runtime+reward unit |
| `plugin_pool` | `declare_state(name, shape, dtype, init)` 分配的持久张量，按 env 行独立 | 各插件（命名空间隔离） |
| `rng` | `seed_offsets (B,) i64 + step_counter` — 设备端确定性派生（hash(seed, step, salt)），不设全局 generator 状态 | runtime 分配 |

约束：

- 插件**不直接持有** `sim.qpos` 等原始字段指针做原地写；写操作必须通过带权限的 mutator 方法（`set_core_state_rows`、`add_ext_force`、`upload_force_schedule`），由后端保证写入落在正确时机。
- `reset_request` 由插件置位，runtime 在 action-step 边界消费（device-side scatter reset）。
- 插件持久状态的 reset 语义：`on_envs_reset(env_ids)` 回调 + 默认行清零约定；不复用"下次 hook 时惰性发现"这类隐式语义。

## 3. 生命周期与权限表

沿用 `batch_plugin.py` 的 hook 序列（已按块边界设计，与子步融合兼容），补齐权限：

| Hook | 时机 | DEVICE 权限 | HOST 适配行为 |
|---|---|---|---|
| on_attach | attach 时 | declare_state/预编译 kernel | 同 |
| set_episode_seeds | reset 前 | 更新 rng salt | 同（host RNG 由插件自持） |
| on_pre_episode | reset 后 | rw: plugin_pool | 物化一次 ctx |
| on_pre_action_step | action 入队后 | rw: io.actions | 可改 actions（写回 device） |
| on_pre_batch_step | 子步块前 | rw: xfrc_pending / schedule | apply_external_force 写回 |
| on_post_batch_step | 子步块后 | rw: 受限投影接口 | set_core_state 写回 |
| on_post_action_step | 每步 | ro；可置 reset_request / 写 reward | 物化一次 ctx（多插件共享缓存） |
| on_post_episode | env 终止后 | ro | 物化 + 终止列表 |
| on_detach | detach | 释放资源 | 同 |

同步纪律：DEVICE 插件在一个 action step 内**不得触发** host↔device 传输；runtime 在 strict 模式下对 DEVICE 插件包 sync 计数器（见 W4），违规即报错。

## 4. 工作包与执行顺序

### W1：设备数据平面 + warp 绑定（`device_state.py`，改动 `warp_simulator.py`）

1. `DeviceBatchState` dataclass + `declare_state` 池；字段布局冻结为文档。
2. warp 后端绑定：`mjw.Data` 字段 → torch 视图（dlpack）；contact flat-packed → padded (B,cap) 视图（复用 `_contacts_padded` 但改为零拷贝/设备内 gather）。
3. mutator 实现：`set_core_state_rows(env_ids, fields)`、`add_ext_force(body, f, t)`、`upload_force_schedule(B,n_steps,nbody,6)`、`set_action_rows`。
4. **设备端观测计算**：把 `_get_robot_view_batch` 从 host 快照路径重写为 torch ops 读 `sim` 视图，输出直接写 `io.obs_a/obs_b`——这是把观测也迁入热路径的关键步骤（当前 warp 侧观测是 host 快照，验证可用、生产不行）。

### W2：`BatchRuntime` + 原生插件基类（`device_runtime.py`、`device_plugin.py`）

5. `BaseDevicePlugin`：plane/required_fields/mutator 需求声明；hook 签名收 `DeviceCtx`（state 视图 + plugin_pool 句柄 + episode 簿记）。
6. `BatchRuntime`：reset（全量/部分）、step 驱动循环、hook 调度（priority）、termination 消费、terminated env 的 device-side 部分 reset、observer dispatcher 的批量化（dispatcher 本身也在设备平面跑）。
7. per-env 终止语义：`reset_request`/`term_reason` 为 per-env 行；KO/timeout 只影响对应行；测试锁定"per-agent 终止不误判整场结束"。

### W3：HOST 转换层（`host_compat.py`）

8. `HostCompatAccessor`：实现现有 `IBatchDataAccessor`（numpy），字段访问惰性 `.cpu().numpy()`，**同一 action step 内缓存**（多插件共享一次物化）；`SyncStats` 计数每次传输。
9. `HostCompatMutator`：`set_core_state`/`apply_external_force`/`set_action` 反向 np→device 写回，计入 sync。
10. `HostCompatPluginAdapter(old_plugin)`：把 `BasePlugin`/`BaseObserverPlugin` 实例挂进新 runtime，hook 签名适配 + plane=HOST 标记；输出中如实标注 `mode=host_compat, syncs_per_step=N`。

### W4：能力注册表 + 测试

11. `capability_registry.py`：旧插件类 → {native_impl | compat_adapter | pending | unsupported} 显式映射表；blueprint 解析时逐类查表，未注册即启动失败（不猜测映射）。
12. **轻量测试后端** `FakeBatchBackend`：numpy/torch-cpu 实现的 `DeviceBatchState` 契约，不依赖 warp 编译——生命周期/hook 顺序/权限/部分 reset/环境隔离测试全部在它上面跑。
13. warp 冒烟测试：真实后端跑通 reset→多步→部分 reset→终止消费。
14. 模板：stateless observer / stateful plugin / 子步 schedule 施力 / reset 四个最小示例（带测试）。

### W5：文档与移交 M4

15. `M3_RESULTS.md`：能力矩阵（目标实验插件逐个标注 plane 与状态）、性能纪律审计结果（DEVICE 路径 sync=0 的证据）、未覆盖项。
16. standup 目标插件初步归类（不转换，M4 做）：`RandomFallenStatePlugin`→native（reset 注入，DEVICE）；`Standup4StageRewarder`→native observer；timeout→内置 `BatchTimeoutPlugin`；`get_broadcastview_image`/VideoRecorder→HOST_SLOW（低频、允许）。

## 5. 放行标准

- FakeBatchBackend 上：hook 顺序、mutator 授予/回收、部分 reset 隔离、per-env 终止、插件状态清理有自动化测试并全通过。
- warp 后端冒烟：DEVICE 插件完整 episode 无 host sync（计数器断言）；HOST 插件同 episode 结果一致（设备原生 timeout vs host 兼容 timeout 输出比对）。
- schedule 施力在**正确子步**生效（对比 fixture 级的单env验证，非事后 history）。
- 能力注册表覆盖 standup blueprint 全部类；未知类拒绝启动有测试。
- 观测设备端实现输出与 M2 已验证的 host 路径在 FP32 容差内一致（复用 fixture 比较器）。

## 6. 明确不做

- 不转换具体任务插件为 native（M4 做；M3 只给框架+注册表+模板）。
- 不接 rollout/PPO（M5）。第一个 collector 只做同步定长模式。
- 不做异步 episode 调度、不做动态 batch resize。
- 不做渲染（VideoRecorder 永远走 HOST_SLOW 路径）。
- 不为"看起来方便"向插件暴露 `wp.array` 或 warp packed 布局。

## 7. 暂停条件

- warp↔torch dlpack 互操作在 stream 同步上出现无法可靠解决的竞态；
- 设备端观测实现无法复用 M2 已验证公式（数值系统性偏移，非 fp32 噪声级）；
- contact padded 化的设备内 gather 成本大到吃掉 warp 优势（M2 已测 packed→padded 转换，预计很小，实测为准）。
