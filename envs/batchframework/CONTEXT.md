# CONTEXT — envs/batchframework

> 类型：指南

> AI-oriented memo。读这个 + DESIGN.md 即可开始干活；细节去
> SEMANTICS.md / MIGRATION_GUIDE.md / PUBLIC_INTERFACE.md。

## 是什么

CPU `envs/framework` 的**设备批量对偶**：同一 env blueprint 在
GPU 上按波（wave）批量 rollout，产出与 CPU collector **同契约的
`Episode`**（v3），`train.py --collector device` 直接消费。
后端默认 mujoco-warp（WarpBackend）；FakeBackend 是契约测试假后端。
目标不是 bit-identical，是同语义 + 可证等价（回放/golden/训练行为）。

**依赖策略：纯 Warp 栈**。requirements.txt 不含 jax/mujoco-mjx；
`mjx_simulator.py`/`probe_e2e_jax.py` 是遗留后备/探针（docstring
已标注），需手工补装才能跑——别把它们当生产入口。

## 心智模型

- **行即 env**：DeviceBatchState 所有张量第一维 (B,)，一行 = 一个
  并行 env。ENDED 行封存冻结直到 collector `reset_rows`——
  **波内复用**是吞吐来源。
- **提出 vs 判定**：`request_termination` 提出即生效即归档；
  env ENDED 由 phase 屏障判——别把两者混成一件事。
- **三个"不静默"**：未注册即拒、不支持即拒、manifest 过期即拒。
  看到 ValueError 先查 REGISTRY/binding，不要想着 fallback；
  畸形 manifest 会 `UserWarning`（不是静默跳过）。
- **蓝图不动**：迁移新单元 = `capability_registry.REGISTRY` 加
  `NATIVE` factory，key 是 blueprint 里的 CPU cls 原文。
- **sim config 三层处置**（MIGRATION_GUIDE §3.3）：binding
  `sim_config_keys` 白名单消费为构造参数 / `config_notes` 显式
  声明忽略 / 否则 `_check_sim_config` 启动即拒——与 audit 的
  unknown 判定同一约定，杜绝静默丢配置。

## 入口

| 干什么 | 去哪 |
|---|---|
| 跑 device collect | `DeviceRollouter`（PUBLIC_INTERFACE §6.3 有片段）/ `probe_e7_baseline.py` |
| 迁移实验 | **MIGRATION_GUIDE.md**（7 步流程）；`migration_audit.py` 出 audit，`migration_manifests/` 存证据 |
| 写插件/observer | 模板 `device_examples.py`；参考 `device_balance.py`/`device_standup.py`/`device_step.py` |
| 理解时序 | `device_runtime.py` 模块 docstring（权威）+ SEMANTICS.md §3 |
| 改随机性 | `device_state.py` RngNamespace/RngView；禁自备 generator |
| 调试 | `debug_capture`（npz 快照）→ `debug_replay`（rerun/replay/cpu_eval） |
| 查支持状态 | `capability_registry.check_support` / `E8_SUPPORT_MATRIX.md` |

## 常见坑（踩过的都登记了）

- `ctx.pstate` 必须先 `declare_state`——partial reset 不清实例属性，
  跨 episode 污染很隐蔽。
- `torch.nonzero`/动态形状/`bool(t)`/`.item()` 在热路径：前者进不了
  CUDA Graph，后者全是 host sync。行号收集用 nonzero+numel 一次同步。
- 覆写 `on_pre/post_phy_step` → 物理图化自动失效（eager）；替代是
  `upload_force_schedule`。
- `ctx.metrics`/`ctx.events`/`ctx.episode_options` 存在但形态不同
  （E9 闭合）：metrics 是声明式张量池（先 `declare_shared`）、events
  是定长数值 journal（emit/since/epoch 游标）、options 仅白名单键
  的 (B,) 行快照——见 BLACKBOARD_DESIGN.md。
- `episode_steps` 步尾增 vs `action_call_index` 步首增——归档/
  边界一律用后者；两值恒等但语义不同。
- manifest 证据 schema 五字段必填（level/passed/input_hash/detail/
  pass_ts）。证据指纹 = `unit_hash` v2（cls + config + **CPU/设备侧
  模块文件 sha256**）——改实现即 stale，须重跑证据再盖章；已知
  残留：跨模块依赖（基类/他模块 import 常量）不覆盖。
- 负例防护有测试：`tests/test_negative_migration.py`（拒迁/stale/
  错误注入定位）——改动注册表/指纹/校验时保持它绿。
- `warp_backend` 已含 `xfrc_applied` vec6 兼容 shim（按目标数组
  自身 shape/dtype 建零）——升 mjw 3.12+ 无需再改此处。
- 测试引用以 `ls tests/` 现状为准。

## 用户意图备忘

- CPU framework 是**行为参照与生产主路径**——设备侧不追求取代它
  的语义地位，追求在同抽象层（collect→Episode）对等替换。
- R 系验收已结算：R1 ✅（E5 链路闭环）；R2 ✅（**等效正确性为准**——
  原 ≥2× 端到端门槛已撤销，不达标事实保留在 M0_BASELINE）；R3 ✅
  （step 实验 L1–L4 + 负例试验闭环，见 R3_RESULTS.md）。
- 挂起待需求（刻意不做，等真实实验需要）：reference/delta_factor、
  policy_eval_mask 接线、facade shim 收口、worker host 瓶颈优化。
- 已知设计内缺口：无渲染（record→CPU replay）、hooks-on 路径
  eager、HOST/HOST_SLOW 适配器休眠。
- 可选升级候选：mjw 3.15（e2e −30%，vec6 shim 已就位；立项清单
  见 PERF_AUDIT §8——注意 pin `warp-lang==1.17`，1.18 需 driver≥13）。
