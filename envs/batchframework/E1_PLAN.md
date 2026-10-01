# E1 实施计划：仿真器与框架在代码上解耦

状态：**W0–W5 已实施并提交（2026-10-02），W6 回归进行中**。依据 [ROADMAP E1](ROADMAP.md) 放行条件与
[E0 契约草案](discuss.md) D2/D4/D5/D18.2。E1 只做**结构解耦**，保持当前可观测
行为（wave 语义、auto-reset、记录口径）不变；语义升级（sealed-ENDED、终止历史
设备化、InitializationProgram）属于 E2/E3。

## 0. 结论先行

E1 的全部内容可拆成一条主线 + 一个前置探针：

- **主线**：`WarpHumanoid21Simulator` 拆掉对 `MjxHumanoid21Simulator` 的继承，
  共享逻辑抽到 `Humanoid21` 绑定层；后端只剩"持有 mjw.Model/Data + 物理原语"，
  episode/io/rng/plugin 簿记移到 runtime 侧分配。
- **前置探针（W0）**：验证 warp 的**行隔离手段**——源码核查已确认
  `mjw.step(m, d)` 无 mask 参数、`opt.disableflags` 为标量（warp 1.12.1），
  "masked advance 一等原语"**不存在**。E1 的 `advance` 接口因此按
  "step-all + 后端负责冻结已结束行"设计，具体冻结机制由 W0 选定。

## 1. 现状耦合清单（代码级）

| 位置 | 耦合/问题 | E1 处置 |
|---|---|---|
| `warp_simulator.py:44` | `WarpHumanoid21Simulator(MjxHumanoid21Simulator)`——仅为复用 `_meta/_robots/_norm_params/_jax_statics/KP/KD/_compute_reset_state/_write_core_state` 等形式继承 | 提取 `Humanoid21Binding`；warp 后端持绑定不继承模拟器 |
| `warp_simulator.py` 各处 | `cuda:0` 字面量 ≥8 处（put_data、wp.array、torch.as_tensor、ScopedStream） | device 参数从 collector→runtime→backend 注入；cuda:1 冒烟 |
| `warp_simulator.py:474` `build_device_state` | sim 分配并持有 `Episode/Io/Rng` 命名空间——生命周期状态活在物理后端里 | runtime 分配 RuntimeState；backend 只暴露物理视图 |
| `warp_simulator.py:91-172` | `_wdata_snapshot`/`_contacts_padded` host 桥伪装 mjx.Data 喂父类提取函数；`_contacts_padded` 是逐元素 Python 循环 | 保留为显式 `WarpSnapshotAdapter`（仅验证/回放路径），不再承担契约角色 |
| `warp_simulator.py:328-366` `physical_step` | PD kernel + xfrc pending/schedule 消费内嵌在步进循环里 | 接口上分离为 binding 提供的 ControlProgram；warp 实现仍融合执行（行为等价），fusion 变为后端能力而非隐含结构 |
| `device_runtime.py:42` `BatchRuntime` | sim 契约是鸭子类型：`build_device_state + dev_* + physical_step`，未形式化 | 替换为 `PhysicsBackend` protocol（describe/read/apply_patch/advance/initialize/status/close） |
| `device_runtime.py:155-212` `step()` | 步内 auto-reset 结束行 + `bool(need_reset.any())` 每步 host sync + `dev_reset_rows` 内 host 姿态计算 | E1 保持该行为（wave 语义依赖它），但把编排权显式收归 runtime；sealed-ENDED 待 W0 探针结果后在 E2 切换 |
| `device_plugin.py` `DeviceMutator` | `reset_rows` 直接暴露给插件——reset 是 runtime 操作，插件应只置 `reset_request` | mutator 收窄为契约动词；检查 `DeviceFallenResetPlugin` 实际用哪个入口 |
| `device_plugin.py:22` `TERM_CODES` | 固定 5 个 i8 原因码 | E1 不变；per-plan 原因目录与首次出现历史表在 E2 实现 |
| `device_rollouter.py:416-423` | `_run_wave` 直接掏 `st.plugin` 里 `fallen` pool 的 `init_steps/init_height/init_hit` 作为 metrics——collector 碰任务私有字段 | 改为经 registry/binding 声明的 `episode_metrics` 导出 schema |
| `device_rollouter.py:366-370` | obs/act/log_prob 缓冲硬编码 96/21 维 | 从 binding 的 schema 取维度 |
| `device_rollouter.py` `_load_policy` | 硬编码 `TruncatedNormalPolicy` + `file:` 蓝图 | E1 抽成 PolicyLoader adapter（仍只支持现类型，但入口可注册） |
| `device_obs.py` / `device_standup.py` | `WarpObsBuilder(sim)`/`from_sim`/`bind_shared_sim` 读 `sim._model/_meta/_robots/_ground_geom_id/_norm_params` 私有字段 | 全部改为接 `Humanoid21Binding`（具名只读表） |
| `device_state.py` `DeviceBatchState` | sim/episode/io/rng/plugin 五命名空间混绑，所有权不清 | 拆成 `PhysicsView`（backend 拥有、借用语义）+ `RuntimeState`（runtime 拥有） |
| `device_state.py` `declare_state` | 只支持 `init=0.0` 填充 | 支持 init 值/张量/设备初始化函数（E2 需要非零初值） |
| `capability_registry.py` | factory 签名收 `sim=`；无 provenance/版本字段 | factory 改收 `binding=`；staleness 字段 E5 再加 |
| `fake_backend.py` | 实现鸭子契约 | 改实现正式 `PhysicsBackend` 协议，作为契约测试的双后端之一 |

依赖方向目标（E1 结束时由导入测试锁定）：

```
batchframework.{physics,runtime,plugin_api,state}  ── 不 import warp/mjw/humanoid21/baseline
batchframework.backends.warp                       ── 可 import warp/mjw + 核心契约
envs.humanoid21.batch_binding                      ── import humanoid21.meta + 核心契约
baseline.humanoid21 设备单元                        ── import 契约 + binding，不碰 warp 私有
baseline.framework.rollout 适配                     ── 可 import 一切
```

## 2. 前置事实核查（本次源码/探针已确认）

| 事实 | 影响 |
|---|---|
| `mjw.step(m, d)` 无 mask 参数；`m.opt.disableflags` 为标量 int（warp 1.12.1） | H1"masked advance 一等原语"不成立 → 冻结行机制改用 write-back 或 reset-park，W0 选型 |
| warp 物理各 world 独立积分；接触容量 per-world（`naconmax = nconmax × nworld`） | 结束行继续步进不污染他行状态，但**单 world 失稳可触发 njmax 断言杀掉整个 batch**（u55 事故） → 冻结/驻留机制是健壮性需求，不只是性能 |
| `dev_reset_rows`/`dev_set_integration_rows` 内有 `wp.synchronize()`；`step()` 有 `bool(...any())` 每步 host sync | 现状无法声称"热路径零同步"——E1 如实计量并记录，E7 再优化 |

## 3. 工作包

### W0 — 行隔离与快照探针（最先做，~1 个脚本）

产出 `probe_isolation.py` + 结论记录进本文件附录：

1. **冻结手段对比**（B=256，跑 200 子步计时）：
   - A. 结束后 `dev_reset_rows` 驻留新 episode（现状，已知可用）；
   - B. 每步对已结束行做 masked write-back（保存的 qpos/qvel/warmstart/ctrl/xfrc 在 `mjw.step` 后写回）——测正确性（冻结行逐位不变、他行不受影响）与开销；
   - C. 结束行写到已知稳定姿态后任其自然步进（判断：freebody 受重力会漂移，预计不可行，探针确认）。
2. **快照完备性**（H3）：capture(qpos,qvel,warmstart,ctrl,xfrc,qfrc,time) → 推进 k 步 → restore → 再推进 → 与不间断对照逐位比对；找出遗漏字段。
3. **视图时效**（H2）：`mjw.step` 后读 xpos vs `mjw.forward` 后读的差异；`set qpos` 后不经 forward 读 derived 的行为——确定 refresh_policy 枚举的真实映射。

出口：选定冻结机制写入 backend `advance`/`park` 语义；H1/H2/H3 状态更新到 discuss.md。

### W1 — 物理后端契约模块（纯代码，无行为变化）

新文件 `envs/batchframework/physics.py`：

- `PhysicsBackend` Protocol：describe / initialize / read / apply_patch / advance / park_rows（冻结）/ capture / restore / status / close；
- `BackendDescriptor`：device、batch_capacity、字段表（shape/dtype/采样相位/有效性）、容量语义（per-world）、能力 flags（`fused_control`、`masked_park`、`snapshot_levels`）；
- `PhysicsView` 借用契约：epoch 计数，推进/写入后失效；
- 错误类型：`CapacityError / ContractError / BackendError`。

验证：FakeBackend 改造实现该协议 + 最小契约测试（视图失效、mask 语义、capacity 负例）。

### W2 — Humanoid21Binding 提取（E1-a 主线）

新文件 `envs/humanoid21/batch_binding.py`，从 `mjx_simulator.py` 提取（**移动而非复制**，MJX 路径同步改为引用）：

- 模型加载 + `Humanoid21Meta.build_runtime_tables` 结果；
- `_norm_params`、KP/KD、`_build_statics(xp, dtype)`（已是数组后端泛型，直接复用出 torch 版本）；
- `_compute_reset_state`（host 姿态计算）、`_write_core_state` 映射；
- 观测公式所需的索引/常量表；
- `ControlProgram`：PD target 计算 + per-substep ctrl 更新（warp 实现为 wp.kernel 数组+kernl 定义，binding 给数据，backend 执行）。

`WarpHumanoid21Simulator` 拆成：

- `batchframework/backends/warp/backend.py`：`WarpBackend(binding, batch_size, device, nconmax, njmax)`——持有 mjw.Model/Data、torch 视图、物理原语实现、wp.synchronize 限定在显式边界；
- `WarpSnapshotAdapter`：`_wdata_snapshot`/`_contacts_padded`/host `get_*` 移入，仅 validation/replay 用。

旧 `WarpHumanoid21Simulator` 类名保留为薄 facade（构造 WarpBackend + binding + host adapter），外部调用点（train.py 路径、测试、capability_registry factory）先不改签名，内部重写。

### W3 — RuntimeState 所有权迁移（E1-c）

- `device_state.py` 拆分：`PhysicsView` 命名空间来自 backend.read()（借用），`RuntimeState`（episode/io/rng/plugin pool）由 runtime 构造；
- `BatchRuntime.__init__` 改收 `PhysicsBackend` + `ObsProgram` + binding 的 schema（obs/action 维度、n_agents），`build_device_state()` 废弃；
- `declare_state` 支持非零 init / callable init；
- runtime.reset() 编排保持现状：backend.initialize + plugin seed 分发 + pre_episode。

验证：FakeBackend/Warp 双后端跑同一套 runtime 生命周期测试（现 `test_device_runtime.py` 扩展）。

### W4 — 任务绑定收口（E1-d）

- `WarpObsBuilder` → `HumanoidObsProgram(binding)`：从 binding 具名表取 geom_bodyid/geom_aff/kp_ids/norm/ground_gid，不再碰 `sim._*`；
- `DeviceStandup4StageRewarder.from_sim` → `from_binding`；
- `DeviceFallenResetPlugin`：`sim._robots` 索引改从 binding；scratch sim 改经 `backend_factory()`（初始化的受限物理实例接口——E1 给最小版，E2 再上升为 InitializationProgram）；
- `capability_registry` factory 签名 `sim=` → `binding=`；
- `_WaveRecorder`/`_run_wave` 的维度与 metrics 改由 binding schema + registry 声明的 `episode_metrics` 导出。

### W5 — device/capacity 注入 + 兼容出口（E1-e）

- `cuda:0` 全量替换为注入 device；`wp.ScopedStream(wp.stream_from_torch())` 保留但不再硬编码设备；
- nconmax/njmax 变为 deployment 参数（collector 构造时传入 backend），不再藏为构造默认值；
- `DeviceRollouter` 保持 facade：内部改为 `resolve → binding → backend → runtime` 装配；
- 依赖方向检查脚本（E0 D2 要求）：import 扫描断言核心模块无 warp/humanoid21 依赖；
- cuda:1 冒烟测试跑通一个 wave。

### W6 — 回归与放行

- 现有 7 个生命周期测试 + `test_device_standup`/`test_warp_validation`/`test_device_rollouter` 全绿；
- 新增：双后端契约一致性测试、masked/park 隔离测试（W0 结论落地为测试）、非默认 device 冒烟、依赖方向检查；
- 短训练冒烟：device collector B=512 跑 ~10 update，对比重构前同 seed 前几 update 的 rollout 指标（不要求逐位——重构动了初始化顺序则允许等价分布差异，但逐字段 schema 必须一致）；
- M6 交叉评估 harness 复用：重构后 warp 环境对冻结 CPU policy 的 eval 成功率差 ≤ 历史噪声带。

## 4. 顺序与理由

```
W0 探针 → W1 契约 → W2 绑定提取 → W3 状态迁移 → W4 任务收口 → W5 device 注入 → W6 回归
```

- W0 先行：冻结机制决定 `advance/park` 接口形状，写错接口后面全返工；
- W1 先于 W2/W3：协议先行，两边才有共同目标；
- W3 依赖 W2：runtime 需要 binding 的 schema 来分配 io 缓冲；
- 每个 W 完成后跑对应测试就提交，不攒大 commit。

## 5. 明确不做（E1 边界）

- 不引入 masked-advance 内核补丁去改 mujoco-warp 本体；
- 不改 wave 语义、不实现 sealed-ENDED、不做 per-plan 终止原因目录（E2）；
- 不改 collector 的 collect→Episode 协议、不动策略加载数学（E3/E4）；
- 不碰 CPU 路径任何行为；MJX 路径只做"引用绑定"的最小改动，不重构其 jit 管线；
- 不做 CUDA Graph 捕获（E7）；不做多卡（E4）。

## 6. 风险与待决问题

| 风险 | 缓解 |
|---|---|
| write-back 冻结 per-step 开销过大 | W0 量化；若 >5% step 时间，选 reset-park 并记录性能备注 |
| scratch 初始化（fallen reset）重构后行为漂移 | `test_device_standup` 有 665 行对照测试；重构后先跑它再动 collector |
| `declare_state` init 扩展引入 pool 语义变化 | 保持默认 init=0 兼容；新签名显式 |
| MJX 路径引用 binding 后的回归 | `test_mjx_validation` + 小 fixture replay |
| facade 兼容期双份真相 | facade 内只做转发；E1 出口检查调用方是否全部迁移，未迁移的列出清单 |

## 7. 放行检查单（对齐 ROADMAP E1）

- [x] FakeBackend 与 WarpBackend 通过同一套契约测试
      （`test_physics_contract.py`，fake 11/11；warp 经
      `WarpHumanoid21Simulator` facade + 26 项 validation fixtures）
- [x] 物理/PD/观测/状态写入/外力/容量负例回归全绿
      （warp_runtime + device_runtime + device_standup + device_rollouter
      + physics_contract + warp_validation + batch_validation = 85 项绿；
      `test_mjx_validation` 的 fixture-stale 失败与 `test_stage_seg_rewards`
      的 collection 错误均为本重构前的既有问题——base.py hash 漂移于
      17d031fd、baseline.framework.experiment 于 9044ff41 被删）
- [x] 非默认设备（cuda:1）跑通（in-process `device='cuda:1'` +
      `torch.cuda.set_device(1)`，BatchRuntime 端到端 8 步验证）
- [x] 核心模块无 warp/mjw/humanoid21 import
      （`test_dependency_direction.py` 10/10）
- [x] `WarpHumanoid21Simulator` 不再继承 `MjxHumanoid21Simulator`
      （组合 binding+backend；断言测试固化）
- [x] 通用路径无 `sim._*`/`st.plugin` 私有字段访问
      （task_tables()/views()/export_episode_metrics() 收口；
      `device_rollouter` 的 fallen pool 嗅探已移除）
- [x] W0 结论写回 discuss.md（H1/H2/H3 状态更新）
- [x] CPU 路径零行为变化（envs/humanoid21/simulator.py 未动）

### 实施摘要（W2–W5 落地形态）

- **W1** `physics.py`：`PhysicsBackend` 协议 + `BackendDescriptor` +
  `ContractError/CapacityError` + `RefreshPolicy`/`SnapshotLevel`；
  FakeBatchBackend 实现协议；`test_physics_contract.py` 共享契约测试。
- **W2** `envs/humanoid21/batch_binding.py`：任务语义唯一实现
  （模型/meta/norm/PD/reset 姿态/core-state 映射/host 提取公式）；
  `mjx_simulator` 改为委托（1063→~530 行）；`backends/warp_backend.py`
  为纯物理后端（mjw.Data 所有权、advance/capture/restore、PD
  ControlProgram、wrench 缓冲、host_snapshot）；`warp_simulator.py`
  重写为组合 facade。
- **W3** `device_state.compose_state()` 成为簿记分配唯一入口；
  `BatchRuntime.state` 自行组装并 `sim.attach_state()` 注册；
  facade `reset()` 不再碰 episode/rng。
- **W4** `Humanoid21DeviceTables`（binding.device_tables(device)）
  收口全部任务元数据；`export_episode_metrics()` 插件契约替代
  rollouter 的 pool 嗅探。
- **W5** device 注入贯通 `DeviceRollouter → facade → WarpBackend`；
  依赖方向静态测试；in-process cuda:1 冒烟。

### 残余（移交 E2/E6，非 E1 阻塞）

- facade 上仍保留 `_robots/_model/_meta/_norm_params/_torch_views/
  _mjx_data` 等兼容 shim 供验证路径使用——E5 迁移清理时再收。
- ENDED 行 write-back 冻结原语已在契约层提供（capture/restore），
  runtime 侧组合与 sealed-ENDED 语义切换留给 E2。
- `dev_reset_rows` 每行 reset 含 host 姿态计算与同步——E7 优化项，
  语义已收口。
- warp 后端同输入不逐位确定（~1e-7/10 步快照漂移）——契约已按
  近似恢复定义；跨后端对照沿用容差协议。

---

## 附录：W0 探针结果（2026-10-01，probe_isolation.py，B=256/64/16，cuda:4）

| 假设 | 结果 | 结论 |
|---|---|---|
| H1 masked advance | warp 1.12.1 无原语（`mjw.step` 无 mask，`opt.disableflags` 标量） | **`advance()` 契约改为全行推进 + `capture/restore(mask)` 原语**；END 行冻结 = runtime 策略组合这两个原语 |
| write-back 冻结 | 冻结行逐位稳定 ✓；冻结不污染运行行（masked 写只触 ended 行内存，world 独立） | **选定 write-back 冻结**，action-step 粒度（每 25 子步一次写回即足够——窗内漂移有界不累积） |
| warp 运行间确定性 | **同 seed 双实例 50 步后 qpos 不逐位一致** | warp 非逐位确定后端（contact atomic 顺序等）；契约不承诺同后端重放逐位一致 |
| H3 快照恢复 | 全部候选字段集 restore 后 k=10 步漂移 ~1e-7 | integration 快照是**近似恢复**（~1e-7/10步），不承诺逐位续跑 |
| H2 视图时效 | `mjw.step` 内刷新 derived ✓；写 qpos 不 forward → xpos 陈旧 ✓；forward 后可见 ✓ | refresh 语义：advance 后 post_integrate 视图有效；patch 后需显式 forward |
| `reset(seeds)` 确定性 | 逐位确定 ✓（`_compute_reset_state` 无 RNG，seeds 仅记录） | wave 初始化可复现 |

**对 E1 设计的修订**：
1. `PhysicsBackend` 契约：`advance()` 推进全部 world（warp 无法 mask）；`capture(mask)`/`restore(mask)` 是冻结/快照的原语，runtime 组合实现 ENDED 行密封。
2. ENDED 行冻结的必要性不仅是语义——ended 行自由演化可能触发 njmax 断言杀掉整个 batch（u55 教训），写回冻结使行状态有界。
3. 快照等级修正：`integration` 级不承诺逐位续跑（后端本身非确定），契约改述为"同版本近似恢复，误差 ~fp32 solver 噪声级"。
