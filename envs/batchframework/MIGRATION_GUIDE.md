# MIGRATION_GUIDE — CPU 实验 → 设备批量框架迁移流程

> 类型：指南

**读者**：AI agent 或工程师，目标是把一个 `envs/framework` 上运行的
实验（env blueprint + 插件 + observer + PPO 实验类）迁移到
`envs/batchframework` 设备路径，产出同契约 `Episode` 供 `train.py
--collector device` 消费。

**核心心智**：
- **蓝图不动**。blueprint 里的 `cls: "module:Class"` 保持 CPU 类名；
  迁移 = 在 `capability_registry.REGISTRY` 给这个 key 注册一个
  `NATIVE` factory，产出设备实现。同一份蓝图可在 `--collector
  cpu|device` 两侧运行（E5 设计）。
- **逐单元迁移**。插件/observer 一个不落：注册表未注册的类 = 启动
  即拒绝，不允许部分迁移跑起来再猜。
- **验证走阶梯**：`unit_replay → wave_contract → e2e_collect →
  train_smoke`，每级证据写 `migration_manifests/<exp>.json`。

---

## 0. 前置判定——先确认可迁，再动手

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 -c "
from envs.batchframework.migration_audit import audit_yaml
rep = audit_yaml('path/to/your_env.yaml')
print(rep.to_markdown())"
```

报告逐 unit 给出：`native`（已有设备实现）/ `pending` / `unsupported` /
`unbound`（simulator 无绑定）。同时检查**结构性阻塞**：

| 阻塞项 | 判定方法 | 现状（见 BATCHFRAMEWORK_AUDIT） |
|---|---|---|
| 插件写 `ctx.metrics` 供 observer 读 | `grep 'ctx\.metrics' <plugin>.py` | **已闭合（E9）**：`ctx.metrics` 共享张量池——见 §1 映射与 BLACKBOARD_DESIGN §1（任意对象→须拆成张量键） |
| 插件 `ctx.events.append` / observer 消费事件 | `grep 'ctx\.events'` | **已闭合（E9）**：`ctx.events.emit/since`——事件须编码为定长数值记录（BLACKBOARD_DESIGN §2） |
| 插件读 `ctx.episode_options` | `grep 'episode_options'` | **已闭合（E9 G4）**：`ctx.episode_options` = {白名单键: (B,) 张量}——白名单之外仍不可见 |
| 插件覆写 `on_pre/post_phy_step` | 逐类检查 | 可走，但**物理图化自动失效**（eager 路径）；优先考虑能否改为 `upload_force_schedule` |
| `get_sensor_data` / 渲染 / broadcastview | `grep 'sensor_data\|broadcastview'` | 无对应物——sensor 需进 obs_builder，渲染走 debug_capture→CPU replay |
| 采样 spec 含 callable ef / reference / delta_factor | 看 Job 生成处 | callable ef：改写为声明式 ef 程序（`ef_programs` 注册表，`obs_threshold` 已支持）即可双后端运行；reference/delta_factor 仍显式拒绝（G6），需扩展 executor ctx 声明 |
| 策略 `policy_class` 非 truncnorm 族 | 导出 payload `policy_class` | 8 族全支持（`_FAMILY_SPECS`）；pre_tanh/未知类显式拒绝——新族接入 = `_FAMILY_SPECS` 一行 + `register_executor` + golden 测试 |

**判定输出**：全部 native → 跳到 §5 直接验证；有 pending → 按 §1–4
逐个转换；命中结构性阻塞 → 先解决阻塞或标记实验不可迁（写
`UNSUPPORTED` 注册项 + 原因，不静默降级）。

---

## 1. ctx API 映射表（改写前先查）

| CPU `SimContext` | 设备 `DeviceCtx` | 注意 |
|---|---|---|
| `ctx.accessor.get_core_state()["qpos"]` | `ctx.sim.qpos` (B,…) 张量 | dict → 类型化命名空间；字段名映射看 `warp_backend.build_sim_namespace` |
| `get_derived_state()["contacts"/"xpos"/...]` | `ctx.sim.contacts_flat` / `ctx.sim.xpos` / `ctx.sim.zup` … | 接触是 dense 固定容量 + `active` mask（不是 `nonzero` 列表） |
| `get_static_data()` | binding `task_tables` | 声明即冻结，构造期注入 |
| `ctx.mutator.set_action(a,b)` | `ctx.mutator.set_action(a,b)` (B,21) | 同语义批量化 |
| `ctx.mutator.apply_external_force(body,f,t)` | `ctx.mutator.add_ext_force(body_id, (B,3) force, torque)` | 作用于下一物理块首子步 |
| 逐子步外力 | `ctx.mutator.upload_force_schedule((B,n,nbody,6))` | **替代子步 hook 的图化兼容路径** |
| `ctx.mutator.reset()` | `ctx.mutator.reset_rows(env_ids)` | 部分行 reset |
| `ctx.request_termination(reason, agent_id)` | `ctx.request_termination(env_ids, reason, agents=[0])` | 提出即生效即归档；`agents=None` 全员 |
| `ctx.agent_terminated["robot_a"]` | `ctx.episode.agent_done[:, 0]` | (B,n_agents) bool |
| `ctx.episode_step` / `ctx.physics_step` | `ep.episode_steps` / `ep.physics_steps` (B,) i32 | 语义已对齐（E8 终止帧契约） |
| `plugin.__dict__` 实例状态 | `ctx.pstate[key]` + `declare_state()` | **必须声明**（partial reset 才会清零行） |
| `np.random` / `set_episode_seed` | `ctx.rng.unit_seed(env_ids, counter)` + `rng_salt` 类属性 | job-keyed splitmix；禁止自备 generator |
| `ctx.metrics[k]`（任意对象） | `ctx.metrics[k]` → (B,*shape) 张量本体（先 `declare_shared`） | 写=原位操作 `[ids]=v`/`.copy_`；结构化数据拆成多键 |
| `ctx.events.append(e)` | `ctx.events.emit(ids, "kind", agent=, value=, aux=)` | 定长记录 [code,agent,value,aux]；消费用 `since(ids, marks)` + epoch 自检 |
| `ctx.episode_options` dict | `ctx.episode_options` → {白名单键: (B,) 张量} | 仅白名单键；非白名单选项进插件 config |

## 2. 写设备单元

模板：`envs/batchframework/device_examples.py`（四个可抄件）；
参考实现：`device_balance.py`、`device_standup.py`。

### 2.1 BaseDevicePlugin 骨架

```python
from envs.batchframework.device_plugin import BaseDevicePlugin, DeviceCtx
import torch

class MyDevicePlugin(BaseDevicePlugin):
    @property
    def name(self): return "my_plugin"
    @property
    def priority(self): return 100        # 同 CPU 语义，大者先跑
    @property
    def rng_salt(self): return 0xABCD     # 用到随机性才声明（得 ctx.rng）
    @property
    def declared_reads(self):             # sim namespace 字段名（白名单校验）
        return ("qpos", "xpos")
    @property
    def declared_writes(self):            # mutator 动词（非 episode 字段！）
        return ("add_ext_force",)

    def declare_state(self, state):
        # per-env 持久张量（partial reset 自动清零行）
        state.declare_state(self.name, "counter", (1,), torch.int32)

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        # reset_env_ids=None 全量；否则仅这些行是新 episode
        ctx.pstate["counter"].zero_()

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        fell = ctx.sim.xpos[:, 0, 2] < 0.5          # (B,) 判定
        ids = fell.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            ctx.request_termination(ids, "fell", agents=[0])
```

要点：
- **全部向量化**：`env_ids = mask.nonzero()` 收行号，操作只落在这些
  行；没有 per-env Python 循环。
- **状态必须 `declare_state`**——插件实例属性在 partial reset 语义下
  会污染下一 episode（设备行是复用的）。
- **reason 是字符串**，registry 分配确定性 code；同 (agent,reason)
  去重归档。
- 终止判定放 `on_post_action_step`（CPU 同位）；子步级提议走
  `on_post_phy_step`（eager-only）或 block 尾屏障。

### 2.2 BaseDeviceObserver 骨架

```python
from envs.batchframework.device_plugin import BaseDeviceObserver
import torch

class MyObserver(BaseDeviceObserver):
    @property
    def name(self): return "my_observer"
    @property
    def output_schema(self):
        # {leaf_key: (dtype, shape_suffix)}——(B,) f32 标量叶写作 (f32, ())
        return {"r_custom": (torch.float32, ())}

    def on_post_action_step(self, ctx):
        # 计算写进自管缓冲；get_output 返回 dict of (B,) 张量
        self._out["r_custom"].copy_(compute(ctx.sim))

    def get_output(self):
        return self._out
```

`output_schema` 是声明式校验（每帧每叶缺失/形状不符即报错）——CPU
`get_output` 的 dict 自由形态在这里被收紧，迁移时把 CPU 返回 dict
的每个 key 写成一条 `{key: (dtype, shape_suffix)}` 条目。

### 2.3 观测/奖励通道

`obs_builder`（如 `device_obs.StandingBalanceObs`）产出 `io.obs_a/b`；
自定义 reward channel = observer 的 `output_schema` 条目，collector
经 `_RecorderAdapter` 写入 Episode `observer_outputs`。

## 3. 注册与接线

### 3.1 capability_registry 加条目

`envs/batchframework/capability_registry.py` 的 `REGISTRY` dict 追加：

```python
"envs.humanoid21.plugins:MyCpuPlugin": CapabilityEntry(
    Capability.NATIVE,
    factory=lambda cfg, **kw: MyDevicePlugin(**_translate_config(cfg)),
    note="DeviceMyPlugin；xx 语义等价说明"),
```

- **key 是 blueprint 里的 CPU cls 原文**——蓝图不改。
- `config_notes`：blueprint config 里**不在**构造签名的键必须逐键
  解释去向（`migration_audit` 会判 unknown 配置失败）。

### 3.2 判定不了的类

如实标 `Capability.UNSUPPORTED`（或 `PENDING`）+ 原因——启动即拒
优于静默降级（框架设计原则）。

### 3.3 binding

humanoid21 已绑（`_Humanoid21WarpBinding`；步态时钟变体
`_GaitClockWarpBinding`）。新任务 = 实现 `DeviceBinding`
（`make_sim`/`io_schema`/`episode_options_keys`/`sim_config_keys`）+
`register_binding(sim_cls, binding)`。

**sim_config 处置语义**：collect 时 `binding.make_sim(B, device,
sim_config=dict(env_bp.simulator.config))`——蓝图 simulator config
原样进绑定。每个键的去向三选一：

| 去向 | 机制 |
|---|---|
| 绑定消费 → 设备 sim 构造参数 | 键列入 `sim_config_keys`；`make_sim` 内 `cfg.pop(k)` 转构造 kwarg（见 `_GaitClockWarpBinding` 消费 `gait_period`） |
| 有解释地忽略 | 键在该 sim 的 `capability_registry` 条目 `config_notes` 中声明去向（如 `debug_torque` = CPU-only 调试打印，设备端无语义） |
| 以上皆非 | `DeviceBinding._check_sim_config` 启动即 `ValueError`——**不许静默丢配置** |

与 audit 的 unknown 判定是同一约定的两层：audit 按 **CPU 类构造
签名**判 `consumed`（静态、离线）；binding 按 `sim_config_keys`
判（运行期、作用于真实 config）。新任务迁移时两侧都要满足。
键同时可作 per-env 覆盖时，另加 `episode_options_keys`——
sim_config 提供静态默认，episode_options 逐行覆盖。

### 3.4 episode_options

`binding.episode_options_keys` 白名单键 → collector 把 per-env 值
广播成张量进 `sim.reset(options)`；同时 runtime 发布
`ctx.episode_options` 行快照（`{key: (B,) 张量}`），hook 内只读
可见（E9 G4）。白名单之外的选项改写进插件 config（静态）。

## 4. 验证阶梯（每级都要留证据）

| 级 | 命令/方式 | 证明 |
|---|---|---|
| `unit_replay` | 手写 Layer A 对照测试（仿 `tests/test_device_standup.py`：伪 accessor 喂相同张量态，CPU 单元 vs 设备单元逐字段对比，容差 ~1e-6） | 单元语义等价 |
| `wave_contract` | `pytest tests/test_wave_contract.py` + 为新单元补 FakeBackend 契约用例 | 波生命周期/种子/导出合法 |
| `e2e_collect` | `probe_e7_baseline.py --env <bp>` 或小型 `DeviceRollouter.collect(jobs)` | 产出契约合法 Episode |
| `train_smoke` | `train.py --experiment <exp> --algo ppo --collector device --collector-batch-size 512 --smoke` | PPO 端到端消费 |

证据写入 `envs/batchframework/migration_manifests/<exp>.json`——
**schema 必填字段**：`{"level","passed","input_hash","detail",
"pass_ts"}`（`pass_ts` ISO 时间戳；缺字段会让 `find_manifest_for`
静默判 manifest 不可用——E8 实际踩过的坑）。`unit_hash` 漂移
（类源码变化）会让旧证据自动失效→`stale`，须重跑。

## 5. 全 native 后的收尾

1. `audit_blueprint` 全绿 + `check_manifest` 无 stale；
2. `pytest tests/` 相关套件；
3. `--collector device --smoke` 通过；
4. 更新 `E8_SUPPORT_MATRIX`/`REGISTRY` note 中的证据指针。

## 6. 陷阱清单

- **`torch.nonzero`/动态形状进不了图路径**：obs/reward 热路径用
  dense + mask（参考 `contact_forces_flat` dense 变体）；nonzero
  只许出现在 host 同步无所谓的冷路径。
- **隐式 host sync**：`bool(tensor)` / `.item()` / `print(tensor)` /
  `if tensor:` 都会同步——热路径禁；判定行号用 `nonzero`+`numel`
  一次同步是可接受模式。
- **`pstate` 忘了 declare_state**：partial reset 后旧值残留 → 跨
  episode 污染，契约测试抓不到（FakeBackend 不做 reset 行清零
  语义时尤其隐蔽）。
- **种子**：禁止 `np.random`/`torch.Generator` 自备——分片重排后
  不复现；一律 `ctx.rng`。
- **`agents=None` 与 `[0]`**：全员终止 vs 单 agent 终止语义不同，
  env ENDED 判定在屏障层。
- **config 键去向**：每个 blueprint config 键必须有接收方或
  `config_notes` 解释——两层都拦：audit 按构造签名判 unknown，
  binding 按 `sim_config_keys` 白名单消费后查 `config_notes`
  （运行期 `ValueError`）。
- **测试引用**：写文档/注释引用测试文件时以 `ls tests/` 现状为准。

## 7. 参考实现索引

| 单元 | 设备实现 | CPU 参照 |
|---|---|---|
| timeout | `device_runtime.DeviceTimeoutPlugin` | `common_plugins.TimeoutPlugin` |
| 失衡终止 | `device_balance.DeviceDualImbalancePlugin` | `humanoid21` balance 插件 |
| 摔倒初始化 | `device_standup` fallen reset | `disturbance_plugins.RandomFallenStatePlugin` |
| 4 段平衡奖励 | `device_standup.DeviceStandup4StageRewarder` | `rewards.standing_balance_4stage` |
| 交叉支撑 | `device_examples.DeviceCrossSupportObserver` | — |
| 子步插件示例 | `device_examples.SubstepProbePlugin` | — |
| 策略 executor（8 truncnorm 族分发） | `policy_executor._FAMILY_SPECS` + `TorchPolicyExecutor` | 训练侧 `baseline/framework/ppo/policies/*_mlp.py`（golden：`tests/test_policy_executor_golden.py`） |
| 步态时钟 simulator（sim_config 消费样板） | `device_step.WarpGaitClockSimulator` + `binding_registry._GaitClockWarpBinding` | `baseline/humanoid21/end2end/gait_clock_simulator.py` |
| 足部接触 observer | `device_step.DeviceFootStateObserver` | `baseline/humanoid21/end2end/foot_state_observer.py` |
