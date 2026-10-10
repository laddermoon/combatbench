# CombatBench 引擎底座 (Framework)

> 类型：指南

本目录包含了 CombatBench 仿真环境的**核心驱动底座**。它被设计为一个极简、纯粹且严格遵循“接口隔离原则（ISP）”与“最小权限原则”的开源扩展框架。

## 🏆 这项成果的价值

本框架是整个项目的能力底座——**环境（#2）、训练（#3）、平台评测（#0）都建立在它的抽象之上**：

- **环境即资产**：`EnvBlueprint`/`ParameterizedEnvBlueprint` 把环境定义序列化成 YAML——环境身份成为一个可搬运的文件，这是提交包、训练复现、回放、批量评测得以成立的前提（见 `BLUEPRINTS.md`）。
- **实验不动世界规则**：观测与奖励全部是只读 observer plugin——换奖励通道 = 换一组 observer，环境代码与计分规则零改动。这是"同一任务多奖励对比""同一实验多策略对比"类实验（成果 #9）的效率来源。
- **后端可换**：`BaseSimulator` 的读写契约让同一套环境语义可以从 CPU MuJoCo 迁移到批量执行后端（`envs/batchframework`，成果 #5）——参考实现与加速路径共享同一环境描述。
- **训练与评测同源**：本地用 `RoundRunner`/`MatchRunner`（`RUNNERS.md`）评估与平台线上评测跑的是同一套环境契约，本地结果对线上排名有预测意义。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| Accessor/Mutator 能力分离 | `IDataAccessor` 永久可读，`IDataMutator` 只在可写生命周期发放——写错误在**挂载期**暴露（`require_mutator` 声明 + 生命周期检查），而不是在万步训练后以幽灵 bug 形式出现 |
| 生命周期权限表 | 可写钩子（重置/动作映射/扰动/约束）与只读钩子（计分/裁决/汇总）在表层写死——奖励计算**结构性不可能**污染物理状态 |
| SimContext 黑板 | 插件之间不直接依赖，经 `metrics`/`events` 通信——插件可独立增删、跨实验复用 |
| observer 只读统一调度 | 观测、奖励、debug view 是同一种抽象（`BaseObserverPlugin`），由 dispatcher 批量驱动——策略视图与世界规则正交演化 |
| 蓝图可序列化 | `file:`/`${DIR}` 机制使"环境+策略"描述可搬运——平台提交、run 复现、批量评估共用同一套描述语言 |

## 🎯 设计理念

1. **底层物理去耦 (Backend Decoupling)**: 不关心你使用的是 MuJoCo、IsaacGym 还是 PyBullet。只要实现 `BaseSimulator` 的五个读写契约，任何物理后端都能无缝接入。
2. **读写权限隔离 (Capability-Based Security)**: 告别在 RL 环境中常见的“状态被意外修改”的幽灵 Bug。引擎会在不同的生命周期精准分发 `IDataAccessor`（只读）和 `IDataMutator`（可写）权限。
3. **世界规则与策略视图分层**: 世界规则继续由 `BasePlugin` 驱动；策略视图由内部 observer dispatcher 统一调度的 `BaseObserverPlugin` 负责。
4. **Runtime First**: `EnvRuntime` 是主要对外接口；它只负责驱动仿真、分发插件、转发双边动作输入，不负责替上层组装策略结果。

---

## 🏗️ 核心抽象

### 1. 物理契约 (`backend.py`)
定义了两个能力接口：
*   **`IDataAccessor`**: 提供对静态配置、核心物理状态 (`core_state`)、派生状态 (`derived_state`) 和传感器的只读访问。
*   **`IDataMutator`**: 提供对核心物理状态和控制动作的写入能力。
*   **`BaseSimulator`**: 物理后端的实现基类，继承了上述两个接口并提供 `physical_step`。

### 2. 黑板与权限管家 (`context.py`)
*   **`SimContext`**: 跨插件流转数据的黑板。
    *   通过 `ctx.accessor` 提供永久的只读访问。
    *   通过 `ctx.mutator` 提供受引擎严格控制的写入能力（未授权时为 `None`）。
    *   提供 `ctx.metrics`（派生指标黑板）和 `ctx.events`（`EventJournal`
        —— append-only 事件日志：episode 内只增不减，清空权归框架，
        "本步事件"由消费者游标差分得出）。
    *   提供 `ctx.request_termination(reason)` 机制用于发起终止提案。
*   **`ReadOnlySimContext`**: 面向 observer plugin 的只读裁剪视图，由内部 dispatcher 统一构造。

### 3. 生命周期插件 (`plugin.py`)
开发者通过继承 `BasePlugin` 并在特定的生命周期挂载逻辑：

| 生命周期 (Hook) | 时机与频率 | 权限状态 (`mutator`) | 典型用例 |
| :--- | :--- | :--- | :--- |
| `on_pre_episode` | 每次 reset 时 | **可用** | 环境重置 (Resetter)、初始状态采样 |
| `on_pre_action_step` | 每个 RL 控制步前 | **可用** | 动作空间映射、动作限幅、控制模式切换 |
| `on_pre_phy_step` | 每个物理细粒度步前 | **可用** | 注入外部扰动力 (Disturbances) |
| `on_post_phy_step` | 每个物理细粒度步后 | **可用** | 状态约束强行拉回 (Constraints) |
| `on_post_action_step` | 每个 RL 控制步结束后 | **不可用** (只读) | 指标统计、犯规/KO判断、计算 Reward |
| `on_post_episode` | episode 终止收尾时 | **不可用** (只读) | 整局日志汇总 |

> **💡 权限双重检查**：如果一个插件希望修改状态，它必须重写 `require_mutator` 属性并返回 `True`，且必须挂载在允许修改的生命周期。

### 4. 统一运行时 (`env_runtime.py`)
*   **`EnvRuntime`**: 对外主接口，负责：
    *   接收 `action_a, action_b`
    *   驱动底层仿真时序
    *   挂载世界插件与 observer plugin
    *   暴露当前共享信息与 observer 输出读取接口
*   **`_RuntimeCore`**: 仅供内部使用的时序执行核心，不作为 framework 公共 API 暴露。

### 5. Observer 插件调度 (`runtime_plugin.py`)
*   **`BaseRuntimeUnit`**: 所有策略侧只读单元的统一基类，显式暴露以下调用时机：
    *   `on_pre_episode(ctx: ReadOnlySimContext)`：在 `EnvRuntime.reset()` 后触发一次
    *   `on_post_action_step(ctx: ReadOnlySimContext)`：在每个 `EnvRuntime.step()` 结束后触发一次
    *   `on_post_episode(ctx: ReadOnlySimContext)`：在 episode 确认终止后触发一次
    *   `on_manual_refresh(ctx: ReadOnlySimContext)`：调用 `runtime.refresh_observers()` 时触发
    *   `get_output()`：返回当前缓存输出
*   **`BaseObserverPlugin`**: 统一的只读 observer plugin 抽象。观测、reward、debug view 都可以实现为这种插件。
*   **`_ObserverDispatcherPlugin`**: 唯一挂到内部 runtime core 的只读调度器，负责：
    *   在关键时机把 `SimContext` 裁剪为 `ReadOnlySimContext`
    *   批量驱动多个 observer plugin
    *   减少 plugin 调用次数和 context 转换次数

---

## 🚀 插件与 RuntimeUnit 开发指南

### 示例 1: 编写一个纯只读的监控插件

如果你的插件只负责看，不负责改，保持 `require_mutator = False`（默认值）。

```python
from framework import BasePlugin, SimContext

class HeightMonitorPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "height_monitor"

    def on_post_phy_step(self, ctx: SimContext):
        # 只能用 accessor 读数据！
        state = ctx.accessor.get_core_state()
        z_height = state['robot_a']['root_position'][2]
        
        # 记录到黑板的 metrics 中供 Reward 插件使用
        ctx.metrics['max_height'] = max(ctx.metrics.get('max_height', 0), z_height)
```

### 示例 2: 编写一个修改物理状态的约束插件

如果你要在物理步后强行拉回状态，必须声明权限。

```python
from framework import BasePlugin, SimContext

class GroundConstraintPlugin(BasePlugin):
    @property
    def require_mutator(self) -> bool:
        return True  # 必须声明我需要写权限

    def on_post_phy_step(self, ctx: SimContext):
        state = ctx.accessor.get_core_state()
        z_height = state['robot_a']['root_position'][2]
        
        if z_height < 0.0:
            # 此时 ctx.mutator 是可用的
            state['robot_a']['root_position'][2] = 0.0
            ctx.mutator.set_core_state(state)
            # 提出犯规警告
            ctx.request_termination("foul_under_ground")
```

---

## 🛠️ 编写 ObserverPlugin 并挂到 EnvRuntime

要将这套引擎对接到策略、对战或训练逻辑，你需要：
1. 提供一个具体的 `BaseSimulator` (如 MuJoCo 版)。
2. 实现一个或多个 `BaseObserverPlugin`。
3. 用 `EnvRuntime` 组装它们。

```python
from framework import EnvRuntime, BaseObserverPlugin, VideoRecorderPlugin


class MyObserverPlugin(BaseObserverPlugin):
    def __init__(self):
        self._output = None

    def on_pre_episode(self, ctx):
        core_state = ctx.accessor.get_core_state()
        self._output = core_state["robot_a"]

    def on_post_action_step(self, ctx):
        core_state = ctx.accessor.get_core_state()
        self._output = core_state["robot_a"]

    def get_output(self):
        return self._output


class MyRewardPlugin(BaseObserverPlugin):
    def __init__(self):
        self._output = 0.0

    def on_pre_episode(self, ctx):
        self._output = 0.0

    def on_post_action_step(self, ctx):
        self._output = -float(ctx.metrics.get("robot_a_clamp_count", 0))

    def get_output(self):
        return self._output

simulator = MujocoSimulator(...)
plugins = [VideoRecorderPlugin(fps=30), HeightMonitorPlugin()]

runtime = EnvRuntime(
    simulator=simulator,
    plugins=plugins,
    observer_plugins={
        "robot_a_obs": MyObserverPlugin(),
        "robot_a_reward": MyRewardPlugin(),
    },
    phy_steps_per_action=10,
    max_steps=1000,
)

runtime.reset()
obs_a, obs_b = runtime.get_observation()
reward = runtime.get_observer_output("robot_a_reward")
over = runtime.is_episode_over()               # True when all agents terminated
reasons = runtime.get_agent_termination()      # {"robot_a": "timeout", ...}
```

## ♻️ 外部适配说明

- `EnvRuntime` 是当前唯一推荐的主入口。
- 如果以后需要 Gymnasium、SB3 或自定义训练器适配，请在 framework 外围单独实现薄适配层。
- framework 内部不再维护旧的 Gym 适配路径。

---

## 🔑 关键点与边界

- **唯一公共入口是 `EnvRuntime`**；`_RuntimeCore` 与 `_ObserverDispatcherPlugin` 是内部实现细节，不是公共 API。
- **写权限是结构性的**：`ctx.mutator` 在只读钩子中为 `None`，此时写入物理状态不可能发生；想用 accessor 读出的 dict 回写绕过该模型属于架构违规。
- **终止走提案制**：`ctx.request_termination(reason)` 只提交提案，仲裁归框架（`TerminationReason` 常量见 `context.py`）。
- **本目录是 CPU 参考路径**：单进程、按回合执行，面向灵活性与正确性。批量/GPU 路径是 `envs/batchframework/`（独立成果，语义对齐、实现不同）。
- **本仓不维护 Gym 适配层**——需要时在 framework 外围自行包装薄适配层。
- 深入契约：`DESIGN.md`（架构规范）· `BLUEPRINTS.md`（YAML schema）· `SEED.md`/`RESET.md`/`EPISODE_OPTIONS.md`（回合控制）· `RUNNERS.md`（评估执行器）。
