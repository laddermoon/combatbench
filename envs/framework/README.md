# CombatBench 引擎底座 (Framework)

本目录包含了 CombatBench 仿真环境的**核心驱动底座**。它被设计为一个极简、纯粹且严格遵循“接口隔离原则（ISP）”与“最小权限原则”的开源扩展框架。

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
    *   提供 `ctx.metrics` 和 `ctx.events` 用于存放派生指标。
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
    *   兼容旧式 `process_data(ctx)` 实现，但新代码推荐直接实现显式生命周期钩子

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
obs = runtime.get_observer_output("robot_a_obs")
reward = runtime.get_observer_output("robot_a_reward")
info = runtime.get_shared_info()
```

## ♻️ 外部适配说明

- `EnvRuntime` 是当前唯一推荐的主入口。
- 如果以后需要 Gymnasium、SB3 或自定义训练器适配，请在 framework 外围单独实现薄适配层。
- framework 内部不再维护旧的 Gym 适配路径。
