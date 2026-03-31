# CombatBench 引擎底座 (Framework)

本目录包含了 CombatBench 仿真环境的**核心驱动底座**。它被设计为一个极简、纯粹且严格遵循“接口隔离原则（ISP）”与“最小权限原则”的开源扩展框架。

## 🎯 设计理念

1. **底层物理去耦 (Backend Decoupling)**: 不关心你使用的是 MuJoCo、IsaacGym 还是 PyBullet。只要实现 `BaseSimulator` 的五个读写契约，任何物理后端都能无缝接入。
2. **读写权限隔离 (Capability-Based Security)**: 告别在 RL 环境中常见的“状态被意外修改”的幽灵 Bug。引擎会在不同的生命周期精准分发 `IDataAccessor`（只读）和 `IDataMutator`（可写）权限。
3. **世界规则与策略视图分层**: 世界规则继续由 `BasePlugin` 驱动；策略视图由 `RuntimeDriverPlugin` 统一调度的 `BaseObserver` / `BaseRewarder` 负责。
4. **Runtime First**: `PolicyRuntime` 是主要对外接口；Gym 适配层仅保留兼容用途，不再作为框架主入口。

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
*   **`ReadOnlySimContext`**: 面向运行时单元的只读裁剪视图，由 `RuntimeDriverPlugin` 统一构造。

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

### 4. 引擎枢纽 (`engine.py`)
*   **`SimEngine`**: 负责驱动 `Simulator`，管理时间线 (`phy_steps_per_action`)，并在正确的时机分发上下文 `SimContext` 和读写权限给所有挂载的插件。

### 5. 运行时驱动 (`runtime_plugin.py` / `policy_runtime.py`)
*   **`BaseRuntimeUnit`**: 所有策略侧只读单元的统一基类，只保留两个方法：
    *   `process_data(ctx: ReadOnlySimContext)`
    *   `get_output()`
*   **`BaseObserver` / `BaseRewarder`**: 观测构造器与奖励构造器。
*   **`RuntimeDriverPlugin`**: 唯一挂到 `SimEngine` 的 runtime plugin，负责：
    *   在关键时机把 `SimContext` 裁剪为 `ReadOnlySimContext`
    *   批量驱动多个 observer / rewarder
    *   减少 plugin 调用次数和 context 转换次数
*   **`PolicyRuntime`**: 对外主接口，直接接收 `action_a, action_b` 并返回双边结果。

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

## 🛠️ 编写 Observer / Rewarder 并挂到 PolicyRuntime

要将这套引擎对接到策略、对战或训练逻辑，你需要：
1. 提供一个具体的 `BaseSimulator` (如 MuJoCo 版)。
2. 实现一个或多个 `BaseObserver` / `BaseRewarder`。
3. 用 `PolicyRuntime` 组装它们。

```python
from framework import PolicyRuntime, BaseObserver, BaseRewarder, VideoRecorderPlugin


class MyObserver(BaseObserver):
    def __init__(self):
        self._output = None

    def process_data(self, ctx):
        core_state = ctx.accessor.get_core_state()
        self._output = core_state["robot_a"]

    def get_output(self):
        return self._output


class MyRewarder(BaseRewarder):
    def __init__(self):
        self._output = 0.0

    def process_data(self, ctx):
        self._output = -float(ctx.metrics.get("robot_a_clamp_count", 0))

    def get_output(self):
        return self._output

simulator = MujocoSimulator(...)
plugins = [VideoRecorderPlugin(fps=30), HeightMonitorPlugin()]

runtime = PolicyRuntime(
    simulator=simulator,
    plugins=plugins,
    observers={"robot_a": MyObserver()},
    rewarders={"robot_a": MyRewarder()},
    phy_steps_per_action=10,
    max_steps=1000,
)

result = runtime.reset()
obs = result["obs"]
info = result["info"]
```

## ♻️ 外部适配说明

- `PolicyRuntime` 是当前唯一推荐的主入口。
- 如果以后需要 Gymnasium、SB3 或自定义训练器适配，请在 framework 外围单独实现薄适配层。
- framework 内部不再维护旧的 Gym 适配路径。
