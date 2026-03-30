# CombatBench 引擎底座 (Framework)

本目录包含了 CombatBench 仿真环境的**核心驱动底座**。它被设计为一个极简、纯粹且严格遵循“接口隔离原则（ISP）”与“最小权限原则”的开源扩展框架。

## 🎯 设计理念

1. **底层物理去耦 (Backend Decoupling)**: 不关心你使用的是 MuJoCo、IsaacGym 还是 PyBullet。只要实现 `BaseSimulator` 的五个读写契约，任何物理后端都能无缝接入。
2. **读写权限隔离 (Capability-Based Security)**: 告别在 RL 环境中常见的“状态被意外修改”的幽灵 Bug。引擎会在不同的生命周期精准分发 `IDataAccessor`（只读）和 `IDataMutator`（可写）权限。
3. **万物皆插件 (Everything is a Plugin)**: 视频录制、超时判断、RL 奖励计算、约束拉回……所有业务逻辑全部被剥离为 `BasePlugin`。
4. **轻薄的转译层**: `CombatGymEnv` 只有不到 80 行代码，它只是把引擎的时间线翻译成标准的 `gymnasium` API。

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

---

## 🚀 插件开发指南

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

## 🛠️ 将 RL 环境运行起来 (`rl_env.py`)

要将这套引擎对接到标准强化学习算法（如 PPO, SAC），你需要：
1. 提供一个具体的 `BaseSimulator` (如 MuJoCo 版)。
2. 实现一个继承自 `BaseRLAdapter` 的类（定义 Observation/Action 空间及 Reward 计算逻辑）。
3. 用 `CombatGymEnv` 组装它们。

```python
import gymnasium as gym
from framework import CombatGymEnv

# 你的具体实现
simulator = MujocoSimulator(...)
rl_adapter = MyRLAdapter(...)
plugins = [VideoRecorderPlugin(fps=30), HeightMonitorPlugin()]

env = CombatGymEnv(
    simulator=simulator,
    rl_adapter=rl_adapter,
    plugins=plugins,
    phy_steps_per_action=10, # 1个控制步等于10个物理步
    max_steps=1000           # 自动挂载 Timeout 插件
)

obs, info = env.reset()
```
