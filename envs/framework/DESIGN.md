# CombatBench Framework 架构设计规范

本框架（CombatBench Framework）旨在为多智能体格斗仿真提供一个**纯粹的物理沙盒**，并为上层强化学习（RL）算法提供**灵活且正交的工程化接口**。

为了避免架构腐化（Bad Smells），所有开发者必须严格遵守以下各组件的职责边界与数据约定。

---

## 一、 核心组件图谱

```mermaid
graph TD
    subgraph 算法与适配层 (Algorithm / Adapter)
        PPO[PPO / GRPO / SAC / IL] --> GymAdapter[可选 Gym Adapter / Wrapper]
        GymAdapter --> PolicyRuntime
    end

    subgraph 策略运行时层 (Policy Runtime Layer)
        PolicyRuntime[PolicyRuntime: 面向策略的主接口]
        ObserverA[Observer A]
        ObserverB[Observer B]
        RewarderA[Rewarder A]
        RewarderB[Rewarder B]

        PolicyRuntime --> ObserverA
        PolicyRuntime --> ObserverB
        PolicyRuntime --> RewarderA
        PolicyRuntime --> RewarderB
    end

    subgraph 物理沙盒层 (Base Sandbox)
        SimEngine[SimEngine: 生命周期与时序调度]
        Plugins[Plugins: 约束、裁决、事件记录]
        BaseSimulator[Simulator: 纯物理引擎后端]

        PolicyRuntime -->|action_a, action_b| SimEngine
        ObserverA -->|读取 IDataAccessor| SimEngine
        ObserverB -->|读取 IDataAccessor| SimEngine
        RewarderA -->|读取 IDataAccessor| SimEngine
        RewarderB -->|读取 IDataAccessor| SimEngine
        SimEngine --> BaseSimulator
        SimEngine --> Plugins
    end
```

---

## 二、 组件定位与职责边界

### 1. BaseSimulator (物理引擎后端)
- **定位**：对底层物理引擎（如 MuJoCo）的极简封装。
- **边界**：
  - ✅ **建议**：只处理关节位置、速度的读写，施加力矩，以及物理步进（`physical_step`）。
  - ✅ **建议**：通过 `IDataAccessor` / `IDataMutator` 暴露标准化的读写接口。
  - ❌ **禁忌**：绝对不要在这里计算得分，不要处理游戏规则，不要知道什么是“一局游戏（Episode）”。

### 2. SimEngine (仿真驱动引擎)
- **定位**：将一个 `Simulator` 与一组世界插件（World Plugins）按固定时序驱动起来。
- **边界**：
  - ✅ **建议**：负责 episode 生命周期、action step / physics step 调度、插件调用顺序、权限授予与回收。
  - ✅ **建议**：对上层暴露稳定的 `SimContext` / `IDataAccessor` 读取入口。
  - ❌ **禁忌**：不要在这里做单视角裁剪、不要在这里做主观奖励、不要实现 Gym 语义。

### 3. BasePlugin (世界插件)
- **定位**：世界规则的裁判与干预器。
- **边界**：
  - ✅ **建议**：处理生命值扣除、判定击倒（KO）、施加防摔倒约束、记录事件（Events）和统计指标（Metrics）。
  - ✅ **建议**：在允许的生命周期中通过 `ctx.mutator` 修改状态或动作，在只读阶段通过 `ctx.accessor` 读取数据。
  - ❌ **禁忌**：不要把“某个算法喜欢什么奖励”写到这里。世界插件只能产出客观事实，不能带有实验特定的价值判断。

### 4. BaseObserver (观测插件)
- **定位**：面向策略侧的只读观测构造器，是一种由 `RuntimeDriverPlugin` 托管的运行时单元。
- **边界**：
  - ✅ **建议**：只暴露一个核心重写入口，由开发者基于 `IDataAccessor` 直接构建某个 agent 的 observation / info view。
  - ✅ **建议**：做 Ego-centric 转换、视角裁剪、特征拼接、调试信息补充。
  - ✅ **建议**：把 `IDataAccessor` 的数据格式视为首要契约，上层逻辑优先围绕这些标准接口开发。
  - ❌ **禁忌**：不要通过 Observer 修改物理状态，不要把它直接挂到 `SimEngine` 的世界插件队列，不要在这里写 reward。

### 5. BaseRewarder (奖励插件)
- **定位**：面向策略侧的只读奖励构造器，是一种由 `RuntimeDriverPlugin` 托管的运行时单元。
- **边界**：
  - ✅ **建议**：只暴露一个核心重写入口，由开发者基于 `IDataAccessor`、`metrics`、`events` 等只读信息为某个 agent 计算 reward。
  - ✅ **建议**：支持稠密奖励、稀疏奖励、课程阶段奖励等实验逻辑。
  - ✅ **建议**：允许直接读取底层标准化数据，而不是被中间层提前裁剪。
  - ❌ **禁忌**：不要通过 Rewarder 修改世界状态，不要将 Rewarder 当成规则插件，不要把胜负裁决逻辑写进 Rewarder。

### 6. RuntimeDriverPlugin (统一运行时驱动插件)
- **定位**：挂载在 `SimEngine` 上的唯一 runtime driver，负责批量驱动多个 Observer / Rewarder。
- **边界**：
  - ✅ **建议**：在 `on_pre_episode`、`on_post_action_step`、`on_post_episode` 统一把 `SimContext` 裁剪为 `ReadOnlySimContext`。
  - ✅ **建议**：同一时机只做一次上下文转换，再批量调用多个 runtime unit，以减少 plugin 调度次数和上下文构造成本。
  - ✅ **建议**：统一托管 A/B 双侧的 observer 和 rewarder，并提供集中取结果的接口。
  - ❌ **禁忌**：不要把策略逻辑拆成多个独立 `BasePlugin` 分别挂到 `SimEngine`；这样会回到高频调度、重复上下文转换的旧问题。

### 7. PolicyRuntime (对外主接口)
- **定位**：框架对开发者提供的主要交互入口。
- **边界**：
  - ✅ **建议**：在 `SimEngine` 之上统一管理双边动作输入、`RuntimeDriverPlugin` 调度、输出结果组装。
  - ✅ **建议**：`step` 直接接收两个参数：一个给 `robot_a`，一个给 `robot_b`。
  - ✅ **建议**：允许为 A/B 分别注入不同的 Observer / Rewarder。
  - ❌ **禁忌**：不要在这里重新实现底层物理逻辑，不要绕过 `SimEngine` 直接控制 `Simulator`。

### 8. Gym Adapter (兼容层)
- **定位**：把 `PolicyRuntime` 或等价 runtime 包装成 Gymnasium 等外部生态需要的接口。
- **边界**：
  - ✅ **建议**：仅作为兼容旧训练代码或外部库的薄适配层存在。
  - ❌ **禁忌**：不要再把 Adapter 当成框架主入口，不要把本应属于 Observer / Rewarder 的核心语义再塞回 Adapter。

---

## 三、 数据格式约定 (The Contracts)

在新架构中，**`IDataAccessor` 是最主要的标准契约**。Observer 与 Rewarder 的实现，应优先围绕它提供的数据格式展开，而不是依赖某个中间层私有结构。

### 1. `IDataAccessor` 的合法读取入口

Observer / Rewarder / RuntimeDriverPlugin 管理的只读运行时单元可以合法依赖以下接口：

```python
accessor.get_static_data()
accessor.get_core_state()
accessor.get_derived_state()
accessor.get_sensor_data()
accessor.get_action()
accessor.get_broadcastview_image()
```

这些接口的职责约定如下：

- **`get_static_data()`**
  - 返回 episode 内通常不变的配置或索引信息。
  - 例如：`robot_info`、body / geom 索引映射、关节索引、arena 常量。

- **`get_core_state()`**
  - 返回最核心、最接近物理引擎的数据。
  - 例如：root pose、joint qpos / qvel、body 的关键状态。

- **`get_derived_state()`**
  - 返回由底层状态进一步加工出的派生信息。
  - 例如：contacts、局部几何关系、碰撞结果、可直接复用的运动学指标。

- **`get_sensor_data()`**
  - 返回传感器层面的读数。
  - 例如：IMU、接触传感器、外力、脚底触地状态。

- **`get_action()`**
  - 返回当前 action step 正在执行的动作。
  - 可用于构建 action smoothness、能耗类 reward 或调试信息。

- **`get_broadcastview_image()`**
  - 返回广播视角图像。
  - 主要用于录像、可视化和调试，不建议作为默认学习输入。

### 2. 世界插件产出的共享黑板

`SimContext` 中的以下字段是跨插件流转的共享客观信息：

```python
ctx.metrics: Dict[str, Any]
ctx.events: List[Any]
ctx.termination_proposals: List[str]
```

- **`metrics`**
  - 必须保存客观的、可解释的标量或结构化统计。
  - 例如：血量、累计伤害、clamp 次数、阶段性计数器。

- **`events`**
  - 保存当前 step 发生的瞬时事件。
  - 例如：命中事件、越界事件、关键状态切换。

- **`termination_proposals`**
  - 保存 episode 终止建议。
  - 例如：`timeout`、`ko`、`foul`。

### 3. PolicyRuntime 的输入约定

`PolicyRuntime.step(...)` 的输入是双边动作，而不是 Gym 风格的单一 action dict 包装层：

```python
action_a = np.ndarray(shape=(21,), dtype=np.float32)
action_b = np.ndarray(shape=(21,), dtype=np.float32)
```

由 `PolicyRuntime` 负责将它们组织为底层 `SimEngine` 所需的 joint action。

### 4. PolicyRuntime 的输出约定

`PolicyRuntime` 对外返回的是**双边、对称、面向策略侧**的结果。一个典型结果至少应包含：

```python
result = {
    "obs": {
        "robot_a": np.ndarray(...),
        "robot_b": np.ndarray(...),
    },
    "reward": {
        "robot_a": float | None,
        "robot_b": float | None,
    },
    "info": {
        "shared": {...},
        "robot_a": {...},
        "robot_b": {...},
    },
    "terminated": bool,
    "truncated": bool,
}
```

- **`obs`** 由对应的 Observer 负责生成。
- **`reward`** 由对应的 Rewarder 负责生成；如果某一侧未配置 Rewarder，可以为 `None`。
- **`info.shared`** 用于放共享的客观信息。
- **`info.robot_a` / `info.robot_b`** 用于放各自视角的补充说明。

### 5. RuntimeDriverPlugin 的最小实现契约

统一 driver 负责托管多个 runtime unit，并在每个关键时机只做一次 `ReadOnlySimContext` 转换。

```python
class BaseRuntimeUnit(ABC):
    def process_data(self, ctx: ReadOnlySimContext) -> None: ...
    def get_output(self) -> Any: ...
```

- **`BaseObserver`** 与 **`BaseRewarder`** 都继承自 `BaseRuntimeUnit`。
- 它们自身不是 `BasePlugin`，而是被 `RuntimeDriverPlugin` 统一托管。
- `RuntimeDriverPlugin` 才是唯一挂到 `SimEngine` 的 runtime 插件。

### 6. 实现原则

- **[首要原则]** 开发者优先关注 `IDataAccessor` 的格式是否标准、稳定、可复用。
- **[Observer 原则]** 观测构建应直接建立在标准读接口之上，而不是依赖脆弱的中间映射层。
- **[Rewarder 原则]** 奖励构建可以读取原始标准数据，但不能反向污染世界状态。
- **[Adapter 原则]** Gym / SB3 / 自定义训练器都只是 `PolicyRuntime` 的外层封装，而不是框架核心。
