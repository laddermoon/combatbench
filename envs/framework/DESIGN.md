# CombatBench Framework 架构设计规范

本框架（CombatBench Framework）旨在为多智能体格斗仿真提供一个**纯粹的物理沙盒**，并为上层强化学习（RL）算法提供**灵活且正交的工程化接口**。

为了避免架构腐化（Bad Smells），所有开发者必须严格遵守以下各组件的职责边界与数据约定。

---

## 一、 核心组件图谱

```mermaid
graph TD
    subgraph 算法与适配层 (Algorithm / Adapter)
        PPO[PPO / GRPO / SAC / IL] --> GymAdapter[可选 Gym Adapter / Wrapper]
        GymAdapter --> EnvRuntime
    end

    subgraph 策略运行时层 (Policy Runtime Layer)
        EnvRuntime[EnvRuntime: 统一 Runtime 主接口]
        ObserverA[ObserverPlugin A]
        ObserverB[ObserverPlugin B]
        ObserverC[ObserverPlugin C]

        EnvRuntime --> ObserverA
        EnvRuntime --> ObserverB
        EnvRuntime --> ObserverC
    end

    subgraph 物理沙盒层 (Base Sandbox)
        RuntimeCore[_RuntimeCore: 内部生命周期与时序调度]
        Plugins[Plugins: 约束、裁决、事件记录]
        BaseSimulator[Simulator: 纯物理引擎后端]

        EnvRuntime -->|action_a, action_b| RuntimeCore
        ObserverA -->|读取 IDataAccessor| RuntimeCore
        ObserverB -->|读取 IDataAccessor| RuntimeCore
        ObserverC -->|读取 IDataAccessor| RuntimeCore
        RuntimeCore --> BaseSimulator
        RuntimeCore --> Plugins
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

### 2. _RuntimeCore (内部仿真驱动核心)
- **定位**：将一个 `Simulator` 与一组世界插件按固定时序驱动起来。
- **边界**：
  - ✅ **建议**：负责 episode 生命周期、action step / physics step 调度、插件调用顺序、权限授予与回收。
  - ✅ **建议**：对上层暴露稳定的 `SimContext` / `IDataAccessor` 读取入口。
  - ❌ **禁忌**：不要把它当成公共 API 暴露，不要在这里做单视角裁剪、不要实现 Gym 语义。

### 3. BasePlugin (世界插件)
- **定位**：世界规则的裁判与干预器。
- **边界**：
  - ✅ **建议**：处理生命值扣除、判定击倒（KO）、施加防摔倒约束、记录事件（Events）和统计指标（Metrics）。
  - ✅ **建议**：在允许的生命周期中通过 `ctx.mutator` 修改状态或动作，在只读阶段通过 `ctx.accessor` 读取数据。
  - ❌ **禁忌**：不要把“某个算法喜欢什么奖励”写到这里。世界插件只能产出客观事实，不能带有实验特定的价值判断。

### 4. BaseObserverPlugin (统一只读观察插件)
- **定位**：面向策略侧的只读输出构造器，是一种由内部 observer dispatcher 托管的运行时单元。
- **边界**：
  - ✅ **建议**：只暴露一个核心重写入口，由开发者基于 `IDataAccessor` 直接构建 observation、reward、debug view 或其它只读输出。
  - ✅ **建议**：做 Ego-centric 转换、视角裁剪、特征拼接、调试信息补充。
  - ✅ **建议**：把 `IDataAccessor` 的数据格式视为首要契约，上层逻辑优先围绕这些标准接口开发。
  - ❌ **禁忌**：不要通过 observer plugin 修改物理状态，不要把它直接挂到世界插件队列。

### 5. _ObserverDispatcherPlugin (内部统一调度器)
- **定位**：挂载在 `_RuntimeCore` 上的唯一只读调度器，负责批量驱动多个 observer plugin。
- **边界**：
  - ✅ **建议**：在 `on_pre_episode`、`on_post_action_step`、`on_post_episode` 统一把 `SimContext` 裁剪为 `ReadOnlySimContext`。
  - ✅ **建议**：同一时机只做一次上下文转换，再批量调用多个 runtime unit，以减少 plugin 调度次数和上下文构造成本。
  - ✅ **建议**：统一托管多种只读输出插件，并提供集中取结果的接口。
  - ❌ **禁忌**：不要把策略逻辑拆成多个独立 `BasePlugin` 分别挂到 `_RuntimeCore`；这样会回到高频调度、重复上下文转换的旧问题。

### 6. EnvRuntime (对外统一 Runtime)
- **定位**：框架对开发者提供的主要交互入口。
- **边界**：
  - ✅ **建议**：在内部 `_RuntimeCore` 之上统一管理双边动作输入、世界插件调度、observer plugin 调度。
  - ✅ **建议**：`step` 直接接收两个参数：一个给 `robot_a`，一个给 `robot_b`。
  - ✅ **建议**：`step` / `reset` 不返回结果；上层通过 observer plugin 输出和共享信息自行组装视图。
  - ❌ **禁忌**：不要在这里重新实现底层物理逻辑，不要绕过内部 `_RuntimeCore` 直接控制 `Simulator`。

### 7. Gym Adapter (兼容层)
- **定位**：把 `EnvRuntime` 或等价 runtime 包装成 Gymnasium 等外部生态需要的接口。
- **边界**：
  - ✅ **建议**：仅作为兼容旧训练代码或外部库的薄适配层存在。
  - ❌ **禁忌**：不要再把 Adapter 当成框架主入口，不要把本应属于 Observer / Rewarder 的核心语义再塞回 Adapter。

---

## 三、 数据格式约定 (The Contracts)

在新架构中，**`IDataAccessor` 是最主要的标准契约**。ObserverPlugin 的实现，应优先围绕它提供的数据格式展开，而不是依赖某个中间层私有结构。

### 1. `IDataAccessor` 的合法读取入口

ObserverPlugin / 内部 dispatcher 管理的只读运行时单元可以合法依赖以下接口：

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

### 3. EnvRuntime 的输入约定

`EnvRuntime.step(...)` 的输入是双边动作，而不是 Gym 风格的单一 action dict 包装层：

```python
action_a = np.ndarray(shape=(21,), dtype=np.float32)
action_b = np.ndarray(shape=(21,), dtype=np.float32)
```

由 `EnvRuntime` 负责将它们组织为底层 joint action。

### 4. EnvRuntime 的输出约定

`EnvRuntime.reset(...)` 与 `EnvRuntime.step(...)` 都不返回值。上层通过 observer plugin 和共享信息接口自行组装视图：

```python
runtime.reset(seed=seed)
obs_a = runtime.get_observer_output("robot_a_obs")
shared_info = runtime.get_shared_info()
terminated, truncated = runtime.get_termination_flags()
```

- **observer output** 由对应的 observer plugin 负责生成。
- **shared info** 用于放共享的客观信息。
- **termination flags** 由 runtime 根据终止原因统一解析。

### 5. ObserverPlugin 的最小实现契约

统一 dispatcher 负责托管多个只读 runtime unit，并在每个关键时机只做一次 `ReadOnlySimContext` 转换。

```python
class BaseRuntimeUnit(ABC):
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None: ...
    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None: ...
    def on_post_episode(self, ctx: ReadOnlySimContext) -> None: ...
    def on_manual_refresh(self, ctx: ReadOnlySimContext) -> None: ...
    def get_output(self) -> Any: ...
```

- **`BaseObserverPlugin`** 继承自 `BaseRuntimeUnit`。
- 它自身不是 `BasePlugin`，而是被内部 dispatcher 统一托管。
- dispatcher 才是唯一挂到内部 runtime core 的只读插件。
- dispatcher 的时机映射是固定的：观察者的 `on_pre_episode` / `on_post_action_step` / `on_post_episode` 分别对应内部 runtime core 的 `on_pre_episode` / `on_post_action_step` / `on_post_episode` 钩子；手动刷新走 `on_manual_refresh`。
- 为兼容旧代码，`process_data(ctx)` 仍可作为后备入口存在，但新实现不应再依赖这个模糊命名。

### 6. 实现原则

- **[首要原则]** 开发者优先关注 `IDataAccessor` 的格式是否标准、稳定、可复用。
- **[ObserverPlugin 原则]** 观测、奖励和其它只读输出都应直接建立在标准读接口之上，而不是依赖脆弱的中间映射层。
- **[只读原则]** observer plugin 可以读取原始标准数据，但不能反向污染世界状态。
- **[Adapter 原则]** Gym / SB3 / 自定义训练器都只是 `EnvRuntime` 的外层封装，而不是框架核心。
