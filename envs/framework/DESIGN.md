# CombatBench Framework 架构设计规范

本框架（CombatBench Framework）旨在为多智能体格斗仿真提供一个**纯粹的物理沙盒**，并为上层强化学习（RL）算法提供**灵活且正交的工程化接口**。

为了避免架构腐化（Bad Smells），所有开发者必须严格遵守以下各组件的职责边界与数据约定。

---

## 一、 核心组件图谱

```mermaid
graph TD
    subgraph 强化学习算法层 (Algorithm)
        PPO/SAC/IL -->|单/多维 Box 空间| Wrapper
    end

    subgraph 接口适配层 (Interface Layer)
        Wrapper[Wrappers: 单视角/Self-Play/奖励计算]
        RewardSystem[Reward Function: 基于指标计算 RL 奖励]
    end

    subgraph 物理沙盒层 (Base Sandbox - CombatGymEnv)
        RLAdapter[RL Adapter: 状态 Ego-centric 转换]
        SimEngine[SimEngine: 生命周期与时序调度]
        Plugins[Plugins: 约束、裁决、事件记录]
        BaseSimulator[Simulator: 纯物理引擎后端]
        
        Wrapper -->|双人动作 Dict| CombatGymEnv
        CombatGymEnv -->|客观观测与指标 Dict| Wrapper
        
        CombatGymEnv --> RLAdapter
        CombatGymEnv --> SimEngine
        SimEngine --> BaseSimulator
        SimEngine --> Plugins
    end
```

---

## 二、 组件定位与职责边界

### 1. BaseSimulator (物理引擎后端)
- **定位**：对底层物理引擎（如 MuJoCo）的极简封装。
- **边界**：
  - ✅ **建议**：只处理关节位置、速度的读写，施加力矩，以及物理步进 (`step`)。
  - ❌ **禁忌**：绝对不要在这里计算得分，不要处理游戏规则，不要知道什么是“一局游戏（Episode）”。

### 2. Plugins (规则与裁决)
- **定位**：世界规则的裁判和旁观者。
- **边界**：
  - ✅ **建议**：处理生命值扣除、判定击倒（KO）、施加防摔倒约束、记录事件（Events）和统计指标（Metrics）。
  - ❌ **禁忌**：**绝对不要在这里计算或返回 RL 奖励 (Reward)**。不同实验的奖励权重不同，Plugin 只能输出客观事实（如 `damage_taken: 10`），不能带有主观价值判断。

### 3. RLAdapter (物理域 -> 感知域桥梁)
- **定位**：机器人的“传感器”和“神经中枢”。负责定义 Action 和 Observation 空间。
- **边界**：
  - ✅ **建议**：必须将全局物理坐标转化为**自我中心化（Ego-centric）**的局部坐标系。对于 `robot_a` 和 `robot_b`，其观测数据的结构必须完全对称。
  - ❌ **禁忌**：不要在这里处理视角提取或丢弃另一个机器人的数据。它必须忠实地返回双人的完整状态字典。

### 4. SimEngine / CombatGymEnv (沙盒引擎)
- **定位**：管理整个双人仿真的生命周期。它呈现的是一个**双人、对称、纯客观**的环境。
- **边界**：
  - ✅ **建议**：接收包含双人动作的 Dict，返回包含双人观测、客观事实的 Dict。
  - ❌ **禁忌**：不要在这里写任何关于单智能体（Single Agent）或自我对弈（Self-Play）的逻辑，不要包含任何 `if mode == "single"` 的判断。

### 5. Wrapper (主观滤镜)
- **定位**：将客观的双人沙盒“伪装”成各种 RL 算法需要的形状（如单人闯关、向量化环境）。
- **边界**：
  - ✅ **建议**：处理字典到数组的转换、视角的截取（如提取 `robot_a`）、调用对手的固定策略。
  - ❌ **禁忌**：不要在 Wrapper 中修改物理状态。

### 6. RewardFunction (奖励计算系统)
- **定位**：将底层 Plugin 输出的客观指标（Metrics/Events）转化为标量（Scalar）的 RL 奖励。
- **边界**：
  - ✅ **建议**：作为一个独立的类，通过增量计算（`curr_info` - `prev_info`）生成致密（Dense）或稀疏（Sparse）奖励。由 Wrapper 调用。
  - ❌ **禁忌**：不要依赖物理状态（如 `qpos`），奖励计算只能依赖 `info` 字典中暴露出来的合法指标。

---

## 三、 数据格式约定 (The Contracts)

在沙盒边界（即 `CombatGymEnv` 的输出），数据结构必须严格遵守以下格式。

### 1. 观测 (Observation)
`obs` 必须是一个包含所有参与者状态的字典，且每个参与者的数组必须已经是 Ego-centric 的。
```python
obs = {
    "robot_a": np.ndarray(shape=(127,), dtype=np.float32),  # 以 A 为中心
    "robot_b": np.ndarray(shape=(127,), dtype=np.float32)   # 以 B 为中心
}
```

### 2. 动作 (Action)
`action` 必须是一个包含所有参与者动作指令的字典。
```python
action = {
    "robot_a": np.ndarray(shape=(21,), dtype=np.float32),
    "robot_b": np.ndarray(shape=(21,), dtype=np.float32)
}
```

### 3. 环境信息 (Info)
这是 Wrapper 和 RewardFunction 获取裁判信息的唯一合法途径。

```python
info = {
    # Metrics：必须是累积的（Cumulative）或绝对的标量值
    "metrics": {
        "health_a": 100.0,
        "health_b": 85.0,
        "damage_taken_a": 0.0,
        "damage_taken_b": 15.0,
        "robot_a_clamp_count": 2,
        # 可以添加如 distance_ab 等物理状态指标，供奖励函数使用
    },
    
    # Events：当前 step 发生的瞬时事件列表
    "events": [
        {
            "type": "hit",
            "attacker": "robot_a",
            "defender": "robot_b",
            "part": "head",
            "damage": 15.0
        }
    ],
    
    # Termination Reasons：记录为何结束
    "termination_reasons": ["Timeout", "KO"]
}
```

### 4. 奖励 (Reward)
在底层的 `CombatGymEnv` 中，通常返回 0 或全 0 的字典（因为物理沙盒不负责价值观）：
```python
reward = {
    "robot_a": 0.0,
    "robot_b": 0.0
}
```
**真正的 RL Reward 由外层的 Wrapper 结合 RewardFunction 动态生成，并替换为算法所需的形式（如单个 `float` 或 `(2,)` 的批次张量）。**
