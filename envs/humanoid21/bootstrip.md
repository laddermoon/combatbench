接下来我要重构Humanoid21的实现。 

首先要先有设计，面向设计来实现。
最基本的设计有几个方面：
1. 各个方法的输入输出应该定义成什么样，数据的形态和含义。
  get_static_data
  get_core_state
  等
  我觉得不应该直接暴露Mujoco的内部数据，而是应该有一定的设计。
  目前的一个想法是将数据进行归一化。 21个控制关节的POS进行归一化输出。 
  每个控制关节都有上下限位，取限位的中间值做为基准，将位置归一化到-1，1
  get_derived_data 输出什么，以及什么形式输出都先规划好，再用Mujoco的数据进行实现
2. 控制模式的设计
  目前考虑输入归一化POS作为动作。
3. 性能指标
  机器人的动作跟随特性，响应特性等。
4. 如何测试
  通过什么样的测试验证实现是可用的。

如何利用现有资源：
1. 大部分的实现有一定的参考价值，并不需要完全抛弃。
2. 但是不要被现在的实现束缚。

---

# Humanoid21 重构设计方案 (Refactoring Design)

## 1. 核心理念与痛点纠正
旧版 `get_core_state` 直接返回了完整的 `data.qpos` 和 `data.qvel`，这不仅包含了双机器人的混合数据，甚至包含了全局绝对坐标（这对大部分局部控制策略来说是无意义的）。
- **屏蔽 MuJoCo 内部全局数组**：策略层绝不应该拿到全局的 `qpos/qvel`。所有状态必须按**单个机器人 (robot_a/robot_b)** 拆分。
- **Core State 定义修正**：`core_state` 应该是一个**能唯一决定单体机器人当前姿态和运动学状态的最小物理集合**。
- **局部坐标系优先**：除了必要的世界参考系（如高度、朝向基准），速度、角速度等均应转换到机器人的**根节点局部坐标系**下。
- **全局归一化**：有明确物理限位的控制量和观测特征，默认归一化到 `[-1, 1]`。

---

## 2. 数据接口设计 (Data Interface)

重构后的底层接口应按照 `robot_a` 和 `robot_b` 分别提供以下视图。所有数组均为 `np.ndarray`：

### 2.1 静态属性 (`get_static_info`)
描述机器人本身的固定参数，在整个 episode 中不会改变。
- **返回类型**: `Dict[str, Any]` (按 robot 返回)
- **包含内容**:
  - `dof_names` (List[str]): 21 个受控自由度的名称列表。
  - `body_names` (List[str]): 躯干部位名称列表。
  - `joint_limits` (ndarray, shape=(21, 2)): 各受控关节的真实物理限位 `[min, max]` (rad)。

### 2.2 核心状态 (`get_core_state`)
**定义**：能唯一决定机器人在空间中“怎么摆放、怎么运动”的最小数据集。注意，这里以 `torso`（实际物理模型中的根节点，带 freejoint）为基准。
- **返回类型**: `Dict[str, np.ndarray]` (按 robot 返回)
- **包含内容**:
  - `root_pos` (3,): Torso 根节点的绝对世界坐标 `(x, y, z)`。
  - `root_rot` (4,): Torso 根节点的绝对姿态四元数。
  - `root_vel_local` (3,): Torso 在**自身局部坐标系**下的线速度（对移动策略至关重要）。
  - `root_angular_vel_local` (3,): Torso 在**自身局部坐标系**下的角速度。
  - `joint_pos_norm` (21,): **归一化到 `[-1, 1]` 的关节位置**。
    - 计算公式: `(qpos - reference) / scale`
  - `joint_vel_norm` (21,): **归一化关节角速度**。
    - 计算公式: `qvel / scale`
    - 物理含义: 当前速度每秒能跨越几个“半量程”。它恰好是 `joint_pos_norm` 对时间的导数，与位置特征在数学上完美自洽。

### 2.3 派生数据 (`get_derived_state`)
**定义**：面向机器学习特征工程、碰撞检测、奖励计算的丰富派生数据。
- **返回类型**: `Dict[str, Any]` (包含全局对抗信息与各机器人的单边视角信息)
- **包含内容**:
  - **全局对抗信息 (Shared / Global)**
    - `torso_distance` (1,): 两个机器人根关节（Torso）之间的欧氏距离。
    - `combat_contacts` (List[Dict]): 双方之间的接触列表及接触受力（例如 `{'body_a': 'head', 'body_b': 'torso', 'force': 150.0}`）。**明确要求：只记录双方机器人之间的物理碰撞，排除与自身的接触。**
  - **单边视角信息 (分别放置在 `robot_a` 和 `robot_b` 键下)**
    - `uprightness` (1,): 直立度。由 Torso 局部 z 轴与世界 z 轴的内积计算（1 表示完全直立，<0 表示倒地）。
    - `feet_forces` (2,): 左、右脚与地面的接触受力大小（用于判断着地和发力支撑）。
    - `opponent_in_local` (Dict): **对手在当前机器人局部坐标系下的完整运动学状态**（对策略博弈感知极度重要）：
      - `pos` (3,): 对手 Torso 在自己局部坐标系下的位置。
      - `rot` (4,): 对手 Torso 相对于自己的局部姿态四元数。
      - `vel` (3,): 对手 Torso 在自己局部坐标系下的线速度。
      - `angular_vel` (3,): 对手 Torso 在自己局部坐标系下的角速度。

---

## 3. 控制模式设计 (Control Mode)

继续采用**归一化目标位置 (Normalized Position Control)**，但彻底隔离底层 `ctrl` 数组的拼接。

### 3.1 接口定义
- `set_action(robot_id: str, action: np.ndarray)`
  - `action` 形状为 `(21,)`，取值范围 `[-1, 1]`。

### 3.2 内部执行逻辑
1. **反归一化**: `Target_rad = action * scale + reference`
2. **PD 计算**: `Torque = KP * (Target_rad - current_qpos) - KD * current_qvel`
3. **安全限幅**: `Ctrl = clip(Torque / Gear, ctrl_range_min, ctrl_range_max)`
4. **底层写入**: 将安全扭矩写入 MuJoCo 中该 robot 对应的 `data.ctrl` 分片中。

### 3.3 参数化
- `KP` (默认 50) 和 `KD` (默认 5) 在初始化时必须可配，且应支持按关节数组配置（例如腿部刚度高，手臂刚度低）。

---

## 4. 性能评估指标 (Performance Metrics)

1. **跟踪误差 (Tracking Error)**:
   - `mean(abs(Target_rad - qpos))`，评估 PD 刚度是否足以驱动当前质量的模型。
2. **响应延迟 (Response Latency)**:
   - 给予阶跃信号（如 0 到 1），测量实际达到 90% 所需的仿真步数。
3. **控制努力 (Control Effort)**:
   - `mean(abs(Torque))`，评估高 KP/KD 是否产生高频震荡或无效力矩消耗。
4. **稳定性边界 (Stability Margin)**:
   - 无外力干扰下，维持 `uprightness > 0.8` 能承受的最大单侧关节指令突变。

---

## 5. 测试验证方案 (Testing Strategy)

### 5.1 单元测试 (Unit Tests)
- **状态纯净度测试**: 改变 robot_b 的状态，断言 robot_a 的 `get_core_state` 的任何字段（包括局部速度、局部方向）都不受影响。
- **归一化满量程测试**: 手动设置底层 `qpos` 为上限，断言 `joint_pos_norm` 全为 `1.0`。
- **控制映射测试**: 输入 `action = 0`，推算预期力矩与实际写入 `data.ctrl` 的值是否一致。

### 5.2 物理级集成测试 (Physics Tests)
- **零动中位悬空**: 关重力，`action=0`。断言机器人数步后平稳收敛至 `joint_pos_norm = 0` 且不抖动。
- **全量程抗压测试**: 输入 `action=1` 或交变正弦波，观测模型是否平滑到达极限，无严重超调或穿模。

### 5.3 强化学习系统测试 (RL Sanity Check)
- **单臂举高测试**: 使用 PPO 训练单边机器人“尽可能把右手举高”。若 100k 步内稳定收敛，证明：局部坐标转换正确、归一化观测有效、动作映射符合物理直觉。
