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

## 1. 核心理念
- **彻底屏蔽 MuJoCo 内部表示**：策略层只看到“具有物理意义和特定取值范围”的特征，不知道 `qpos`、`qvel`、`id` 等底层细节。
- **全局归一化 (Normalized By Default)**：凡是有明确边界的物理量（如关节位置、控制信号），在暴露给策略或从策略接收时，一律归一化到 `[-1, 1]`。
- **面向强化学习优化**：数据结构应当是扁平化、定长的 `numpy.ndarray` 或明确的 `Dict[str, ndarray]`，避免深层嵌套和变长列表。

---

## 2. 数据接口设计 (Data Interface)

重构后的 `HumanoidRobot` 应该提供以下几类清晰的数据视图：

### 2.1 静态属性 (`get_static_info`)
描述机器人本身的固定参数，在整个 episode 中不会改变。
- **返回类型**: `Dict[str, Any]`
- **包含内容**:
  - `dof_names` (List[str]): 21 个受控自由度的名称列表，保证顺序一致性。
  - `body_names` (List[str]): 躯干部位名称列表（用于识别受击部位）。
  - `joint_limits` (ndarray, shape=(21, 2)): 各关节真实的物理限位 `[min, max]` (rad)。

### 2.2 核心状态 (`get_core_state`)
描述机器人本体当前的运动学和动力学最小完备集。全部使用**机器人局部坐标系**（除根节点绝对位置外）。
- **返回类型**: `Dict[str, np.ndarray]`
- **包含内容**:
  - `root_pos` (3,): 骨盆(pelvis)的绝对坐标 `(x, y, z)`。
  - `root_rot` (4,): 骨盆的绝对姿态四元数 `(w, x, y, z)` 或 `(x, y, z, w)`。
  - `root_vel_local` (3,): 骨盆在**自身局部坐标系**下的线速度。
  - `root_angular_vel_local` (3,): 骨盆在**自身局部坐标系**下的角速度。
  - `joint_pos_norm` (21,): **归一化到 `[-1, 1]` 的关节位置**。
    - 计算公式: `(qpos - reference) / scale`
  - `joint_vel` (21,): 关节真实角速度（rad/s）。（考虑后续是否需要根据 max_vel 也做归一化）

### 2.3 派生数据 (`get_derived_data`)
由核心状态衍生出的、对策略决策（如对打）强相关的特征。
- **返回类型**: `Dict[str, np.ndarray]`
- **包含内容**:
  - `head_pos_global` (3,): 头部的绝对坐标（用于计算距离）。
  - `com_pos_global` (3,): 质心绝对坐标。
  - `facing_dir` (3,): 躯干当前的绝对朝向向量（例如骨盆的局部 x 轴在世界坐标系下的投影）。
  - `uprightness` (1,): 直立度。可以通过躯干局部 z 轴与世界 z 轴的内积计算（1 表示完全直立，<0 表示倒地）。
  - `contact_forces` (N,): 各个关键部位（如双脚、双拳）受到的接触力大小或向量。

---

## 3. 控制模式设计 (Control Mode)

如 `CONTROLSPEC.md` 所述，继续采用**归一化目标位置 (Normalized Position Control)** 模式，内部走 PD 控制。

### 3.1 接口定义
- `set_action(action: np.ndarray)`
  - `action` 形状为 `(21,)`，取值范围 `[-1, 1]`。
  - 含义：期望关节到达的归一化目标位置。

### 3.2 内部执行逻辑
1. **反归一化**: `Target_rad = action * scale + reference`
2. **PD 计算**: `Torque = KP * (Target_rad - qpos) - KD * qvel`
3. **输出限制**: `Ctrl = clip(Torque / Gear, ctrl_range_min, ctrl_range_max)`
4. **底层写入**: 写入 MuJoCo 的 `data.ctrl`。

### 3.3 参数暴露
- `KP` 和 `KD` 应当作为 `HumanoidRobot` 初始化时的可配参数（可支持 array 格式，允许不同关节有不同的刚度）。

---

## 4. 性能评估指标 (Performance Metrics)

为了验证控制逻辑和动力学特性的合理性，我们需要定义并在调试时收集以下指标：

1. **跟踪误差 (Tracking Error)**:
   - `mean_abs_error = mean(abs(Target_rad - qpos))`
   - 评估底层 PD 控制器的刚度是否足够，以及在动作剧烈时是否会严重滞后。
2. **响应延迟 (Response Latency)**:
   - 给予一个阶跃信号（如从 0 突变到 1），测量关节实际达到 0.9（90%）所需的时间（步数）。
3. **控制努力 (Control Effort)**:
   - `mean_torque = mean(abs(Torque))`
   - 评估机器人是否因为高 KP/KD 而产生高频震荡或消耗过大的不合理力矩。
4. **稳定性边界 (Stability Margin)**:
   - 在无外力干扰下，机器人维持直立（如 `uprightness > 0.8`）所能承受的最大初始关节扰动。

---

## 5. 测试验证方案 (Testing Strategy)

要证明重构后的 Humanoid21 是可用的，必须通过以下层次的测试：

### 5.1 单元测试 (Unit Tests)
- **数据结构一致性**: 断言 `get_core_state()` 返回的字典 keys 和 shape 与设计严格一致。
- **归一化逻辑测试**: 
  - 手动将底层的 `qpos` 设置为上限，断言 `joint_pos_norm` 全为 `1.0`。
  - 手动将底层的 `qpos` 设置为下限，断言 `joint_pos_norm` 全为 `-1.0`。
- **控制逻辑映射测试**:
  - 调用 `set_action(ones)`，推演计算公式，断言写入 `data.ctrl` 的力矩值符合预期的 `KP` 和 `KD` 计算结果。

### 5.2 物理级集成测试 (Physics Integration Tests)
1. **中位悬空测试 (Zero-Action Hang Test)**:
   - 禁用重力或将机器人固定在半空。
   - 输入 `action = 0`。
   - 预期结果：经过短暂几步后，所有关节收敛并静止在 `joint_pos_norm = 0`，`Tracking Error` 趋近于 0。
2. **极限活动范围测试 (Range of Motion Test)**:
   - 输入 `action = 1`。
   - 预期结果：所有关节平滑移动到其物理上限，无严重超调和剧烈震荡。

### 5.3 强化学习系统测试 (RL Sanity Tests)
- **过拟合测试 (Overfit / Sanity Check)**:
   - 挂载一个新的 EnvRuntime，使用单智能体训练一个简单的任务（如“尽可能把右手举高”或“保持直立不要倒”）。
   - 预期结果：使用 PPO/GRPO，能在 100k steps 内明显看到奖励上升并收敛，证明状态观测无缺失、动作映射符合物理直觉。
