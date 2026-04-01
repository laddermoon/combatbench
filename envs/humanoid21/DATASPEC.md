# Humanoid21 数据规范 (Data Specification)

## 1. 核心理念 (Core Principles)
- **按主体隔离**: 策略层绝不能获得包含双机器人的混合数据（如全局 `qpos`）。所有方法必须返回 `Dict[str, np.ndarray]`，并在外层按 `robot_a` 和 `robot_b` 区分。
- **局部坐标系优先**: 除非必要（如朝向、高度），否则机器人的速度、角速度及对手的相对位置，一律转换到以自身 `Torso` 为原点的局部坐标系下。
- **全局归一化**: 具有物理限位的观测特征（位置、速度等）默认映射到 `[-1, 1]` 的无量纲区间。

---

## 2. 静态属性 (Static Properties)
**接口**: `get_static_info()`
- **定义**: 描述单个机器人的固定参数集。
- **结构**:
  - `dof_names` (List[str], len=21): 受控自由度名称。
  - `body_names` (List[str]): 躯干与肢体部位名称。
  - `joint_limits` (ndarray, shape=(21, 2)): 受控关节真实的物理限位 `[min, max]` (rad)。

---

## 3. 核心状态 (Core State)
**接口**: `get_core_state()`
- **定义**: 能唯一决定单体机器人在空间中“姿态与本体运动”的最小数据集。基准为物理模型的根节点（带有 `freejoint` 的 Torso）。
- **结构**:
  - **`root_pos`** (3,): Torso 的绝对世界坐标 `(x, y, z)`。
  - **`root_rot`** (4,): Torso 的绝对姿态四元数。
  - **`root_vel_local`** (3,): Torso 在**自身局部坐标系**下的线速度。
  - **`root_angular_vel_local`** (3,): Torso 在**自身局部坐标系**下的角速度。
  - **`joint_pos_norm`** (21,): **归一化关节位置** `[-1, 1]`。
    - 计算公式: `(qpos - reference) / scale`
  - **`joint_vel_norm`** (21,): **归一化关节角速度**。
    - 计算公式: `qvel / scale`
    - 物理含义: 当前速度每秒能跨越的“半量程”数。它是 `joint_pos_norm` 对时间的精确导数。

---

## 4. 派生数据 (Derived State)
**接口**: `get_derived_state()`
- **定义**: 面向机器学习特征工程、碰撞检测和奖励计算的丰富感知数据。
- **结构**: 包含全局对抗信息与单边视角信息。

### 4.1 全局对抗信息 (Shared / Global)
放置在字典的最外层，供环境或中心化评论家(Critic)使用：
- **`torso_distance`** (1,): 双方 Torso 之间的欧氏距离。
- **`combat_contacts`** (List[Dict]): 双方实体之间的物理接触及受力列表。
  - 格式示例: `{'body_a': 'head', 'body_b': 'torso', 'force': 150.0}`
  - **规则**: 仅记录双发机器人之间的碰撞，必须排除机器人与自身的接触。

### 4.2 单边视角信息 (Per-Robot Views)
分别放置在 `robot_a` 和 `robot_b` 的键下，供策略网络感知博弈态势：
- **`uprightness`** (1,): 直立度。由 Torso 局部 z 轴与世界 z 轴的内积计算（1=直立，<0=倒地）。
- **`feet_forces`** (2,): 左脚、右脚与地面的接触受力大小。
- **`facing_dir_local`** (3,): 机器人局部坐标系的“面朝”向量（通常是局部 x 轴或 y 轴）在世界系下的投影。
- **`opponent_in_local`** (Dict[str, ndarray]): 对手 Torso 在**当前机器人局部坐标系**下的完整运动学状态：
  - `pos` (3,): 对手位置（局部相对向量）。
  - `rot` (4,): 对手相对于自身的局部姿态四元数。
  - `vel` (3,): 对手的局部线速度。
  - `angular_vel` (3,): 对手的局部角速度。
