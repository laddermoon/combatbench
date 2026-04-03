# Humanoid21 数据规范 (Data Specification)

## 1. 核心理念 (Core Principles)
- **按主体隔离**: 策略层绝不能获得包含双机器人的混合数据（如全局 `qpos`）。所有方法必须返回 `Dict[str, np.ndarray]`，并在外层按 `robot_a` 和 `robot_b` 区分。
- **局部坐标系优先**: 除非必要（如朝向、高度），否则机器人的速度、角速度及对手的相对位置，一律转换到以自身 `Torso` 为原点的局部坐标系下。
- **全局归一化**: 具有物理限位的观测特征（位置、速度等）默认映射到 `[-1, 1]` 的无量纲区间。

---

## 2. 静态属性 (Static Properties)
**接口**: `get_static_data()`
- **定义**: 描述单个机器人的固定参数集。
- **结构**:
  - `dof_names` (List[str], len=21): 受控自由度名称。
  - `body_names` (List[str]): 躯干与肢体部位名称。
  - `joint_limits` (ndarray, shape=(21, 2)): 受控关节真实的物理限位 `[min, max]` (rad)。

---

## 3. 核心状态 (Core State)
**接口**: `get_core_state()`
- **定义**: 能唯一决定单体机器人在空间中"姿态与本体运动"的最小数据集。基准为物理模型的根节点（带有 `freejoint` 的 Torso）。
- **结构**:
  - **`root_pos`** (3,): Torso 的绝对世界坐标 `(x, y, z)`。
  - **`root_rot`** (4,): Torso 的绝对姿态四元数 `[w, x, y, z]`。
  - **`root_vel_local`** (3,): Torso 在**自身局部坐标系**下的线速度。
  - **`root_angular_vel_local`** (3,): Torso 在**自身局部坐标系**下的角速度。
  - **`joint_pos_norm`** (21,): **归一化关节位置** `[-1, 1]`。
    - 计算公式: `(qpos - reference) / scale`
    - `reference` 为关节上下限的中间值，`scale` 为关节总行程的 1/2。
  - **`joint_vel_norm`** (21,): **归一化关节角速度**。
    - 计算公式: `qvel / scale`
    - 物理含义: 当前速度每秒能跨越的"半量程"数。它是 `joint_pos_norm` 对时间的精确导数。

**模块一：本体感知 (42维)**

| 数据项 | 维度 | 说明 |
|---|---|---|
| 归一化关节角度 | 21 | `joint_pos_norm` |
| 归一化关节角速度 | 21 | `joint_vel_norm` |

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
  - **规则**: 仅记录双方机器人之间的碰撞，必须排除机器人与自身的接触。

### 4.2 单边视角信息 (Per-Robot Views)
分别放置在 `robot_a` 和 `robot_b` 的键下，供策略网络感知博弈态势。

#### 4.2.1 模块二：全局状态 (13维)
**接口**: `robot_view['root_state']`

| 数据项 | 维度 | 说明 |
|---|---|---|
| `height` | 1 | Z轴高度，判断是否倒地 |
| `local_orientation` | 6 | 世界坐标四元数 → 局部旋转矩阵（取前两列） |
| `linear_vel` | 3 | 全局坐标系下的线速度 |
| `angular_vel` | 3 | 全局坐标系下的角速度 |

#### 4.2.2 模块三：触觉力反馈 (2维)
**接口**: `robot_view['feet_forces']`

| 数据项 | 维度 | 说明 |
|---|---|---|
| `feet_forces` | 2 | 左脚、右脚与地面的接触受力大小 |

#### 4.2.3 模块四：对手观测 (39维)
**原则**: "以我为中心" + "动静结合"
**坐标系**: 所有对手坐标均转换至自身（Ego）的局部坐标系下。

##### 4.2.3.1 对手基础位姿 (9维)
**接口**: `robot_view['opponent_basic_pose']`

| 数据项 | 维度 | 物理含义 |
|---|---|---|
| `relative_pos` | 3 | 对手根关节 - 自身根关节（局部坐标系） |
| `relative_vel` | 3 | 相对速度（局部坐标系） |
| `face_vector` | 3 | 对手朝向的单位向量在Ego坐标系中，与自身同向为 (1,0,0) |

##### 4.2.3.2 对手关键点位置 (15维)
**接口**: `robot_view['opponent_keypoint_pos']`

| 数据项 | 维度 | 作用 |
|---|---|---|
| `head` | 3 | 头部中心点 |
| `hand_right` | 3 | 右手中心点 |
| `hand_left` | 3 | 左手中心点 |
| `foot_right` | 3 | 右脚中心点 |
| `foot_left` | 3 | 左脚中心点 |

##### 4.2.3.3 对手关键点速度 (15维)
**接口**: `robot_view['opponent_keypoint_vel']`

| 数据项 | 维度 |
|---|---|
| `head` | 3 |
| `hand_right` | 3 |
| `hand_left` | 3 |
| `foot_right` | 3 |
| `foot_left` | 3 |

---

## 5. 观测空间总结 (Observation Space Summary)

**总维度**: 96 维（每个机器人）

| 模块 | 维度 | 接口 | 说明 |
|------|------|------|------|
| 模块一：本体感知 | 42 | `get_core_state()[robot_id]['joint_pos_norm']`<br>`get_core_state()[robot_id]['joint_vel_norm']` | 关节角度和角速度 |
| 模块二：全局状态 | 13 | `get_derived_state()[robot_id]['root_state']` | 高度、朝向、速度 |
| 模块三：触觉力反馈 | 2 | `get_derived_state()[robot_id]['feet_forces']` | 足底受力 |
| 模块四：对手观测 | 39 | `get_derived_state()[robot_id]['opponent_basic_pose']`<br>`get_derived_state()[robot_id]['opponent_keypoint_pos']`<br>`get_derived_state()[robot_id]['opponent_keypoint_vel']` | 对手位姿、关键点 |
| **完整观测** | **96** | `get_derived_state()[robot_id]['observation']` | 所有模块平铺后的完整观测 |

**完整观测获取**:

完整 96 维观测直接包含在 `get_derived_state()[robot_id]['observation']` 中：

```python
derived_state = sim.get_derived_state()

# robot_a 完整观测 (96维) - 直接获取
robot_a_obs = derived_state['robot_a']['observation']  # 96维，包含所有模块
```

如果需要单独访问各模块的数据：

```python
derived_state = sim.get_derived_state()

# 模块一：本体感知 (42维) - 需要从 get_core_state 获取
core_state = sim.get_core_state()
joint_pos_norm = core_state['robot_a']['joint_pos_norm']  # 21维
joint_vel_norm = core_state['robot_a']['joint_vel_norm']  # 21维

# 模块二：全局状态 (13维)
root_state = derived_state['robot_a']['root_state']

# 模块三：触觉力反馈 (2维)
feet_forces = derived_state['robot_a']['feet_forces']

# 模块四：对手观测 (39维)
opponent_basic = derived_state['robot_a']['opponent_basic_pose']  # 9维
opponent_keypoint_pos = derived_state['robot_a']['opponent_keypoint_pos']  # 15维
opponent_keypoint_vel = derived_state['robot_a']['opponent_keypoint_vel']  # 15维
```

**数据结构层级**:
```
get_derived_state()
├── torso_distance (全局)
├── combat_contacts (全局)
└── robot_a / robot_b
    ├── root_state (模块二: 13维)
    │   ├── height
    │   ├── local_orientation
    │   ├── linear_vel
    │   └── angular_vel
    ├── feet_forces (模块三: 2维)
    ├── opponent_basic_pose (模块四.1: 9维)
    ├── opponent_keypoint_pos (模块四.2: 15维)
    ├── opponent_keypoint_vel (模块四.3: 15维)
    ├── observation (96维平铺: 模块一+二+三+四)
    ├── uprightness (兼容旧版)
    └── opponent_in_local (兼容旧版)
```
