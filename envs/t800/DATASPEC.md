# T800 数据规范 (Data Specification)

## 1. 核心理念 (Core Principles)

- **按主体隔离**: 所有读取接口必须按 `robot_a` / `robot_b` 分桶，禁止把两台机器人原始状态混杂暴露给策略层。
- **局部坐标系优先**: 速度、相对位姿、对手信息默认在 ego 局部坐标系表示，减少策略对全局朝向的耦合。
- **归一化优先**: 与关节限位直接相关的特征（关节位置/速度）默认提供归一化表示。
- **Sandbox 闭包**: 观察者插件通过 `IDataAccessor`/`IDataMutator` 访问数据，不依赖 MuJoCo 私有字段。

---

## 2. 静态属性 (Static Properties)

**接口**: `get_static_data()`

### 2.1 按机器人分离字段 (`result['robot_a']` / `result['robot_b']`)

| 键 | 类型 | 说明 |
|---|---|---|
| `dof_names` | `List[str]`，长度 25 | 受控自由度短名（不含 `_red/_blue`） |
| `controlled_joint_names` | `List[str]`，长度 25 | 受控关节全名（带颜色后缀） |
| `joint_limits` | `ndarray(shape=(25,2))` | 受控关节物理限位 `[min,max]` (rad) |

### 2.2 全局字段

| 键 | 类型 | 说明 |
|---|---|---|
| `dt` | `float` | 物理子步时长，当前为 `0.002` |
| `ground_geom_name` | `str` | 地面 geom 名，默认 `"ground"` |

### 2.3 受控关节顺序（固定）

`dof_names` / `controlled_joint_names` / `joint_limits` 的顺序必须一致：

1. `J00_HIP_PITCH_L`
2. `J01_HIP_ROLL_L`
3. `J02_HIP_YAW_L`
4. `J03_KNEE_PITCH_L`
5. `J04_ANKLE_PITCH_L`
6. `J05_ANKLE_ROLL_L`
7. `J06_HIP_PITCH_R`
8. `J07_HIP_ROLL_R`
9. `J08_HIP_YAW_R`
10. `J09_KNEE_PITCH_R`
11. `J10_ANKLE_PITCH_R`
12. `J11_ANKLE_ROLL_R`
13. `J12_TORSO_YAW`
14. `J13_SHOULDER_PITCH_L`
15. `J14_SHOULDER_ROLL_L`
16. `J15_SHOULDER_YAW_L`
17. `J16_ELBOW_PITCH_L`
18. `J17_ELBOW_YAW_L`
19. `J18_SHOULDER_PITCH_R`
20. `J19_SHOULDER_ROLL_R`
21. `J20_SHOULDER_YAW_R`
22. `J21_ELBOW_PITCH_R`
23. `J22_ELBOW_YAW_R`
24. `J23_HEAD_PITCH`
25. `J24_HEAD_YAW`

---

## 3. 核心状态 (Core State)

**接口**: `get_core_state()`

每个机器人返回：

| 键 | 类型 | 说明 |
|---|---|---|
| `root_pos` | `ndarray(3,)` | 根 body 世界坐标 `(x,y,z)` |
| `root_quat_wxyz` | `ndarray(4,)` | 根姿态四元数 `[w,x,y,z]` |
| `root_vel` | `ndarray(3,)` | 根线速度（世界系） |
| `root_ang_vel` | `ndarray(3,)` | 根角速度（世界系） |
| `joint_pos` | `ndarray(25,)` | 受控关节角 |
| `joint_vel` | `ndarray(25,)` | 受控关节角速度 |
| `joint_pos_norm` | `ndarray(25,)` | 归一化关节角，`(qpos-ref)/scale` |
| `joint_vel_norm` | `ndarray(25,)` | 归一化关节角速度（实现可按 `qvel/scale` 或阶段性简化） |

---

## 4. 派生数据 (Derived State)

**接口**: `get_derived_state()`

每个机器人至少提供：

| 键 | 类型 | 说明 |
|---|---|---|
| `observation` | `ndarray(104,)` | 平铺后的完整观测向量 |

推荐扩展（与 Humanoid21 对齐）：

| 键 | 类型 | 说明 |
|---|---|---|
| `root_state` | `Dict` | 模块二：`height` + `local_orientation(6)` + `linear_vel(3)` + `angular_vel(3)` |
| `feet_forces` | `ndarray(2,)` | 模块三：左右足底受力 |
| `opponent_basic_pose` | `Dict` | 模块四.1：相对位置/速度/朝向 |
| `opponent_keypoint_pos` | `Dict` | 模块四.2：头/双手/双脚关键点位置 |
| `opponent_keypoint_vel` | `Dict` | 模块四.3：头/双手/双脚关键点速度 |

---

## 5. 传感器数据 (Sensor Data)

**接口**: `get_sensor_data()`

每个机器人返回：

| 键 | 类型 | 说明 |
|---|---|---|
| `force_left_foot` | `ndarray(3,)` | 左足 force 传感器向量 |
| `force_right_foot` | `ndarray(3,)` | 右足 force 传感器向量 |
| `feet_forces` | `ndarray(2,)` | 左/右足受力模长 |

对应 XML 传感器：
- `force_left_foot_{red/blue}`
- `force_right_foot_{red/blue}`

---

## 6. 观测空间总结 (Observation Space Summary)

### 6.1 总维度

T800 单机器人观测维度：**104**

| 模块 | 维度 | 说明 |
|---|---:|---|
| 模块一：本体感知 | 50 | `joint_pos_norm(25) + joint_vel_norm(25)` |
| 模块二：根状态 | 13 | `height(1) + orientation6d(6) + linear_vel(3) + angular_vel(3)` |
| 模块三：触觉反馈 | 2 | 左右足底受力 |
| 模块四：对手观测 | 39 | 基础位姿(9) + 关键点位置(15) + 关键点速度(15) |
| **合计** | **104** | 平铺向量 |

### 6.2 observation 拼接顺序（固定）

```text
[0:25)    joint_pos_norm
[25:50)   joint_vel_norm
[50:51)   root_height
[51:57)   root_orientation_local6d
[57:60)   root_linear_vel_local
[60:63)   root_angular_vel_local
[63:65)   feet_forces
[65:104)  opponent_features(39)
```

---

## 7. 与 Humanoid21 DATASPEC 的关系

- **一致点**：
  - 接口分层：`static/core/derived/sensor`
  - 按机器人隔离
  - observation 由模块拼接
- **差异点**：
  - DOF: `21 -> 25`
  - Observation: `96 -> 104`
  - 关节命名体系由 Humanoid 命名改为 `Jxx_*` 工业命名

---

## 8. 实现状态与后续建议

当前 T800 simulator 已覆盖核心契约（`BaseSimulator` 所需接口），但若要与 Humanoid21 的高级分析插件完全平替，建议后续补齐：

1. `body_names` / `body_masses_by_name` / `joint_names` / `root_joint_name`
2. `keypoint_body_names` / `keypoint_joint_names`
3. 结构化接触列表 `contacts`
4. `root_state`/`opponent_*` 的字典化字段（当前可由 `observation` 反推，但插件直接读取会更稳）

本文件先作为 **T800 v1 数据契约**，确保训练和基础插件可落地；上述扩展可按需求逐步迭代到 v2。
