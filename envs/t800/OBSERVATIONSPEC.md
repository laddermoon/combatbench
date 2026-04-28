# T800 观测规范 (Observation Specification)

## 1. 总体目标

在保持 Humanoid21 观测设计思想（本体 + 根状态 + 触觉 + 对手）的前提下，适配 T800 的 25 自由度和传感器布局，形成用于对抗任务的统一观测协议。

---

## 2. 观测向量结构（推荐）

```
Observation (104 维)
│
├── 模块一：本体感知 (Proprioception) - 50维
│   ├── 归一化关节角度 (25维)
│   └── 归一化关节角速度 (25维)
│
├── 模块二：根状态 (Root State) - 13维
│   ├── 高度 + 局部朝向 (7维)
│   └── 局部线速度 + 角速度 (6维)
│
├── 模块三：触觉/力反馈 (Tactile) - 2维
│   └── 左右足底接触力标量 (2维)
│
└── 模块四：对手观测 (Opponent) - 39维
    ├── 对手基础位姿 (9维)
    ├── 对手关键点位置 (15维)
    └── 对手关键点速度 (15维)
```

**总计：`50 + 13 + 2 + 39 = 104`**

---

## 3. 模块细节

### 3.1 模块一：本体感知（50维）

#### 3.1.1 归一化关节角度（25维）
- 来源：受控关节 `qpos`
- 归一化：
  ```python
  joint_pos_norm = (qpos - reference) / scale
  joint_pos_norm = clip(joint_pos_norm, -1, 1)
  ```

#### 3.1.2 归一化关节角速度（25维）
- 来源：受控关节 `qvel`
- 归一化建议：
  - 使用每关节速度上限或统一经验上限做缩放
  - 最终 clip 到 `[-1, 1]`

---

### 3.2 模块二：根状态（13维）

- `height`（1维）：机体根部高度（Z）
- `orientation_local6d`（6维）：根部姿态的 6D 表示（旋转矩阵前两列）
- `linear_vel_local`（3维）：根部线速度在自体坐标系下表示
- `angular_vel_local`（3维）：根部角速度在自体坐标系下表示

> 不暴露绝对 `x/y` 位置，减少策略对地图绝对坐标的依赖。

---

### 3.3 模块三：触觉/力反馈（2维）

- `left_foot_force`（1维）
- `right_foot_force`（1维）

推荐从以下传感器提取（取模长并归一化）：
- `force_left_foot`
- `force_right_foot`

> 来源：`t800/xml/serial_sensors.xml`

---

### 3.4 模块四：对手观测（39维）

与 Humanoid21 保持同样语义，全部转换到 ego 局部坐标系：

1. **对手基础位姿（9维）**
   - 相对位置（3）
   - 相对速度（3）
   - 对手朝向向量（3）

2. **对手关键点位置（15维）**
   - 头（3）
   - 左手、右手（6）
   - 左脚、右脚（6）

3. **对手关键点速度（15维）**
   - 与位置对应的 5 个关键点速度

---

## 4. 关键点映射（T800 建议）

建议在 `get_static_data()['keypoint_body_names']` 中固定以下语义映射：

- `torso` -> `LINK_BASE` 或 `LINK_TORSO_YAW`
- `head` -> `LINK_HEAD_YAW`（若层级不同可用 `LINK_HEAD_PITCH`）
- `hand_left` -> `LINK_ELBOW_YAW_L`（或末端 site）
- `hand_right` -> `LINK_ELBOW_YAW_R`（或末端 site）
- `foot_left` -> `LINK_ANKLE_ROLL_L`（或足底传感 site 对应 body）
- `foot_right` -> `LINK_ANKLE_ROLL_R`

最终以可稳定读取的 body/site 为准，核心是“语义不变、命名可替换”。

---

## 5. 数据契约建议

为了与 framework / plugin 体系一致，建议 `derived_state[robot_id]` 至少包含：

- `observation`: `np.ndarray(shape=(104,), dtype=np.float32)`
- `joint_pos_norm`: `(25,)`
- `joint_vel_norm`: `(25,)`
- `root_height`: `float`
- `root_orientation_local6d`: `(6,)`
- `root_linear_vel_local`: `(3,)`
- `root_angular_vel_local`: `(3,)`
- `feet_forces`: `(2,)`
- `opponent_features`: `(39,)`

---

## 6. 与 Humanoid21 的关系

- **保持一致**：
  - 四模块观测思想
  - 对手信息的 ego 坐标表达
  - 不提供绝对平面位置
- **主要差异**：
  - 本体关节由 21 -> 25，观测总维度由 96 -> 104

若需要复用 96 维老策略，可在上层提供一个 observation adapter（例如去除头部相关维度），不建议在底层环境中丢失原生信息。
