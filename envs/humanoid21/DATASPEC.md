# Humanoid21 数据接口规范

本文档严格定义 `MujocoCombatSimulator` 实现的 `IDataAccessor` 和 `IDataMutator` 接口方法的输入输出格式。

**所有定义均基于 `battle_v1.xml` 完全确定，无任何推断或不确定的内容。**

---

## 模型基本信息

| 属性 | 值 | 说明 |
|------|-----|------|
| `nq` (qpos 维度) | 56 | 位置数组长度 |
| `nv` (qvel 维度) | 54 | 速度数组长度 |
| `nu` (actuator 数量) | 42 | 执行器数量（每机器人 21 个） |
| `nbody` (body 数量) | 33 | 刚体数量 |
| `njnt` (joint 数量) | 44 | 关节数量 |
| `ngeom` (geom 数量) | 44 | 几何体数量 |

---

## IDataAccessor 接口（只读）

### 1. `get_static_data() -> Dict[str, Any]`

获取静态配置数据（不随仿真状态变化）。

**输出格式：**
```python
{
    'dt': 0.002,                   # float，物理时间步（秒）
    'robot_info': {
        'robot_a': RobotInfo,
        'robot_b': RobotInfo
    }
}
```

**RobotInfo 结构：**
```python
{
    'body_id': int,                 # pelvis body id (robot_a=4, robot_b=20)
    'root_jnt_id': int,             # root free joint id (robot_a=0, robot_b=22)
    'qpos_adr': int,                # qpos 中 root 位置的起始索引 (robot_a=0, robot_b=28)
    'qvel_adr': int,                # qvel 中 root 速度的起始索引 (robot_a=0, robot_b=27)
    'suffix': str,                  # '_red' 或 '_blue'
    'actuators': List[int],         # 21 个执行器 id (robot_a=0..20, robot_b=21..41)
    'qpos_indices': List[int],      # 21 个关节在 qpos 中的索引
    'qvel_indices': List[int],      # 21 个关节在 qvel 中的索引
    'jnt_ranges': List[ndarray],    # 21 个 [lower, upper] 边界（弧度）
    'ctrl_ranges': List[ndarray],   # 21 个 [lower, upper] 控制边界（总是 [-1.0, 1.0]）
    'qpos0': List[float]            # 21 个关节的参考位置（弧度）
}
```

**代码依据：** `simulator.py:126-130`

---

### 2. `get_core_state() -> Dict[str, Any]`

获取核心物理状态（位置、速度、时间）。

**注意：** 核心状态只包含原始的 qpos 和 qvel 向量。机器人的位置、姿态等结构化数据可以通过 qpos/qvel 和 robot_info 中的索引计算得到。

**输出格式：**
```python
{
    'qpos': ndarray,                # shape=(56,), dtype=float64
    'qvel': ndarray,                # shape=(54,), dtype=float64
    'time': float,                  # 仿真时间（秒）
}
```

**代码依据：** `simulator.py:136-142`

**如何获取机器人数据：**

使用 `get_static_data()` 获取 robot_info，然后用索引访问 qpos/qvel：

```python
static_data = simulator.get_static_data()
state = simulator.get_core_state()

robot_a_info = static_data['robot_info']['robot_a']
qpos_adr = robot_a_info['qpos_adr']
qvel_adr = robot_a_info['qvel_adr']

# 获取 robot_a 的 root 位置和姿态
root_position = state['qpos'][qpos_adr:qpos_adr+3]      # [x, y, z]
root_orientation = state['qpos'][qpos_adr+3:qpos_adr+7]  # [w, x, y, z]
root_linear_velocity = state['qvel'][qvel_adr:qvel_adr+3]     # [vx, vy, vz]
root_angular_velocity = state['qvel'][qvel_adr+3:qvel_adr+6]   # [ωx, ωy, ωz]
```

---

### 3. `get_derived_state() -> Dict[str, Any]`

获取派生状态（碰撞、位置等）。

**输出格式：**
```python
{
    'contacts': List[Contact],      # 碰撞列表
    'robot_a': {
        'xpos': ndarray,            # shape=(33, 3), 所有点的位置
        'xvelp': ndarray,           # shape=(33, 3), 所有点的线速度
        'xquat': ndarray,           # shape=(33, 4), 所有点的四元数
    },
    'robot_b': {}                   # 空（当前实现未填充）
}
```

**Contact 结构：**
```python
{
    'geom_a': int,                   # geom1 id
    'geom_b': int,                   # geom2 id
    'body_a': int,                   # body1 id
    'body_b': int,                   # body2 id
    'position': ndarray,            # shape=(3,), 碰撞位置
    'normal': ndarray,              # shape=(3,), 碰撞法向量
    'impulse': float                 # 冲量大小（标量）
}
```

**代码依据：** `simulator.py:173-203`

---

### 4. `get_sensor_data() -> Dict[str, Any]`

获取传感器数据。

**输出格式：**
```python
{
    'sensordata': ndarray            # shape=(0,), dtype=float64（空数组，无传感器定义）
}
```

**代码依据：** `simulator.py:205-206`

---

### 5. `get_action() -> Dict[str, Any]`

获取当前动作。

**输出格式：**
```python
{
    'robot_a': ndarray,              # shape=(21,), dtype=float64
    'robot_b': ndarray               # shape=(21,), dtype=float64
}
```

**代码依据：** `simulator.py:208-209`

---

### 6. `get_broadcastview_image() -> Any`

获取广播视角图像。

**输出：**
```python
ndarray,  # shape=(720, 1280, 3), dtype=np.uint8（RGB 图像）
```

**代码依据：** `simulator.py:347-424`

---

## IDataMutator 接口（可写）

### 1. `set_core_state(state: Dict[str, Any]) -> None`

设置核心物理状态。

**输入 `state` 的必需键：**
```python
{
    'qpos': ndarray,                # shape=(56,), 完整位置数组
    'qvel': ndarray                 # shape=(54,), 完整速度数组
}
```

**输入 `state` 的可选键：**
```python
{
    'time': float,                  # 仿真时间（可选）
}
```

**行为说明：**
1. `qpos` 和 `qvel` 必须是完整的 56/54 维数组（直接替换整个数组）
2. 如果需要修改特定机器人的状态，请先获取 qpos/qvel，使用 robot_info 中的索引修改相应位置，然后调用 set_core_state
3. 最后调用 `mujoco.mj_forward()` 刷新正向运动学缓存

**代码依据：** `simulator.py:155-161`

**示例：**
```python
# 修改 robot_a 的 root 位置
state = simulator.get_core_state()
qpos = state['qpos'].copy()
qpos_adr = robot_info['robot_a']['qpos_adr']
qpos[qpos_adr:qpos_adr+3] = [1.0, 2.0, 1.282]  # 新位置 [x, y, z]
state['qpos'] = qpos
simulator.set_core_state(state)
```

---

### 2. `set_action(action: Dict[str, Any]) -> None`

设置动作。

**输入格式：**
```python
{
    'robot_a': ndarray | None,      # shape=(21,), 值域 [-1, 1]
    'robot_b': ndarray | None       # shape=(21,), 值域 [-1, 1]
}
```

**行为说明：**
1. 如果值为 `None`，跳过该机器人
2. 如果值为 `ndarray`，先裁剪到 `[-1.0, 1.0]`，再转换为目标关节位置
3. 如果 `_pd_initialized` 为 `False`，此方法不执行任何操作

**代码依据：** `simulator.py:211-226`

---

## 完全确定的维度和索引

### qpos 布局（56 维）

| 索引 | 说明 | 索引 | 说明 |
|------|------|------|------|
| 0:7 | robot_a root (7 DOF) | 28:35 | robot_b root (7 DOF) |
| 7 | abdomen_z_red | 35 | abdomen_z_blue |
| 8 | abdomen_y_red | 36 | abdomen_y_blue |
| 9 | abdomen_x_red | 37 | abdomen_x_blue |
| 10 | hip_x_right_red | 38 | hip_x_right_blue |
| 11 | hip_z_right_red | 39 | hip_z_right_blue |
| 12 | hip_y_right_red | 40 | hip_y_right_blue |
| 13 | knee_right_red | 41 | knee_right_blue |
| 14 | ankle_y_right_red | 42 | ankle_y_right_blue |
| 15 | ankle_x_right_red | 43 | ankle_x_right_blue |
| 16 | hip_x_left_red | 44 | hip_x_left_blue |
| 17 | hip_z_left_red | 45 | hip_z_left_blue |
| 18 | hip_y_left_red | 46 | hip_y_left_blue |
| 19 | knee_left_red | 47 | knee_left_blue |
| 20 | ankle_y_left_red | 48 | ankle_y_left_blue |
| 21 | ankle_x_left_red | 49 | ankle_x_left_blue |
| 22 | shoulder1_right_red | 50 | shoulder1_right_blue |
| 23 | shoulder2_right_red | 51 | shoulder2_right_blue |
| 24 | elbow_right_red | 52 | elbow_right_blue |
| 25 | shoulder1_left_red | 53 | shoulder1_left_blue |
| 26 | shoulder2_left_red | 54 | shoulder2_left_blue |
| 27 | elbow_left_red | 55 | elbow_left_blue |

### qvel 布局（54 维）

| 索引 | 说明 | 索引 | 说明 |
|------|------|------|------|
| 0:6 | robot_a root (6 DOF) | 27:33 | robot_b root (6 DOF) |
| 6 | abdomen_z_red | 33 | abdomen_z_blue |
| 7 | abdomen_y_red | 34 | abdomen_y_blue |
| 8 | abdomen_x_red | 35 | abdomen_x_blue |
| 9 | hip_x_right_red | 36 | hip_x_right_blue |
| 10 | hip_z_right_red | 37 | hip_z_right_blue |
| 11 | hip_y_right_red | 38 | hip_y_right_blue |
| 12 | knee_right_red | 39 | knee_right_blue |
| 13 | ankle_y_right_red | 40 | ankle_y_right_blue |
| 14 | ankle_x_right_red | 41 | ankle_x_right_blue |
| 15 | hip_x_left_red | 42 | hip_x_left_blue |
| 16 | hip_z_left_red | 43 | hip_z_left_blue |
| 17 | hip_y_left_red | 44 | hip_y_left_blue |
| 18 | knee_left_red | 45 | knee_left_blue |
| 19 | ankle_y_left_red | 46 | ankle_y_left_blue |
| 20 | ankle_x_left_red | 47 | ankle_x_left_blue |
| 21 | shoulder1_right_red | 48 | shoulder1_right_blue |
| 22 | shoulder2_right_red | 49 | shoulder2_right_blue |
| 23 | elbow_right_red | 50 | elbow_right_blue |
| 24 | shoulder1_left_red | 51 | shoulder1_left_blue |
| 25 | shoulder2_left_red | 52 | shoulder2_left_blue |
| 26 | elbow_left_red | 53 | elbow_left_blue |

### robot_a 索引

| 属性 | 值 |
|------|-----|
| body_id | 4 |
| root_jnt_id | 0 |
| qpos_adr | 0 |
| qvel_adr | 0 |
| suffix | '_red' |
| actuators | [0, 1, 2, ..., 20] |

### robot_b 索引

| 属性 | 值 |
|------|-----|
| body_id | 20 |
| root_jnt_id | 22 |
| qpos_adr | 28 |
| qvel_adr | 27 |
| suffix | '_blue' |
| actuators | [21, 22, 23, ..., 41] |

### robot_a qpos/qvel 索引（21 个关节）

| 索引 | 关节 | qpos | qvel |
|------|------|------|------|
| 0 | abdomen_z | 7 | 6 |
| 1 | abdomen_y | 8 | 7 |
| 2 | abdomen_x | 9 | 8 |
| 3 | hip_x_right | 10 | 9 |
| 4 | hip_z_right | 11 | 10 |
| 5 | hip_y_right | 12 | 11 |
| 6 | knee_right | 13 | 12 |
| 7 | ankle_y_right | 14 | 13 |
| 8 | ankle_x_right | 15 | 14 |
| 9 | hip_x_left | 16 | 15 |
| 10 | hip_z_left | 17 | 16 |
| 11 | hip_y_left | 18 | 17 |
| 12 | knee_left | 19 | 18 |
| 13 | ankle_y_left | 20 | 19 |
| 14 | ankle_x_left | 21 | 20 |
| 15 | shoulder1_right | 22 | 21 |
| 16 | shoulder2_right | 23 | 22 |
| 17 | elbow_right | 24 | 23 |
| 18 | shoulder1_left | 25 | 24 |
| 19 | shoulder2_left | 26 | 25 |
| 20 | elbow_left | 27 | 26 |

### robot_b qpos/qvel 索引（21 个关节）

| 索引 | 关节 | qpos | qvel |
|------|------|------|------|
| 0 | abdomen_z | 35 | 33 |
| 1 | abdomen_y | 36 | 34 |
| 2 | abdomen_x | 37 | 35 |
| 3 | hip_x_right | 38 | 36 |
| 4 | hip_z_right | 39 | 37 |
| 5 | hip_y_right | 40 | 38 |
| 6 | knee_right | 41 | 39 |
| 7 | ankle_y_right | 42 | 40 |
| 8 | ankle_x_right | 43 | 41 |
| 9 | hip_x_left | 44 | 42 |
| 10 | hip_z_left | 45 | 43 |
| 11 | hip_y_left | 46 | 44 |
| 12 | knee_left | 47 | 45 |
| 13 | ankle_y_left | 48 | 46 |
| 14 | ankle_x_left | 49 | 47 |
| 15 | shoulder1_right | 50 | 48 |
| 16 | shoulder2_right | 51 | 49 |
| 17 | elbow_right | 52 | 50 |
| 18 | shoulder1_left | 53 | 51 |
| 19 | shoulder2_left | 54 | 52 |
| 20 | elbow_left | 55 | 53 |

### Body 列表（33 个）

| ID | 名称 | ID | 名称 |
|----|------|----|------|
| 0 | world | 17 | torso_blue |
| 1 | torso_red | 18 | head_blue |
| 2 | head_red | 19 | waist_lower_blue |
| 3 | waist_lower_red | 20 | pelvis_blue |
| 4 | pelvis_red | 21 | thigh_right_blue |
| 5 | thigh_right_red | 22 | shin_right_blue |
| 6 | shin_right_red | 23 | foot_right_blue |
| 7 | foot_right_red | 24 | thigh_left_blue |
| 8 | thigh_left_red | 25 | shin_left_blue |
| 9 | shin_left_red | 26 | foot_left_blue |
| 10 | foot_left_red | 27 | upper_arm_right_blue |
| 11 | upper_arm_right_red | 28 | lower_arm_right_blue |
| 12 | lower_arm_right_red | 29 | hand_right_blue |
| 13 | hand_right_red | 30 | upper_arm_left_blue |
| 14 | upper_arm_left_red | 31 | lower_arm_left_blue |
| 15 | lower_arm_left_red | 32 | hand_left_blue |
| 16 | hand_left_red | |

---

## 关键实现细节

### 核心状态访问方式

**核心状态只包含原始数组：** `qpos` 和 `qvel` 是完整的 MuJoCo 状态向量。要获取特定机器人的数据，需要：

1. 从 `get_static_data()` 获取 `robot_info`
2. 使用 `qpos_adr` 和 `qvel_adr` 索引访问对应机器人的数据

示例：
```python
static_data = simulator.get_static_data()
state = simulator.get_core_state()

# 获取 robot_a 的 root 数据
info_a = static_data['robot_info']['robot_a']
qpos_adr_a = info_a['qpos_adr']
qvel_adr_a = info_a['qvel_adr']

root_pos = state['qpos'][qpos_adr_a:qpos_adr_a+3]      # [x, y, z]
root_quat = state['qpos'][qpos_adr_a+3:qpos_adr_a+7]   # [w, x, y, z]
root_lin_vel = state['qvel'][qvel_adr_a:qvel_adr_a+3]     # [vx, vy, vz]
root_ang_vel = state['qvel'][qvel_adr_a+3:qvel_adr_a+6]   # [ωx, ωy, ωz]
```

### 四元数格式

所有四元数均使用 **wxyz** 顺序：`[w, x, y, z]`

### 坐标系

- **位置**：全局坐标系，单位为米
- **速度**：全局坐标系，线速度单位为 m/s，角速度单位为 rad/s
- **角度**：所有关节边界和位置均使用弧度（radians）

### PD 控制参数

- **Kp** = 4.0（比例增益）
- **Kd** = 0.4（微分增益）
- **目标速度** = 0

---

## XML 文件位置

- **场景文件**: `envs/humanoid21/battle_v1.xml`
- **实例化**：
```python
from humanoid21.simulator import MujocoCombatSimulator

sim = MujocoCombatSimulator()  # 自动使用 battle_v1.xml
```
