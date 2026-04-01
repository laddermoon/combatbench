# Humanoid21 数据接口规范

本文档严格定义 `MujocoCombatSimulator` 实现的 `IDataAccessor` 和 `IDataMutator` 接口方法的输入输出格式。

所有定义均基于 `simulator.py` 的实际代码推断，不包含猜测或未验证的内容。

---

## IDataAccessor 接口（只读）

### 1. `get_static_data() -> Dict[str, Any]`

获取静态配置数据（不随仿真状态变化）。

**输出格式：**
```python
{
    'dt': float,                    # 物理时间步（秒），默认 0.002
    'robot_info': {
        'robot_a': RobotInfo,
        'robot_b': RobotInfo
    }
}
```

**RobotInfo 结构：**
```python
{
    'body_id': int,                 # pelvis body id
    'root_jnt_id': int,             # root free joint id
    'qpos_adr': int,                # qpos 中 root 位置的起始索引
    'qvel_adr': int,                # qvel 中 root 速度的起始索引
    'suffix': str,                  # '_red' 或 '_blue'
    'actuators': List[int],         # 21 个执行器 id
    'qpos_indices': List[int],      # 21 个关节在 qpos 中的索引
    'qvel_indices': List[int],      # 21 个关节在 qvel 中的索引
    'jnt_ranges': List[ndarray],    # 21 个 [lower, upper] 边界
    'ctrl_ranges': List[ndarray],   # 21 个 [lower, upper] 控制边界
    'qpos0': List[float]            # 21 个关节的参考位置
}
```

**代码依据：** `simulator.py:126-130`

---

### 2. `get_core_state() -> Dict[str, Any]`

获取核心物理状态（位置、速度、时间）。

**输出格式：**
```python
{
    'qpos': ndarray,                # MuJoCo 完整位置数组，shape=(nq,), dtype=float32
    'qvel': ndarray,                # MuJoCo 完整速度数组，shape=(nv,), dtype=float32
    'time': float,                  # 仿真时间（秒）
    'robot_a': RobotCoreState,
    'robot_b': RobotCoreState
}
```

**RobotCoreState 结构：**
```python
{
    'root_position': ndarray,       # shape=(3,), [x, y, z]
    'root_orientation': ndarray,    # shape=(4,), [w, x, y, z] 四元数
    'root_linear_velocity': ndarray,     # shape=(3,), [vx, vy, vz]
    'root_angular_velocity': ndarray,    # shape=(3,), [ωx, ωy, ωz]
}
```

**代码依据：** `simulator.py:132-149`

---

### 3. `get_derived_state() -> Dict[str, Any]`

获取派生状态（碰撞、位置等）。

**输出格式：**
```python
{
    'contacts': List[Contact],
    'robot_a': {
        'xpos': ndarray,             # shape=(nbody, 3), 所有点的位置
        'xvelp': ndarray,            # shape=(nbody, 3), 所有点的线速度
        'xquat': ndarray,            # shape=(nbody, 4), 所有点的四元数
    },
    'robot_b': {}                    # 空（当前实现未填充）
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
    'sensordata': ndarray            # MuJoCo 传感器数据数组
}
```

**代码依据：** `simulator.py:205-206`

---

### 5. `get_action() -> Dict[str, Any]`

获取当前动作。

**输出格式：**
```python
{
    'robot_a': ndarray,              # shape=(action_dim,), 默认 (21,)
    'robot_b': ndarray               # shape=(action_dim,), 默认 (21,)
}
```

**代码依据：** `simulator.py:208-209`

---

### 6. `get_broadcastview_image() -> Any`

获取广播视角图像。

**输出：**
- 成功：`ndarray`，shape=(720, 1280, 3), dtype=np.uint8（RGB 图像）
- 失败：`ndarray`，shape=(720, 1280, 3), dtype=np.uint8（全零图像）

**代码依据：** `simulator.py:347-424`

---

## IDataMutator 接口（可写）

### 1. `set_core_state(state: Dict[str, Any]) -> None`

设置核心物理状态。

**输入 `state` 的必需键：**
```python
{
    'qpos': ndarray,                # 必需，完整位置数组
    'qvel': ndarray                 # 必需，完整速度数组
}
```

**输入 `state` 的可选键：**
```python
{
    'time': float,                  # 可选，仿真时间
    'robot_a': {
        'root_position': ndarray,       # 可选，shape=(3,)
        'root_orientation': ndarray,    # 可选，shape=(4,)
        'root_linear_velocity': ndarray,    # 可选，shape=(3,)
        'root_angular_velocity': ndarray,   # 可选，shape=(3,)
    },
    'robot_b': { ... }             # 结构同 robot_a
}
```

**行为说明：**
1. `qpos` 和 `qvel` 必须是完整的 MuJoCo 数组（直接替换整个数组）
2. 如果提供 `robot_a`/`robot_b` 的结构化数据，会同步回 `qpos`/`qvel` 的相应位置
3. 最后调用 `mujoco.mj_forward()` 刷新正向运动学缓存

**代码依据：** `simulator.py:151-171`

---

### 2. `set_action(action: Dict[str, Any]) -> None`

设置动作。

**输入格式：**
```python
{
    'robot_a': ndarray | None,      # shape=(action_dim,), 值域建议 [-1, 1]
    'robot_b': ndarray | None       # shape=(action_dim,), 值域建议 [-1, 1]
}
```

**行为说明：**
1. 如果值为 `None`，跳过该机器人
2. 如果值为 `ndarray`，先裁剪到 `[-1.0, 1.0]`，再转换为目标关节位置
3. 如果 `_pd_initialized` 为 `False`，此方法不执行任何操作

**代码依据：** `simulator.py:211-226`

---

## 无法推断的部分

以下内容无法从代码中严格推断，依赖于 MuJoCo 模型文件：

1. **`qpos`/`qvel` 的具体维度**：由 MuJoCo 模型决定
2. **`sensordata` 的具体内容和维度**：由 MuJoCo XML 中的传感器定义决定
3. **`robot_info` 中各列表的实际长度**：虽然代码注释为 21，但实际由 XML 中的 controlled_joints 决定

---

## 关键实现细节

### 状态同步双向机制

`set_core_state()` 支持两种修改方式：

1. **直接修改数组**：提供完整的 `qpos`/`qvel` 数组
2. **结构化修改**：提供 `robot_a`/`robot_b` 的结构化数据，会自动同步到数组

示例：
```python
# 方式 1：直接修改
simulator.set_core_state({
    'qpos': new_qpos_array,
    'qvel': new_qvel_array
})

# 方式 2：结构化修改
simulator.set_core_state({
    'qpos': current_qpos,  # 仍需提供完整数组
    'qvel': current_qvel,
    'robot_a': {
        'root_position': [1.0, 0.0, 1.282]  # 只修改 root 位置
    }
})
```

### 四元数格式

所有四元数均使用 **wxyz** 顺序：`[w, x, y, z]`

### 坐标系

- **位置**：全局坐标系，单位为米
- **速度**：全局坐标系，线速度单位为 m/s，角速度单位为 rad/s
