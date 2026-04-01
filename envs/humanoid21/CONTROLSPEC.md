# Humanoid21 控制规范

## 控制方式

### 输入
- **Action**: 归一化的关节控制信号，范围 `[-1, 1]`
  - `Action = 0` → 关节保持在中位
  - `Action = 1` → 关节移动到上限
  - `Action = -1` → 关节移动到下限

### 参数定义

| 参数 | 说明 | 公式 |
|------|------|------|
| `Down` | 关节下限 | 从 MuJoCo 模型的 `jnt_range[i][0]` 获取 |
| `Up` | 关节上限 | 从 MuJoCo 模型的 `jnt_range[i][1]` 获取 |
| `Reference` | 参考位置（关节中位） | `Reference = (Down + Up) / 2` |
| `Scale` | 动作缩放因子 | `Scale = (Up - Down) / 2` |
| `Qcurrent` | 当前关节位置 | 从 `data.qpos` 获取 |
| `QVel` | 当前关节速度 | 从 `data.qvel` 获取 |
| `KP` | 比例增益 | 默认值 `50.0` |
| `KD` | 微分增益 | 默认值 `5.0` |

### 目标位置计算

```
Target = Reference + Action * Scale
```

**说明**：
- `Reference` 是关节限位的中间值
- `Scale` 是关节范围的一半
- 当 `Action = 1` 时，`Target = Reference + Scale = Up`（上限）
- 当 `Action = -1` 时，`Target = Reference - Scale = Down`（下限）

### PD 控制力矩计算

```
Torque = KP * (Target - Qcurrent) - KD * QVel
```

**简化写法**：
```
Torque = (Action * Scale + Reference - Qcurrent) * KP - QVel * KD
```

### 执行流程

1. **接收 Action** (`[-1, 1]`)
2. **计算目标位置**: `Target = Reference + Action * Scale`
3. **PD 控制输出力矩**: `Torque = KP * (Target - Qcurrent) - KD * QVel`
4. **考虑 Actuator Gear**: `Ctrl = Torque / Gear`
5. **应用限幅**: `Ctrl = clip(Ctrl, ctrl_range[0], ctrl_range[1])`
6. **写入 MuJoCo**: `data.ctrl[actuator_id] = Ctrl`

### 示例

以 `abdomen_z` 关节为例：
- `Down = -0.7854` rad (-45°)
- `Up = 0.7854` rad (+45°)
- `Reference = 0.0` rad
- `Scale = 0.7854` rad

| Action | Target 位置 | 说明 |
|--------|-------------|------|
| -1.0 | -0.7854 rad | 下限（-45°） |
| -0.5 | -0.3927 rad | 下半区中点（-22.5°） |
| 0.0 | 0.0 rad | 中位（0°） |
| 0.5 | 0.3927 rad | 上半区中点（+22.5°） |
| 1.0 | 0.7854 rad | 上限（+45°） |

### 限制

**所有关节必须具有有限的上下限**。如果某个关节的限位为无限（`-inf` 或 `+inf`），初始化将抛出异常。

### 默认参数

- `KP = 50.0`
- `KD = 5.0`
- 可在创建 `MujocaCombatSimulator` 时通过参数修改
