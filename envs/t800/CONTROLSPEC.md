# T800 控制规范 (Control Specification)

## 1. 控制模式：归一化位置控制 (Normalized Position Control)

策略层与底层物理引擎之间，通过**归一化关节目标位置**进行通信。  
T800 的所有受控关节统一映射到 `[-1, 1]` 的无量纲区间。

### 1.1 动作接口 (Action Interface)

- **输入**: `action`，形状为 `(25,)` 的 `numpy.ndarray`
- **取值范围**: `[-1, 1]`
- **物理含义**:
  - `action[i] = 0.0` -> 关节目标位于中位 `Reference`
  - `action[i] = 1.0` -> 关节目标位于上限 `Up`
  - `action[i] = -1.0` -> 关节目标位于下限 `Down`

---

## 2. 受控关节定义（固定顺序）

动作向量维度与以下 25 关节严格一一对应，不允许重排：

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

> 来源：`t800/xml/serial_actuators.xml`

---

## 3. 映射与计算逻辑 (Mapping & Calculation)

### 3.1 静态参数准备

每个受控关节在初始化时，从 MuJoCo 模型读取：

- `Down = jnt_range[i][0]`（rad）
- `Up = jnt_range[i][1]`（rad）
- `Reference = (Down + Up) / 2`
- `Scale = (Up - Down) / 2`

### 3.2 控制流程

1. **反归一化**
   ```python
   target_rad = action * scale + reference
   ```

2. **PD 控制律**
   ```python
   torque = kp * (target_rad - qpos) - kd * qvel
   ```

3. **输出限幅**
   ```python
   ctrl = clip(torque / gear, ctrl_range_min, ctrl_range_max)
   ```

4. **执行写入**
   将 `ctrl` 写入对应 `data.ctrl` 切片，索引顺序必须与上面的 25 关节一致。

---

## 4. KP/KD 设定（参照 Humanoid21 思路）

### 4.1 设计思路

参照 `humanoid21/simulator.py` 的“**分组刚度 + 固化数组**”策略，T800 同样采用：

1. **本体参数固化**：`KP/KD` 写死在 simulator，不作为外部可调超参。
2. **按关节功能分层**：
   - 腿部承重关节（hip/knee）高刚度
   - 踝关节中等刚度（保留柔顺性）
   - 躯干 yaw 中高刚度（抗扭）
   - 手臂中等刚度（兼顾攻击与稳定）
   - 头部低刚度（避免高频抖动）
3. **与执行器能力对齐**：结合 T800 的 `ctrlrange/actuatorfrcrange`（415/370/222/160/52）分组定档，防止弱关节过驱。

### 4.2 推荐初始值（可直接落地）

以下数组顺序严格对应第 2 节的 25 个关节：

```python
# T800 25-DOF 初始 PD 参数（建议）
KP = np.array([
    # J00~J05: 左腿
    220.0, 180.0, 180.0, 240.0, 120.0, 100.0,
    # J06~J11: 右腿
    220.0, 180.0, 180.0, 240.0, 120.0, 100.0,
    # J12: 躯干 yaw
    260.0,
    # J13~J17: 左臂
    140.0, 130.0, 120.0, 110.0, 80.0,
    # J18~J22: 右臂
    140.0, 130.0, 120.0, 110.0, 80.0,
    # J23~J24: 头部
    60.0, 60.0
], dtype=np.float32)

KD = np.array([
    # J00~J05: 左腿
    26.0, 20.0, 20.0, 28.0, 12.0, 10.0,
    # J06~J11: 右腿
    26.0, 20.0, 20.0, 28.0, 12.0, 10.0,
    # J12: 躯干 yaw
    30.0,
    # J13~J17: 左臂
    14.0, 13.0, 12.0, 11.0, 8.0,
    # J18~J22: 右臂
    14.0, 13.0, 12.0, 11.0, 8.0,
    # J23~J24: 头部
    6.0, 6.0
], dtype=np.float32)
```

### 4.3 为什么比 Humanoid21 略高（腿/躯干）

- T800 下肢 `actuatorfrcrange` 高（髋膝到 415）且 `armature` 明确建模，允许比 Humanoid21 稍高的腿部增益来提升站立和抗冲击。
- 躯干仅 1 个 yaw 自由度（J12），需承担较大抗扭任务，因此单轴 `KP/KD` 设为中高档。
- 头部执行器能力较弱（52），增益必须保守，避免持续饱和。

### 4.4 接受标准（第一轮）

第一轮参数确认建议使用与 Humanoid21 一致的工程口径：

1. **静站稳定**：10s 内不倒，关节无持续发散震荡。
2. **跟踪误差**：主要关节稳态误差多数落在 `0.05~0.1 rad` 内。
3. **控制饱和率**：弱关节（肘 yaw、头部）不长期顶满 `ctrlrange`。
4. **受扰恢复**：轻推后 1~2 秒内回稳，不出现“头部高频摆振”。

### 4.5 调参优先级

若出现问题，按以下顺序调整：

1. 先调 `KD` 抑制振荡（每次 +10%）
2. 再调 `KP` 提升跟踪（每次 +10%）
3. 头部与肘 yaw 单独降 `KP/KD`（若饱和或抖动）
4. 踝关节优先通过 `KD` 调整落地稳定性

---

## 5. 与 Humanoid21 的兼容关系

- **相同点**：控制语义完全一致（归一化位置 -> PD -> 限幅）
- **差异点**：`ACTION_DIM` 从 21 改为 25，新增头部 2 自由度 + 手臂 yaw 自由度参与控制

若后续需要复用 21 维策略，可在策略层做动作适配器；环境底层保持 25 维原生控制，不做裁剪。
