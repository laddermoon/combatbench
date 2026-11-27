# Humanoid28 控制契约 (CONTROLSPEC)

## 1. 频率

| 量 | 值 |
|---|---|
| 物理频率 | 500 Hz（`dt = 0.002 s`） |
| 决策频率 | 20 Hz |
| 每 action step 物理子步 | 25 |

决策 #6：与 humanoid21 一致（**不是** MimicKit 上游的 120/30 或 240/30）。

## 2. Action 格式（决策 #4：与 MimicKit 一致）

`action` 为 28 维 float32，**弧度制 PD 位置目标**，顺序同
`Humanoid28Meta.CONTROLLED_JOINTS`（上游 XML 深度优先顺序：

```
abdomen_x/y/z, neck_x/y/z,
right_shoulder_x/y/z, right_elbow,
left_shoulder_x/y/z, left_elbow,
right_hip_x/y/z, right_knee, right_ankle_x/y/z,
left_hip_x/y/z, left_knee, left_ankle_x/y/z
```

**没有归一化层**：`action[i] = 0.5` 就是目标角度 0.5 rad。
因此 `np.zeros(28)` 不是"中立动作"，而是"T-pose 手臂侧平举"目标；
保持 reset 姿态应使用 `get_action()` 的当前值。

## 3. Action Bounds（MimicKit 语义，决策 #4）

输入逐关节裁剪到 bounds。bounds 不是机械限位，而是 MimicKit 的扩展限位
（`char_env.py` 同款公式）：

- **单曲柄组**（elbow、knee）：以机械中点为中心，半幅 × **1.4**
- **多自由度组**（abdomen xyz、neck xyz、shoulder xyz、hip xyz、ankle xyz）：
  取组内最大绝对机械限位 `maxabs`，对称扩展为 ±**1.2** × maxabs

精确数值见 `robot_actuation.json` 与 `meta.compute_action_bounds()`，
测试中有手算对照（`test_action_bounds_mimickit_exact`）。

后果：PD 目标**可以超出机械限位**——这是 MimicKit 刻意设计，使关节贴近
限位时仍能输出满力矩。

## 4. PD 控制律（决策 #5：与 MimicKit 一致）

每个物理子步、每个关节：

```
tau  = kp * (target_pos − q) − kd * qd
tau  = clamp(tau, −gear, +gear)         # 扭矩上限 = motor gear
ctrl = tau / gear  (写入 actuator ctrl，ctrlrange = ±1)
```

- `kp` = 上游 XML `stiffness` 属性值
- `kd` = 上游 XML `damping` 属性值
- 扭矩上限 = 上游 motor `gear`（abdomen/hip 200, knee 150, ankle 90,
  shoulder 100, elbow 70, neck 50 Nm）

与 MimicKit Newton engine `pd_explicit`  kernel 公式相同。

**关键**：XML 中的 `stiffness/damping` 在 MimicKit 语义里就是 PD 增益，
被动刚度/阻尼已在生成 XML 时清零（与 Newton `_build_controls` 行为一致）。
KP/KD/gear 是机器人固化的内在物理属性，不作为策略层可调参数暴露。

`armature`（0.01–0.02）是真实动力学参数，保留原值。

## 5. 接触（决策 #7 的落地解释）

- 机器人 geom `condim=3`（带摩擦；上游 XML 字面 `condim=1` 在 Isaac Gym
  语义下并不表示无摩擦——Isaac Gym 接触始终带摩擦，MuJoCo 字面复制会得到
  错误的无摩擦机器人间接触）
- friction/solimp/solref 照抄上游 XML 原值（`1.0 0.05 0.05`,
  solimp `.9 .99 .003`, solref `.015 1`）
- 地面为 MuJoCo 平面，接触参数同上
