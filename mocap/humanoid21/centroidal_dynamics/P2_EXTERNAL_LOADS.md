# P2 验证报告：外部作用规范化

状态：完成  
范围：`external_loads.py`、`test_external_loads.py`

## 已实现接口

`ExternalLoadNormalizer.snapshot(robot_id)` 返回 `LoadSnapshot`，包含：

- 作用于目标 robot subtree 的 `ExternalLoad` 列表；
- 目标 robot 的 `GeneralizedLoad` 列表；
- 世界系重力向量；
- 接触数量、内部接触数量、非目标接触数量、其他机器人 applied-load 诊断。

`ExternalLoad` 的统一形式：

\[
F_{world}\ \text{作用于}\ p_{world},\qquad
T_{world}\ \text{为同参考点自由力偶}
\]

关于全身 COM 的外力矩：

\[
\tau_{COM}=(p_{world}-c)\times F_{world}+T_{world}
\]

## 接触分类

当前明确分类：

- `contact_ground`：目标机器人 geom 与 `ground` 接触；
- `contact_wall`：目标机器人 geom 与非 ground 环境 geom 接触；
- `contact_opponent`：目标机器人 geom 与对手机器人 geom 接触；
- self-contact：同一 robot 内部接触，不进入整体外力；
- 不相关接触：不属于目标 robot，不进入目标 robot 外力。

未映射 geom、未知 affiliation 或非法 contact pair 会抛出 `UnknownExternalSource`。

## 接触方向

MuJoCo `mj_contactForce` 返回 contact-frame wrench。本实现沿用现有 `contacts_vec` 约定：

```text
contact force -> geom2: +force_world
contact force -> geom1: -force_world
```

测试逐项对照了 `get_derived_state(['contacts'])['contacts']['force_world']`。

`condim=3` 下独立 contact torque 不属于适用域；若非零，抛出 `OutOfDomainContact`。

## `xfrc_applied`

受控探针确认：

- `apply_external_force()` 写入 `data.xfrc_applied[body]`；
- `mj_forward()` 后，该 wrench 贡献进入 `data.qfrc_smooth`；
- 广义力映射与 `mj_jacBodyCom` 的 COM Jacobian 完全一致；
- `data.qfrc_applied` 不包含 `xfrc_applied` 的映射结果；
- `physical_step()` 后 `xfrc_applied` 被清零。

因此规范化时：

- `point_world = data.xipos[body_id]`
- `force_world = xfrc[:3]`
- `torque_world = xfrc[3:]`

## `qfrc_applied`

目标 robot 的非零 `qfrc_applied` 被记录为 `GeneralizedLoad`，但默认 `strict_generalized=True` 时抛出 `UnsupportedGeneralizedLoad`。

原因：qfrc 是广义力，不唯一对应一个空间外力/力矩；在没有明确受力 body 和 Jacobian 之前，不能把它作为质心外力项参与收支。

## 测试覆盖

当前新增测试验证：

- ground contact 的数量、作用点、方向和 `contacts_vec` 一致；
- `xfrc_applied` 的 COM 作用点、world wrench 和广义力映射；
- `physical_step()` 后 applied wrench 被清除；
- 目标 robot 非零 `qfrc_applied` fail loud；
- 未映射 geom fail loud；
- 非零 contact torque fail loud。

当前结果：

```text
18 passed
```
