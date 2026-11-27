# Humanoid28 数据接口契约 (DATASPEC)

结构与 humanoid21 DATASPEC 同构；本文只记契约要点与 humanoid28 特有差异。
数据接口分层纪律（accessor/mutator 白名单、backend 内部不外泄）与
humanoid21 完全相同。

## 1. Action

见 CONTROLSPEC.md：28 维弧度制 PD 位置目标。`set_action` 接受
ndarray / torch.Tensor / list，逐关节裁剪到 action bounds；`None` 跳过该机器人。

## 2. 静态数据 `get_static_data()`

per-robot：`dof_names`（28）、`body_names`、`body_masses_by_name`、
`joint_names`、`controlled_joint_names`、`root_joint_name`、
`keypoint_body_names`/`keypoint_joint_names`、
`joint_limits`（机械限位 rad）、`action_bounds`（low/high, rad）。
global：`dt`、`ground_geom_name/id`、`geom_id_to_name`、`body_id_to_name`、
`body_id_to_aff`、`geom_id_to_aff`。

keypoint bodies：`head / right_hand / left_hand / right_foot / left_foot`
（side-first 命名，与 MimicKit 一致）。

## 3. 核心状态 `get_core_state()` / `set_core_state()`

per-robot：

| 键 | 形状 | 含义 |
|---|---|---|
| `root_pos` | (3,) | pelvis 世界系位置 |
| `root_rot` | (4,) | pelvis 四元数 [w,x,y,z] |
| `root_vel_local` | (3,) | pelvis 机体系线速度 |
| `root_angular_vel_local` | (3,) | pelvis 机体系角速度 |
| `joint_pos` | (28,) | **原始弧度**（无归一化） |
| `joint_vel` | (28,) | **原始弧度/秒** |

与 humanoid21 的差异：键名为 `joint_pos`/`joint_vel`（非 `_norm` 后缀），
数值为原始物理值。`set_core_state` 支持按 robot_id 部分写回，语义与读取对称。

## 4. 派生数据 `get_derived_state(fields)`

`fields` 子集：`root_distance`、`contacts`（SoA，键名即 `contacts`——
humanoid21 的文档漂移 `contacts_vec` 在本环境不存在）、`robot_a`、`robot_b`。

per-robot view 含：`root_state`（height/projected_gravity/linear_vel/angular_vel/
arena_center_local）、`feet_forces`（[right, left]，体重归一化无因次）、
`opponent_basic_pose`、`opponent_keypoint_pos/vel`、`observation`（110 维）、
`uprightness`、`opponent_in_local`，以及 `body_xpos/xipos/xquat/
body_linvel_world/body_angvel_world/joint_world_anchor` 等世界系原始量。

### relative_vel 契约（决策 #14，明确且正确）

```
relative_vel = R_ego⁻¹ · (v_opp_world − v_self_world)
```

其中两个速度均为 root freejoint 的世界系线速度（同一信源）。
注意：这是"相对速度向量的机系表示"，**不是** `relative_pos` 在机系下的
严格时间导数（后者还需 ω×r 修正项）。与 humanoid21 的历史实现
（仅旋转对手绝对速度、不减自身速度）不同——那是已确认的错误语义。

## 5. 观测

110 维平铺向量，布局见 OBSERVATION_zh.md。

## 6. 传感器

当前 arena 未定义任何 sensor，`get_sensor_data()` 恒返回 `{}`。
