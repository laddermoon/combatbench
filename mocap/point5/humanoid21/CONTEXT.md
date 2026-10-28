# CONTEXT.md — mocap/point5/humanoid21/

## 一句话

P5 → Humanoid21 的**纯几何姿态补全**适配层：五点定义（landmarks）+
正运动学（fk）+ 姿态渲染 + 自然姿态/刚度配置，目标是三阶段求解器
（脚→头→手）把 P5 变成全身代表姿态。计划见 `POSE_COMPLETION_PLAN.md`。

## 数据切面

`point5_from_view(view, robot_id)`：`get_derived_state` 的 per-robot view
（body_xpos/body_xquat/body_linvel_world）→ `Point5State`。
脚点=踝关节中心（landmarks +0.06z），不是 body 原点。

## 调关节的经验（重要，反复用到）

**Humanoid21 关节轴是斜的/镜像的，符号方向不直观，任何角度调整必须渲染验证。**

已踩过的方向坑：

| 关节 | 结论 |
|---|---|
| shoulder2 | **增大（往正向）= 手臂内收贴身；减小（更负）= 肩外翘后掰**。v5→v6 改 -0.4→-0.55 反而更翘，正确方向是 -0.25 |
| elbow | 负值弯肘；肘部朝向由 shoulder 组合决定，肘在身体后面前掰时用搜索定位 |
| 肩组合 | "手臂自然下垂"不是 0 位：需要 (s1≈0.45, s2≈-0.25, e≈-1.0) 这类非零组合 |
| knee | 负值屈膝；踝 ankle_y 正值补偿保持脚平 |

方法：改姿态参数 → `render_pose()` 出正/侧/斜图 → 人眼迭代。
肩部方向这类斜轴别猜，用 FK 数值搜（见 pose 迭代历史）。

## 姿态约定

- root_pos z 不固定——屈膝时 root 要降，用踝中心 0.087 反解贴地高度
- 自然姿态 `q⁰` 在 `pose_config.yaml`，用户看图迭代后冻结（当前 v7 待确认）
- 踝 4 关节固定不优化；刚度分组 torso3/leg2/arm1 为初值
