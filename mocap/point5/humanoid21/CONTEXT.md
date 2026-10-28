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

权威语义表：`joints/JOINTSPEC.md`（用户逐关节看 ±1 渲染图标注的），
配套图片在同目录。摘要（neg/pos = 归一化 ±1）：

- knee: neg=收缩弯曲, pos=伸直（单向关节，0≈已伸直到位）
- hip_y: neg=抬腿, pos=放下
- abdomen_y: neg=上身前弯, pos=后仰；abdomen_z: 扭腰；abdomen_x: 侧弯
- shoulder1: neg=抬臂, pos=内收；shoulder2: neg=向后摆, pos=向前摆
- elbow: neg=伸直, pos=收回（注意：与直觉相反！）
- 踝 4 关节不在 17 关节语义表里（脚姿态忽略）

方法：改姿态参数 → `render_pose()` 出正/侧/斜图 → 人眼迭代。
肩部方向这类斜轴别猜，用 FK 数值搜（见 pose 迭代历史）。

## 姿态约定

- root_pos z 不固定——屈膝时 root 要降，用踝中心 0.087 反解贴地高度
- 自然姿态 `q⁰` 在 `pose_config.yaml`，用户看图迭代后冻结（当前 v7 待确认）
- 踝 4 关节固定不优化；刚度分组 torso3/leg2/arm1 为初值
