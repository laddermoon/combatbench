# CONTEXT.md — mocap/point5/humanoid21/

## 一句话

P5 → Humanoid21 的**纯力学姿态补全**：无重刚体链 + 铰链弹簧 +
5点硬约束 → 平衡姿态。核心是 `mechanics.py`——自包含的机械结构
（16刚体/21铰链/5标记点）+ 弹簧表 + `solve_equilibrium()`，
运行时零 MuJoCo 依赖。计划见 `POSE_COMPLETION_PLAN.md`。

## 力学模型（mechanics.py，权威）

```
刚体树（16节，MuJoCo 语义）+ 铰链（锚点/轴/限位）
弹簧：中心=0（无载位形=自然姿态），law∈{linear,quadratic}，非对称k
求解：全零出发 → 钉脚→头→手，每步5cm准静态拉 → min E(q) s.t. 硬约束
solve_equilibrium(5点) → {q_norm, q_rad, root, landmarks, errors, ...}
纯函数：无外部状态、无自然姿态配置、不 import mujoco
```

刚度表（验证过）：abdomen线性20、hip/knee二次1、ankle二次0.7、
elbow线性1、shoulder线性0.4。fk.py/render.py 仅作显示工具。

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
- 自然姿态 = 弹簧零位 = 全零姿态（定义式，不需要配置/渲染迭代）
- 踝 4 关节固定不优化；刚度分组 torso3/leg2/arm1 为初值
