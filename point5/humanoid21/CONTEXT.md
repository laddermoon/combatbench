# CONTEXT.md — point5/humanoid21/

## 一句话

P5 → Humanoid21 的**纯力学姿态补全** + 打击规划器 + web 编辑器：
无重刚体链 + 铰链弹簧 + 5 点硬约束 → 平衡姿态。核心是
`mechanics.py`——自包含机械结构（16 刚体/21 铰链/5 标记点）+
弹簧表 + `solve_equilibrium()`/`check_legal()`，运行时零 MuJoCo 依赖。
上游契约见 `../DESIGN.md`（v2 局部系）。

## 力学模型（mechanics.py，权威）

- 刚体树（16 节，MuJoCo 语义）+ 铰链（锚点/轴/限位）常量表，
  FK 纯 numpy；`Skeleton.landmarks()` → 5 点。
- 弹簧：中心=0（无载位形=自然姿态，定义式），law∈{linear,quadratic}，
  非对称 k。刚度表（验证过）：abdomen 线性20、hip/knee 二次1、
  ankle 二次0.7、elbow 线性1、shoulder 线性0.4。
- `solve_equilibrium(local_p5)` 两段锁定求解（输入已在躯干局部系，
  躯干 +x 必须朝 +X）：
  - **stage1**：钉双脚+头+躯干朝向 → 解 root + 腿 + 腹 + **踝**
    （`_BODY_IDX`=15 关节，SLSQP 最小弹簧能）。先一次性直达，不满足
    再走 5cm 准静态 continuation。
  - **stage2**：冻结 stage1，只动 `abdomen_z` + 6 臂关节；闭式 3-DOF
    臂 IK（`_arm_ik` 至多 4 候选）+ AZ 一维扫描取能量最小。肩部拉拽
    允许扭腰并改变最终 torso yaw——有意设计。
  - 返回 `{root_pos, root_quat_wxyz, q_norm, q_rad, landmarks,
    point_errors, satisfied, failed_stage(stage1_body|stage2_hands),
    energy}`；失败返回 best-effort + 明确标注，不混"不可达"与"没解到"。
- `check_legal(p5)`：<1ms 必要条件包络预筛（`_LEGAL_BOUNDS` 由全
  关节盒 200k 采样测得）。`legal=False` 是定论；`True` 仍需完整求解。
- `_stage2_core` 单独导出：冻结 stage1 身体下验证手目标（studio 拖动
  时实时用）。

## 数据切面

`point5_from_view(view, robot_id)`（adapter.py）：`get_derived_state`
的 per-robot view（body_xpos/body_xquat/body_linvel_world）→ 局部系
`Point5State`。脚点=踝关节中心（landmarks +0.06z，站姿 z≈0.087），
不是 foot body 原点；缺数据 fail-loud，不编造点。

## 打击规划器（产出 P5Clip，sim-unaware 几何规划）

- `punch_planner.plan_punch(initial_p5, target, hand)`：5 帧
  start→swing→load→extend→strike；jab/cross 扭转，诚实可达性判决
  （out_of_sector/out_of_reach → task=None）。
- `stand_punch.plan_stand_punch(target, hand)`：站姿前手拳 4 帧
  coil→swing→land→strike（刃架 30°→60°、committed step、拳打到全臂
  伸展）。`stand_p5()` = arena spawn 站姿（root z=1.282）的局部 P5。
- 两者给每帧 `meta["specs"]` 写 {w,tol} 供下游评分。

## 调关节的经验（重要，反复用到）

**Humanoid21 关节轴是斜的/镜像的，符号方向不直观，任何角度调整必须渲染验证。**

权威语义表：`joints/JOINTSPEC.md`（用户逐关节看 ±1 渲染图标注的），
配套图片在同目录。摘要（neg/pos = 归一化 ±1）：

- knee: neg=收缩弯曲, pos=伸直（单向关节，0≈已伸直到位）
- hip_y: neg=抬腿, pos=放下
- abdomen_y: neg=上身前弯, pos=后仰；abdomen_z: 扭腰；abdomen_x: 侧弯
- shoulder1: neg=抬臂, pos=内收；shoulder2: neg=向后摆, pos=向前摆
- elbow: neg=伸直, pos=收回（注意：与直觉相反！）
- 踝 4 关节不在 17 关节语义表里（JOINTSPEC 未标注脚关节方向）

方法：改姿态参数 → `render_pose()` 出正/侧/斜图 → 人眼迭代。
肩部方向这类斜轴别猜，用 FK 数值搜（见 pose 迭代历史）。

## 可视化与编辑

- `render.py` `render_pose()` / `clip_render.py`：clip → MP4，逐帧
  equilibrium solve + MuJoCo 模型渲染 + 5 点叠加（`MUJOCO_GL=egl`）。
- `studio/server.py`：web 编辑器（Flask SPA），3D 拖点实时约束——
  拖头/脚走 check_legal+stage1，拖手走 stage2 解析验证。
  `PYTHONPATH=. python3 point5/humanoid21/studio/server.py --dir <workdir> --port 8901`
- `fk.py` `Humanoid21FK` / `render.py`：MuJoCo 显示工具（仅展示用）。

## 姿态约定

- root_pos z 不固定——屈膝时 root 要降，用踝中心 0.087 反解贴地高度
- 自然姿态 = 弹簧零位 = 全零姿态（定义式，不需要配置/渲染迭代）
- 求解在 v2 局部系内进行；世界放置由调用方加 frame pose 提升
