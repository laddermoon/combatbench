# CONTEXT.md — point5/humanoid21/

## 一句话

P5 → Humanoid21 的**纯力学姿态补全**（`solve_p5_pose`）+
SupportGeometry → Pose 的**参考姿态合成**（`solve_support_pose`）
+ 姿态/状态数据类（`robot_state.py`）+ 支撑分析（`support_state.py`）。
核心是 `mechanics.py`——自包含机械结构（16 刚体/21 铰链/5 标记点）
+ 弹簧表 + 质量表 + 反解器，运行时零 MuJoCo 依赖（FK 纯 numpy）。
反解设计见 `infer_pose_from_p5.md` / `reference_pose_for_support_geometry.md`；
P5 协议见 `../p5.py`。

## 力学模型（mechanics.py，权威）

- 刚体树（16 节，MuJoCo 语义）+ 铰链（锚点/轴/限位）常量表，
  `Skeleton.body_transforms()` / `landmarks()` 纯 numpy FK。
- `Skeleton.anchors_jac(q, joint_names)` → 关节锚点世界坐标 +
  (n,3,21) 解析雅可比（仅祖先链列非零）。**踝心 = ankle_y 上游锚点**，
  踝关节旋转下不动——P5 脚点语义。
- 弹簧：归一化坐标中心 s=0 = 关节域中点（注意≠解剖零位）；
  law∈{linear,quadratic}，非对称 k。abdomen 线性20、hip/knee 二次1、
  ankle 二次0.7、elbow 线性1、shoulder 线性0.4。
- `solve_p5_pose(target (5,3))` 三阶段（输入标准 P5 帧，yaw≡0）：
  - **stage1**：pitch/roll + 11 关节（腰3+髋膝8），双踝心−头相对
    向量 6 硬约束，min 弹性能。LM 求可行 → KKT-QP 抛能量；
    几何初值（头踝倾斜→躯干角、余弦定理→膝角、踝向→髋角）+ 直膝
    变体 + 确定性扰动。多初值后残差仍超 tolerance ⇒ 几何不可达。
  - **stage2**：冻结躯干，每臂闭式 3-DOF IK（`_arm_ik` 至多 4 候选，
    按 (vio, energy) 选支）；不可达 best-effort + relaxed 标记，
    **不无解**。
  - **stage3**：踝 = 弹簧中心（纯先验）。root 平移由头绝对位置恢复，
    朝向 = 阶段一 pitch/roll + 输入 yaw。
  - 实测：随机可达 60/60，中位 ~27ms、P95 ~38ms（<100ms 预算内）；
    雅可比有限差分 1e-9。
- `BODY_MASS`：单台质量表（16 体 mass+局部质心偏移，XML 导出，
  总 40.844kg）。`Skeleton.support_jac()` 单次遍历出 com/头/锚点
  + 全部雅可比；J_com[:,j] = (M_sub/M)·axis×(com_sub−anchor)。
- `solve_support_pose(com_t, ankle_r, ankle_l)` 单阶段（标准几何帧）：
  23 变量（pitch/roll+全 21 关节——com 受全关节影响，勿照搬
  _S1_IDX）对 6 硬约束（踝心−com 相对向量）；目标
  min −w_head·|头−com| + E_spring + w_arm·Σ_arm(qn−ref)²，
  同 LM→KKT 骨架；root 由 com 恢复。
  **已知形态**：|头−com| 无"向上"偏好 → 产出 ~0.2rad 前倾+腰弯
  的"杠杆伸展"姿（设计文档记了 max head_z 备选变体）。实测
  com 误差 0、踝 ~2cm、~50ms；骨架 com 对 MuJoCo 1.7e-7。
  `SupportGeometry.reference_pose()` 为入口（不可达 raise）。

## 数据类

- `robot_state.py`：`Pose(qpos, root_position, root_orientation)`
  root 朝向 = [pitch,roll,yaw]，R=Rz(yaw)·Ry(pitch)·Rx(roll) 标准分支。
  `infer_from_p5(p5)` 走 standardize → solve_p5_pose → _se2 反变换；
  `to_p5()`/`standardize()`/`reanchor()`/`to_support_geometry()`/
  `render()`/`inertia_matrix()`（后两者走 MuJoCo）。
  `RobotState(Pose)` 加 qvel/root 速度/全身角动量，`from_obs()`；
  `take_action(action,dt)`=模拟器真值推演（共享 Humanoid21
  Simulator 懒加载、robot_b 停置远处、同帧输出），**返回新
  对象**供候选动作在同一底本上评估，~20-35ms/控制周期。
- `support_state.py`：`SupportGeometry`（COM+踝心支撑点+踝高+yaw）
  / `SupportState`（+COM 速度、角动量、capture point）。
  `SupportState.drift(dt, alpha=None)`：重力+接触约束演化
  （0脚弹道/1脚自由铰/2脚踝线铰链），动量形式积分（能量误差
  ~1e-4），返 DriftReport 含贴地/摩擦/切向三项假定体检；
  `alpha`=(3,) 世界系绕支点角加速度 = 踝矩通道控制输入
  （单位同 lip_analyze 的 DirectionAlpha.ankle；pivot 丢 ẑ
  分量、hinge 取 e·α、ballistic 作绕 com 外力矩 H+=J·α·dt；
  不裁剪——预算归 lip_analyze）；
  模式 t=0 定死不切、**摆脚再触地不入模型**（规则），
  但 pivot 报 swing_min_clearance（<0=本会触地，纯诊断）；
  `can_balance()`：0 步可捕获性——ξ 落在可控 COP 界内
  （pivot: min(τ_act,N·margin)/N ~5cm 执行器限；hinge:
  载重分摊=多边形垂直轴全长，沿轴锁死不查）；
  `path_to(aim)`：纯几何脚步序列（相邻差一脚、落点在不动脚
  step_radius 内），贪心交替、aim 站位过宽/不收敛 raise；
  `reference_pose()`：几何→参考 Pose（见 solve_support_pose）。
- `stand_punch.py`：站姿前手拳规划 —— **当前 import 断链**（见下）。

## 调关节的经验（重要，反复用到）

**Humanoid21 关节轴是斜的/镜像的，符号方向不直观，任何角度调整必须渲染验证。**

权威语义表：`joints/JOINTSPEC.md`（用户逐关节看 ±1 渲染图标注的），
配套图片在同目录。摘要（neg/pos = 归一化 ±1）：

- knee: neg=收缩弯曲, pos=伸直（单向关节，0≈已伸直到位）
- hip_y: neg=抬腿, pos=放下
- abdomen_y: neg=上身前弯, pos=后仰；abdomen_z: 扭腰；abdomen_x: 侧弯
- shoulder1: neg=抬臂, pos=内收；shoulder2: neg=向后摆, pos=向前摆
- elbow: neg=伸直, pos=收回（注意：与直觉相反！）
- 踝 4 关节不在 JOINTSPEC 里（未标注脚关节方向，读模型轴定义）

方法：改姿态参数 → 渲染出正/侧/斜图 → 人眼迭代。
肩部方向这类斜轴别猜，用 FK 数值搜。

## 测试

`test_mechanics.py`（pytest，13 项）：
结构/弹簧/零目标恢复/确定性/不可达语义/下蹲腿吸能（P5）+
支撑反解的 com/踝恢复、确定性、不可达、屈腿恢复（纯骨架自洽）。

## ⚠ 当前断链（迁移中的陈旧引用，勿按此编码）

- `studio/server.py`：web 编辑器，import 已删的 `solve_equilibrium`/
  `_stage2_core`/`check_legal`——启动即 ImportError，待迁移或重写。
- `stand_punch.py` + `test_stand_punch.py`：按旧 P5 API 写
  （`P5(pos=)`/`Transfer`/`P5Clip(Base,Transfers)`，`Transfer` 类
  已不存在）——整模块待按新 `p5.py` API 重写。
- `executor/` 子树：多处 `from point5.humanoid21.pose import Pose`
  （pose.py 已改名为 `robot_state.py`）与 `point5.contracts`
  （已删；残留于 `executor/contracts.py`）——待迁移。
- 已删文件：pose.py, clip_render.py, render.py, fk.py, adapter.py,
  punch_planner.py, eval_com_model.py, test_pose.py。

## 姿态约定

- P5 标准帧：踝中点=XY 原点、躯干 +x、z 向上、yaw≡0；
  `P5.standardize()` 负责归一。解出后用 `Pose._se2` 逆变换回原帧。
- root_pos z 不固定——屈膝时 root 要降，由头/踝几何反解得到。
- 弹簧中心 = 归一化 0 = 关节域中点；本模型踝部中点恰≈rad 0。
- P5 踝心 = `ankle_y` 关节锚点（非 foot body 物质点、非 ankle_x）。
