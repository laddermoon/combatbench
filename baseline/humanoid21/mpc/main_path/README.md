# Humanoid21 第一版主路径规划器

> 范围：通用 reduced-order main-path planning。

本目录实现主路径规划，不实现关节动作。整体结构是：

```text
MainPathRequest
  = 当前 MainPathState + 目标位置
    ↓
SupportSequencePlanner
  = 生成左右脚未来的支撑点与支撑时间区间
    ↓
MainPathTrajectoryPlanner
  = 由支撑序列生成 c,l,L 和双脚等效受力
    ↓
MainPath
```

`SupportInterval` 是规划中间结果，不是用户输入。

## 1. 默认规划请求

第一版只实现一个默认指令：**走到一个位置并停住保持平衡**。

```python
MainPathRequest:
    current_state: MainPathState
    target_pos_world: np.ndarray  # 主要使用水平 XY 分量
```

`MainPathState` 是本模块自己的最简状态，不复用旧的 `CentroidalState`
契约：

```python
MainPathState:
    time: float
    com_pos: np.ndarray          # world (3,)
    lin_mom: np.ndarray          # world (3,)
    ang_mom: np.ndarray          # about com_pos, world (3,)
    heading_yaw: float           # body yaw in world XY
    left_point_world: np.ndarray   # equivalent point, world (3,)
    right_point_world: np.ndarray  # equivalent point, world (3,)
    left_force_world: np.ndarray   # external force on robot, world (3,)
    right_force_world: np.ndarray  # external force on robot, world (3,)
    left_support_active: bool
    right_support_active: bool
```

`heading_yaw` 是必需的最小上下文：线动量为零时没有方向信息，而且
即使线动量非零，它的方向也不等于身体朝向。左右脚等效点和当前受力
同样属于这个约化状态的边界条件；没有它们，第一阶段无法知道当前支
撑结构。

`*_support_active=False` 时，对应外力必须为零；对应点仍可表示当前
脚部等效位置，但不参与支撑动力学。当前 `heading_yaw` 同时作为左/
右脚语义和最终站姿宽度方向的参考；请求暂不指定目标朝向。

质量、重力、时间步和模型几何放在 `MainPathModel` / planner config
中，不属于瞬时状态，也不在本目录重新复制一套历史契约。

`target_pos_world` 表示期望最终停止的水平位置。最终质心高度不由
用户指定，而由当前任务的能力包络或默认站立/屈膝高度确定。

规划请求不包含：

- 当前关节状态；
- 完整 root pose；
- 全身惯量；
- 手动指定的支撑区间；
- 对手状态；
- 全身动作目标；
- 目标朝向。

## 2. 第一阶段：支撑序列规划器

`SupportSequencePlanner` 的职责是：

```python
plan_support_sequence(request) -> SupportSchedule
```

它根据当前 `(c,l,L)`、身体朝向、左右脚等效点、支撑状态，以及目标
位置和步长/步宽/时长等能力参数，生成未来每只脚的支撑位置和支撑
时间区间。

输出形式：

```python
SupportInterval:
    foot: Literal["left", "right"]
    start_time: float
    end_time: Optional[float]
    point_world: np.ndarray  # (3,)

SupportSchedule:
    intervals: Tuple[SupportInterval, ...]
```

示例输出：

```text
left:
  [0.0, 0.8] -> pL0
  [1.4, ∞)   -> pL1

right:
  [0.0, 0.5] -> pR0
  [1.0, 1.4] -> pR1
  [2.0, ∞)   -> pR2
```

`end_time=None` 表示这是最后支撑区间，持续到规划结束；该脚不再换
支撑点，规划器应在末端检查是否可以保持平衡。

### 支撑序列的性质

- 一只脚可以生成多个支撑区间，表示重复迈步；
- 两脚之间没有数据层关联约束；
- 某只脚没有区间覆盖的时间表示该脚不支撑；
- 两脚都没有覆盖的时间表示无支撑，数据层允许；
- 支撑区间内该脚的 `point_world` 固定；
- 不在支撑区间内的脚，外力必须为零；
- 序列本身不包含“准备迈步”“重心转移”等阶段语义。

### 数据校验

只允许字段级 sanity check：

- `foot` 合法；
- `start_time` 有限；
- `end_time` 为 `None` 或有限值；
- 非空 `end_time` 必须大于 `start_time`；
- `point_world` 为有限 `(3,)`；
- 同一只脚的区间不得重叠，`end_time=None` 后不能有后续区间。

左右脚之间的时序组合是否可行，由规划器检查，不写入 schema。

### 默认生成策略

第一版支撑序列规划器使用确定性启发式：

1. 用 `heading_yaw` 建立 body XY frame，取 `com_pos` 到
   `target_pos_world` 的水平位移并投影到前向/左向分量；
2. 若位移小于 `target_tolerance`，直接生成保持当前左右脚点的开放
   式区间；
3. 否则以当前左右脚点为起点，以目标位置周围沿 `heading_yaw` 展开
   的双脚站姿为终点；
4. 按 `max_step_length` 分别计算左右脚需要的步数；
5. 侧向目标优先移动对应侧的脚，纵向目标默认右脚先动；
6. 交替生成左右脚落点和接触/摆动时间；
7. 最后一对支撑区间以 `end_time=None` 结束。

第一版没有目标朝向，因此最终站姿保持当前 `heading_yaw`。

TODO: 后续加入 `target_yaw` 和 nominal centerline 后，再把曲线路径、
缓慢转身、脚部朝向与最终站姿绑定到支撑序列规划中。

## 3. 第二阶段：主路径轨迹规划器

细化实施方案见 [STAGE2_PLAN.md](STAGE2_PLAN.md)。当前已实现第一版
连续路径生成器和独立验证器。

`MainPathTrajectoryPlanner` 消费：

```python
MainPathTrajectoryRequest:
    current_state: MainPathState
    support_schedule: SupportSchedule
    target_pos_world: np.ndarray
```

这里的 `support_schedule` 是第一阶段的输出，不是外部输入。
`MainPathModel` 与 `MainPathConstraints` 是 planner 配置，不属于请求。

调用结果是结构化的 `PlanningResult`：

```python
PlanningResult:
    success: bool
    path: Optional[MainPath]
    report: ValidationReport
    diagnostics: dict
```

`MainPath` 的核心语义是 `MainPathState` 在时间维度上的连续表示：

```python
state = main_path.state_at(t)
```

对定义域 `[start_time, end_time]` 内任意 `t` 都返回完整的
`MainPathState`，包含质心、动量、角动量、朝向、左右脚等效点、左右脚
受力和支撑标志。某个脚 inactive 时，对应 force 严格为零；摆动脚的
等效点仍有连续几何参考，但不参与动力学。

当前连续表示采用支撑事件对齐的 `PathPiece`：

- 每只脚的外力在段内线性的；
- `l(t)` 由合外力解析积分为二次曲线；
- `c(t)` 由 `l(t)/m` 再积分为三次曲线；
- `L(t)` 由等效接触点关于 COM 的力矩积分得到；
- `state_at(t)` 是唯一的对外查询接口，不从另一套采样数组插值。

活动支撑脚的 `point_world` 是该脚的等效受力点。它可以在支撑点周围
的脚底尺寸包络内连续移动，用于实现简化 COP 调节；第一阶段给出的
`SupportInterval.point_world` 仍是脚底几何支撑中心。摆动期或脚不承
载时，外力为零，等效点只作为连续几何参考。

开放式最后支撑区间会额外使用 `terminal_extension` 做收敛检查：路径
不能只到第一阶段 `schedule.horizon` 就声称停稳，而必须在扩展窗口末
端满足位置、线动量、角动量、合力和合力矩条件。

`MainPath` 的核心语义是 `MainPathState` 在时间维度上的连续表示。
离散数组只是实现和检查方式；对外契约要求：

```python
state = main_path.state_at(t)
```

对定义域内任意 `t` 都返回一个完整的 `MainPathState`。返回值必须包含
质心、动量、角动量、朝向、左右脚等效点、左右脚受力和支撑标志。某个
脚 inactive 时，对应 force 必须为零。

“整条时间线上的状态对机器人可行”是指它必须持续满足当前 reduced
model 和任务能力包络：支撑点与 active flag 一致、受力在约束内、状态
变化满足主路径动力学、没有伪造的瞬时跳变。它仍不单独证明所有关节
动作可实现；全身可实现性由后续参考姿态和 branch simulation 验证。

`*_active=False` 时，对应 `force_world` 必须为零；对应 `point` 可表
示当前等效脚部位置，但不参与支撑动力学。

## 4. 动力学

每个采样时刻的活动支撑集合由 `SupportSchedule` 决定。动力学为：

\[
\dot c=\frac{l}{m}
\]

\[
\dot l=mg+f_L+f_R
\]

\[
\dot L=(p_L-c)\times f_L+(p_R-c)\times f_R
\]

无支撑段有：

\[
f_L=f_R=0,\qquad \dot l=mg,\qquad \dot L=0
\]

规划器必须保证整个连续表示满足这些方程；节点检查只是其中一部分。
对任意时间区间，状态差分须与区间内速度、合力和力矩的积分一致。

## 5. 能力与受力约束

第一版能力包络由 `MainPathConstraints` 显式配置，默认值从
`GaitModel` 推导并用 metadata 记录来源：

```python
MainPathConstraints:
    friction
    max_normal_force
    max_force
    max_force_rate
    max_com_height_deviation
    max_com_speed
    max_ang_mom
    max_com_to_support_xy
    max_cop_x
    max_cop_y
    max_swing_speed
    swing_lift_height
    nominal_com_height
    tracking_time_constant
    angular_tracking_time_constant
    terminal_extension
    integration_dt / constraint_dt / residual tolerances
```

对每只活动支撑脚检查：

\[
0\le f_z\le f_{z,\max}
\]

\[
\sqrt{f_x^2+f_y^2}\le\mu f_z
\]

\[
|f|\le f_{\max},\qquad |\dot f|\le\dot f_{\max}
\]

等效受力点还必须留在名义支撑点周围的 `max_cop_x / max_cop_y` 包络
内。当前假设脚底局部轴与世界 XY 对齐；非零 yaw 下的 COP 局部坐标映
射仍属于后续细化项。

第一版不引入接触 torque，也不规划惯量张量；角动量只能通过等效受力
点位置、脚部受力分配和质心运动调节。

## 6. 目标与可行性

默认目标包含：

- 到达目标水平位置附近的可停稳区域；
- 最终线动量和角动量接近零；
- 路径平滑，脚部受力变化不过快；
- 支撑脚受力满足约束；
- 最终开放式支撑区间内存在可行平衡受力。

最终平衡的主路径层判定条件为：

\[
l\approx0,\qquad L\approx0
\]

且在当前最终支撑点下存在一组可行脚部受力，使：

\[
f_L+f_R=-mg
\]

\[
(p_L-c)\times f_L+(p_R-c)\times f_R=0
\]

这不证明全身动作可行；最终能否站立必须由下游全身仿真验证。

## 7. 求解与验证方式

当前生成器是透明的参考/反馈式第一版，不是通用 NLP 求解器：

1. 用支撑事件和控制周期构造连续 `PathPiece` 网格；
2. 根据活动支撑点生成 COM 参考；
3. 用位置/速度反馈计算期望接触力，并按时限平滑；
4. 按支撑切换规则保持力连续，离地端清零，新落地端从零加载；
5. 在双脚支撑时分配总力，并用差动切向力辅助调节 `L_z`；
6. 在脚底包络内移动等效受力点，用 COP 偏移调节 `L_x/L_y`；
7. 将候选交给独立 `MainPathValidator`。

求解器与验证器分离。`PlanningResult.success=True` 只表示候选通过
独立验证，不代表优化器自评通过。验证器支持任意区间：

```python
report = validator.check_interval(path, t_a, t_b)
```

它把区间切到 piece 边界，对状态差分和动力学积分比较，同时检查：
力连续性、inactive 零力、法向力、摩擦锥、最大力、力变化率、COM
高度/速度、COM 到支撑距离、等效点脚底包络、角动量包络、摆动脚速
度和终端保持。

失败结果包含结构化 `ValidationIssue(code, message, time, interval,
measured, limit, details)`，可区分前置输入冲突、动力学残差、能力约
束和终端收敛失败。

## 8. 第三阶段：运动参照（MotionReference）

第三阶段是第二阶段之上的**纯函数推演**，不做搜索也不做 IK 求解：

```text
(t, MainPath, MotionReferenceParams)
    -> MotionReferenceState    # 参数化关键形态参照
    -> sample_pose(t)          # 代表性的完整骨架状态（采样接口）
```

`MotionReference` 消费的是已经通过独立验证的 `MainPath`，其第一个状
态与机器人当前主路径状态一致；参照因此天然从机器人当前状态出发。

参照采用显式参数化的协调规则，而不是隐式约束集合：

- `stride_delta = (p_left - p_right) · forward` 度量左右脚前后差；
- 骨盆扭转 `= clip(k_hip * stride_delta)`，躯干反向扭转
  `= clip(-k_torso * stride_delta)`；
- 摆臂异侧协调：左脚在前时左臂向后、右臂向前，幅度
  `clip(k_arm * stride_delta)`，同时给出摆臂角速度参考；
- 肘角、躯干前倾、屈膝量、摆动脚 pitch 都是模板参数；
- `arm_lift` 是唯一自由调节通道：按当前 `ang_mom` 水平推导出限幅
  目标，用于转动性能调节；
- 骨盆位置取 `com_pos` 的固定竖直偏移；
- 支撑脚的脚底中心取第一阶段名义支撑点，等效受力点（COP 偏移）作
  为独立字段输出。

姿态模板由 `MotionReferenceParams.for_template()` 提供：

- `walk`：自然步态，摆臂与异侧协调正常，肘角较小；
- `boxing_guard`：拳击架，肘角大、手臂抬起、摆臂幅度受限。

`sample_pose(t)` 把参照实例化为一个**代表性的完整骨架**
（`SampledPose.keypoints`）：骨盆、胸、头、肩、肘、手、髋、膝、踝、
脚尖。腿部用两段连杆 IK 朝 heading 前方屈膝，手臂由摆角/抬角/肘角
唯一确定。采样状态只用于检查和可视化，不是关节轨迹命令。

`sim_playback.py` 进一步把采样骨架映射到真实 Humanoid21 模型：
`ReferenceIKSolver` 用阻尼最小二乘把骨盆/脚/手/头关键点解到完整
`qpos`（含脚部与躯干的旋转任务和站姿正则项），`render_reference_in_sim`
逐帧写入仿真器并用广播相机渲染 MP4。离线渲染需要
`MUJOCO_GL=egl`（`render_cases.py` 与测试 `conftest.py` 已默认设置）。

第三阶段仍不输出力矩、关节角命令或 MuJoCo 状态；下游动作层围绕这
些参照搜索自己的动作。

## 9. 与下游的边界

`MainPath` 只回答：

> 为了走到目标位置并停住，整体质心、动量、角动量和双脚等效外力
> 应该怎样随时间变化。

下游再做：

```text
MainPath
  -> 生成全身参考姿态
  -> 在参考附近搜索动作
  -> Observation-only branch simulator 验证
  -> 输出实际控制 action
```

下游可以反馈要求重新规划，但全身姿态和动作实现不属于本模块。

## 10. 当前实现与文件划分

当前文件：

```text
contracts.py           # MainPathState / MainPathRequest / SupportInterval / SupportSchedule
gait_model.py          # 从 Humanoid21 站姿提取步态数值包络
support_sequence.py    # MainPathState + target_pos -> SupportSchedule
constraints.py         # MainPathModel 与能力包络
trajectory.py          # SupportSchedule + 当前状态 -> 连续 MainPath
validation.py          # 任意区间动力学与能力包络验证器
diagnostics.py         # ValidationIssue / ValidationReport / PlanningResult
reference.py           # 第三阶段：MotionReference / 参数化模板 / 骨架采样
sim_playback.py        # 采样骨架 -> MuJoCo qpos IK -> 仿真器渲染
visualize.py           # 支撑序列 / 主路径 / 运动参照 MP4
render_cases.py        # 批量生成代表性用例视频
test_support_sequence.py
test_trajectory.py
test_reference.py
```

`render_cases.py` 会为三个用例同时生成第一阶段支撑序列回放、第二阶
段验证后的 `MainPath` 回放和第三阶段骨架采样回放。

当前从 `standing` 姿态提取的默认 gait envelope：

```text
mass                    40.844 kg
stance_width             0.180 m
foot_length              0.265 m
foot_width               0.094 m
leg_length               0.830 m
nominal_com_height       0.850 m
nominal_step_length      0.344 m
max_step_length          0.477 m
double_support_duration  0.150 s
swing_duration           0.350 s
settle_duration          0.600 s
control_dt               0.050 s
```

生成可视化用例：

```bash
PYTHONPATH=. python3 -m baseline.humanoid21.mpc.main_path.render_cases
```

输出目录：

```text
baseline/humanoid21/mpc/main_path/visualizations/
├── hold.mp4
├── forward_step.mp4
├── diagonal_two_steps.mp4
├── hold_main_path.mp4
├── forward_step_main_path.mp4
├── diagonal_two_steps_main_path.mp4
├── hold_reference.mp4
├── forward_step_reference.mp4
└── diagonal_two_steps_reference.mp4
```

第一阶段视频仍只展示二维支撑回放：脚在支撑区间内按颜色绘制，离地时
不绘制；视频时间与机器人时间轴一致。

第二阶段 `*_main_path.mp4` 使用 `path.state_at(t)` 逐帧查询，显示：

- COM 投影轨迹与当前 COM；
- 当前活动脚底矩形和实际等效受力点；
- 左右脚支撑区间；
- `|l|/m` 与 `|L|/(m·leg)` 曲线；
- 左右脚合力与法向力曲线。

第三阶段 `*_reference.mp4` 直接用 MuJoCo 仿真器渲染：每帧对
`reference.sample_pose(t)` 的骨架关键点解 `ReferenceIKSolver`，把完
整 `qpos` 写入仿真器后用广播相机成像，可以直接检查采样参照在真实
模型上是否合理（脚是否落地、膝肘方向、手臂形态）。

`animate_motion_reference` 保留 matplotlib 调试视图：俯视脚位/COM/
骨盆、侧视与正视骨架棍图、摆臂/屈膝/脚高曲线。

PNG 静态支撑序列图接口仍保留，便于单帧检查和测试。

当前测试覆盖：

- Humanoid21 站姿几何与 gait envelope 提取；
- `MainPathState` 中朝向、脚点、支撑标志与受力校验；
- 朝向决定左右脚几何和侧向首迈脚；
- 目标很近时生成开放式保持区间；
- 前向目标生成右脚先动的交替支撑序列；
- 更远目标自动增加左右脚步数；
- `SupportInterval` 字段级校验；
- `MainPath.state_at(t)` 完整状态查询与越界错误；
- 静态保持和短步/斜向主路径的独立验证；
- 任意时间区间动力学残差；
- 无支撑弹道段解析积分；
- 单支撑期 inactive 脚零外力；
- 终端位置、线动量、角动量与静态力矩；
- 结构化前置冲突与内部能力违规诊断；
- `MotionReferenceState` 完整字段与模板参数；
- 步态异侧协调（摆臂/骨盆/躯干扭转方向）；
- 参照随时间的连续性；
- 骨架采样的一致性（两段腿长、支撑踝高度、关键点完备）；
- 仿真器 IK 求解残差与 MP4 渲染输出；
- PNG 与 MP4 可视化输出。
