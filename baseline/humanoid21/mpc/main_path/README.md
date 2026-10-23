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

第一版没有目标朝向，因此最终站姿保持当前 `heading_yaw`。需要转向、
背向移动或指定目标朝向的任务，应在后续请求中加入 `target_yaw` 或
单独的转向目标。

## 3. 第二阶段：主路径轨迹规划器

`MainPathTrajectoryPlanner` 消费：

```python
MainPathTrajectoryRequest:
    current_state: MainPathState
    support_schedule: SupportSchedule
    target_pos_world: np.ndarray
    constraints: MainPathConstraints
```

这里的 `support_schedule` 是第一阶段的输出，不是外部输入。

它输出：

```python
MainPath:
    times: np.ndarray              # (N,)
    com_pos: np.ndarray            # (N,3)
    lin_mom: np.ndarray            # (N,3)
    ang_mom: np.ndarray            # (N,3)
    left_point: np.ndarray         # (N,3)
    left_force: np.ndarray         # (N,3)
    right_point: np.ndarray        # (N,3)
    right_force: np.ndarray        # (N,3)
    left_active: np.ndarray        # (N,), bool
    right_active: np.ndarray       # (N,), bool
    diagnostics: dict
```

`*_active=False` 时，对应 `force_world` 必须为零；对应 `point` 不参与
动力学。

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

规划器必须保证输出满足离散版本的这些方程，而不是只生成几何上
平滑的质心曲线。

## 5. 受力约束

第一版使用可配置的脚部等效受力限制：

```python
FootLoadLimits:
    max_force: float
    max_force_rate: float
    friction: float
    min_normal_force: float
    max_normal_force: float
```

对每只活动支撑脚：

\[
f_{z,\min}\le f_z\le f_{z,\max}
\]

\[
\sqrt{f_x^2+f_y^2}\le\mu_{\rm safe}f_z
\]

\[
|f|\le f_{\max},\qquad |\dot f|\le\dot f_{\max}
\]

第一版不引入接触 torque，也不规划惯量张量。

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

## 7. 求解方式

第一版建议使用离散求解：

1. 用 `CONTROL_DT=0.05s` 生成时间网格；
2. `SupportSequencePlanner` 先生成 `SupportSchedule`；
3. 从 schedule 求每个采样点的活动支撑集合；
4. 以 `c,l,L` 为轨迹变量施加离散动力学约束；
5. 对每个采样点求解左右脚受力分配；
6. 最小化平滑性、受力变化和与边界目标的偏差；
7. 不可行时返回结构化诊断，不输出伪造路径。

失败诊断至少包含：

- 失败时刻；
- 当前活动支撑集合；
- 违反的约束；
- 目标位置是否超出步长/受力/平衡包络；
- 可调整的边界余量。

## 8. 与下游的边界

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

## 9. 当前实现与文件划分

第一阶段 `SupportSequencePlanner` 已实现，当前文件：

```text
contracts.py           # MainPathState / MainPathRequest / SupportInterval / SupportSchedule
gait_model.py          # 从 Humanoid21 站姿提取步态数值包络
support_sequence.py    # MainPathState + target_pos -> SupportSchedule
visualize.py           # 支撑点俯视图、支撑时间轴 PNG 与通用 MP4 回放
render_cases.py        # 批量生成代表性用例视频
test_support_sequence.py
```

后续第二阶段文件仍为：

```text
trajectory.py          # SupportSchedule + 当前状态 -> MainPath
force_allocation.py    # 双脚等效受力分配与约束检查
diagnostics.py         # 结构化失败原因与余量
```

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
└── diagonal_two_steps.mp4
```

视频是这一阶段结果的通用展示形式：

- 左上：左右脚支撑点俯视回放、当前活动脚和 `heading_yaw`；
- 右上：左右脚支撑区间时间轴和当前时间；
- 左下：活动支撑脚数量与左右脚支撑占空曲线；
- 右下：左右脚剩余规划脚程曲线。

这里不绘制尚未规划的 `c,l,L` 轨迹，避免把支撑序列结果伪装成完整
主路径。PNG 静态图接口仍保留，便于单帧检查和测试。

当前测试覆盖：

- Humanoid21 站姿几何与 gait envelope 提取；
- `MainPathState` 中朝向、脚点、支撑标志与受力校验；
- 朝向决定左右脚几何和侧向首迈脚；
- 目标很近时生成开放式保持区间；
- 前向目标生成右脚先动的交替支撑序列；
- 更远目标自动增加左右脚步数；
- `SupportInterval` 字段级校验；
- PNG 与 MP4 可视化输出。
