# Humanoid21 第一版主路径规划器

> 范围：通用 reduced-order main-path planning。
> 输入是逐脚独立的支撑区间；输出是质心、线动量、角动量和双脚等效
> 受力随时间变化的主路径。

本目录只负责主路径规划，不定义某个具体任务，也不生成关节动作。它
描述的只是：在指定的脚部支撑位置与时间区间下，机器人整体的
`com_pos / lin_mom / ang_mom` 应当怎样变化，以及每只脚需要提供什么
等效外力。

## 1. 抽象边界

规划状态严格为：

\[
z=(c,l,L)
\]

| 字段 | 含义 | 坐标系 / 形状 |
|---|---|---|
| `com_pos` \(c\) | 全身质心位置 | world `(3,)` |
| `lin_mom` \(l\) | 全身线动量 | world `(3,)` |
| `ang_mom` \(L\) | 关于 `com_pos` 的角动量 | world `(3,)` |
| `time` | 主路径时间 | scalar |

每只脚使用一个等效支撑点和一个等效外力：

| 字段 | 含义 |
|---|---|
| `point_world` | 脚部等效外力作用点 |
| `force_world` | 该脚施加在机器人上的外力 |

质量、重力和模型时间步来自 `CentroidalModelSpec`，不是规划变量。

第一版不规划：

- 关节位置、关节速度或 root pose；
- marker/keypoint 轨迹；
- 身体姿态或惯量张量；
- 脚部姿态、接触法向独立变化；
- 接触扭矩；
- 对手动作；
- 全身动作可行性保证。

这些内容由下游的参考姿态生成和全身控制/仿真验证补齐。

## 2. 支撑输入

输入是左右脚各自独立的 `SupportInterval` 序列：

```python
SupportInterval:
    foot: Literal["left", "right"]
    start_time: float
    end_time: Optional[float]
    point_world: np.ndarray  # (3,)
```

示例：

```text
left:
  [0.0, 0.8] -> pL0
  [1.4, ∞)   -> pL1

right:
  [0.0, 0.5] -> pR0
  [1.0, 1.4] -> pR1
  [2.0, ∞)   -> pR2
```

规则：

- 一只脚可以有多个支撑区间，表示重复迈步；
- `end_time=None` 表示这是最后支撑区间，持续到规划 horizon；
- 两脚之间不做数据层关联约束；
- 某只脚没有区间覆盖的时间表示该脚不支撑；
- 两脚都没有覆盖的时间表示无支撑，数据层允许；
- 支撑区间内，该脚的 `point_world` 固定；
- 不在支撑区间内的脚，外力必须为零。

数据校验只允许字段级 sanity check：

- `foot` 合法；
- `start_time` 有限；
- `end_time` 为 `None` 或有限值；
- 非空 `end_time` 必须大于 `start_time`；
- `point_world` 为有限 `(3,)`；
- 同一只脚的区间不得重叠，且 `end_time=None` 后不能有后续区间。

是否允许无支撑段、是否要求最终平衡、以及支撑时序是否合理，属于
规划器和任务约束，不属于输入 schema。

## 3. 动力学

对每个采样时刻，活动支撑集合由所有覆盖该时刻的 `SupportInterval`
推出。没有激活区间的脚，其 `force_world=0`。

系统动力学为：

\[
\dot c=\frac{l}{m}
\]

\[
\dot l=mg+f_L+f_R
\]

\[
\dot L=(p_L-c)\times f_L+(p_R-c)\times f_R
\]

其中未支撑的脚对应的 \(p,f\) 不参与求和。

当双脚都不支撑时：

\[
f_L=f_R=0
\]

因此线动量按重力演化：

\[
\dot l=mg
\]

由于没有外力矩作用，关于质心的角动量保持恒定：

\[
\dot L=0
\]

## 4. 主路径输出

规划器输出一条时间索引的 `MainPath`：

```python
MainPath:
    times: np.ndarray              # (N,)
    com_pos: np.ndarray            # (N,3)
    lin_mom: np.ndarray            # (N,3)
    ang_mom: np.ndarray            # (N,3)
    left_point: np.ndarray         # (N,3), invalid when inactive
    left_force: np.ndarray         # (N,3)
    right_point: np.ndarray        # (N,3), invalid when inactive
    right_force: np.ndarray        # (N,3)
    left_active: np.ndarray        # (N,), bool
    right_active: np.ndarray       # (N,), bool
    diagnostics: dict
```

语义：

- `*_active=False` 时，对应 `*_force` 必须为 0；
- 活动支撑脚的 `point` 等于其当前 `SupportInterval.point_world`；
- 非活动脚的 `point` 不参与动力学，可以保留最后一次位置用于诊断，
  但不能被解释成支撑点；
- 所有状态和外力都是 world-frame 数量；
- `ang_mom` 是关于 `com_pos` 的角动量。

## 5. 规划器职责

`MainPathPlanner` 的职责是：

1. 校验和排序每只脚的 `SupportInterval`；
2. 在给定 horizon 和 `dt` 上生成活动支撑集合；
3. 从初始 `CentroidalState` 开始生成 `c,l,L`；
4. 为每个采样点求解满足动力学与约束的脚部外力；
5. 检查不可行支撑组合并返回结构化诊断；
6. 输出可用于下游参考姿态生成和全身仿真的 `MainPath`。

它不负责：

- 判断输入属于“走路、跑步、转身或打击”哪种任务；
- 生成关节或 root 轨迹；
- 判断全身姿态是否美观；
- 证明全身控制器一定能实现该路径；
- 替代完整仿真验证。

## 6. 受力约束

第一版脚部受力使用保守、可配置的约束集合：

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
|f|\le f_{\max}
\]

\[
|\dot f|\le\dot f_{\max}
\]

第一版不引入接触 torque。若以后恢复脚底压力中心能力，可以把
`point_world` 从固定点扩展为允许在缩小支撑域内变化的变量；这仍然是
规划参数变化，不改变 `SupportInterval` 输入格式。

## 7. 平衡与开放式支撑段

开放式 `SupportInterval.end_time=None` 是“该脚之后不再换支撑点”的
输入事实。规划器可以把最终开放式区间解释为末端保持目标，但这仍然是
规划层语义。

对于最终保持窗口，规划目标为：

\[
l\approx0,\qquad L\approx0
\]

且存在一组满足受力约束的支撑力，使：

\[
f_L+f_R=-mg
\]

\[
(p_L-c)\times f_L+(p_R-c)\times f_R=0
\]

这只是主路径层的可行条件。最终能否站稳，必须由下游全身仿真验证。

## 8. 第一版求解方式

第一版建议使用离散轨迹求解，而不是先做复杂连续优化：

1. 按 `CONTROL_DT=0.05s` 生成时间网格；
2. 由支撑区间生成每个采样时刻的 `left_active/right_active`；
3. 以 `c,l,L` 为轨迹变量，施加离散动力学约束；
4. 对每个支撑组合求解脚部受力分配；
5. 最小化平滑性、受力变化和与任务配置中少量边界目标的偏差；
6. 不可行时返回原因，不输出伪造路径。

边界条件可以包括：

- 初始 `CentroidalState`；
- 最终 `lin_mom/ang_mom` 目标；
- 可选的初始或最终 `com_pos` 包络；
- 可选的质心高度包络。

主路径的内部实现可以先用小规模 QP / least-squares。支撑几何或受力
约束不可行时，应 fail loud，并返回：

- 失败采样点；
- 活动支撑集合；
- 违反的约束；
- 可用的约束余量。

## 9. 与下游的边界

`MainPath` 只回答：

> 在这些支撑位置和时间下，整体质心运动与脚部等效外力应该是什么。

下游模块再做：

```text
MainPath
  -> 经验函数生成全身参考姿态
  -> 在参考附近搜索动作
  -> Observation-only branch simulator 验证
  -> 输出实际控制 action
```

下游可以要求主路径重规划，例如调整时间、步长或受力约束，但下游失败
不改变本目录的职责边界。

## 10. 建议文件划分

```text
contracts.py          # SupportInterval / FootLoadLimits / MainPath
support_schedule.py   # interval validation + active-contact lookup
centroidal_solver.py  # c,l,L 与脚部受力的离散求解
diagnostics.py        # feasibility issues / margins / failure reasons
test_contracts.py
test_support_schedule.py
test_centroidal_solver.py
```

测试应覆盖：

- 单脚重复支撑区间；
- 双脚任意重叠；
- 双脚同时不支撑时按弹道规则演化；
- 开放式最终支撑区间；
- 非支撑脚 force 恒为零；
- 摩擦力、法向力和力变化率约束；
- 不可行支撑组合的结构化失败；
- 输出路径满足离散动力学。
