# Humanoid21 简化支撑主路径 MPC

> 状态：设计契约  
> 第一轮目标：从屈膝双脚站立出发，完成一次重心转移和短距离迈步，末端重新站稳。

本目录实现我们约定的最简底层规划抽象。设计原则是：主路径只规划
质心运动、动量、角动量和双脚等效受力；关节姿态、身体协调和动作细节由
后续参考姿态生成与仿真搜索补齐。

## 1. 第一轮任务定义

输入只包含每只脚的支撑时间区间和支撑位置。最小示例：

```text
left:
  [t0, t_end] -> p_left_0

right:
  [t0, t1]     -> p_right_0
  [t2, t_end]  -> p_right_1
```

这里 `t1 < t2` 的区间空缺表示右脚不在支撑中；左脚全程支撑。输入
只记录位置和时间边界，不包含“准备离地”“重心转移完成”“要求平衡”
等阶段语义，也不直接指定关节轨迹。

第一轮只允许平地、屈膝姿态附近、无腾空的小步。跑步、对手接触和
大幅转向不在第一版范围内。

## 2. 规划状态与外力

主路径状态严格按照简化抽象定义：

\[
z=(c,l,L)
\]

字段含义：

| 字段 | 含义 | 形状 |
|---|---|---|
| `com_pos` \(c\) | 全身质心，world frame | `(3,)` |
| `lin_mom` \(l\) | 全身线动量 | `(3,)` |
| `ang_mom` \(L\) | 关于 `com_pos` 的角动量 | `(3,)` |
| `time` | 参考时间 | scalar |
| `left_contact` | 左脚作用点与外力 | point + force |
| `right_contact` | 右脚作用点与外力 | point + force |

每只脚使用一个等效作用点。支撑阶段内作用点固定；摆动阶段该脚
外力必须为零。质量是模型参数，不是规划变量。

系统动力学：

\[
\dot c=\frac{l}{m}
\]

\[
\dot l=mg+f_L+f_R
\]

\[
\dot L=(p_L-c)\times f_L+(p_R-c)\times f_R
\]

第一轮不规划惯量张量、关节状态、全身姿态或接触扭矩。

## 3. 支撑区间与任务结束条件

支撑输入的数据形式只包含三个信息：脚、时间区间、世界坐标点。

```python
SupportInterval:
    foot: Literal["left", "right"]
    start_time: float
    end_time: float
    point_world: np.ndarray   # (3,)
```

同一只脚的区间不得重叠。某只脚没有覆盖到的时间段表示该脚不支撑。
任意采样时刻的活动支撑集合由所有覆盖该时刻的区间推出：

- 双脚都在区间内：双脚支撑；
- 只有左脚或右脚：单脚支撑；
- 两只脚都没有：腾空（第一轮拒绝）。

输入本身不包含平衡标记。第一轮任务的“初始站稳、末端站稳”由任务
配置单独给出，例如 `settle_start` / `settle_end` 或最终保持时长；
在这些窗口内，规划目标为：

\[
l\approx0,\qquad L\approx0
\]

且在当前支撑点下存在一组可行脚部受力，使：

\[
f_L+f_R=-mg
\]

\[
(p_L-c)\times f_L+(p_R-c)\times f_R=0
\]

每只支撑脚还必须满足后续定义的受力约束。最终能否保持，由全身
branch simulation 和实际 rollout 验收，而不是由支撑输入声明。

## 4. 主路径生成

输入：

```text
初始 CentroidalState
左右脚 SupportInterval 列表
任务结束与最终保持窗口
质心参考高度 / 高度包络
每脚允许作用区域与受力约束
阶段持续时间边界
```

输出：

```python
CentroidalPath:
    times: np.ndarray
    com_pos: np.ndarray
    lin_mom: np.ndarray
    ang_mom: np.ndarray
    left_point: np.ndarray / mask
    left_force: np.ndarray
    right_point: np.ndarray / mask
    right_force: np.ndarray
    phase: support phase per sample
```

第一版使用固定或弱约束的质心高度，并将 \(L\) 的参考值设为零。可以采用
离散轨迹求解：以 `CONTROL_DT=0.05s` 为主路径采样间隔，在满足动力学的
前提下求解一条平滑、可行的 \(c(t),l(t)\)，再用一个小规模分配问题求
\(f_L(t),f_R(t)\)。

双脚受力分配优先满足：

\[
f_L+f_R=m(\ddot c-g)
\]

\[
(p_L-c)\times f_L+(p_R-c)\times f_R\approx0
\]

并满足每只支撑脚：

\[
f_z\ge0
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

如果不能求出可行轨迹，主路径生成应 fail loud，而不是返回“几何平滑但
动力学不可实现”的曲线。允许的自动调整只能是已声明范围内的步长时间
或质心包络调整。

## 5. 从主路径生成全身参考姿态

主路径之后，使用经验函数生成协调参考姿态：

```python
ReferencePosePath:
    times
    root_pos / root_rot reference
    joint_pos_norm reference
    left/right foot pose reference
    support phase
```

第一版经验模板：

1. 初始与最终姿态使用固定屈膝 standing pose；
2. 根据质心参考估计 root/pelvis 位置；
3. 摆动脚生成“抬起—前移—放下”的 world-frame 轨迹；
4. 用下肢运动学求解髋、膝、踝的参考角度；
5. 躯干保持直立或小幅前倾；
6. 手臂使用固定保护姿态或小幅协调摆动；
7. 生成参考姿态后计算实际全身 COM，并做小量迭代修正。

参考姿态是动作搜索的名义中心，不等于可直接发送的 PD target。承重、
加减速和落地可能需要 target 与当前关节姿态存在偏差。

## 6. 动作搜索与执行

动作层继续使用 observation-only branch simulator，不读取真实 `MjData`
或 `get_core_state()`。

流程：

```text
Observation
  -> 解码自身状态
  -> 生成 / 对齐当前参考姿态
  -> 在参考附近生成少量协调动作候选
  -> branch rollout
  -> 选择 cost 最低候选
  -> 输出当前 action
```

第一版候选不应在所有 21 个关节上独立大范围扰动，而是围绕参考动作
搜索少量协调参数，例如：

- root/pelvis 前后、左右、上下偏移；
- 支撑腿膝踝协调修正；
- 摆动脚进度和落点修正；
- 躯干/手臂小范围协调修正。

评分项至少包括：

- `com_pos` 与 `lin_mom` 跟踪；
- 末端 `lin_mom` / `ang_mom` 是否消除；
- root 姿态与 joint tracking；
- 脚是否按计划支撑/摆动；
- 落地时刻、落点与落地速度；
- action 平滑度与偏离参考的量；
- 摔倒、非预期接触和 solver 异常。

每个候选必须在 branch 中以闭环方式运行；如果实际状态与参考偏差过大，
应提前失败并允许重规划。

## 7. 计划状态与复用

策略内部保存：

- 左右脚 `SupportInterval` 列表；
- 当前 `CentroidalPath`；
- 当前 `ReferencePosePath`；
- 当前执行阶段与时间；
- 最近一次 branch rollout 的预测状态；
- 当前选中的低层修正参数。

默认按控制周期复用当前计划和修正参数；满足以下任一条件才重规划：

- 到达下一阶段边界；
- 质心/动量跟踪误差超阈值；
- 脚接触时序与计划不一致；
- 仿真评分预测末端无法站稳；
- 到达固定 replan interval。

## 8. 第一轮验收

第一轮验收任务：

```text
初始：屈膝双脚站立
动作：向支撑脚转移重心，另一只脚向前迈一小步
末端：双脚支撑，并在落脚后继续稳定至少 2 秒
```

建议初始指标：

- 全程不摔倒；
- 迈步脚只在计划窗口内离地；
- 摆动脚在计划时间窗内落地；
- 末端双脚接触并保持；
- `lin_mom` / `ang_mom` 收敛到小阈值；
- root height 与 uprightness 保持在屈膝域内；
- 记录质心跟踪误差、落脚误差、规划耗时和动作修正幅度。

阈值先标定，再写入测试，不预先假定为物理保证。

## 9. 建议文件划分

```text
support_plan.py       # SupportInterval / phase derivation / validation
centroidal_path.py    # c,l,L 与脚部受力的主路径生成
reference_pose.py     # 主路径 -> 协调全身参考姿态
step_policy.py        # Observation-only 动作搜索 Policy
cost.py               # branch rollout tracking / terminal cost
test_support_plan.py
test_centroidal_path.py
test_reference_pose.py
test_step_policy.py
```

后续实现应保持这层边界：主路径只处理 `c,l,L` 与脚部受力；任何关节
姿态或动作细节都只能出现在 `reference_pose.py` 和 `step_policy.py`。
