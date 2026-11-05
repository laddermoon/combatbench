# tools/ — 无状态物理先验工具集

执行器流水线的"先验层"：**每个模块都是纯函数，P5/状态进、结果出**，
不持有任何执行进度。算法1（`centroidal_planner.py`）和算法2
（`action_computer.py`）按编号调用它们；工具自身可以独立改进实现
而不影响管线语义。

P5 约定：5 个关键点的局部坐标 `(5,3)`——头、左手、右手、左脚、
右脚，重力对齐局部系（x 前、z 上）。

## 工具清单

| # | 文件 | 功能 | 接口 |
|---|---|---|---|
| 1 | `com_solver.py` | P5 局部姿态 → 推荐质心位置 | `com_estimate(p5) -> (3,)`；`mass_estimate(p5) -> MassProxy` |
| 2 | `pose_solver.py` | P5 局部姿态 → Humanoid21 参考关节姿态 + (21,) 动作 | `Humanoid21PoseSolver().solve(p5) -> PoseResult{qpos(21,), action(21,), satisfied, energy}` |
| 3 | `inertia_solver.py` | P5 局部姿态 → 绕质心转动惯量 | `inertia_estimate(p5) -> (3,3) kg·m²`；`inertia_per_mass`；`gyration_z` |
| 4 | `capability_solver.py` | P5 → 质心变化能力预期（LIP/捕获点模型） | `capability(p5) -> Capability`；`margin(d)`/`a_max(d)`/`v_max(d)` |
| 5 | `landmark_viz.py` | LandMark 锚点序列 → MP4 走查动画 | `render_landmarks(landmarks, out_mp4)` |
| 6 | `cop_solver.py` | P5 脚点 → CoP 可行域多边形 | `cop_ranges(p5) -> [poly|None ×2]`；`foot_cop_range(foot_xy)` |
| 7 | （在 `../support_planner.py`） | clip → 支撑转移段检出 | `support_transitions(clip) -> List[SupportTransition]` |
| 8 | `start_solver.py` | Observation → 起点 LandMark | `extract_landmark(obs, decoder) -> LandMark` |

## 各工具说明

### 工具1 `com_solver` — P5 → 推荐质心

包 `point5.mass_model.mass_proxy`：无模型的"虚拟体"质心估计器
（对姿态解出的 P5 ~4cm RMSE）。算法1 拿它定锚点 `com`；算法2 的
反蹲下限也用它评估 rollout 状态。

### 工具2 `pose_solver` — P5 → 全身参考姿态/动作

包 `mechanics.solve_equilibrium`（两段式均衡求解，~0.2s/次）。
输出生成器 PD 目标（`(21,)` 归一化），是算法2 里 `a_ref` 的
唯一来源——锚点的 P5 意图经它翻译成关节空间参考。
`solve_fast`/`build_lut` 预留量化热启动表接口。

### 工具3 `inertia_solver` — P5 → 转动惯量

从 mass_proxy 取绕质心 (3,3) 惯量矩阵（P5 局部系），乘 MASS_KG
换算绝对量。`gyration_z` 给 z 向回转半径。算法1 估计角动量/
姿态恢复能力时用。

### 工具4 `capability_solver` — P5 → 质心能力预期

物理经验模型（LIP/捕获点）：支撑域=接地脚 patch 凸包；
方向 d 上加速度权限 `a(d) = g·margin(d)/z0`（封顶 μg），
速度上限 `v(d) = sqrt(g/z0)·margin(d)`。负 margin = 静态不稳。
算法1 判"当前能不能迈出下一步/要不要先矫正"；算法2 用
`_hull2d`/`_PATCH` 算捕获点出域罚。

### 工具5 `landmark_viz` — LandMark 序列 → MP4

算法1 输出的走查动画：每个锚点驻留 ~0.6s——蓝球=com+垂线、
橙/绿 patch=右/左脚着地（透明度∝force_ratio）、浮空脚=小球标记、
红点=CoP、青色箭头=vel 约束、拖尾+序号字幕。直观检查锚点序列
的支撑事件/姿态意图对不对。

### 工具6 `cop_solver` — 脚点 → CoP 可行域

足底 patch 几何（x∈[-0.07,+0.14]，半宽 0.057）。脚踝高度判
接触形态：平贴 → 全 patch 多边形；翘边 → CoP 退化为接触边集合；
离地 → None。支撑规划和锚点 CoP 项都用它。

### 工具7 `support_transitions` — clip → 支撑转移段

**实现在 `../support_planner.py`**（与脚规划器同文件，未独立成
模块）：逐段检出关键帧间的摆动脚/支撑脚，返回需要支撑转移的段
列表；None = 无转移。是 `plan_footsteps` 的前置检出。

### 工具8 `start_solver` — Observation → 起点 LandMark

当前状态即第一个锚点：com 由工具1 从 FK 解码 P5 估；vel 读
obs[46:49] 转局部系；force_ratio 读 obs[52:54]；CoP≈承重脚 xy；
foot_pos=解码脚 xy；torso_yaw=None（v1 不用）。

## 调用关系

```
gen_action(ctx)
├── 算法1 centroidal_planner
│     ├── 工具8 extract_landmark(obs)     起点
│     ├── 工具7 support_transitions(clip) 支撑段检出 → plan_footsteps
│     ├── 工具1/4/6  com/margin/CoP       锚点合成
│     └── gen_landmarks                    帧驱动锚点序列
└── 算法2 action_computer
      ├── 工具2 pose_solver               锚点P5 → a_ref
      └── 工具1/4  com/hull                rollout 平衡评分
```
