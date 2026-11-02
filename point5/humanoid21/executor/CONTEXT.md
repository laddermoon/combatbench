# CONTEXT.md — point5/humanoid21/executor/

## 一句话

P5 指令的**无状态伺服后端**：`Executor.gen_action(Context) -> Action`。
Executor 是策略后端——只能看到 policy 的 96 维 observation，拿不到
sim 内部结构；只持有自己的内部仿真器，不维护任何执行进度。

## 依赖结构（大逻辑）

```
gen_action(ctx)
  0  _verify_anchor      clip 首帧必须锚定 obs 解出的当前 P5（已实现）
  1  obs -> P5            _extract_p5（stub，obs 解码器）
  2  obs -> 质心状态      CentroidalPlanner.extract_centroidal_state（stub）
  3  算法1 centroidal     CentroidalPlanner.plan（stub）
  4  算法2 action         ActionComputer.compute（stub）
      CNext -> P5Next -> [可选 Humanoid21RefNext(工具2)] -> Action
```

| 件 | 文件 | 状态 |
|---|---|---|
| 算法1 质心规划 | `centroidal_planner.py` | stub |
| 质心状态表示 | `centroidal.py` | ✅ `z=(c,l,L)` 9维 + `com_vel=l/m` |
| 算法2 动作计算 | `action_computer.py` | stub |
| 工具1 P5→推荐质心 | `com_solver.py` → `point5.mass_model` | 已实现 |
| 工具2 P5→参考姿态 | `pose_solver.py` → `mechanics.solve_equilibrium` | solve 已接通；LUT 加速 stub |
| obs→P5 解码 | `executor._extract_p5` | stub（按 OBSERVATION_zh 布局） |

## 契约（contracts.py）

- `Observation` = policy 收到的 96 维平铺向量（不是 derived dict）。
- `Context` = {observation, clip, meta}；clip 是调用方每次锚定当前
  状态切好的**剩余指令**。
- `Action` = (21,) 归一化 PD 目标，即 `sim.set_action` 输入。
- 锚定校验容差 `anchor_atol = 1e-6`；不一致抛 `AnchorError`，
  Executor 不补接。

## 质心状态（centroidal.py）

唯一保留的核心表示：`z = (c, l, L)`（质心位置/线动量/角动量，
`com_vel = l/m`）。接触模式、落脚点、阶段时间属伴随信息。
旧 centroidal_dynamics 包的其余实现全部删除——过度复杂。

## 已证结论（前一轮失败迭代，勿重踩）

- 本机**没有任何姿态开环 PD 稳定**（4 个关键帧姿态全部 ~1.15s 内倒）
  ——动作层必须持续主动平衡。
- CEM+PD 可保持双足站立、可完成单脚摆动到达，但迈步落地相在
  预算内找不到稳定序列——算法2 是主战场。
- IK 姿态本身能正确放脚（放置后漂移 <2cm），问题只在动态转移。
