# 算法2 · 信息流分解

> 输入：`Observation, Clip, SupportPlan` → 输出：`Action (21,)`
> SupportPlan = 统一首帧 P5 系的 SupportState 序列（com + feet + yaw）
> 规范来源：`point5/humanoid21/bootstrip.md`

## 信息项（I）

| # | 名称 | 内容 | 来源 |
|---|---|---|---|
| I1 | 当前 P5 | 当前五点姿态 (5,3) | 取Clip.Base |
| I2 | 下一锚点的 P5 | 下一 SupportState 对应帧的五点姿态，`clip.p5[st.for_frame]` | `F1(plan,clip)` |
| I3 | 下一帧目标 P5 | 本控制步应到达的五点姿态 | `F2(I1, I2, I8, I6, I7)` |
| I4 | 当前姿态 | Pose（qpos + root_z + g_body + 速度） | `Pose.from_obs(obs)` |
| I5 | 目标姿态 | Pose | `F3(I3, I4, I8)` |
| I6 | 当前支撑态 | SupportState，首帧 P5 系 | obs 解码（f12）为主，plan[0] 备选 |
| I7 | 锚点支撑态 | 下一 SupportState，首帧 P5 系 | SupportPlan |
| I8 | 下一帧支撑态目标 | 本控制步的 SupportState 目标 | `F4(plan)`（I6/I7/I9 内部） |
| I9 | 下下锚点支撑态 | 下下个 SupportState，前瞻输入 | SupportPlan |
| R | 结果动作 | 归一化 PD 目标 (21,) | `F5(I8, I5)` |
