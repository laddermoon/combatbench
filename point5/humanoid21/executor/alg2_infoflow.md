# 算法2 · 信息流分解

> 输入：`Observation, Clip, Landmarks` → 输出：`Action (21,)`
> 规范来源：`point5/humanoid21/bootstrip.md`

## 信息项（I）

| # | 名称 | 内容 | 来源 |
|---|---|---|---|
| I1 | 当前 P5 | 当前五点姿态 (5,3) | Observation 解码；clip kf0 亦有同一份 |
| I2 | 下一 landmark 的 P5 | 下一个锚点对应的五点姿态 | `F1(landmarks,clip)`  |
| I3 | 下一帧目标 P5 | 本控制步应到达的五点姿态 | `F2(I1, I2, I8)` |
| I4 | 当前姿态 | 关节位置 (21,) | obs[0:21] |
| I5 | 目标姿态 | 关节空间参考 (21,) | `F3(I3, I4, I8?)` |
| I6 | 当前质心位置 | com (3,) | obs 解码（工具1/8） |
| I7 | 锚点质心位置 | 下一 landmark 的 com (3,) | Landmarks |
| I8 | 下一帧质心目标 | 本控制步的质心位置目标 | `F4(I6, I7)` |
| R | 结果动作 | 归一化 PD 目标 (21,) | `F5(I8, I5)` |
