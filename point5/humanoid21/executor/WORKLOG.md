# Executor 工作日志

## 2026-11-04（至 ~18:00）

### 本轮完成

**算法2（ActionComputer）首版落地**——`compute(ctx, landmarks, wsim) -> (21,)`

结构按商定的 Min/Subject-to 滚动时域：
- `obs_to_core_state(obs)`：96维 obs → `set_core_state` 载荷（x̂₀ 重建）。
  root x/y 观测没有→置0（局部规划不需要）；yaw 不可观→只保留 tilt。
  sqrt 压缩反解：关节 `sign·(2v)²`、角速度 `sign·2v²`。
- `_fk_decode(wsim, obs)`：解码 rollout 中每步观测 → 世界系关键点 +
  局部系原点/yaw（frame0 锚定基准）
- `_landmark_error`：D_LM——com(6×) + foot xy(4×) + vel(2×，仅约束时)
- CEM：分段常值动作（8步/4chunk），24种群×4迭代×6精英，min-over-horizon
  （tau 隐式 = argmin_t，未显式优化到达时刻）
- rollout 间 `set_core_state` 快照恢复；只执行 u₀

### 验证

| 目标锚点 | hold基线 | CEM |
|---|---|---|
| lm[1] coil | 0.011 | 0.006 |
| lm[2] 左脚离地 | 0.071 | 0.043 |
| lm[3] swing姿态 | 0.101 | 0.019 |

compute 耗时 ~2.5s/次（4×24×8 rollout 的 FK 解码是瓶颈）。

### 已知问题 / TODO
- 只打分最近锚点（landmarks[1]），无多锚点次序项、无 P5 项、无动作平滑项
- `vel_w` 未接入 landmark_error（观测可得但没用）
- CEM 随机种子未固定，结果每次略不同
- 无 torso_yaw / force_ratio / cop 误差项
- 误差权重拍值，未按量纲归一
