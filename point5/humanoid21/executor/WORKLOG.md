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

### 18:00 追加

- **回放发现**：u0 瞬时能逼近锚点(step1 0.0028)但持 8 步发散下沉——符合预期（真实使用每步重规划）
- **热启动负结果**：用工具2 solve_equilibrium 的目标帧均衡姿态做 CEM mean 反而更差（cold 0.0038 vs warm 0.0067）——静态姿态不含动态转移信息，陷局部最优
- **cost 补齐**：加 `force_ratio` 误差项（obs[52:54] 直接读实测触地力，3× 权重）——swing/离地锚点的核心就是卸载，之前完全没用到
- **LandMark 加 `frame` 元信息**（实现细节）：锚点记录来源 kf 下标，供算法2 的 P5 项/种子用

### ~18:30

- **多锚点顺序打分 `_seq_score`**：rollout 步推进模型——状态误差 < reach_eps 即"到达"并切下一锚点（隐式 tau），residual = 当前+后续锚点误差和 - 进度奖励；CEM 主循环已接入
- git push 故障排查：新 shell 无 ssh-agent → `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519` 恢复

### ~19:00 闭环测试

- 4步连续 gen_action：com_z 缓降 1.278→1.238，第4步 lms 7→10（矫正锚点出现）——**机器人在缓慢失衡**，cost 缺直立项
- 待改：加 com_z/upright 维持惩罚到打分

### ~19:30 平衡项修复

- 加 upright 惩罚：tilt²(2×) + com_z<0.87 下沉(1.5×)
- 重跑闭环：com_z 1.282→1.258 缓稳、lms 恒 7（不再触发矫正）——CEM 不再把机器人往倒里选
- 遗留：无 P5 姿态项、无 cop 项、速度约束未接入 vel_w

### 备注：git push 间歇失败（publickey）

- 现象：`ssh -T git@github.com` 认证成功但 `git push` 间歇性 publickey denied
- 原因：每个 exec shell 是新进程，不继承 ssh-agent；`eval "$(ssh-agent -s)" && ssh-add` 同一命令链内才有效
- 现状：commit 均在本地安全（main ahead of origin），稍后重推即可

### ~20:00 闭环偏离诊断

- 重测闭环：直立项生效（z 稳 1.22-1.28）但 dist->lm1 单调增大 0.066→0.115——**机器人漂离锚点**
- 根因分析：seq_score 的 reach_eps=0.06 短时窗内够不着锚点时，min(errs)几乎不变 → 打分退化成"不倒即可"，无收敛梯度
- 待改：打分改为进步导向（终态距离+到达奖励）而非二元到达

### ~20:40 进步导向打分修复漂移

- 改 seq_score：end_err(终态距锚点) + min(errs) - 进度奖励，替代原"min+tail"二元到达
- 效果：dist->lm1 从单调增大(0.066→0.115)变为震荡收敛(0.054→0.063)——优化器有梯度了
- 遗留：收敛仍慢（短时窗0.4s内进步有限）、P5姿态项未加

### ~21:00 长时窗无改善

- 试 _HORIZON_ACTIONS=16/CHUNK=4（0.8s 视野）：dist->lm1 0.062→0.074 震荡——**瓶颈不在时窗长度**
- 结论：算法2 v1 能"不倒+有梯度"，但单锚点 CEM 收敛慢；下一步该考虑：
  (a) P5 姿态项引导身体形态（出拳手臂）
  (b) 锚点子目标间用 pose_solver 的帧姿态做中间引导（非 mean 而是 waypoint）
  (c) 多候选动作族（非纯随机采样）

### 本轮总结
- 算法1 完整落地：obs→start LandMark→plan_footsteps(capture物理+矫正before1+回位for帧)→gen_landmarks(帧驱动锚点)
- 算法2 v1 落地：obs→sim注入+CEM rollout搜索+D_LM(com/foot/force_ratio/upright)+多锚点顺序打分
- 已知局限：闭环收敛慢、无P5姿态项、CEM 2.6s/次偏慢

## E2E: ExecutorPolicy + RoundRunner video pipeline (~00:30)

- `e2e_policy.py`: Policy ABC wrapper. Per act(): decode obs→local P5,
  rebuild remaining clip (kf0=current P5 + kfs[idx+1:]), gen_action,
  advance idx on Transfer.dur budget, hold last action when clip ends.
  KNOWN GAP v1: naive re-anchor → displacement targets drift with the
  robot; needs persistent virtual anchor for faithful semantics.
- `e2e_env.yaml`: sim + FrozenRobotPlugin(robot_b) + Timeout(120).
  No reward observers — policy eats raw 96-dim obs.
- BUGFIX support_planner: Footstep for_frame/before_frame were int=0
  defaults; walk-step emit had neither → KeyError:0 in gen_landmarks
  on first real obs. Now Optional=None + walk gets before_frame=seg.
- Run: `python -m envs.framework.round_runner --env-blueprint
  .../e2e_env.yaml --policy-a-blueprint .../e2e_policy.yaml
  --policy-b-blueprint policy/blueprints/random.yaml
  --video .../punch.mp4 --seed 0`
- Result: 120 steps OK, robot_a upright + visibly steps/reaches in
  punch window, no fall. Video: /data1/dev/combatbench_executor/e2e/punch.mp4
