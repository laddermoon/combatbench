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

## 算法2 稳定性攻关 (~02:30)

诊断链：
- trivial "stand" clip 也倒 → 不是跟踪问题，是基础能力问题
- 常值保持动作 ~35 步倒；CEM 输出 ~16 步倒 → 搜索主动挑更烂的解
- wsim 注入后 rollout 与真实 sim **零发散** → 模型完美，纯搜索/cost 问题
- 300 个 hold-perturb 候选里有近水平的解 → 好解存在
- A/B: hold 0.396 vs CEM 1.478 → **CEM 从不给 null 候选打分**，漂进更烂区

修复序列：
1. 平衡项移出 _landmark_error，per-step 累加（tilt²/root_z下限/vz下沉/
   捕获点出支撑域）；landmark 走查不再 min()
2. mean 平铺 hold-pose（原为 0 填充=关节归中折叠）
3. 注入 hold+mean 候选保底
   → 24步→48步，仍会瘫
4. pose 先验：a_ref=锚点 frame→clip kf pos→pose_solver 参考动作，
   CEM mean=当前→a_ref 斜坡
   → 52步倒，0.75 蹲姿能稳 12 步但起不来
5. **反馈式 rollout（关键）**：候选=5组增益 k，wsim 内逐步闭环
   u=clip(a_ref+k(a_ref−q))。84维→5维，候选天然抗扰
   → **stand 场景 80 步全活**，com_z 稳 0.835，tilt 有界 ~9°

原理：开环动作带里没有"主动扶正"方向；跟踪控制器参数空间才有。
下一步：punch 场景验证。

## punch 场景攻关 (~03:30)

- clip 按 dur 很快耗尽 → policy 退化重放 last_action（无反馈）→ 塌
  → 耗尽后改喂"hold clip"（kf0=当前 + 重复帧）保持控制器在线
- hold 目标锚定当前歪斜姿态 → 棘轮倒塌
  → 改为耗尽时冻结姿态 → 仍慢漂（耗尽时已 14° 前倾，目标本身歪）
  → **改为收势语义：hold 目标=stand_p5 站姿**（拳击=出拳→收回站架）
- 结果：punch 冲刺 tilt→11° 后 **主动收复 com 0.82→0.865 tilt→1.7°**，
  之后 86+ 步稳悬 ~0.83，tilt 缓慢爬行 ~10°（未倒，趋势待收敛）
- 剩余：慢速漂移；pose 跟踪表达不了迈步——单姿态 ref 没有时间结构
  （swing→land 是两拍）。后续候选：分段 a_ref + 可搜索切换时刻。

## 漂移攻关 (~05:00)

- 加 δ 残差通道（候选 [k(5), d(5)]）：tilt 能还手了，但换成"下蹲求稳"
  → com_z 持续走低
- 反直觉教训：root_z 下限（0.84）打不到——obs[45] 是 root(1.28) 不是
  com(0.868)，floor 触发太晚
- 多次试验方差大（32-112 步倒）→ CEM 改固定种子消运气成分
- **结构性修复：rollout 内 ref 跟着锚点走查推进**——a_refs[每锚点]，
  到达切下一个，swing→land 两拍终于能表达
- **决定性：H=8→16（0.8s 看到缓慢垮塌后果）+ 单向 com_z 下限项 +
  com 权重 6→10**
  → **punch 场景 120 步全活**，com_z 稳 0.85-0.86，tilt 1-6° 有界
- 耗时 ~9.6s/compute（H=16 + pop32），可接受用于 dev 验证

## 进度门控 (~05:50)

- 时间驱动 clip 推进的缺陷证实：物理没跟上，dur 预算照烧 → 迈步
  窗口 ~16 步就过期，脚没动意图就没了
- 改为到达门控：P5 err<0.30 且过半 dur 才翻页；兜底 3×dur 或 1.2s
  → 左脚 fl_x −0.05→+0.10 真迈步了（之前完全不动），tilt 冲 23°
    收回 2°，40 步全活且收在站姿
- 诊断新发现：迈步需要的时间远超 clip dur（0.1s 的帧物理上要 1s+），
  门控让物理节奏主导意图节奏，符合"携带信息量"原则

## 迈步推进 (~06:10)

- rollout 内 ref 切换卡在难锚点（卸载锚点 reach_eps 到不了 → ref 永
  远停在 lm1）→ 加时间兜底：到达 OR dwell≥6步 切下一锚点
- budget=H/剩余≈3 太急（跟踪没开始 ref 就切）→ 固定 6 步（0.3s）
- budget=6 结果：fl_x 峰值 +0.11，tilt 峰 28° 后收回 ~4°，com_z 回
  0.849 站直收尾；迈步仍不完整（+0.11/0.35）
- 测试 15/15 通过

## 评分/执行对齐 (~06:40)

- 结构 bug：rollout 的 ref 走查有 dwell 兜底会前进，但 _seq_score
  只认 reach_eps——优化目标锚点1 vs 控制器执行锚点3 错位
- _seq_score 改同款 reach-or-dwell 走查 + 每步 0.25 跟踪项
- 结果：tilt 更敢冲（峰 33°）但仍能收回 ~3°，40 步全活收尾直立
- v4 视频（gate+H16 版）：f45 左腿真抬起迈步、f60 手臂前伸、
  120 步全活——动作序列可见但 tilt 尖峰狼狈

## v5 回归 + tilt 硬门 (~07:20)

- v5 视频（score对齐+dwell兜底）：中途实际倒了——对齐让优化器敢放
  更大 tilt 赌注，尖峰失控
- 反制：tilt 权重 6→9 + 40·max(0,tilt−25°)² 硬门——对齐保留但尖峰
  收边
- 40 步 diag 带门全活收尾直立；待 v6 视频验证

## v5/v6 回归分析 (~07:50)

- v5（score对齐+dwell）：视频中途倒——对齐让优化器敢押大 tilt
- v6（+tilt软门 9+40×>25°）：仍倒（f45 起躺平）——软门不够硬
- v4（gate+H16，无对齐/dwell）：唯一活 120 步的视频，动作可见
- 对策：rollout tilt>35° 直接 abort 判 +inf——绝对安全界，不许
  为锚点赌过界姿态。v7 渲染验证中
- 教训：目标对齐和动作冒险是一体两面，必须配硬安全界

## 跟踪项回退 (~08:10)

- v7（对齐+abort+跟踪项 0.25e）：视频仍倒——abort 挡 rollout 内过
  界但真实跨步重规划照样漂过去；跟踪奖励本身是冒险源
- 决策：删 0.25e 每步跟踪项，保留 schedule-aware 末端误差+tgt 推
  进——对齐收益在、冒险激励无
- 40 步 diag：tilt 多在 15° 内（峰 27° 收），fl_x +0.127，收敛好转
- v8 渲染验证中

## v4 语义回退 (~05:00)

- 视频统计：v4（reach-only）唯一活 120 步；v5-v9 全部中途倒——
  dwell 兜底（无论 6/14 步）都会把 ref 推进到身体做不到的锚点
- 固定阈值 abort 的瘫痪模态：倾过阈值后全候选立即死→fallback→
  自由落体；改自适应也救不了（候选空间里没有收角动量的方向）
- 回退：ref 走查 reach-only + abort 放宽 ~40°+6°（只杀真不可恢复）
  + 去 tilt 硬门 → 40 步 diag：tilt 峰 31° 收回 ~12°，迈步 +0.14
- 核心权衡记录：迈步完整性和存活是此消彼长，v1 选存活
