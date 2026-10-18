# 分层控制训练日志（s42i 标准步态线）

> 最后更新：2026-10-16 ~13:00
> 当前在跑：`hier_s2_balance_s42i_a`（GPU0）

## 训练目标

低层策略统一为 **66 维输入**（57 本体 + 9 维指令，v2 契约，见 `COMMAND_DESIGN.md`），指令当前全零。能力分阶段：

- **Stage 1**：倒地 → 起身 → 进入 s42i 标准动态踱步（φ² 门控片段奖励）
- **Stage 2**：12 级推力课程（40N→100N）下保持/恢复标准步态
- 后续（未开始）：S3 平移+转向 → S4 手部到点 → S5 腿部踢点 → S6 高度控制

## 标准步态来源

- 源策略：`baseline/runs/balance_step_s42i` 最终导出（99 维 GaitClock 包装，抗推 level 11 通关策略）
- 采集：12 集无推力 rollout → `assets/standard_stepping_clips.npz`（5 clips × 40 帧）
- 步态特征：高频小碎步，半周期 lag=5（0.25s），root_z≈1.20m，抬脚 ~10cm
- 参考视频：`/data1/dev/policy_capture/s42i_stepping/ep_3000.mp4` ~ `ep_3003.mp4`

## 奖励结构（S1 / S2）

| 通道 | 奖励 | actor_weight |
|---|---|---|
| `r_potential` | `0.01 × φ(t)` 站立势能 | 1.0（S2 为 3.0） |
| `r_std_clip` | `clip + pose + 0.5·cad + 0.5·h_term` | `3·φ²`（S2 为 `2·φ²·(非受推中)`） |
| `r_fall`（仅 S2） | 摔倒 + 靠墙 + 受推窗口惩罚 | 2.0 |

片段匹配：70 维约减动态对空间（腿 q/qv + 重力 + 脚相对 + 触地），库内白化，σ=8；pose 项原始加权空间 σ=0.9；cadence = 滞后 5 的承重脚自相关；h_term 目标 1.20m。

## 运行历史

| Run | 时间段 | 时长 | Updates | 结果 | 备注 |
|---|---|---|---|---|---|
| `hier_s1_standup_step_s42a` | 01:52–03:07 | ~1h15m | 5→480 | success=1.0, clip=0.02, cycles=10, h=1.18 | 旧 step_mbs 片段库；收敛到**宽蹲姿踏步**局部最优 |
| `hier_s1_standup_step_s42b` | 03:07–03:58 | ~51m | 485→820 | success=1.0, clip=0.63, cad=0.95, h=1.265 | u475 ckpt 续训；σ 4→8、pose 权重↑、新增身高项 → 直立步态 ✅（仍是旧步态参考） |
| `hier_s2_balance_s42a` | 03:59–10:36 | ~6h37m | 825→1995 | **level 9**（100N×26-30 步），recovery 0.71–0.86 | 从 s42b 续训；推到 level 9 |
| `hier_s2_balance_s42b` | 10:37–11:47 | ~1h10m | 1995→2125 | level 10，recovery 0.76 | 换 s42i 片段库后续训；clip 0.19（旧步态向新流形迁移中）；**因配置变更崩溃终止** ❌ |
| `hier_s1b_step_s42a` | 10:44–11:44 | ~1h | 5→130 | success=1.0, clip=0.69 | **错误实验**：站姿出生而非倒地出生，方向偏差，已废弃删除 ❌ |
| `hier_s1_s42i_scratch_65d` | 11:44–11:47 | 3m | 1→2 | — | 65 维启动后即切 66 维，废弃 |
| **`hier_s1_s42i_scratch`** | 11:47–12:47 | **~1h** | 1→380 | **success=1.0, clip=0.607, cad=0.96, h=1.195, cycles=17.2** | ✅ **Stage 1 通过**（66 维从零，倒地→起身→s42i 踱步） |
| **`hier_s2_balance_s42i_a`** | 12:49–15:40 | ~2h51m | 381→845 | level 4，recovery 起步 | 空心推力 bug（见事故记录 5），弃 run 保 checkpoint |
| **`hier_s2_balance_s42i_b`** | 15:42–23:10 | ~7.5h | 845→1680 | **level 11 封顶（最高级），recovery 0.948** | ✅ **Stage 2 收敛完成**，手动终止；最强策略=u1635 |

## Stage 2 最终收敛结果（2026-10-16 23:10 手动停止）

- **最强策略**：`baseline/runs/hier_s2_balance_s42i_b/checkpoints/checkpoint_u01635.pt`
  （已自动导出至 `baseline/runs/hier_s2_balance_s42i_b/policy/`，与 u01635 权重逐一比对一致）
- **峰值 eval（u1635）**：`recovery=0.948`（100N 推力课程最高级 L11，~2s 持续推下 95% 恢复）、
  `push_falls=40/764`、`success=1.0`、`clip=0.388`、`cad=0.610`
- 课程 12 级全部打穿（L0-L11）；L11 下 recovery 从 0.82 → 0.948 缓慢爬升后趋平，收益递减故停
- 用作 S3/S4 的初始化 checkpoint：`hier_s3_move_*`、`hier_s4_punch_*` 各 run 均从 s2b 系 checkpoint 续训

## 实验目录与关键产物

```
baseline/runs/hier_s1_s42i_scratch/        # Stage 1 成果（66 维）
  checkpoints/checkpoint_u00380.pt         # ← Stage 2 的初始化 checkpoint
  videos/u00375.mp4 等                     # 起身+踱步 eval 视频

baseline/runs/hier_s2_balance_s42i_b/      # Stage 2 最终收敛 run（已完成）
  policy/                                  # ← 最强策略（u1635，recovery=0.948）
  checkpoints/checkpoint_u01635.pt         # ← 对应 checkpoint（含优化器状态）
  train.log  videos/

baseline/runs/hier_s2_balance_s42i_a/      # Stage 2 前段 run（空心推力 bug，止于 u845）
  train.log  checkpoints/  videos/

baseline/runs/hier_s2_balance_s42a/        # 旧迁移线 level-9 存档（65 维，step_mbs 步态）
baseline/humanoid21/hierarchy/assets/
  standard_stepping_clips.npz              # s42i 片段库
  standard_stepping_clips.png              # 诊断图
```

## 事故与教训记录

1. **s2b 崩溃（非训练失败）**：s2b 运行中修改 `hier_s2_balance_env.yaml` 增加 `cmd_version: "v1"`，rollout worker（fork 继承旧代码）读到新蓝图后 `TypeError`。**教训：改动在跑实验引用的配置文件前，必须先停相关进程或确认兼容。**
2. **s1b 方向偏差**：误将"从头开始"实现为随机权重+站姿出生（只训保持踱步），实际要求是倒地出生+起身+踱步。已删除实验文件；正确做法就是直接重跑 `hier_s1_standup_step_mbs`（随机初始化本就内建）。
3. **蹲姿盆地修复**（s42a→s42b）：转移匹配 σ=4 下梯度 underflow，加宽到 8 并增加 pose 权重 0.3→1.0、收窄 σ_pose 1.5→0.9、新增身高项（目标 1.28→后改 1.20）后 clip 从恒 0 升到 0.63。
4. **步态源纠正**：最初片段库来自 `step_mbs_s42`（慢速高抬腿），用户指定改用 `balance_step_s42i`（抗推训练出的高频小碎步），同步更新 cadence_lag 8→5、身高目标 1.28→1.20。
5. **空心推力课程（s42i_a，重大）**：`StandingTriggeredForcePlugin` 的站立触发要求 `root_pos[2] > 1.15` 连续 20 步不中断；s42i 动态踱步 h≈1.17 且 ~20-25% 步数跌破 1.15，`standing_timer` 峰值只有 17-18 → 推力几乎从未施加（eval pushes=0-5/128 eps，旧步态同配置为 70+）。后果：L0→L6 晋级靠"偶发 1-2 推恰好没摔"，s2 全程实际无推恢复训练。修复（commit 见下）：插件加 `standing_grace_steps`（连续低于阈值 N+1 帧才重置计时器，默认 0 保持旧语义），s2 蓝图改 `standing_height_threshold: 1.10` + `grace: 5`。修复后 eval pushes≈1547。run `hier_s2_balance_s42i_b` 从 u845 checkpoint 重启，level patch 回 4。

## Stage 2 监控要点

- `recovery`：受推后恢复率，晋级阈值 0.8（连续 2 次 eval）
- `level` / `force`：课程进度，目标 level 11（100N，最长推力段）
- `clip` / `cad`：被推后回到 s42i 步态的质量（65 维旧线在 level 9-10 时约 clip 0.18、cad 0.43；66 维新线待观察）
- `success` / `falls`：起身与摔倒率

监控命令：

```bash
grep "\[eval" baseline/runs/hier_s2_balance_s42i_a/train.log | tail -3
```

---

# 基础技能训练（skill_train_mbs，S2 基座之上）

> 最后更新：2026-10-18
> 当前在跑：`skill_train_s2b_u5855_p1000`（GPU0，从 u5855 热恢复）

## 概况

11 个技能（move×4 / turn×2 / jab×2 / heavy×2 / step）复合训练，job 级
`skills` 标签决定奖励通道挂载（详见 `SKILL_TRAINING_PLAN.md` v7）。
从 S2 收敛 checkpoint `u1635`（recovery 0.948）热启动。

## 运行历史

| Run | Updates | 结果 | 备注 |
|---|---|---|---|
| `skill_train_s2b_u1635e` | 1→3910 | quality 0.02→0.63 | 首版（warmup 初始化前）；`max_updates=4000` 到顶正常退出 |
| `skill_train_s2b_u4000b_ext` | 4000→5855 | quality →0.774 | ext_only 版；`r_drift` 权重 2→3 后 fwd_dyaw 1.4→0.4；u5855 早停（patience 400 触发） |
| `skill_train_s2b_u5855_p1000` | 5855→ | — | patience 放宽至 1000 继续磨 heavy strike |

## 收敛水位（u5850 附近）

| 技能族 | 指标 |
|---|---|
| move | v_along fwd 0.86 / back 0.55 / left 0.39 / right 0.61 m/s；dyaw fwd 0.4 / left 0.23 / right 0.38 rad |
| turn | w left 5.3 / right 4.5 rad/s，vxy<0.05 m/s |
| strike | jab kp 0.15/0.13、hand 0.09-0.10；heavy kp 0.27-0.28、hand 0.13/0.29 |
| step | stood 1.0，d_pos ~0.13m，d_yaw ~0.2rad（80 步/4s） |
| falls | 3-5% |

验收视频：`baseline/humanoid21/hierarchy/videos/20261018_071515/`（u5840，quality=0.775）

## 事故与教训记录（技能线）

1. **fork worker 读新配置崩溃（两次）**：运行中的 run 修改 yaml/构造参数（`ext_only`），
   fork worker 读新蓝图 TypeError 静默死。教训同事故 1：改配置前先停 run 或热恢复。
2. **`max_updates` 静默到顶**：run 到 u4000 正常退出但无提示，恢复 run 从 u4001 秒退。
   查因方向：先查 `max_updates`/`early_stop` 再查崩溃。
3. **早停误判**：`quality` 复合分不含 dyaw，朝向改善不计入 → patience 400 触发时
   strike kp/q 仍在缓降。已将 `_no_improvement_limit` 放宽至 1000。
4. **朝向漂移套利（已治）**：move 技能长期以朝向漂移换速度（fwd_dyaw 峰值 1.4 rad）。
   干预路径：dyaw 系数 0.5→1.5 → r_drift 权重 1→2 → 2→3，最终 fwd_dyaw 收敛 ~0.4，
   speed 无损。

监控命令：

```bash
grep "\[eval" baseline/runs/skill_train_s2b_u5855_p1000/train.log | tail -3
```

## 打击技能：放开约束 + 右臂局部最优攻坚（u6285 起）

### 干预 1：放开"怕摔"约束（commit `04b36b89`）

诊断（逐帧 rollout）：strike ep 的 `r_fall`/`r_potential` 在 `t<k_land` 生效 +
`r_strike` 自身带 `φ²` 门控 → 重心前移会让打击梯度二次方衰减。policy 学到
"宁可打不到也不能掉重心"（phi 全程顶 0.98+）。重拳另有时序问题：clip 6 帧
到位（0.3s），开立步站架物理上来不及。

修复：strike ep 的 r_fall/r_potential 全窗置零；r_strike 去 φ² 门控；
重拳参考时钟半速（clip_rates=0.5，12 帧到位）；heavy_r 采样 1.5。

效果：jab hand →0.06-0.07，heavy_l hand →0.08，heavy_r hand 0.30→0.24。

### 干预 2：右臂局部最优（commit `a061f52c`+`6ebe69e1`）

逐关节诊断发现 heavy_r 的 `elbow_right` 在 k_land 误差 **57°**——
手臂根本不伸，而"不出拳"本身 kp 反而更低（不动身体各关键点误差小），
kp 指标对"没出拳"完全失明。jab_r 手臂能伸，排除硬件/clip/目标点不对称。

修复：heavy_r 采样 1.5→3.0；W_HAND 0.35→0.50；新增 `r_arm` 项
（出拳手臂 shoulder1/2/elbow 独立 pose 跟踪，W_ARM=0.30——手臂误差在
21 维 pose 均值里被稀释）；eval strike 分并入 hand_err。

效果（u7310 起）：heavy_r hand 0.24→**0.14** 持续下探，elbow_right
误差 57→49°；heavy_l hand →0.055 最佳水位。quality 0.835 new_best。

监控命令：

```bash
grep "\[eval" baseline/runs/skill_train_s2b_u7310_armterm/train.log | tail -3
```

验收视频：`videos/20261018_122822/`（u7454，heavy_r 右臂半伸、肘仍弯 ~45°）

### 阶段收敛（u8000 终态，run `skill_train_s2b_u7540_arm50`）

| 技能 | 指标 |
|---|---|
| jab_l/r | hand 0.067-0.075 双侧对称，kp ~0.15 |
| heavy_l/r | hand **0.055 / 0.050**（右侧反超），kp 0.18/0.15，q 14.3/12.5 |
| move ×4 | v fwd 0.90 / back 0.55 / left 0.61 / right 0.73；dyaw ≤0.31 |
| turn ×2 | w 5.4 / 5.0 rad/s 对称，vxy ~0.05 |
| step | stood 1.0，4s 漂移 ~0.14-0.21m |
| falls | 2-4% |
| quality | 峰值 0.848（含 hand_err 口径） |

heavy_r 局部最优全程：hand_err 0.30→0.24（放开约束）→0.20（采样×3+W_HAND↑）
→0.14（r_arm 0.3）→0.10（r_arm 0.5）→0.05 收敛。elbow_right 误差 57°→~30°
以内。经验：**多关节均值型跟踪项会稀释单关节误差，关键末端/关节链需要独立项。**

验收视频：`videos/20261018_141445/`（u8000 导出，11 技能全渲）

## 出拳提速：单调进度跟踪（commit `44372bce`，u8000 起）

**问题**：严格 clip 时钟把速度上限钉在 mocap 节奏——"超前参考帧"同样计
误差，只有 margin 帧的"早到早赚"给速度留梯度；且重拳全程半速时钟等于
直接训练半速拳。实测 u8000 策略落点耗时 heavy ~13 步 / jab ~14+ 步。

**机制**（P2）：进度指针 k 每步在 `[k, k+lookahead=2]` 内取完整 reward
最高的 ref 帧，只进不退——速度上限变为物理极限；到达 k_land 后每步
+0.25 hold bonus（早到=多赚）。站架慢不罚（指针会等）。clip_rates
半速机制删除；eval 新增 `land_t`（首次到达步号）量化拳速。

**效果**（u8015→8060，~60 update）：

| 技能 | land_t 初值 → 现值 | 名义 clip 节奏 |
|---|---|---|
| jab_l | 15.1 → ~10.0 | 11 |
| jab_r | 11.0 → ~10.1 | 11 |
| heavy_l | 12.3 → ~10.0 | 6 |
| heavy_r | 11.5 → **~9.0** | 6 |

hand_err/kp 无牺牲（jab ~0.07、heavy ~0.055），move/turn/step/falls
零回归。下降趋平——~9-10 步 ≈0.45-0.5s 接近"踏步→站架→出拳"的
物理下限。

验收视频：`videos/20261018_233504/`（u8103 导出，heavy_r/jab_l）
