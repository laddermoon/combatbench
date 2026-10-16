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
| **`hier_s2_balance_s42i_a`** | 12:49– | 进行中 | 381→ | 待出结果 | 从 scratch u380 续训，12 级推力课程从 level 0 开始 |

## 实验目录与关键产物

```
baseline/runs/hier_s1_s42i_scratch/        # Stage 1 成果（66 维）
  checkpoints/checkpoint_u00380.pt         # ← Stage 2 的初始化 checkpoint
  videos/u00375.mp4 等                     # 起身+踱步 eval 视频

baseline/runs/hier_s2_balance_s42i_a/      # Stage 2 当前 run
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

## Stage 2 监控要点

- `recovery`：受推后恢复率，晋级阈值 0.8（连续 2 次 eval）
- `level` / `force`：课程进度，目标 level 11（100N，最长推力段）
- `clip` / `cad`：被推后回到 s42i 步态的质量（65 维旧线在 level 9-10 时约 clip 0.18、cad 0.43；66 维新线待观察）
- `success` / `falls`：起身与摔倒率

监控命令：

```bash
grep "\[eval" baseline/runs/hier_s2_balance_s42i_a/train.log | tail -3
```
