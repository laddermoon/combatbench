# Follow-Experiment 训练流水账

目标跟随课程训练的追加式决策日志。记录每次关键观察、参数调整和进度快报。

---

## [2026-06-15 03:00] 训练启动与代码修复

### 启动配置
- 命令: `PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train --v2 --experiment follow --resume-from .../curriculum_balance_recover_plus_20260612_103559/checkpoints/checkpoint_u10000.pt`
- 使用 `--v2` flag（sub-episode segmentation，排除 fallback 步）
- 从 balance_recover_plus u10000 checkpoint 恢复
- `nohup` 防止 SIGHUP 导致进程退出

### 代码修复 (在启动前完成)

1. **policy_loss 报告 Bug**: `ppo_trainer_v2.py` 中 `pol_losses.extend()` 在 early-stop `break` 之后，导致 early-stop 时 policy_loss 总是报 0.0。实际梯度步骤已执行，只是 loss 未记录。修复：将 extend 移到 break 之前。

2. **监控脚本**: 创建 `analyze_follow_logs.py`，跟踪 hold_ratio / survived / primary_ratio + reward 分解 + PPO 健康。

### 首次崩溃分析
- 第一次启动（无 nohup）只跑了 4 个 update (u10000-u10004) 就异常终止
- 日志无 traceback，无报错，进程直接消失
- 原因推测：终端关闭导致 SIGHUP
- 解决：改用 `nohup` 重启

---

## [2026-06-15 03:05] 首次进度快报 (u10000 → u10010)

### 核心指标
- **Update**: 10010 (跑了 10 代)
- **Level**: 0 (对手静止, speed=0.0 m/s)
- **hold_ratio**: 0.007 → 0.111 (15x 提升!)
  - series: [0.007, 0.014, 0.015, 0.025, 0.038, 0.045, 0.059, 0.069, 0.101, 0.111]
- **survived**: ~0.51 (稳定)
- **primary_ratio**: 1.000 (gating 从未切换到 fallback)
- **mean episode length**: ~112 步 / 200

### Reward 趋势
- **r_radial**: -0.005 → +0.0004 (从远离对手变为接近!)
- **EV(r_radial)**: -0.59 → +0.68 (critic 学会预测接近回报)
- **EV(r_tangential)**: -0.34 → +0.71
- **EV(r_fall)**: +0.84 → +0.998 (非常强)

### PPO 健康
- policy_loss: 非零 (0.05-0.25)，Bug 修复生效
- epochs_done: 1/4 (每次 early stop，KL ≈ 0.05-0.15 > target)
- std_mean: 0.671 (探索能力正常)
- std_min: 0.165 (在 log_std_min=-1.8 硬限处，正常)

### 决策
- **hold_ratio 趋势非常好，不需要干预**
- early-stop 每次 epoch 0 是因为从 balance recovery checkpoint 微调，初始 KL 天然很大
- 继续观察，等待 hold_ratio 达到 0.5 (PROMOTE_HOLD_RATIO) 以触发晋级
