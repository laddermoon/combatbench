# V2 basic_balance_v2 训练时间记录

## 实验信息

- **实验名**: `v2_basic_balance_v2`
- **算法**: PPO
- **框架版本**: V2 (ExperimentV2 + Trajectory)
- **启动时间**: 2026-08-06 18:20:22
- **终止时间**: 2026-08-06 20:45 (手动终止)
- **GPU**: CUDA_VISIBLE_DEVICES=0
- **PID**: 1808063

## 训练目录

- **Run dir**: `baseline/runs/train_v2_basic_balance_v2_ppo_20260806_182022/`
- **实验文件**: `baseline/experiments_v2/exp_basic_balance_v2.py`

## 时间节点记录

| Update | 累计时间 | 区段耗时 | Eval Survived | Survival Rate |
|--------|-----------------|--------------|---------------|---------------|
| 100 | 0h06m31s | 0h06m31s | 0/16 | 0.000 |
| 200 | 0h15m14s | 0h08m43s | 0/16 | 0.000 |
| 300 | 0h26m39s | 0h11m25s | 0/16 | 0.000 |
| 400 | 0h47m43s | 0h21m03s | 16/16 | 1.000 |
| 500 | 1h22m21s | 0h34m37s | 16/16 | 1.000 |
| 600 | 1h57m56s | 0h35m35s | 16/16 | 1.000 |
| 677 (终止) | 2h25m30s | 0h27m33s | 16/16 | 1.000 |

## 关键观察

1. **前 300 updates** (0h26m): 机器人尚未学会站立，survival_rate=0
2. **update 400** (0h47m): 突破性进展，survival_rate 从 0 跳到 1.0，16/16 全部存活
3. **update 400+**: 训练时间显著增加（每 100 updates 从 ~10min 增到 ~35min），因为 episode 长度从 ~20 步增长到 ~200 步（跑满 horizon），rollout 计算量增大
4. **update 677**: 训练已收敛，policy entropy=-6.84，critic EV 全部 >0.73

## 时间增长原因

前 300 updates episode 平均长度约 20 步，rollout 很快。400 之后 episode 跑满 200 步，rollout 时间增长约 10 倍，导致每 100 updates 耗时从 ~7min 增到 ~35min。
