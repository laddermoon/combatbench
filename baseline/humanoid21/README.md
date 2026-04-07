# Standing Policy Training

训练一个让 21-DOF 人形机器人保持站立的策略，使用 GRPO (Group Relative Policy Optimization) 算法。

## 功能

- **训练目标**: 学习让机器人在无外部干扰下保持直立站立
- **训练算法**: GRPO (Group Relative Policy Optimization)
- **对称自博弈**: 双机器人同时使用同一策略训练
- **并行采集**: 支持多进程并行进行 episode rollout
- **自动评估**: 定期评估并保存最佳模型
- **策略导出**: 自动生成可加载的 combatbench policy

## 快速开始

```bash
# 直接运行（使用默认配置）
python standing.py

# 或指定并行工作进程数（默认为 CPU 核心数的一半）
STANDING_ROLLOUT_WORKERS=8 python standing.py
```

## 训练输出

训练完成后，在 `runs/standing_<timestamp>/` 目录下生成：

```
runs/standing_20240101_120000/
├── best_model.pt         # 最佳模型检查点
├── final_model.pt        # 最终模型
├── policy/               # 导出的策略目录（可直接被 combatbench 加载）
│   ├── policy.py         # 策略实现
│   └── model.pt          # 模型权重
├── checkpoints/          # 定期保存的检查点
├── history.json          # 训练历史记录
└── config.json           # 训练配置
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STANDING_ROLLOUT_WORKERS` | `min(64, cpu_count/2)` | 数据采集并行进程数 |
| `STANDING_EVAL_WORKERS` | `min(rollout_workers, 16)` | 评估并行进程数 |

## 使用训练好的策略

```bash
# 在 run_round.py 中使用
python envs/humanoid21/run_round.py \
  --policy-a baseline/humanoid21/runs/standing_<timestamp>/policy \
  --policy-b random \
  --video match.mp4
```

## 关键配置

- **Episode 时长**: 5 秒 (100 steps @ 20Hz)
- **初始距离**: 1.5 ~ 3.5 米（随机）
- **Group size**: 8
- **Episodes per update**: 256
- **最大更新数**: 10000

## 奖励机制

- 每步奖励: +1（保持站立）
- 终止条件: 倒下（高度 < 1.10m 或直立度 < 0.8，持续 3 步）
- 目标: 最大化站立时长
