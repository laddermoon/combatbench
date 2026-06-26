# V2 Baseline 训练指南

本文档说明如何使用 V2 课程学习框架训练 Humanoid21 基线策略。

## 概述

V2 在 V1 的基础上改进了训练循环，引入 **sub-episode 分段**：当门控网络判断需要平衡恢复介入时，自动截断轨迹并独立计算 GAE，避免状态不连续导致的梯度错误。

训练流程分为四阶段：**平衡恢复 → 门控网络 → 跟踪对手 → 对抗**，使用 V2 专用实验配置。

## 前置条件

```bash
cd /data1/mono/things/combatbench
export PYTHONPATH=.
```

## 可用实验

V2 相关实验：

| 实验名 | 阶段 | 说明 |
|--------|------|------|
| `basic_balance_v2` | 平衡 | V2 基础平衡训练 |
| `balance_recover_v2` | 平衡 | V2 扰动恢复训练 |
| `balance_recover_plus_v2` | 平衡 | V2 增强扰动恢复 |
| `follow_v2` | 跟踪 | V2 跟踪对手训练 |
| `fight_v2` | 对抗 | V2 对抗打击训练 |
| `fight_v2_oppopool` | 对抗 | V2 对手池对抗训练 |

## 训练流程

### 阶段 1：平衡恢复

从零开始训练基础平衡：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment basic_balance_v2 \
  &> basic_balance_v2.log &
```

继续训练扰动恢复：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment balance_recover_v2 \
  --resume-from baseline/humanoid21/runs/curriculum_basic_balance_v2_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> balance_recover_v2.log &
```

继续训练增强扰动恢复：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment balance_recover_plus_v2 \
  --resume-from baseline/humanoid21/runs/curriculum_balance_recover_v2_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> balance_recover_plus_v2.log &
```

**阶段完成标准**：最大扰动下存活率 ≥ 85%。

**参考训练时间**（单卡 4090）：
- `basic_balance_v2`：约 7 小时
- `balance_recover_v2`：约 12 小时

### 阶段 2：门控网络

平衡阶段完成后，必须训练门控网络，用于后续阶段的平衡安全切换。门控网络判断当前状态是否危险，在需要时自动切换到平衡恢复策略。

**1. 收集门控数据**：

```bash
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 100000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data_v2 \
  --policy-path baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_<timestamp>/policy_exports/uXXXX
```

**2. 训练门控模型**：

```bash
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir baseline/humanoid21/curriculum/gating_data_v2 \
  --output-dir baseline/humanoid21/curriculum/gating_model_v2
```

### 阶段 3：跟踪对手

从平衡恢复的 checkpoint 继续训练：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment follow_v2 \
  --resume-from baseline/humanoid21/runs/curriculum_balance_recover_plus_v2_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> follow_v2.log &
```

### 阶段 4：对抗

从跟踪的 checkpoint 继续训练对抗能力：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment fight_v2 \
  --resume-from baseline/humanoid21/runs/curriculum_follow_v2_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> fight_v2.log &
```

进阶：在对抗基础上添加对手池自博弈：

```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train \
  --experiment fight_v2_oppopool \
  --resume-from baseline/humanoid21/runs/curriculum_fight_v2_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> fight_v2_oppopool.log &
```

## 监控训练

```bash
# 平衡阶段
python3 baseline/humanoid21/curriculum/analyze_logs.py basic_balance_v2.log --watch

# 跟踪阶段
python3 baseline/humanoid21/curriculum/analyze_follow_logs.py follow_v2.log --watch

# 对抗阶段
python3 baseline/humanoid21/curriculum/analyze_fight_logs.py fight_v2.log --watch
```

## 生成验证视频

```bash
python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_<name>_<timestamp>/policy_exports/uXXXX/policy_blueprint.yaml \
  --policy-b-blueprint policy/blueprints/random.yaml \
  --video output.mp4
```

## Smoke 测试

```bash
PYTHONPATH=. python3 -m baseline.humanoid21.curriculum.train \
  --experiment basic_balance_v2 \
  --smoke
```

## 关键参数

| 参数 | 说明 |
|------|------|
| `--experiment` | 实验名称（必填） |
| `--resume-from` | 从指定 checkpoint 恢复训练 |
| `--smoke` | 快速测试模式 |
| `--run-name` | 自定义运行名称 |
