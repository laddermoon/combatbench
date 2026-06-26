# V1 Baseline 训练指南

本文档说明如何使用课程学习框架训练 Humanoid21 V1 基线策略。

## 概述

V1 基线采用**四阶段课程学习**，逐步从平衡能力到对抗能力：

1. **平衡恢复（Balance Recover）** — 在随机扰动下保持站立
2. **门控网络（Gating Network）** — 训练状态危险判别器，用于后续阶段的平衡安全切换
3. **跟踪对手（Follow）** — 接近对手到有效距离
4. **对抗（Fight）** — 在保持平衡的前提下打击对手

每个阶段从前一阶段的 checkpoint 继续训练，策略能力逐步叠加。

## 前置条件

```bash
cd /data1/mono/things/combatbench
export PYTHONPATH=.
```

## 可用实验

```bash
python3 baseline/humanoid21/curriculum/train.py --list-experiments
```

V1 相关实验：

| 实验名 | 阶段 | 说明 |
|--------|------|------|
| `basic_balance` | 平衡 | 基础平衡训练 |
| `balance_recover` | 平衡 | 扰动恢复训练 |
| `balance_recover_plus` | 平衡 | 增强扰动恢复 |
| `balance_recover_plus_refine` | 平衡 | 多级扰动防遗忘 |
| `follow` | 跟踪 | 跟踪对手训练 |
| `fight` | 对抗 | 对抗打击训练 |

## 训练流程

### 阶段 1：平衡恢复

从零开始训练平衡恢复策略：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment basic_balance \
  &> balance.log &
```

继续训练扰动恢复：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment balance_recover \
  --resume-from baseline/humanoid21/runs/curriculum_basic_balance_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> balance_recover.log &
```

继续训练增强扰动恢复：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment balance_recover_plus \
  --resume-from baseline/humanoid21/runs/curriculum_balance_recover_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> balance_recover_plus.log &
```

**阶段完成标准**：随机最大扰动下存活率 ≥ 85%。

### 阶段 2：门控网络

平衡恢复训练完成后，需要训练门控网络（Gating Network），用于在后续阶段判断状态是否危险并自动切换到平衡恢复策略。

**2.1 收集门控数据**

使用弱化版平衡策略收集数据，生成正负样本：

```bash
PYTHONPATH=. python3 baseline/humanoid21/curriculum/collect_gating_data.py \
  --num-episodes 10000 \
  --noise-std 0.08 \
  --workers 48 \
  --output-dir baseline/humanoid21/curriculum/gating_data \
  --policy-path baseline/humanoid21/runs/curriculum_balance_recover_plus_<timestamp>/policy_exports/uXXXX
```

- `--noise-std`：动作噪声标准差，控制弱化程度（0.08 为默认，增大则门控更敏感，减小则更极限）
- 数据中 Safe/Unsafe 帧比例接近 5:5 或 4:6 为最佳

**2.2 训练门控模型**

```bash
PYTHONPATH=. python3 baseline/humanoid21/curriculum/train_gating_network.py \
  --epochs 500 \
  --batch-size 4096 \
  --hidden-dims 512 256 128 \
  --lr 5e-4 \
  --data-dir baseline/humanoid21/curriculum/gating_data \
  --output-dir baseline/humanoid21/curriculum/gating_model
```

训练完成后门控模型永久冻结，后续阶段仅推理不更新。

### 阶段 3：跟踪对手

从平衡恢复的 checkpoint 继续训练跟踪能力：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment follow \
  --resume-from baseline/humanoid21/runs/curriculum_balance_recover_plus_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> follow.log &
```

**阶段完成标准**：Episode 中保持在有效距离内的比例达标后，对手移动速度加快进入下一级课程。

### 阶段 4：对抗

从跟踪的 checkpoint 继续训练对抗能力：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment fight \
  --resume-from baseline/humanoid21/runs/curriculum_follow_<timestamp>/checkpoints/checkpoint_uXXXX.pt \
  &> fight.log &
```

## 监控训练

每个阶段有对应的日志分析工具：

```bash
# 平衡阶段
python3 baseline/humanoid21/curriculum/analyze_logs.py balance.log --watch

# 跟踪阶段
python3 baseline/humanoid21/curriculum/analyze_follow_logs.py follow.log --watch

# 对抗阶段
python3 baseline/humanoid21/curriculum/analyze_fight_logs.py fight.log --watch
```

`--watch` 参数会持续刷新显示。

## 生成验证视频

```bash
python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint baseline/humanoid21/runs/curriculum_<name>_<timestamp>/policy_exports/uXXXX/policy_blueprint.yaml \
  --policy-b-blueprint policy/blueprints/random.yaml \
  --video output.mp4
```

## Smoke 测试

快速验证训练流程是否正常（2 个 update，8 个 episode）：

```bash
python3 baseline/humanoid21/curriculum/train.py \
  --experiment basic_balance \
  --smoke
```

## 训练产物

训练输出目录结构：

```
baseline/humanoid21/runs/curriculum_<name>_<timestamp>/
├── config.json              # 实验配置快照
├── checkpoints/             # 训练 checkpoint（.pt）
├── policy_exports/          # 导出的推理策略目录
│   └── uXXXX/
│       ├── policy.py        # 策略代码
│       ├── model.pt         # 模型权重
│       └── policy_blueprint.yaml
├── eval/                    # 评估结果
└── summary.json             # 训练摘要
```

## 关键参数

| 参数 | 说明 |
|------|------|
| `--experiment` | 实验名称（必填） |
| `--resume-from` | 从指定 checkpoint 恢复训练 |
| `--smoke` | 快速测试模式 |
| `--run-name` | 自定义运行名称 |
