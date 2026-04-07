# Standing Policy Training

训练一个让 21-DOF 人形机器人保持站立的策略，使用 GRPO (Group Relative Policy Optimization) 算法。

本目录包含两个训练脚本，按顺序使用：

## 训练流程

### 1. 静态站立训练 (`standing.py`)

**训练目标**: 学习让机器人在无外部干扰下保持直立站立

**功能特性**:
- **训练算法**: GRPO (Group Relative Policy Optimization)
- **对称自博弈**: 双机器人同时使用同一策略训练
- **并行采集**: 支持多进程并行进行 episode rollout
- **自动评估**: 定期评估并保存最佳模型
- **策略导出**: 自动生成可加载的 combatbench policy

### 2. 扰动站立训练 (`standing_with_turbulence.py`)

**训练目标**: 学习机器人在持续风扰动下保持直立站立

**功能特性**:
- **外部扰动**: 使用 `ContinuousWindPlugin` 模拟持续风力，包含随机阵风
- **Warm Start**: 支持从静态站立模型初始化，加速收敛
- **鲁棒性训练**: 提升策略在真实环境中的抗干扰能力
- **继承特性**: 继承 `standing.py` 的所有训练特性（并行采集、自动评估等）

**承接关系**: 扰动站立训练**必须**从静态站立模型 warm start，否则难以收敛。

## 快速开始

### 第一阶段：训练静态站立模型

```bash
# 直接运行（使用默认配置）
python standing.py

# 或指定并行工作进程数（默认为 CPU 核心数的一半）
STANDING_ROLLOUT_WORKERS=8 python standing.py
```

等待训练完成，记录下生成的运行目录，例如：`runs/standing_20240101_120000/`

### 第二阶段：训练扰动站立模型

```bash
# 使用静态站立模型作为初始化
STANDING_TURBULENCE_INIT_MODEL=runs/standing_20240101_120000/best_model.pt \
  python standing_with_turbulence.py

# 或同时指定并行工作进程数
STANDING_TURBULENCE_INIT_MODEL=runs/standing_20240101_120000/best_model.pt \
  STANDING_ROLLOUT_WORKERS=8 \
  python standing_with_turbulence.py
```

## 训练输出

### 第一阶段输出 (`standing.py`)

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

### 第二阶段输出 (`standing_with_turbulence.py`)

训练完成后，在 `runs/standing_<timestamp>/` 目录下生成相同结构的内容，但模型具备抗风扰动能力。

**注意**: 第二阶段的 `policy/` 目录包含的是扰动鲁棒性策略，推荐用于实际对抗环境。

## 环境变量

### 通用环境变量（两个脚本均支持）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STANDING_ROLLOUT_WORKERS` | `min(64, cpu_count/2)` | 数据采集并行进程数 |
| `STANDING_EVAL_WORKERS` | `min(rollout_workers, 16)` | 评估并行进程数 |

### 扰动站立专用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STANDING_TURBULENCE_INIT_MODEL` | 空 | 初始化模型路径（推荐使用静态模型的 best_model.pt） |
| `STANDING_INIT_MODEL` | 空 | 备用初始化模型路径（当 `STANDING_TURBULENCE_INIT_MODEL` 未设置时使用） |

## 使用训练好的策略

### 静态站立策略（无扰动环境）

```bash
# 在 run_round.py 中使用静态站立策略
python envs/humanoid21/run_round.py \
  --policy-a baseline/humanoid21/runs/standing_<timestamp>/policy \
  --policy-b random \
  --video match.mp4
```

### 扰动鲁棒策略（推荐用于实际对抗）

```bash
# 使用扰动站立策略（具备更强的环境适应能力）
python envs/humanoid21/run_round.py \
  --policy-a baseline/humanoid21/runs/standing_<timestamp>/policy \
  --policy-b random \
  --video match.mp4
```

**策略选择建议**:
- **静态站立策略**: 适用于无扰动环境，作为评估基准
- **扰动鲁棒策略**: 适用于实际对抗环境，能更好地应对对手的推搡和碰撞

## 关键配置

### 通用配置（两个脚本相同）

- **Episode 时长**: 5 秒 (100 steps @ 20Hz)
- **初始距离**: 1.5 ~ 3.5 米（随机）
- **Group size**: 8
- **Episodes per update**: 256
- **最大更新数**: 10000

### 扰动站立专用配置 (`standing_with_turbulence.py`)

- **扰动类型**: 持续风力 (`ContinuousWindPlugin`)
- **风向**: [1.0, 0.35, 0.0] 归一化方向
- **基础风力**: 5.0 N
- **阵风概率**: 3% (每步)
- **阵风倍数**: 2.0x

## 奖励机制

- 每步奖励: +1（保持站立）
- 终止条件: 倒下（高度 < 1.10m 或直立度 < 0.8，持续 3 步）
- 目标: 最大化站立时长
