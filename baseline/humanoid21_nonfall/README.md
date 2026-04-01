# Humanoid21 Non-Fall Baseline

基于新框架 (`envs/framework` + `envs/humanoid21`) 的 GRPO 训练 baseline。

## 架构

```
baseline/humanoid21_nonfall/
├── __init__.py           # 模块导出
├── reward_config.py      # 奖励配置和计算函数
├── rewarder.py           # Humanoid21Rewarder (BaseObserverPlugin)
├── opponents.py          # 对手策略 (Standing, Random, ScriptedActive)
├── gym_adapter.py        # Gym 适配器 (SingleAgentAttackerEnv)
├── grpo.py              # GRPO 算法核心实现
└── train_grpo.py        # 训练入口脚本
```

## 与原 baseline 的关系

| 原实现 (mujoco21dof_nonfall) | 新实现 (humanoid21_nonfall) |
|------------------------------|------------------------------|
| `CombatGymEnv` | `MujocoCombatSimulator` + `EnvRuntime` |
| `env_wrapper.py` | `gym_adapter.py` |
| `reward.py` | `reward_config.py` + `rewarder.py` |
| `opponents.py` | `opponents.py` (复用接口) |
| `grpo.py` | `grpo.py` (核心算法复用) |
| `train_grpo.py` | `train_grpo.py` (适配新框架) |

## 使用方法

### 基础训练

```bash
# 使用默认参数训练
python -m combatbench.baseline.humanoid21_nonfall.train_grpo

# 指定运行名称和输出目录
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --run-name my_experiment \
    --output-dir outputs/humanoid21
```

### 训练参数

```bash
# 训练规模
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --total-timesteps 500000 \
    --n-envs 4 \
    --episodes-per-update 64

# 网络结构
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --hidden-sizes 512 512 256 \
    --log-std-init -0.5

# 优化参数
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --learning-rate 1e-4 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --max-grad-norm 0.5
```

### 环境参数

```bash
# 约束模式
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --non-fall-pitch-limit-deg 5.0 \
    --non-fall-roll-limit-deg 5.0

# 对手配置
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --opponent standing \
    --eval-opponent random

# 战斗参数
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --initial-distance 2.0 \
    --match-duration 10.0 \
    --damage-scale 100.0
```

### 奖励配置

#### 攻击模式 (attack)

```bash
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --curriculum-stage attack \
    --damage-reward-scale 1.0 \
    --hit-reward-scale 0.35 \
    --approach-reward-scale 0.8 \
    --win-bonus 2.0
```

#### 距离阶段模式 (distance_stage1)

```bash
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --curriculum-stage distance_stage1 \
    --distance-stage-target-distance 0.55 \
    --distance-stage-reward-mode step_delta \
    --distance-stage-reward-power 2.0
```

### 恢复训练

```bash
python -m combatbench.baseline.humanoid21_nonfall.train_grpo \
    --resume-from outputs/humanoid21/grpo_humanoid21_001/checkpoints/grpo_actor_100000.pt
```

## 输出结构

```
runs/
└── grpo_humanoid21_001/
    ├── config.json           # 训练配置
    ├── summary.json          # 训练总结
    ├── checkpoints/          # 定期检查点
    │   ├── grpo_actor_20000.pt
    │   └── ...
    ├── best_model/           # 最佳模型
    │   └── best_model.pt
    ├── eval/                 # 评估结果
    │   ├── eval_10000.json
    │   ├── evaluations.npz
    │   └── ...
    └── tensorboard/          # TensorBoard 日志
        └── ...
```

## 核心组件说明

### Humanoid21Rewarder

继承自 `BaseObserverPlugin`，从 `ReadOnlySimContext` 提取指标并计算奖励：

- 从 `ctx.accessor` 获取机器人状态
- 从 `ctx.metrics` 获取伤害、clamp 次数等
- 从 `ctx.events` 获取命中事件
- 支持两种课程模式：`attack` 和 `distance_stage1`

### SingleAgentAttackerEnv

Gym 适配器，将 `EnvRuntime` 包装为单智能体环境：

- `robot_a` 是训练的策略
- `robot_b` 是对手策略
- 返回标准的 `(obs, reward, terminated, truncated, info)` 元组

### GRPO 算法

- `GRPOActor` - 策略网络（带 tanh 压缩的高斯策略）
- `GRPORolloutCollector` - 经验收集器
- `optimize_grpo()` - GRPO 优化步骤
- `evaluate_grpo_actor()` - 策略评估

## 依赖

- `mujoco` - 物理引擎
- `gymnasium` - Gym API
- `torch` - 深度学习框架
- `stable-baselines3` - VecEnv 支持
- `tensorboard` - 训练监控
- `scipy` - 空间变换
