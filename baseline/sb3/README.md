# CombatBench SB3 Baseline

这是一个放在 `things/combatbench/combatbench/baseline/sb3` 下的 **Stable-Baselines3 PPO baseline**。

它的目标不是一次性解决完整格斗策略，而是先给出一个 **可训练、可验证、可对战回放** 的起点，让两个机器人先学会稳定站立，再继续学会靠近、对准、出手和维持对战姿态。

## Baseline 特点

- 使用 `CombatGymEnv` 作为底层环境
- 使用 **共享策略 (shared policy)** 的对称自博弈封装
- 训练时同一个动作会同时发送给两个机器人
- 奖励函数同时考虑：
  - 站立高度
  - 身体竖直度
  - 双脚触地
  - 面向对手
  - 合适对打距离
  - 造成伤害 / 受到伤害
  - 出界惩罚
  - 动作幅度与动作突变惩罚
- 提供三类入口：
  - `validate_env.py`
  - `train.py`
  - `evaluate.py`

## 目录说明

```text
baseline/sb3/
├── __init__.py
├── README.md
├── TRAINING_GUIDE_zh.md
├── evaluate.py
├── normalization.py
├── policies.py
├── rewards.py
├── selfplay_env.py
├── train.py
└── validate_env.py
```

## 运行前提

建议在 `things/combatbench` 目录下运行下面这些命令。

依赖至少应包含：

- `mujoco`
- `gymnasium`
- `numpy`
- `torch`
- `stable-baselines3`
- `scipy`
- `opencv-python`

## 1. 先验证环境和封装

```bash
python3 -m combatbench.baseline.sb3.validate_env
```

这个脚本会做三件事：

- 检查原始环境是否能 `reset/step`
- 检查 SB3 baseline 包装环境是否能正常返回观测/奖励
- 调用 `stable_baselines3.common.env_checker.check_env`

## 2. 训练阶段一：站立预训练

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 1000000 \
  --run-name stand_v1
```

产物会保存到：

```text
combatbench/baseline/sb3/runs/stand_v1/
```

其中最重要的是：

- `run_config.json`
- `checkpoints/`
- `best_model/`
- `model_final.zip`

## 3. 训练阶段二：对打微调

```bash
python3 -m combatbench.baseline.sb3.train \
  --phase fight \
  --timesteps 2000000 \
  --run-name fight_v1 \
  --init-model combatbench/baseline/sb3/runs/stand_v1/model_final.zip
```

推荐思路是：

- 先把站立学稳
- 再把站立模型作为 fight 阶段初始化
- fight 阶段重点学习距离、朝向和出手收益

## 4. 自博弈评估

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode selfplay \
  --model combatbench/baseline/sb3/runs/fight_v1/model_final.zip \
  --phase fight \
  --episodes 5 \
  --duration 15
```

## 5. 跑一场双边模型对战并导出视频

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode match \
  --model combatbench/baseline/sb3/runs/fight_v1/model_final.zip \
  --duration 15 \
  --video combatbench/baseline/sb3/runs/fight_v1/demo_match.mp4
```

默认情况下，`--mode match` 会让红蓝双方都加载同一个模型。

如果你想让两个不同模型对打：

```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode match \
  --model path/to/red_model.zip \
  --model-b path/to/blue_model.zip \
  --duration 15
```

## 关键实现说明

- `normalization.py`
  - 对 127 维观测做稳定缩放
  - 尽量把不同量纲压到 `[-1, 1]`

- `rewards.py`
  - 定义 `stand` 和 `fight` 两套奖励配置
  - `stand` 更强调稳定站立
  - `fight` 更强调距离、朝向和伤害交换

- `selfplay_env.py`
  - 把双机器人环境包装成 SB3 可直接训练的单智能体环境
  - 返回归一化后的 `robot_a_obs`
  - 把共享策略动作同时发给两个机器人
  - 用双边 shaped reward 的平均值作为训练回报

- `policies.py`
  - 把 SB3 的 `.zip` 模型适配成 `tools/run_match.py` 可调用的策略对象

## 已知限制

- 这是 **baseline**，不是最终版比赛策略
- 当前训练封装是 **共享策略 + 对称动作训练**，更偏向“先学会站着打起来”
- 它不是完整的零和博弈自博弈系统，因此最终会存在一定协同性偏差
- 评估模式下让同一模型独立控制双方时，行为分布会比训练时更复杂

## 建议训练顺序

- 先跑 `validate_env.py`
- 再跑 `stand` 阶段
- 观察是否能稳定延长 episode
- 再用站立模型初始化 `fight` 阶段
- 最后用 `evaluate.py --mode match` 录视频看效果

更详细的设计和训练说明见：

- `TRAINING_GUIDE_zh.md`
