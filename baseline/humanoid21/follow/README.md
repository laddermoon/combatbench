# Follow 训练说明

## 概述

单 agent 跟随实验：训练人形机器人追逐一个脚本控制的移动靶，同时保持平衡。

对手由 `RandomMovePlugin` 控制（teleport 式移动，不会摔倒），学习策略直接面对环境，**不使用 MixedPolicy / 不使用 Fallback**——机器人必须自己学会在追击中保持平衡。

## 文件

| 文件 | 作用 |
|---|---|
| `follow_env.yaml` | 环境蓝图（插件 + observer 配置） |
| `baseline/experiments_v2/exp_follow.py` | V2 实验定义（奖励、课程、eval） |

## 环境配置

- **对手**：`RandomMovePlugin`，在 arena 内随机移动，始终面向训练机器人，保持 `min_avoid_distance=1.2m` 避碰距离
- **终止**：`ImbalanceTerminationPlugin`，训练机器人失衡即终止 episode
- **Observer**：
  - `cross_support` — CrossSupportBalanceRewarder（交叉支撑平衡，r_cross 来源）
  - `height_phi` — HeightPhiObserver（φ = uprightness × height/standing_height，r_fall 和 actor_weight 门控来源）
  - `approach_velocity` — ApproachVelocityRewarder（记录双方 xy 位置，r_radial/r_tangential 计算来源）

## 奖励结构（4 channel）

| Channel | 奖励 | actor_weight | 说明 |
|---|---|---|---|
| `r_fall` | `0.01 × φ(t)` 每步 | 固定 3.0 | 密集生存奖励，无终局惩罚。φ 高（站得稳）时奖励高 |
| `r_cross` | CrossSupportBalanceRewarder 原始输出 | `1.0 × φ²` | 平衡维护。快摔倒时（φ 低）权重自动降低 |
| `r_radial` | 径向接近速度（朝对手移动的速度分量） | `3.0 × φ²` | 核心跟随信号。仅在区外（>0.9m）生效 |
| `r_tangential` | 切向移动惩罚（横向绕圈） | `1.0 × φ²` | 抑制无效绕圈。仅在区外生效 |

### φ² 门控设计

`φ = uprightness × (height / 1.28)`，范围 [0, 1]。

- φ 高（站得稳）→ r_cross/r_radial/r_tangential 权重正常，机器人被引导追击 + 维持平衡
- φ 低（快摔倒）→ 三个 channel 权重趋零，r_fall 主导，机器人专注不摔倒

这与 `exp_balance_recover.py` 的设计一致：平衡快失控时不该再管追击和姿态细节，优先活下来。

### r_radial / r_tangential 计算

来自 `compute_radial_tangential_rewards()`（`baseline/humanoid21/rewards/follow_opponent.py`）：

1. 对自身 xy 轨迹做居中移动平均（窗口 17 步 ≈ 0.85s），消除步态摆动
2. 居中差分得每步净位移向量
3. 分解为径向（朝对手）和切向（横向）分量
4. 仅在区外（distance > 0.9m）给信号，区内静默

## 课程调度

8 级对手移动速度，从静止到快速：

```
Level 0: 0.0 m/s (静止靶)
Level 1: 0.1 m/s
Level 2: 0.2 m/s
Level 3: 0.3 m/s
Level 4: 0.4 m/s
Level 5: 0.5 m/s
Level 6: 0.6 m/s
Level 7: 0.7 m/s
```

**升级条件**：`hold_ratio ≥ 0.5` 且连续 1 次 eval 通过
- `hold_ratio` = episode 中距离对手 ≤ 1.1m 的步占比

**Early stop**：200 次 eval 无改善 + 最少 600 updates

## PPO 参数

| 参数 | 值 | 说明 |
|---|---|---|
| learning_rate | 3e-5 | 保守学习率 |
| target_kl | 0.05 | KL 早停阈值 |
| update_epochs | 4 | 每次更新的 epoch 数 |
| minibatch_size | 16384 | 大 batch 稳定训练 |
| entropy_coef | 1.5e-3 | 鼓励探索 |
| log_std_min | -1.8 | 限制策略熵下限 |
| episodes_per_update | 1024 | 每次 update 的 episode 数 |
| eval_episodes | 128 | 每次 eval 的 episode 数 |
| eval_interval | 2 | 每 2 updates eval 一次 |
| max_updates | 20000 | 最大训练 updates |

## Warm Start

从平衡恢复训练的 checkpoint 启动，让策略先具备平衡能力再学追击。

```bash
cd /data1/mono/things/combatbench

# 从 recovery_v1_gen3 最新 checkpoint warm start
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 baseline/framework/train.py \
  --experiment follow --algo ppo \
  --resume-from baseline/runs/recovery_v1_gen3/checkpoints/checkpoint_u00445.pt \
  --background
```

**为什么需要 warm start**：follow 实验没有 MixedPolicy / Fallback 安全护盾，策略直接面对环境。如果从随机初始化开始，机器人还没学会站立就会不断摔倒，r_radial/r_tangential 的梯度信号被淹没。从平衡恢复 checkpoint 启动，策略一开始就能站稳（φ≈1），追击奖励的梯度可以立即生效。

**checkpoint 来源**：`recovery_v1_gen3`（`v2_weighted_impulse` 实验，第三代迭代训练），该策略在脉冲扰动下已具备强平衡能力。

## 启动训练

```bash
cd /data1/mono/things/combatbench

# 后台训练
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 baseline/framework/train.py \
  --experiment follow --algo ppo --background

# 前台训练（调试用）
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 baseline/framework/train.py \
  --experiment follow --algo ppo

# Smoke test（2 updates, 8 episodes, 快速验证）
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 baseline/framework/train.py \
  --experiment follow --algo ppo --smoke
```

## 监控

```bash
# 查看训练日志
tail -f baseline/runs/train_follow_ppo_*/train.log

# 查看训练视频
ls baseline/runs/train_follow_ppo_*/videos/

# 停止训练
kill <pid>  # pid 在 train.log 开头或 run_dir/pid 文件中
```

## 与 V1 follow_v2 的区别

| | V1 (`curriculum/experiments/exp_follow_v2.py`) | V2 (`experiments_v2/exp_follow.py`, name=`follow`) |
|---|---|---|
| 框架 | CombatExperimentBase (V1) | CombatExperimentV2Base (V2) |
| 安全护盾 | MixedPolicy + Gating MLP + Fallback | 无，策略直接面对环境 |
| Channel 数 | 9 (r_fall, r_cross, r_joint, r_vel, r_tilt, r_foot, r_radial, r_tangential, r_gate) | 4 (r_fall, r_cross, r_radial, r_tangential) |
| r_fall 设计 | 每步 0.01 + 终局 ±1 | `0.01 × φ(t)` 每步，无终局惩罚 |
| 平衡 channel | r_joint/r_vel/r_tilt/r_foot 独立惩罚 | r_cross 用 φ² 门控，替代 4 个姿态 channel |
| 追随 channel 权重 | 固定 (3.0, 1.0) | φ² 门控 (3.0×φ², 1.0×φ²) |
| r_gate | 有（MixedPolicy 切换惩罚） | 无（无 MixedPolicy） |
| 课程 | 相同 8 级 | 相同 8 级 |
