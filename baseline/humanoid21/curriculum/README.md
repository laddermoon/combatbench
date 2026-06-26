# Curriculum — 课程学习训练框架

## 目录用途

`curriculum/` 是 Humanoid21 基线策略的核心训练目录，实现了基于 PPO 的课程学习框架。通过将格斗任务拆解为多个阶段（平衡 → 门控 → 跟踪 → 对抗），从简单技能开始逐步叠加难度，让策略循序渐进地掌握完整能力。

## 框架架构

### `framework/` — 通用训练框架

训练框架与具体实验解耦，新增实验只需一个配置文件，无需修改框架代码。

| 文件 | 说明 |
|------|------|
| `config.py` | `Experiment` 抽象基类，定义实验接口：reward keys、权重调度、reward 提取、评估指标、episode 分段 |
| `ppo_trainer.py` | PPO 训练器：buffer、update、rollout helpers，支持 sub-episode 分段（排除 fallback 策略介入的帧） |
| `training_loop.py` | 训练循环：checkpoint 管理、视频渲染、评估调度 |

### `experiments/` — 实验配置

每个 `exp_*.py` 文件导出 `EXPERIMENT: Experiment`，定义该实验的奖励方案、课程调度和环境配置。通过 `__init__.py` 自动注册。

### 其他文件

| 文件 | 说明 |
|------|------|
| `train.py` | 统一训练 CLI 入口，通过 `--experiment` 选择实验 |
| `fight_mixed_policy.py` | 混合策略：主学习策略 + 冻结恢复策略，通过 Gating MLP 切换 |
| `mixed_policy.py` | 混合策略：主学习策略 + 冻结恢复策略，通过 Gating MLP 切换 |
| `weakened_policy.py` | 弱化策略包装器，对导出策略的动作添加高斯噪声 |
| `collect_gating_data.py` | 门控数据收集脚本 |
| `collect_gating_data_refine.py` | 多级扰动门控数据收集脚本 |
| `train_gating_network.py` | 门控网络训练脚本 |
| `analyze_training.py` | 通用训练日志分析工具（支持所有实验） |

### Sub-episode 分段

训练框架支持 **sub-episode 分段**：
- 当门控网络判断需要平衡恢复介入时，自动截断轨迹
- 平衡恢复策略介入的帧被排除，不参与 PPO 梯度更新
- 每个分段独立计算 GAE，避免状态不连续导致的梯度错误
- 通过 `Experiment.prepare_training_segments()` 实现分段逻辑（默认返回完整 episode，需分段的实验覆盖此方法）

### 实验配置

Baseline V1 实验（不带 `_v2` 后缀）使用 4 个 reward（`r_fall`, `r_cross`, `r_relation`, `r_damage`），Baseline V2 实验使用 6 个 reward（`r_fall`, `r_cross`, `r_damage`, `r_hold`, `r_radial`, `r_tangential`），奖励方案更精细。

### CLI 使用

- 所有实验统一通过 `--experiment` 选择，无需额外 flag
- 需要分段的实验（如跟踪/对抗阶段）通过覆盖 `prepare_training_segments()` 自动启用 sub-episode 分段

## 训练指导文档

- **Baseline V1 训练指南**：[`TRAINING_V1.md`](TRAINING_V1.md)
- **Baseline V2 训练指南**：[`TRAINING_V2.md`](TRAINING_V2.md)
