# Humanoid21 Baseline

Humanoid21 环境的基线训练实现，包含课程学习（Curriculum Learning）训练框架、奖励插件、环境蓝图和实验配置。

## 目录结构

```
baseline/humanoid21/
├── blueprints/        # 环境蓝图（YAML）
├── curriculum/        # 课程学习训练框架
├── plugins/           # 训练用环境插件
├── rewards/           # 奖励插件
├── tests/             # 单元测试
└── runs/              # 训练产物（gitignored）
```

### `blueprints/`

环境蓝图 YAML 文件，定义训练环境配置（插件组合、参数、初始策略等）。

| 文件 | 说明 |
|------|------|
| `basic_balance_env.yaml` | 基础平衡训练环境 |
| `basic_balance_v2_env.yaml` | V2 基础平衡环境 |
| `balance_recover_env.yaml` | 平衡恢复训练环境 |
| `balance_recover_v2_env.yaml` | V2 平衡恢复环境 |
| `balance_recover_plus_v2_env.yaml` | 增强版平衡恢复环境 |
| `standup_env.yaml` | 起身训练环境 |
| `follow_env.yaml` | 跟踪对手训练环境 |
| `follow_v2_env.yaml` | V2 跟踪对手环境 |
| `fight_env.yaml` | 对抗训练环境 |
| `fight_v2_env.yaml` | V2 对抗环境 |
| `fight_mixed.yaml` | 混合策略对抗环境（参数化蓝图） |
| `mixed.yaml` | 混合策略环境（参数化蓝图） |
| `init_policy.yaml` | 初始策略蓝图 |

### `curriculum/`

课程学习训练框架，支持四阶段训练（平衡 → 门控网络 → 跟踪 → 对抗）。详见 [`curriculum/README.md`](curriculum/README.md)。

**主要文件**：

| 文件 | 说明 |
|------|------|
| `train.py` | 统一训练 CLI 入口，通过 `--experiment` 选择实验配置 |
| `train_gating_network.py` | Gating MLP 分类器训练脚本 |
| `fight_mixed_policy.py` | 混合策略：主学习 Fight 策略 + 冻结 Follow 策略 + 恢复策略，通过 Gating MLP 切换 |
| `mixed_policy.py` | 混合策略：主学习策略 + 冻结恢复策略，通过 Gating MLP 切换 |
| `weakened_policy.py` | 弱化策略包装器，对导出策略的动作添加可调高斯噪声 |
| `collect_gating_data.py` | 使用弱化策略收集 Gating 分类器训练数据 |
| `collect_gating_data_refine.py` | 多级扰动覆盖的 Gating 数据收集 |
| `analyze_logs.py` | 通用训练日志监控工具 |
| `analyze_fight_logs.py` | Fight 实验日志分析工具 |
| `analyze_follow_logs.py` | Follow 实验日志分析工具 |
| `analyze_standup_logs.py` | Standup 实验日志分析工具 |

**`curriculum/framework/`** — 通用训练框架，详见 [`curriculum/README.md`](curriculum/README.md)。

**`curriculum/experiments/`**

实验配置注册表，自动发现 `exp_*.py` 文件。每个文件导出 `EXPERIMENT: ExperimentConfig`。

| 文件 | 说明 |
|------|------|
| `exp_basic_balance.py` | 基础平衡实验 |
| `exp_basic_balance_v2.py` | Baseline V2 基础平衡实验 |
| `exp_balance_recover.py` | 平衡恢复实验 |
| `exp_balance_recover_v2.py` | Baseline V2 平衡恢复实验 |
| `exp_balance_recover_plus.py` | 增强版平衡恢复实验 |
| `exp_balance_recover_plus_refine.py` | 多级扰动课程（防遗忘） |
| `exp_balance_recover_plus_v2.py` | Baseline V2 增强版平衡恢复实验 |
| `exp_standup.py` | 起身训练实验 |
| `exp_follow.py` | 跟踪对手实验 |
| `exp_follow_v2.py` | Baseline V2 跟踪对手实验 |
| `exp_fight.py` | 对抗实验 |
| `exp_fight_v2.py` | Baseline V2 对抗实验 |
| `exp_fight_v2_oppopool.py` | Baseline V2 对手池对抗实验 |

### `plugins/`

训练用环境插件，用于控制对手行为和自定义终止条件。

| 文件 | 说明 |
|------|------|
| `standing_termination.py` | 站立/平衡实验的终止条件插件 |
| `balance_score_termination.py` | 平衡评分终止插件 |
| `imbalance_termination.py` | 失衡终止插件 |
| `random_move.py` | 对手随机移动插件（用于跟踪训练） |

### `rewards/`

奖励插件，实现各训练阶段的奖励函数。

| 文件 | 说明 |
|------|------|
| `balance.py` | 平衡分析奖励（基于支撑面投影） |
| `cross_support.py` | 交叉支撑平衡奖励 |
| `standing_posture.py` | 站立姿态评分 |
| `posture_reward.py` | 姿态诊断观测器（记录 4 项姿态指标） |
| `action_limit.py` | 动作限制（关节姿态）奖励 |
| `follow_opponent.py` | 跟踪对手奖励（距离控制） |
| `opponent_relation.py` | 对手关系奖励（相对位置/朝向） |
| `damage.py` | 净伤害奖励（造成伤害 - 受到伤害） |
| `standup.py` | 起身势能奖励（分段势能函数） |

### `tests/`

| 文件 | 说明 |
|------|------|
| `test_curriculum_gate.py` | 课程门控测试 |
| `test_fight_mixed_policy.py` | 混合对抗策略测试 |

## 训练流程

课程学习按四阶段递进：

1. **平衡（Balance）** — 学会站立不倒
2. **门控网络（Gating）** — 训练状态危险判别器
3. **跟踪（Follow）** — 接近对手到有效距离
4. **对抗（Fight）** — 在保持平衡的前提下打击对手

详细训练说明请参考：
- [Baseline V1 训练指南](curriculum/TRAINING_V1.md)
- [Baseline V2 训练指南](curriculum/TRAINING_V2.md)

```bash
# 列出可用实验
python3 baseline/humanoid21/curriculum/train.py --list-experiments

# 运行指定实验
python3 baseline/humanoid21/curriculum/train.py --experiment basic_balance
```
