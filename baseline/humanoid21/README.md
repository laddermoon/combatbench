# Humanoid21 Baseline

Humanoid21 环境的基线训练实现：奖励插件、环境插件、环境蓝图、训练实验，
以及课程学习阶段的遗留资产。

> **当前训练路径**：新增实验与训练走
> [`../experiments_ppo/`](../experiments_ppo/README.md) 注册表 +
> `baseline/framework/train.py`。`curriculum/` 是旧一代训练框架的遗留
> （见下文），不再维护。

## 目录结构

```
baseline/humanoid21/
├── blueprints/        # 环境/初始策略蓝图（YAML，~57 个）
├── rewards/           # 奖励与诊断 observer（22 个模块）
├── plugins/           # 训练用环境插件（终止条件、对手行为、扰动）
├── tests/             # 单元测试
├── curriculum/        # ⚠️ 遗留：旧训练框架/脚本/历史记录（注册表已失效）
├── balance_recover/   # 平衡恢复子项目（脚本、数据、文档）
├── end2end/           # 端到端方案备忘（分段条件奖励思路）
├── fight/             # 对抗训练子目录
├── follow/            # 跟随训练子目录
└── runs/              # 训练产物（gitignored）
```

### `blueprints/`

环境蓝图与初始策略蓝图 YAML。按用途分三类：

| 类别 | 文件（举例） | 说明 |
|------|------|------|
| 任务环境（`*_env.yaml`） | `basic_balance_env.yaml`、`basic_balance_v2_env.yaml`、`standup_env.yaml`、`follow_env.yaml`、`fight_env.yaml`、`balance_recover*_env.yaml` 等 ~45 个 | 各训练阶段的环境配置（插件组合、参数） |
| 参数化蓝图 | `fight_mixed.yaml`、`fight_mixed_v2.yaml`、`mixed.yaml`、`hybrid_env.yaml`、`standup_fallback.yaml` | 带 `parameters:` 节的可物化蓝图 |
| 初始策略（`init_policy*.yaml`） | `init_policy.yaml` + 12 个 `init_policy_<family>.yaml` | 各策略族的初始权重策略蓝图 |

### `rewards/`

奖励/诊断 observer 插件，按实验蓝图组合使用。主要实现：

| 文件 | 类 | 说明 |
|------|-----|------|
| `cross_support.py` | `CrossSupportBalanceRewarder` | 交叉支撑平衡奖励 |
| `balance.py` | `BalanceValueRewarder` | 基于支撑面投影的平衡分析 |
| `standing_posture.py` | `StandingPostureRewarder` | 站立姿态评分 |
| `posture_reward.py` | `PostureRewarder` | 姿态诊断观测（4 项指标） |
| `action_limit.py` | `ActionLimitRewarder` | 关节姿态限位奖励 |
| `follow_opponent.py` | `InZoneHoldRewarder` | 跟随/驻留奖励 |
| `opponent_relation.py` | `OpponentRelationRewarder` | 相对位置/朝向奖励 |
| `damage.py` | `NetDamageRewarder` | 净伤害奖励（造成 − 承受） |
| `punch_motion.py` | `PunchMotionRewarder` | 出拳动作奖励 |
| `rollover.py` | `RolloverRewarder` | 翻身奖励 |
| `standup.py` ~ `standup_v3.py`、`standup_4stage.py`、`standup_energy.py`、`standup_repro_v2.py` | `Standup*Rewarder` | 起身势能奖励的历代实现（v1/v2/v2_r7/v2_r10/v3/四阶段/能量版/复现版） |
| `standing_balance_3stage.py` / `standing_balance_4stage.py` | `StandingBalance*Rewarder` | 分阶段站立平衡奖励 |
| `phase_observer.py` | `PhaseObserver` | 相位观测器 |
| `fall_contact_observer.py` | `FallContactObserver` | 倒地接触观测 |
| `wall_contact.py` | `WallContactObserver` | 靠墙接触观测 |
| `distance_potential.py` | （函数模块） | 距离势能奖励计算工具函数 |

### `plugins/`

训练用环境插件（终止条件、对手控制、扰动初始化）：

| 文件 | 类 | 说明 |
|------|-----|------|
| `standing_termination.py` | `StandingTerminationPlugin` | 站立/平衡终止条件 |
| `standup_termination.py` | `StandupTerminationPlugin` | 起身终止条件 |
| `standup_4stage_termination.py` | `Standup4StageTerminationPlugin` | 四阶段起身终止 |
| `standup_energy_termination.py` | `StandupEnergyTerminationPlugin` | 能量式起身终止 |
| `imbalance_termination.py` | `ImbalanceTerminationPlugin` | 失衡终止 |
| `balance_score_termination.py` | `BalanceScoreTerminationPlugin` | 平衡评分终止 |
| `random_fall.py` | `RandomFallenStatePlugin` | 随机倒伏初始状态 |
| `random_move.py` | `RandomMovePlugin` | 对手随机移动（跟随训练靶） |
| `height_observer.py` / `height_phi_observer.py` / `height_phi_min_observer.py` | `HeightObserver` / `HeightPhiObserver` / `HeightPhiMinObserver` | 高度/φ 健康指标观测 |

### `tests/`

当前无测试文件。原 `test_curriculum_gate.py`（守护已删除的
`CurriculumStageGate`）与 `test_fight_mixed_policy.py`（依赖已不存在的
历史 runs/ 导出）于 2026-10 审计中删除——它们守护的对象均属 Legacy 资产。

### `curriculum/` ⚠️ 遗留目录

旧一代（v1）课程学习训练框架的遗留资产。`curriculum/experiments/` 注册表
import 已删模块 `baseline.framework.experiment`，**当前不可运行**；
顶层脚本（gating 训练、混合策略、数据收集等）与多份历史训练记录保留备查。
详见 [`curriculum/README.md`](curriculum/README.md)。

## 训练流程

当前标准路径（PPO v2 框架）：

```bash
# 列出可用实验
PYTHONPATH=. python3 baseline/framework/train.py --list-experiments

# 运行指定实验
PYTHONPATH=. python3 baseline/framework/train.py --experiment basic_balance --smoke
```

实验定义在 [`../experiments_ppo/`](../experiments_ppo/README.md)，框架机制见
[`../framework/ppo/GUIDE.md`](../framework/ppo/GUIDE.md)。

历史四阶段课程（平衡 → 门控 → 跟踪 → 对抗）的过程记录在
`curriculum/TRAINING_V1.md` / `TRAINING_V2.md` /
`STANDUP_V2_TRAINING_HISTORY.md`。
