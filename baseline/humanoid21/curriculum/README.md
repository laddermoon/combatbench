# Curriculum — 课程学习训练框架（遗留）

> **⚠️ 历史档案**：本目录是旧一代（v1）训练框架的遗留目录，**当前不可运行**。
>
> - 实验注册表 `curriculum/experiments/` import 已删除的
>   `baseline.framework.experiment`，加载即 `ModuleNotFoundError`。
> - 原 `framework/` 子目录（`config.py` / `ppo_trainer.py` /
>   `training_loop.py`）与统一入口 `train.py` 已删除，v1 框架代码存档在
>   [`../../framework/obsolete/`](../../framework/obsolete/)。
> - 保留下来的：gating/混合策略脚本（见下）、gating 模型与数据目录、
>   以及完整的训练过程记录文档——作为历史证据与可复用思路留存。
>
> **当前训练路径请使用** [`../../experiments_ppo/`](../../experiments_ppo/README.md)
> + `baseline/framework/train.py`。

## 目录用途（历史定位）

`curriculum/` 曾是 Humanoid21 基线策略的核心训练目录：把格斗任务拆解为
多个阶段（平衡 → 门控 → 跟踪 → 对抗）逐步叠加难度。该机制已由 v2 框架
（`baseline/framework/ppo/` + `experiments_ppo/`）取代——实验现在通过
`build_trajectories()` 里的 `actor_weight` 直接做课程调度，不再需要框架级
课程钩子。

## 现存内容

### 策略与脚本（存活，但多数依赖旧接口）

| 文件 | 说明 |
|------|------|
| `train_gating_network.py` | 门控网络（Gating MLP 分类器）训练脚本 |
| `collect_gating_data.py` / `collect_gating_data_refine.py` | 门控数据收集脚本 |
| `fight_mixed_policy.py` / `fight_mixed_policy_v2.py` | 混合策略：主学习策略 + 冻结恢复策略，经 Gating MLP 切换 |
| `mixed_policy.py` / `height_switch_policy.py` / `hybrid_actor.py` / `standup_fallback_policy.py` | 各类策略组合/切换包装器 |
| `weakened_policy.py` | 弱化策略包装器（对导出策略动作加高斯噪声） |
| `run_4stage_chain.py` / `run_orig_chain.py` / `run_repro_chain.py` | 分阶段训练链的调度脚本 |
| `gating_model*/`、`gating_data*/` | 训好的门控模型与收集数据 |

### `experiments/` — 旧实验注册表（已失效）

原自动发现 `exp_*.py` 的注册表，导出 `EXPERIMENT` 配置。**当前 import 即崩**
（依赖已删的 `baseline.framework.experiment`），仅供考古。

### 训练记录文档（历史档案，有参考价值）

| 文档 | 内容 |
|------|------|
| `TRAINING_V1.md` | Baseline V1 四阶段课程训练过程 |
| `TRAINING_V2.md` | Baseline V2 训练过程 |
| `STANDUP_V2_TRAINING_HISTORY.md` | standup_v2 的完整训练史 |
| `REVIEW_SUMMARY.md` / `experiments/REVIEW_SUMMARY.md` | 复盘记录 |
| `experiments/STANDUP_ORIG_*.md` | 原始 standup 奖励分析 |

### 历史机制说明（仅供读旧代码时参考）

- **Sub-episode 分段**：门控判定需要平衡恢复介入时截断轨迹，恢复策略
  介入的帧不参与 PPO 更新，每段独立算 GAE——v1 通过
  `Experiment.prepare_training_segments()` 实现；v2 中等价能力由实验在
  `build_trajectories()` 里自行切分完成。
- **奖励 channel**：V1 实验 4 个 channel（r_fall/r_cross/r_relation/
  r_damage），V2 扩到 6 个（+r_hold/r_radial/r_tangential）。v2 框架的
  channel 机制沿用了这套思路（见 `ppo/GUIDE.md`）。
