# THOUGHTS_AND_EXP

## 2026-03-23 Phase 0 kickoff

- 目标：先把 nonfall baseline 的基础工程链路打通，再进入训练 wrapper 和 reward 设计。
- 当前调研结论：
  - `CombatGymEnv` 已支持 `non_fall_mode` 与 pitch/roll 限制。
  - `RoundRunner` 与 `tools/run_round.py` 已具备回放与视频导出链路。
  - 训练 baseline 目前基本空白，`CombatGymEnv.step()` 仍返回全零 reward。
- 本阶段动作：
  - 打通 nonfall pitch/roll 参数传递。
  - 去掉对旧 baseline phase 配置的依赖。
  - 建立本目录的 README 与实验记录文件。
- 预期下一步：
  - 进入单智能体 wrapper 设计。
  - 明确 attacker-only reward shaping 字段。

## 2026-03-23 Phase 1 wrapper scaffold

- 本阶段目标：先建立一个可直接给 SB3 使用的单智能体训练入口。
- 当前实现：
  - 新增 `env_wrapper.py`，只对外暴露 `robot_a` 的 observation/action。
  - wrapper 内部自动调度 `robot_b` 对手策略。
  - 默认 reward 暂时只返回 `damage_dealt`，后续再扩展完整 shaping。
  - 新增 `opponents.py`，支持 `standing`、`random`、`scripted_active` 三类对手。
- 当前判断：
  - 先把 observation/action/info 接口稳定下来，比现在就开始调 reward 更重要。
  - 需要在下一阶段把距离、朝向、动作活跃度等 shaping 正式接进 reward。

## 2026-03-23 Phase 2 reward shaping v0

- 本阶段目标：让单智能体 wrapper 默认就具备可训练的 attacker reward。
- 当前实现：
  - 新增 `reward.py`，把 reward 逻辑从 wrapper 中拆出来。
  - 当前 reward 由以下部分组成：
    - `damage_dealt`
    - `damage_received_penalty`
    - `hit_reward`
    - `approach_reward`
    - `facing_reward`
    - `facing_delta_reward`
    - `action_magnitude_reward`
    - `action_delta_reward`
    - `inactivity_penalty`
    - terminal `win_bonus` / `loss_penalty`
  - wrapper 额外输出了 `horizontal_distance_delta`、`facing_delta`、`win`、`loss` 等指标。
- 当前判断：
  - 这版 reward 明确偏向“主动进攻”，但只给了很轻的受伤惩罚。
  - 下一步需要进入训练脚本阶段，通过真实 rollout 观察这些项是否平衡。

## 2026-03-23 Phase 3 SB3 training scaffold

- 本阶段目标：提供一个真正可启动训练的 PPO 入口。
- 当前实现：
  - 新增 `train_sb3.py`。
  - 已接入 `SingleAgentAttackerEnv`。
  - 已支持：
    - run directory
    - `run_config.json`
    - checkpoint 保存
    - `best_model` 保存
    - tensorboard 输出目录
    - resume from checkpoint/model
- 当前判断：
  - 现在已经具备从环境到 PPO 的最小训练闭环。
  - 下一步需要补 checkpoint 到 `BaseCombatPolicy` 的适配，以及视频评估脚本。
