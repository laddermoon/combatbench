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

## 2026-03-23 Phase 4 evaluation scaffold

- 本阶段目标：让训练产出的 PPO checkpoint 可以直接进入现有 round runner 与视频链路。
- 当前实现：
  - 新增 `policy_adapter.py`，提供 `SB3PPOCombatPolicy`。
  - 新增 `eval_policy.py`，支持：
    - 指定 checkpoint
    - 指定对手
    - 固定 seed
    - 可选视频输出
    - JSON summary 输出
  - `RoundRunner.run()` 新增了可选 `seed` 参数。
- 当前判断：
  - 现在已经具备训练后评估、导出视频、汇总结果的最小闭环。
  - 下一步主要是文档收口和一次真实的小规模训练验证。

## 2026-03-23 Phase 5 README consolidation

- 本阶段目标：把目录文档收口成“照着跑就能开始用”的状态。
- 当前实现：
  - README 已补齐：
    - 环境准备
    - nonfall smoke test
    - PPO 训练命令
    - checkpoint 评估命令
    - 直接通过 `run_round.py` 加载训练模型的方式
 - 当前判断：
   - 现在从代码结构和文档结构上都已经具备可交付雏形。
   - 真正还缺的是一次实际训练 run 和对应的视频结果验证。

## 2026-03-23 Smoke training validation

 - 本次动作：运行了一次小规模 PPO smoke 训练，用来验证训练产物链路。
 - 训练命令要点：
   - `total_timesteps=1024`
   - `n_steps=128`
   - `batch_size=64`
   - `checkpoint_freq=256`
   - `eval_freq=256`
   - `match_duration=2`
   - `opponent=standing`
 - 产物目录：
   - `baseline/mujoco21dof_nonfall/runs/smoke_train_20260323_153807`
 - 已确认生成：
   - `run_config.json`
   - `summary.json`
   - `final_model.zip`
   - `checkpoints/ppo_attacker_{256,512,768,1024}_steps.zip`
   - `best_model/best_model.zip`
   - `eval/evaluations.npz`
 - 日志观察：
   - rollout `ep_rew_mean` 大约在 `3.3` 左右
   - eval `mean_reward` 大约在 `2.2 ~ 2.4`
   - 脚本端到端成功结束，没有出现训练入口或保存回调错误
 - 当前判断：
   - 训练/保存/评估的工程链路已经打通。
   - 下一步应该直接用 `eval_policy.py` 对这个 smoke checkpoint 导出一段视频，看动作是否已经有明显进攻意图。

 ## 2026-03-23 Smoke evaluation validation

 - 本次动作：对 smoke 训练产生的 `final_model.zip` 跑了 1 局评估，并导出了视频与 summary。
 - 评估产物：
   - `baseline/mujoco21dof_nonfall/runs/smoke_train_20260323_153807/smoke_eval.mp4`
   - `baseline/mujoco21dof_nonfall/runs/smoke_train_20260323_153807/smoke_eval_summary.json`
 - 评估结果摘要：
   - `episodes=1`
   - `mean_steps=40`
   - `winner=draw`
   - `robot_a damage_dealt=0.0`
   - `robot_b damage_dealt=0.0`
   - `video_frames=58`
 - 当前判断：
   - `policy_adapter.py`、`eval_policy.py`、视频保存链路都已经工作正常。
   - 1024 steps 的 smoke 训练只验证工程链路，不足以学出有效攻击，这个结果是合理的。
   - 下一步应该开始一次更长的训练，并对视频里的动作活跃度和接近行为做针对性观察。
