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

 ## 2026-03-23 First formal training attempt

 - 本次动作：运行了第一次比 smoke 更正式的 PPO 训练，并完成了训练后评估。
 - 训练配置要点：
   - `run_name=formal_try1`
   - `total_timesteps=20000`
   - `n_steps=256`
   - `batch_size=128`
   - `checkpoint_freq=5000`
   - `eval_freq=5000`
   - `eval_episodes=3`
   - `match_duration=5`
   - `opponent=standing`
   - `non_fall_mode=true`
 - 训练产物目录：
   - `baseline/mujoco21dof_nonfall/runs/formal_try1_20260323_154933`
 - 训练阶段观察：
   - rollout `ep_rew_mean` 从大约 `6.6` 上升到 `8.3` 左右
   - 末次 eval callback `mean_reward` 大约为 `7.55`
   - `best_model`、`checkpoint`、`final_model` 都已正常生成
 - 训练后评估：
   - summary: `baseline/mujoco21dof_nonfall/runs/formal_try1_20260323_154933/formal_eval_standing_summary.json`
   - video: `baseline/mujoco21dof_nonfall/runs/formal_try1_20260323_154933/formal_eval_standing.mp4`
   - 对 `standing` 跑了 `3` 局，结果全部 `draw`
   - `mean_robot_a_damage_dealt=0.0`
   - `mean_steps=100`
   - 日志中的双方距离大约仍在 `1.98m`
 - 当前判断：
   - 这次正式训练已经证明 reward 和 PPO 可以稳定优化出更高训练回报，但当前 shaping 还不足以把策略推到“真正打中对手”。
   - 更具体地说，策略似乎学到了一些让 reward 上升的行为，但还没有学出足够的接近幅度和有效攻击动作。
   - 下一步优先级应该是：
     - 增强接近/出招相关奖励
     - 延长训练时长
     - 视情况把对手继续保持为 `standing`，先把“能打到人”这个问题解掉

 ## 2026-03-23 Tight 5-degree nonfall + reward shaping attempt

 - 本次动作：
   - 把 non-fall 的 pitch/roll 默认限位统一收紧到 `±5°`
   - reward 新增/增强了几类项：
     - 更强的 approach reward
     - close-distance reward
     - retreat penalty
     - upright reward / upright delta reward
     - tilt penalty
     - 更强的 hit / action-delta 激励
 - 训练配置：
   - `run_name=formal_try2_tight5`
   - `total_timesteps=30000`
   - `match_duration=5`
   - `opponent=standing`
   - `initial_distance=1.6`
   - non-fall 使用新的 `5/5` 默认限位
 - 训练产物目录：
   - `baseline/mujoco21dof_nonfall/runs/formal_try2_tight5_20260323_164301`
 - 训练阶段观察：
   - rollout `ep_rew_mean` 从开局大约 `11.6` 提升到结束时大约 `12.9`
   - 中间 eval callback 大约有：
     - `10k -> mean_reward ≈ 10.5`
     - `25k -> mean_reward ≈ 14.3`
     - `30k -> mean_reward ≈ 14.1`
   - 相比上一轮训练，训练回报提升明显
 - 训练后评估：
   - summary: `baseline/mujoco21dof_nonfall/runs/formal_try2_tight5_20260323_164301/formal_eval_standing_summary.json`
   - video: `baseline/mujoco21dof_nonfall/runs/formal_try2_tight5_20260323_164301/formal_eval_standing.mp4`
   - 对 `standing` 评估 `3` 局，结果全部 `draw`
   - `mean_robot_a_damage_dealt=0.0`
   - `mean_steps=100`
   - 但日志中的最终距离大约下降到 `1.43m`
 - 当前判断：
   - 这次改动是有效的，但有效性目前主要体现在“更愿意接近目标”和“训练回报更高”，还没有转化成有效命中。
   - 当前瓶颈已经从“几乎不接近”转成了“接近了，但没有形成有效攻击接触”。
   - 下一步优先级应该是：
     - 明确鼓励前向拳击/上肢接触
     - 针对进入近距离后的有效动作继续加 shaping
     - 继续保持 `standing` 对手，直到先学会稳定命中

## 2026-03-23 Stage-1 minimal reward attempt (distance delta + facing only)

- 本次动作：
  - 把第一阶段 reward 极简化为两项：
    - `distance_reward`: 只根据 `distance_error_delta = prev_error - current_error` 给线性奖励/惩罚
    - `facing_reward`: 保留朝向对手的稠密 shaping
  - 删除第一阶段其余所有项：到达奖励、区间保持奖励、过冲惩罚、upright/tilt、动作幅度/变化惩罚等
  - 明确要求 reward 与动作结果做因果对齐，按动作执行后的距离误差变化计算
- 训练配置：
  - `run_name=distance_stage1_linear_200k`
  - `total_timesteps=200000`
  - `n_steps=1024`
  - `batch_size=256`
  - `learning_rate=3e-4`
  - `ent_coef=0.01`
  - `match_duration=5`
  - `opponent=standing`
  - `distance_stage_target_distance=0.4`
- 训练产物目录：
  - `baseline/mujoco21dof_nonfall/runs/distance_stage1_linear_200k_20260323_175259`
- 训练阶段观察：
  - 训练在用户要求下中途停止，停止点大约为 `120k` timesteps
  - rollout `ep_rew_mean` 从开局大约 `2.5` 上升到 `8.6~8.7`
  - `120k` eval callback `mean_reward ≈ 7.65`
  - 但用户观察到没有明显行为改善，因此不继续跑满 `200k`
  - PPO 稳定性指标明显偏激进：
    - `clip_fraction` 上升到大约 `0.51~0.55`
    - `approx_kl` 上升到大约 `0.09~0.10`
    - `std` 上升到大约 `1.23~1.24`
- 当前判断：
  - 这版极简 reward 没有带来足够清晰的行为改进，至少在当前超参下没有体现出“更快、更稳地到 0.4m”这一目标。
  - 更大的问题不一定只在 reward，本轮 PPO 更新也偏猛，导致大量样本被 clipping，可能降低了优化有效性。
  - 下一步应优先保持 reward 不变，先把 PPO 调到更保守稳定的区域：
    - 显著增大 `batch_size` 降低梯度方差
    - 同步增加 rollout 覆盖的 episode 数
    - 降低 `learning_rate`
    - 降低 `ent_coef`
    - 增加 `target_kl` 约束更新幅度
