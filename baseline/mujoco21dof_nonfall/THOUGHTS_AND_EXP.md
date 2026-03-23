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
