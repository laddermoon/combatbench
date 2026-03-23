 # Mujoco21dof Nonfall Baseline

 本目录用于实现 `CombatBench` 的 `mujoco21dof` nonfall baseline。

 ## 当前状态

 当前已完成的基础工作：

 - 已确认 `CombatGymEnv` 原生支持：
   - `non_fall_mode`
   - `non_fall_pitch_limit_deg`
   - `non_fall_roll_limit_deg`
 - 已打通 round runner 与 CLI 的 nonfall 参数传递链路。
 - 已移除对旧 phase baseline 配置模块的运行时依赖。
 - 已建立本目录的计划文件与实验记录文件。

 当前尚未完成：

 - 单智能体训练 wrapper
 - reward shaping
 - SB3 训练脚本
 - checkpoint 评估适配层

 ## 相关文件

 - `bootstrip.md`：任务目标与约束
 - `PLAN.md`：实现计划
 - `THOUGHTS_AND_EXP.md`：实验记录，只追加

 ## 当前可用的 nonfall 验证命令

 在项目根目录 `things/combatbench/` 下，可以先用现有 round runner 验证 nonfall 模式与视频链路：

 ```bash
 python tools/run_round.py \
   --policy-a combatbench.policy.RandomCombatPolicy \
   --policy-b combatbench.policy.StandingCombatPolicy \
   --duration 10 \
   --non-fall-mode \
   --non-fall-pitch-limit-deg 15 \
   --non-fall-roll-limit-deg 10 \
   --video outputs/nonfall_smoke.mp4
 ```

 这条命令当前的用途是：

 - 验证 `CombatGymEnv` 的 nonfall 参数生效
 - 验证 `RoundRunner` / CLI / 视频导出链路可用
 - 作为后续训练脚本完成前的基础 smoke test

 ## 实验记录约定

 所有实验思路、现象与结论统一追加到 `THOUGHTS_AND_EXP.md`。

 要求：

 - 只追加，不回写历史记录
 - 记录奖励修改点
 - 记录训练配置变化
 - 记录视频观察结论

 ## 下一步

 下一阶段将实现：

 1. attacker-only 单智能体训练环境包装
 2. 进攻导向奖励函数
 3. SB3 PPO 训练脚本
 4. 模型评估与视频导出工具
