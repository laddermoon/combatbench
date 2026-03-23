 # Mujoco21dof Nonfall Baseline
 
 本目录用于实现 `CombatBench` 的 `mujoco21dof` nonfall baseline。
 
 ## 当前状态
 
 当前目录已经具备一条最小可用的 nonfall baseline 闭环：
 
 - 单智能体 attacker 训练环境：`env_wrapper.py`
 - 对手策略工厂：`opponents.py`
 - attacker reward shaping：`reward.py`
 - SB3 PPO 训练脚本：`train_sb3.py`
 - PPO checkpoint 策略适配：`policy_adapter.py`
 - 评估与视频导出脚本：`eval_policy.py`
 - 计划与实验记录：`PLAN.md`、`THOUGHTS_AND_EXP.md`
 
 当前还**没有**做的一件事是：
 
 - 在这个 README 里附带一组经过充分训练验证的推荐超参数
 
 也就是说，工程链路已经打通，但还需要继续做真实训练与调参。
 
 ## 相关文件
 
 - `bootstrip.md`：任务目标与约束
 - `PLAN.md`：实现计划
 - `THOUGHTS_AND_EXP.md`：实验记录，只追加
 - `env_wrapper.py`：单智能体 attacker 训练环境
 - `opponents.py`：对手策略定义与工厂
 - `reward.py`：reward shaping
 - `train_sb3.py`：SB3 PPO 训练入口
 - `policy_adapter.py`：把 PPO checkpoint 适配成 `BaseCombatPolicy`
 - `eval_policy.py`：评估与视频导出入口
 
 ## 环境准备
 
 在 `things/combatbench/` 目录下，至少需要这些依赖：
 
 ```bash
 pip install mujoco gymnasium numpy opencv-python imageio egl stable-baselines3 torch scipy
 ```
 
 如果你的环境已经能运行 `tools/run_round.py`，通常只需要确认 `stable-baselines3` 已安装即可。
 
 ## 基础 smoke test
 
 在项目根目录 `things/combatbench/` 下，可以先验证 nonfall 模式与视频链路：
 
 ```bash
 python3 tools/run_round.py \
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
 - 作为训练前的基础 smoke test
 
 ## 启动训练
 
 下面是一条最小训练命令：
 
 ```bash
 python3 baseline/mujoco21dof_nonfall/train_sb3.py \
   --run-name ppo_attacker_smoke \
   --total-timesteps 200000 \
   --match-duration 10 \
   --control-frequency 20 \
   --opponent standing \
   --initial-distance 2.0
 ```
 
 训练输出会默认写到：
 
 - `baseline/mujoco21dof_nonfall/runs/<run-name>_<timestamp>/run_config.json`
 - `baseline/mujoco21dof_nonfall/runs/<run-name>_<timestamp>/checkpoints/`
 - `baseline/mujoco21dof_nonfall/runs/<run-name>_<timestamp>/best_model/`
 - `baseline/mujoco21dof_nonfall/runs/<run-name>_<timestamp>/tensorboard/`
 - `baseline/mujoco21dof_nonfall/runs/<run-name>_<timestamp>/final_model.zip`
 
 常见训练参数：
 
 - `--opponent standing|random|active`
 - `--n-envs`
 - `--n-steps`
 - `--batch-size`
 - `--learning-rate`
 - `--checkpoint-freq`
 - `--eval-freq`
 - `--disable-non-fall-mode`
 
 ## 评估 checkpoint 并导出视频
 
 训练结束后，可以用 `eval_policy.py` 评估：
 
 ```bash
 python3 baseline/mujoco21dof_nonfall/eval_policy.py \
   --model-path baseline/mujoco21dof_nonfall/runs/<run>/final_model.zip \
   --opponent standing \
   --episodes 3 \
   --seed 0 \
   --non-fall-mode \
   --video outputs/nonfall_eval.mp4 \
   --summary-json outputs/nonfall_eval_summary.json
 ```
 
 这条命令会：
 
 - 加载 PPO checkpoint
 - 使用指定对手跑若干局
 - 第 1 局可选导出视频
 - 打印并可选写出 JSON summary
 
 ## 直接用 `run_round.py` 加载训练好的模型
 
 如果你想直接复用统一的 round runner CLI，也可以通过 `policy_adapter.py` 加载模型：
 
 ```bash
 python3 tools/run_round.py \
   --policy-a "combatbench.baseline.mujoco21dof_nonfall.policy_adapter.SB3PPOCombatPolicy?model_path=baseline/mujoco21dof_nonfall/runs/<run>/final_model.zip&deterministic=true" \
   --policy-b combatbench.policy.StandingCombatPolicy \
   --duration 10 \
   --non-fall-mode \
   --non-fall-pitch-limit-deg 15 \
   --non-fall-roll-limit-deg 10 \
   --video outputs/nonfall_roundrunner.mp4
 ```
 
 这种方式的优势是：
 
 - 直接复用现有统一工具
 - 更方便和其它策略做对战对比
 - 更接近后续对外演示和提交流程
 
 ## 实验记录约定
 
 所有实验思路、现象与结论统一追加到 `THOUGHTS_AND_EXP.md`。

 要求：

 - 只追加，不回写历史记录
 - 记录奖励修改点
 - 记录训练配置变化
 - 记录视频观察结论
 
 ## 当前建议的下一步
 
 当前工程已经具备训练和评估脚本，下一步最有价值的是：
 
 1. 跑一次小规模真实训练，确认 `runs/` 输出完整。
 2. 用 `eval_policy.py` 导出第一版视频。
 3. 根据视频现象继续调 reward 权重和 opponent 课程。
