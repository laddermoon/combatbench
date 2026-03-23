# Mujoco21dof Nonfall Baseline PLAN

## 1. 任务目标

基于 `things/combatbench/envs/combat_gym.py` 中已有的 `non_fall_mode` 能力，交付一套可以训练出“看起来在主动进攻和打斗”的 21DOF humanoid baseline，并满足以下目标：

1. 能训练出一个可用的 nonfall 进攻型策略。
2. 能稳定生成评估视频，证明策略会靠近、出招、造成伤害，而不是长期僵直或保持同一姿势。
3. 在实现过程中补齐当前仓库缺失的训练基础设施，并修复影响训练/评估闭环的关键问题。
4. 最终交付可以照着执行的训练/评估文档。
5. 按 `bootstrip.md` 的要求，新增 `THOUGHTS_AND_EXP.md`，并且只追加记录实验思路与结果。

## 2. 调研结论

### 2.1 当前已有的可复用能力

- **环境主实现已存在**
  - `envs/combat_gym.py` 中的 `CombatGymEnv` 已支持：
    - 双机器人对战环境
    - 每个机器人 `127` 维 observation
    - 每个机器人 `21` 维归一化 action
    - `non_fall_mode`、`non_fall_pitch_limit_deg`、`non_fall_roll_limit_deg`
    - 视频帧缓存与 `save_video()`
  - `non_fall_mode` 的实现方式是在每个 physics step 后对 root roll/pitch 做 clamp，再 `mj_forward()`。

- **控制接口比纯黑盒环境更强**
  - `CombatGymEnv` 目前已经暴露：
    - `set_controller_reference_positions()`
    - `set_controller_action_scale()`
    - `set_controller_gains()`
    - `set_robot_joint_positions()`
  - 这意味着 baseline 可以直接基于现有 residual-to-target-position + PD torque 控制链路来做，而不必重写底层控制器。

- **视频/评估基础链路已存在**
  - `envs/round_runner.py` + `tools/run_round.py` 已经形成完整 round 执行链路。
  - 当前可以直接复用其视频录制逻辑来做 baseline 验证与可视化交付。

- **策略接口已明确**
  - `policy/base.py` 定义了统一 `BaseCombatPolicy.act(obs, info)` 接口。
  - `policy/random.py` 和 `policy/standing.py` 可作为最小参考实现。

- **胜负与伤害逻辑已存在**
  - `core/scoring.py` 已基于碰撞 impulse 和命中部位实现 HP 下降与 KO/时间结束判定。
  - 这可以直接作为“进攻型奖励”的核心来源。

### 2.2 当前缺失或不一致的部分

- **训练 baseline 实际上还是空的**
  - `baseline/` 目录当前只有 `baseline/mujoco21dof_nonfall/`。
  - 该目录下只有：
    - `bootstrip.md`
    - 空的 `README.md`
  - 仓库里目前没有真正可运行的 PPO / SB3 / self-play 训练实现。

- **环境目前不能直接用于 RL 训练**
  - `CombatGymEnv.step()` 当前返回的 `reward` 固定为：
    - `{'robot_a': 0.0, 'robot_b': 0.0}`
  - 也就是说环境已经能模拟、计分、终止，但还没有训练所需的 reward wrapper / single-agent wrapper。

- **项目文档和代码里有过时引用**
  - `CLAUDE.md`、`README.md`、`README_zh.md` 仍提到 `baseline/sb3`、`baseline/selfplay_hp`。
  - `envs/round_runner.py` 的 `_configure_phase()` 也尝试导入不存在的 `combatbench.baseline.sb3.selfplay_env`。
  - 这说明现有文档与代码结构存在漂移，实施时要按“当前真实目录”而不是按旧文档假设推进。

- **Nonfall 参数链路没有完全接通**
  - `tools/run_round.py` 已解析：
    - `--non-fall-mode`
    - `--non-fall-pitch-limit-deg`
    - `--non-fall-roll-limit-deg`
  - 但 `RoundRunner` 当前只接收并传递了 `non_fall_mode`，没有把 pitch/roll limit 继续传到 `CombatGymEnv`。
  - 这是一个应优先修掉的工程缝隙。

- **实验记录文件还不存在**
  - `THOUGHTS_AND_EXP.md` 当前不存在，需要创建，并严格按 append-only 方式维护。

## 3. 推荐实现路线

### 3.1 总体路线

优先采用 **单智能体 attacker baseline + 固定/脚本化对手 + SB3 PPO** 的路线，而不是一开始就上完整 self-play。

原因：

1. `bootstrip.md` 明确允许“当前只考虑进攻不考虑防守，也就是只奖励进攻”。
2. 当前仓库没有任何现成训练框架；如果直接上 self-play，建设成本和调试成本都过高。
3. 当前任务的第一目标是“先跑出一个明显在打斗的 baseline 并出视频”，不是立即追求博弈最优。
4. SB3 PPO 是最短路径，适合先打通训练-评估-视频闭环。

### 3.2 训练对象定义

第一版 baseline 建议定义为：

- **学习者**：`robot_a`（attacker）
- **对手**：初期用 `StandingCombatPolicy` 或轻量脚本策略；后续按实验再升级到“主动但弱”的 scripted opponent
- **训练目标**：
  - 更快接近对手
  - 面向对手
  - 主动挥臂/躯干发力
  - 产生有效命中并造成 HP 下降
  - 避免长时间静止或重复同一动作

### 3.3 为什么不直接改 `CombatGymEnv` 本体的 reward

不建议直接把训练奖励硬编码进 `CombatGymEnv`：

1. 该环境目前更像“底层仿真环境”，适合保持通用。
2. baseline 需求明显带实验性质，奖励函数需要快速迭代。
3. 用 wrapper 方式更容易替换奖励版本、对手策略和 episode 配置。

因此更合理的结构是：

- `CombatGymEnv` 保持底层环境职责
- baseline 目录中新增单智能体训练 wrapper，负责：
  - 单侧 observation/action 暴露
  - 进攻型 reward 构造
  - 对手策略调度
  - 训练期统计信息汇总

## 4. 分阶段计划

## Phase 0：修复训练前的工程缝隙

### 目标

让 nonfall baseline 目录具备最基本的可开发状态，并消除明显的接口断裂。

### 工作项

1. 创建 `THOUGHTS_AND_EXP.md`，作为实验追加日志。
2. 修复 `tools/run_round.py -> RoundRunner -> CombatGymEnv` 的 nonfall pitch/roll limit 传参链路。
3. 处理 `RoundRunner._configure_phase()` 对不存在模块的陈旧引用：
   - 要么删除 phase 依赖
   - 要么降级为明确的 no-op，并给出清晰错误信息
4. 把 `baseline/mujoco21dof_nonfall/README.md` 变成真实可用文档，而不是空文件。

### 完成标准

- 使用 CLI 能显式控制 nonfall mode 和 pitch/roll 限制。
- 评估工具链不再依赖不存在的 baseline 模块。
- baseline 目录的文档骨架建立完成。

## Phase 1：建立单智能体训练包装层

### 目标

在不破坏底层环境的前提下，提供一个 SB3 可以直接消费的单智能体 Gym 环境。

### 工作项

1. 新增 baseline wrapper，建议职责包括：
   - 只暴露 `robot_a` 的 observation
   - 只接收 `robot_a` 的 action
   - 在内部调用对手策略产出 `robot_b` action
   - 输出标量 reward
   - 输出训练监控用 info
2. 明确 episode 配置：
   - 初始距离
   - match duration
   - control frequency
   - nonfall 参数
   - 是否固定 spawn facing
3. 设计对手策略接口：
   - 先支持 `standing`
   - 再支持轻量 scripted opponent（例如有限幅度挥臂/逼近）
4. 把训练期关键统计统一整理到 `info`：
   - damage dealt
   - damage received
   - distance to opponent
   - facing score
   - action magnitude / action delta
   - hit count / successful contact count

### 完成标准

- baseline wrapper 可通过 `reset()/step()` 稳定运行。
- SB3 可以直接接入，不需要再适配 Dict 双智能体接口。
- `info` 中能拿到足够做 reward 分析和 debug 的字段。

## Phase 2：实现进攻型奖励函数

### 目标

构建符合 `bootstrip.md` 目标的奖励，使学习者明显更倾向于“主动攻击”，并避免僵直。

### 奖励设计原则

- **主奖励只围绕进攻展开**
- **防守不是主目标**
- **动作活跃度必须显式建模**
- **奖励项要尽量可解释和可 ablation**

### 建议的奖励组成

1. **造成伤害奖励（主项）**
   - 对方 HP 下降 / 造成有效命中时给正奖励
   - 这是最核心的训练信号

2. **接近奖励**
   - 鼓励缩短与对手的水平距离
   - 防止学成“原地挥手”

3. **朝向奖励**
   - 鼓励 torso/forward vector 面向对手
   - 提高接触概率

4. **出招活跃度奖励**
   - 奖励动作幅值、动作变化量、关键关节（如肩、肘、髋、躯干）活跃
   - 解决“长时间固定姿态”的问题

5. **命中事件奖励**
   - 对有意义的 contact / hit event 给小额稀疏奖励
   - 帮助在纯 HP reward 稀疏时稳定学习

### 建议避免的设计

- 第一版不要把“防守/少挨打”设为主要负奖励，否则会和“主动进攻”目标冲突。
- 第一版不要引入过重的姿态稳定惩罚，否则会把 nonfall mode 的优势抵消掉。

### 完成标准

- 随机初始化训练中，reward 曲线能区分“无效站桩”和“主动接近/出拳”。
- 视频中能观察到明显比随机/站立更主动的动作模式。

## Phase 3：实现 SB3 训练脚本与配置

### 目标

把 baseline 变成一个可以直接启动、保存 checkpoint、恢复训练、记录日志的训练工程。

### 工作项

1. 使用 SB3 PPO 搭建训练脚本。
2. 加入基础配置项：
   - total timesteps
   - num envs / vec env
   - learning rate
   - rollout length
   - batch size
   - gamma / gae / clip range
   - seed
   - device
3. 加入训练产物输出：
   - checkpoint
   - best model
   - tensorboard log
   - config snapshot
4. 对接 baseline wrapper。
5. 提供至少一组默认可跑配置，优先追求：
   - 能起训练
   - 能看到策略逐渐接近和出手
   - 不是一开始就追求最终最优超参

### 完成标准

- 一条命令能启动训练。
- 训练过程中能定期保存 checkpoint。
- 训练结束后能拿 checkpoint 做评估和视频导出。

## Phase 4：实现评估、视频与策略适配

### 目标

把训练得到的模型接入现有 round runner / CLI，形成完整可视化验收链路。

### 工作项

1. 实现模型到 `BaseCombatPolicy` 的适配层。
2. 提供评估脚本：
   - 加载 checkpoint
   - 指定对手
   - 跑固定 seed 场景
   - 输出结果统计
   - 生成 mp4 视频
3. 提供至少两类评估：
   - 对 `StandingCombatPolicy`
   - 对轻量 active/scripted opponent
4. 如有必要，增加 deterministic / stochastic 推理开关。

### 完成标准

- 可以用训练出来的 checkpoint 直接运行 `round runner` 风格评估。
- 能稳定输出 mp4。
- 视频中能清楚看到策略存在进攻意图，而不是仅仅保持平衡或抖动。

## Phase 5：文档与实验沉淀

### 目标

保证这个 baseline 对后续复现和继续迭代是友好的，而不是一次性代码。

### 工作项

1. 把 `baseline/mujoco21dof_nonfall/README.md` 写成可操作文档，至少包含：
   - 环境准备
   - 训练命令
   - 评估命令
   - 视频生成命令
   - 常见问题
2. 创建并持续追加 `THOUGHTS_AND_EXP.md`：
   - 每次实验的想法
   - 奖励修改点
   - 成功与失败现象
   - 下一步判断
3. 如发现主仓库文档中关于 baseline 结构的旧描述会误导使用者，再决定是否同步修正文档。

### 完成标准

- 新人只看 README 就能启动训练和评估。
- 实验记录可追溯奖励设计和关键决策。

## 5. 建议文件布局

下面是推荐新增/修改的最小文件集合（以最小可用 baseline 为目标）：

- `baseline/mujoco21dof_nonfall/PLAN.md`
- `baseline/mujoco21dof_nonfall/THOUGHTS_AND_EXP.md`
- `baseline/mujoco21dof_nonfall/README.md`
- `baseline/mujoco21dof_nonfall/env_wrapper.py`
- `baseline/mujoco21dof_nonfall/reward.py`
- `baseline/mujoco21dof_nonfall/opponents.py`
- `baseline/mujoco21dof_nonfall/train_sb3.py`
- `baseline/mujoco21dof_nonfall/eval_policy.py`
- `baseline/mujoco21dof_nonfall/policy_adapter.py`

如果实现过程中发现文件太碎，也可以合并成更少的文件，但职责边界最好仍保持清晰。

## 6. 风险与注意事项

### 风险 1：仅靠 HP 下降信号太稀疏

- **现象**：训练长时间没有有效梯度，策略学成站桩或轻微抖动。
- **应对**：加入接近、朝向、命中事件、动作活跃度等辅助 shaping。

### 风险 2：对手太弱导致策略学到无意义 exploit

- **现象**：只对站桩对手有效，换一个有动作的对手就失效。
- **应对**：先打通 standing，对稳定后再增加 scripted active opponent 作为第二阶段课程。

### 风险 3：nonfall 掩盖姿态问题，导致动作不自然

- **现象**：策略依赖 root 姿态夹紧，产生奇怪但有效的攻击动作。
- **应对**：在 reward 和评估中关注动作自然性与视频效果，不只看伤害。

### 风险 4：现有文档/代码的旧引用继续误导后续开发

- **现象**：新代码又去兼容不存在的 `baseline.sb3` 结构。
- **应对**：以当前真实目录为准，必要时同步修正文档。

## 7. 首个可交付版本的验收标准

第一阶段不追求“强策略”，而追求“可用 baseline”。验收标准建议定义为：

1. 能启动训练并稳定跑完一段 timesteps。
2. 能保存 checkpoint 并恢复。
3. 能生成评估视频。
4. 视频中能观察到：
   - 机器人主动接近对手
   - 存在明显出招动作
   - 有一定概率命中并造成伤害
   - 行为显著优于 `StandingCombatPolicy` 和简单随机动作
5. 文档足够让后续人按步骤复现。

## 8. 建议的执行顺序

实际开始编码时，建议严格按以下顺序推进：

1. 先修 nonfall 参数链路和陈旧引用。
2. 再创建 `THOUGHTS_AND_EXP.md` 与完善 README 骨架。
3. 然后实现单智能体 wrapper。
4. 再实现 reward 与 scripted opponent。
5. 再接 SB3 PPO 训练脚本。
6. 最后做评估、视频导出和文档收口。

这个顺序的核心原因是：**先把训练闭环搭起来，再做超参和奖励迭代**，避免一开始就陷入策略细节调参而没有稳定的基础设施。
