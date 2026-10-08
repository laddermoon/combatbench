# SAC 独立训练框架 Roadmap

> 状态：总体路线图与阶段一结果已获用户批准；W1–W8/A1–A8 已完成，阶段一收口。未开始本轮算法实现。Shannon 熵 SAC 保留为基线，用户已批准 uncertainty 为独立替代路线，见 DECISIONS.md 的 A4。
> 本文顶部为本轮有效路线图；下方「历史参考区」完整保留旧规划，不构成本轮约束。
> 原始需求：[bootstrip.md](bootstrip.md)。阶段一详细计划：[DECISIONS.md](DECISIONS.md)。

## 本轮目标与成功标准

建设独立、SAC 原生、可验证、可诊断的训练体系，达到现有 PPO 框架的工程深度，而不是修补旧 SAC 到能跑，也不是将 PPO 换一个 loss。

最终必须同时交付：

1. 在 `baseline/experiments_sac/` 实现与 PPO `exp_standup.py`、`exp_basic_balance.py` 任务语义同等的两个实验，并稳定训练成功。
2. SAC 自有 debug 系统能够展示整体进度、关键数据，并沿真实训练计算链下钻。
3. 两个实验支持替换策略；重点保留并适配 TruncNorm 的八格策略体系。
4. 实验具有标准化探索旋钮与 SAC 原生优化手段，可调度、可观测、可恢复。

以上不是以接口数、页面数或 smoke 通过替代的验收。SAC 不预设必须比 PPO 更快或更好。

## 设计边界

- PPO 与 SAC 算法框架保持独立，不建设通用 PPO/SAC 大框架。需要借用的策略、算法接口、诊断工具采用复制后适配，不反向改造 PPO 来迁就 SAC。
- 使用现有 `EnvRuntime`、Humanoid21 和正式环境数据契约，不重造环境。采集层、CLI、策略导出等必须审查直接及传递依赖，不能从共享工具隐式引入 PPO 内部契约。
- 旧 `framework/sac`、`experiments_sac`、历史决策和历史训练只作为参考；逐项核实后决定借用、改写或不采用。
- 保留 TruncNorm 八格的网络与分布设计：单分量/混合、shared/state σ、bounded/unbounded σ；SAC 专属梯度路径另行验证。
- 先建立可退化到标准 SAC 的正确性基线，再引入多通道及优化机制。多 critic 不等于照搬 PPO 的 advantage 归一化与合并。
- Shannon 熵 SAC 保留为基线；uncertainty 为用户批准的独立替代路线，明确区分 U-bonus 与 U-floor，actor/target/系数控制一致改写，不默认叠加。先同策略做对照，再判断是否值得推广，不把已有 PPO 的 U 测试当 SAC 收敛证据（详见 A4）。
- 实验拥有任务语义，框架拥有学习与运行机制。相同任务不要求相同学习超参数；不得为了收敛偷偷更换重置分布、奖励事实或评估任务。
- 诊断从第一条训练闭环开始；完整界面可以后置，数据来源与关键中间量不能后置。
- PER、REDQ、复杂 n-step、自动 relabel、异步采集、buffer reset 等不作为默认功能清单，按任务证据决定是否引入。
- 坚持 fail loud：缺失必需数据、接口不符、未支持配置必须显式失败；不靠零回退、吞异常或截断掩盖问题。

## 阶段总览

| 阶段 | 主要目标 | 出口 | 当前状态 |
|---|---|---|---|
| 1 | 设计边界与验收口径 | 经审阅的数学、接口、数据与验收契约 | 完成；W1–W8/A1–A8 已收口并获用户批准 |
| 2 | 最小可信 SAC 闭环 | 单通道正确性、基础诊断、恢复与环境 smoke | 完成；`G2.0`～`G2.6` 已通过，fake env、Humanoid21 两任务 smoke、完整 resume 与 `sac_dump_v1` recompute 均有证据 |
| 3 | 多 critic 与两个目标实验 | 任务语义对齐、多通道测试、持续训练链路 | 完成；`P3-AUDIT-0`～`P3-RUN-1` 与 `G3.6` 已通过，`sac_transition_v2/replay_v2/dump_v2`、A3 共同 twin-pair、两实验多 round 训练/resume、v2 dump recompute 均有证据 |
| 4 | 八格策略与探索/优化控制 | 全策略可替换，控制含义明确且经过验证 | 未开始 |
| 5 | SAC 完整 debug | run 到更新、样本及来源帧的可核对分析链 | 未开始 |
| 6 | 两任务稳定收敛 | 多 seed、独立评估、策略产物与替代策略验证 | 未开始 |
| 7 | 工程收口 | 可重现、完整恢复、独立性与长期运行验收 | 未开始 |

## 阶段 1：确定设计边界与验收口径

**工作：** 固定两个任务的语义与评估协议；设计实验、策略、transition/replay 契约；确定通道 Q、熵项、actor 权重与 critic 掩码语义；区分行为策略/训练策略/评估策略；定义环境步、agent transition、梯度步和 UTD；裁决旧代码的利用范围。

**成功标准：**

- 核心数据流、目标函数和所有必需字段有明确含义，单通道可回到标准 SAC。
- 多通道、混合策略、探索与 replay 的高风险问题有明确验证方案及支持边界。
- 训练成功门槛、评估 seeds、预算和持续达标条件在正式训练前冻结。
- 未决项显式标注阻塞范围；不要求此阶段解决所有性能优化细节。
- 阶段结果经用户审阅后，才进入阶段二实现。

## 阶段 2：建立最小可信 SAC 闭环

**工作：** 先适配一个单分量 TruncNorm；实现基础 replay、双 Q、target Q、actor、固定/自动温度；接通采集、训练、评估、导出与恢复；从首版记录 Q、TD error、熵、α、梯度、replay 和实际更新量。

**成功标准：**

- 可控小问题上的 Bellman target、actor 梯度方向和温度调整方向通过验证。
- 截断分布在边界与极端参数下数值稳定，动作采样与评分一致。
- replay 环形覆盖、真正终止/timeout、next observation 和双 agent 边界通过测试。
- Humanoid21 端到端 smoke 通过，每次更新的数据来源可解释。
- 完整续训与仅加载模型的 warm-start 明确区分；恢复契约有测试，不能将清空 replay 称为等价续训。

此阶段不要求机器人已学会站起。

## 阶段 3：完成多 critic 与两个目标实验

**工作：** 实现经裁决的多通道学习；接入 SAC standup/basic_balance；对齐双 agent 边界、奖励、权重与评估；处理历史事实与当前训练控制参数的版本语义；补齐各通道 Q/target/TD/actor 贡献诊断。

**成功标准：**

- 同批 episode 的任务事实、奖励与终止边界按约定对拍；不复制旧实现中的静默回退。
- 单通道退化、多通道屏蔽、actor 权重为零等行为有测试；零 actor 权重不意外关闭有效 critic 学习。
- 两个实验能够持续训练、评估、保存和恢复，样本来源可追踪。
- 不将 loss 有限或下降当作任务成功。

## 阶段 4：完成八格策略与标准化探索/优化控制

**工作：** 将八格复制适配到 SAC 自有策略体系；实现并验证 A4 的 mixture 分量枚举梯度与稳定 TN 数值内核；分离采集探索、π 正则与优化控制。保留 Shannon 基线，独立验证用户批准的 U-bonus/U-floor 路线；原生 peak/L2 度量显式标记，先固定同一策略做正则对照。reference/delta 探索暂不引入，不把 uncertainty 与自动温度默认叠加。

**成功标准：**

- 八格均通过分布、梯度、导出一致性、私有 RNG 重放及短训验收。
- 均值、σ、混合权重都能得到正确学习信号，不能只看梯度非零。
- 两实验通过配置换策略，无需修改 trainer。
- 旋钮的定义、范围、调度时钟、默认行为和跨策略差异明确，实际生效值可观测、可恢复。
- 不把策略 uncertainty 等同于 Shannon 熵；关键控制手段有对照验证。

八格全部支持不等于立即执行八格 × 两任务 × 多 seed 的完整收敛矩阵；训练级覆盖由阶段六分层安排。

## 阶段 5：建成 SAC 原生完整 debug 系统

**分析链：** episode → transition → replay 写入/保留/抽样 → Bellman target → 各通道 Q → actor 的 Q 与熵梯度 → 温度/target 更新 → 策略变化与真实评估。

**工作：** run 总览、指标目录、多 run 比较；预约/按需截面捕获；replay 年龄、来源、行为策略版本与复用诊断；样本级 target/TD/梯度分析；策略与 Q 的更新前后比较；CLI/HTTP/viewer 共用 SAC 内部同一分析实现。

**成功标准：**

- 异常曲线可以追到具体更新、稳定样本 ID 及 episode/agent/frame；不能将 replay 槽位当永久身份。
- 截面足以离线核对指定更新的关键计算，缺失字段显式不可用。
- 可定位注入的典型故障：终止掩码错误、旧数据过多、温度失控、通道压制、Q 异常等。
- 开关诊断不改变训练 RNG 与更新结果，常规及重型诊断成本可测。
- 达到 PPO debug 的分析深度，不照搬 GAE、ratio、clip 等不属于 SAC 的页面。

## 阶段 6：诊断驱动两任务稳定收敛

**工作：** 按假设 → 证据 → 单变量调整 → 复验推进；建立可重复基线，再验证替代策略；分别记录样本效率、计算成本、稳定性和策略质量；观察双 agent replay 中的对手分布变化。新增优化机制须有独立证据与消融。

**验收方向（数值协议在阶段一冻结）：**

- 两任务各至少三个训练 seed，在独立评估 seeds 上持续达标，而非只选最佳一次。
- standup 保留原 `success` 定义，并检查最终站立质量，排除短暂站起即倒下。
- basic_balance 以存活率为主，检查双通道行为与视频，不偷换成 step 的完整迈步任务。
- 成功率/存活率 ≥90% 是候选门槛，结合 PPO 参考表现确认；不是已验证结论，不得训练后降低门槛。
- 默认策略完成多 seed 收敛；至少一种有实质结构差异的替代策略在两个任务完成训练级验证，具体 seed 数与预算提前确定。
- 交付可加载、部署、评估的策略，提供完整训练与失败记录，不只交日志或挑选成功 seed。

## 阶段 7：工程收口与整体回归

**工作：** 核查依赖独立性、配置有效性、测试和错误处理；验证 replay、优化器、target、温度、全套 RNG、实验/调度状态恢复；明确 checkpoint/导出/debug 工件版本；验证长期运行、资源释放与产物体积；完善接入和使用规范。

**成功标准：**

- 固定命令、配置与代码版本可重现两个验收实验。
- 在声明的确定性条件下，连续训练与分段恢复通过一致性验证。
- SAC 不依赖 PPO 算法内部模块，PPO 原路径无行为回归。
- 不支持的功能明确拒绝，无接受配置却不生效的路径。
- 四项用户成功标准逐条关联测试、run、评估及策略产物。

## 推进和变更规则

- 每阶段开始只细化该阶段的执行计划；结束汇报已验证事实、未决问题和路线调整。
- 阶段二开始诊断，阶段三开始真实任务反馈；阶段五是补齐深度而非从零建设 debug。
- 后续证据推翻前置假设时，返回对应阶段修正，不以无限调参或放宽验收替代修复。
- 路线批准不代表旧 SAC 决策重新生效，也不代表本轮任何阶段已经实现。
- 历史资料中的性能倍数、梯度占比、探索解释等结论，必须重新核对条件和证据。

---

# 历史参考区：旧 SAC V2 规划（不作为本轮决策）

以下保留旧文全文。旧文的「一期/二期」「Phase」与上方七阶段无对应关系；旧任务、共享策略方案和优化优先级不自动继承。

# SAC V2 框架规划（初版）

> 状态：规划阶段，未开始实现。
> 定位：不是"与 PPO V2 共用实验的 SAC"，而是"最能发挥和优化 SAC 能力的独立框架"。
> 日期：2026-08-27

---

## 0. 设计取向

借鉴 PPO V2 的**设计深度和取向**，但不追求接口对齐、不追求实验共用。

PPO V2 的最大杠杆是**优势组合**（framework 唯一真正做决定的地方是 `combined_adv`）。
SAC 的最大杠杆不在那里 —— **SAC 的最大杠杆是 replay 分布本身**。

PPO 的训练分布被算法钉死（=当前策略），实验只能通过 reward 和 `actor_weight` 施加影响。
SAC 的训练分布是一个**可以被设计的对象**：什么数据进来、保留多久、怎么分层、怎么采样、
能不能重标注 —— 这些全都是自由度，而且每一个都比 reward shaping 更有力。

因此核心抽象不是 `build_trajectories → 一次性 update`，而是：

```
多源数据摄入  →  带标签的、可分层的、可重标注的 Replay  →  按通道定制的采样  →  高 UTD 的悲观更新
```

`build_trajectories` 在这个架构里退化成"其中一个数据源的适配器"，而不是唯一入口。

---

## 1. SAC 天然能做、PPO 结构上做不到的六件事

按对当前实验的实际价值排序。

### 1.1 对手/脚本策略的数据是免费的，而现在被全部丢掉了

`follow_v2` 和 `fight` 用 `agent_used="random"`，`build_trajectories` 只为
`episode_options["agent_id"]` 那一个 agent 建轨迹 —— 另一个机器人 600 步 × 1024 episode
的数据被直接扔了。

对 SAC 这些数据不是"别的策略的数据"，是**同一个 MDP、同一组 reward channel、
来自一个通常比当前学习者更强的策略**的 off-policy 数据。价值高于自己采的数据：

- 数量直接 ×2
- 覆盖学习者还到不了的状态区域
- 对手池里的策略是冻结的历史最优，本质是准专家数据

`fight` 里 `r_damage_dealt` 这类通道，学习者早期几乎产生不了正样本；
而对手的视角里全是。这是 PPO 结构上拿不到的东西（on-policy 要求数据来自当前策略）。

**投入产出比最高的单个特性。**

### 1.2 分层 replay 可以替代 `RandomFallenStatePlugin` 这类环境侧 hack

`RandomFallenStatePlugin` 存在的唯一原因是：PPO 一旦学会站立，就再也采不到倒地状态，
于是遗忘起立能力 —— 所以必须由环境强行注入倒地初始态。

SAC 不需要环境配合：让 buffer 保证 STANDUP 相位的 transition 占比不低于某个下限即可。
数据留在 buffer 里，不需要环境重新生成。

后果：
- 起立/平衡/跟随/打击可以用**自然的状态分布**训练，而不是人为拼接的重置分布
  （后者本身是 sim2real 和策略连贯性的隐患）
- `standup_step_v3` 那套 plateau 检测 + 相位硬切换的复杂度，一部分可以从
  "per-frame actor_weight 门控"下沉成"buffer 分层" —— 更简单也更直接

### 1.3 `∂Q/∂a` 可测量 —— 让 `actor_weight` 从"猜"变成"闭环"

整个规划里最重要的技术点，也是对 PPO 框架那个未归一化 `combined_adv` 问题的根本性升级。

PPO 只能拿到标量优势，所以只能 z-score 它的值域；但真正决定策略更新的是**动作梯度**，
两者尺度不成比例。SAC 里 `Q_c(s,a)` 对 `a` 可微，于是可以直接测量每个通道对策略梯度的
实际贡献：

```python
g_c = ∂Q_c(s, a_π) / ∂a
ŝ_c = running_RMS(‖g_c‖)              # 每通道动作梯度尺度
actor_loss = α·logπ − Σ_c  w_c(s) · Q_c(s, a_π) / ŝ_c
```

于是 `w_c` **字面意义上就是该通道在策略梯度中的占比**。`aw=3.0` vs `1.0` 精确等于 3:1
的影响力，可测量、可验证、跨实验可迁移。再把 `Σ_c w_c` 归一化到常数，**学习率就和
"这个实验有几个通道、门控开了多少"彻底解耦**。

工程上不贵：`ŝ_c` 只是个标量统计量，每 K 步在子样本上用一次 `autograd.grad` 估计即可，
其余步用 running 值。

**附带产出一个 PPO 永远给不了的诊断**：每个通道**实际实现的**策略梯度占比。
日志里能直接打印 `r_fall: 41% | r_face: 2.3% | r_damage_dealt: 0.4%`。

### 1.4 每个通道可以有自己的采样分布

通道之间的活跃区域差异极大：`r_damage_*` 只在 `dist ≤ 0.9m` 有意义，
`r_potential` 只在倒地时有意义。PPO 只有一个 batch，所以稀疏通道的 Q 被 20:1 地稀释
在无关状态上。

SAC 里每个通道的 Q 是独立的学习问题，**可以从各自关心的状态子集采样**。
`r_damage_dealt` 的 Q 就在打击距离内的 transition 上训练，样本效率提升一个量级。

### 1.5 `n_step` 是 SAC 版的 `gae_lambda` —— 而且 per-channel 更有价值

SAC 默认 1-step TD，偏差小方差小但信息传播慢。因为从 trajectory 摄入数据，可以连续存储
并计算 n-step 目标。通道配置自然变成：

```python
SACRewardChannel(name="r_damage_dealt", gamma=0.90, n_step=10, n_critics=5, in_target_min=2)
SACRewardChannel(name="r_left_foot",    gamma=0.90, n_step=1,  n_critics=2, in_target_min=2)
```

稀疏的伤害奖励要大 n（快速传播）+ 强悲观（防高估）；密集的足高 shaping 要 n=1 + 弱悲观。
**per-channel 的偏差-方差-悲观三元组**，比 PPO 的 per-channel λ 表达力更强，
因为它同时控制了 off-policy 特有的高估问题。

### 1.6 可以从 buffer 里的状态重置环境

`IDataMutator.set_core_state()` 已经存在。稀疏存储 transition 的 `core_state`
（qpos/qvel，每 k 帧一个），就能把 episode 重置到 buffer 里的任意历史状态。

应用场景：`fight` 里"即将被击中"、"在打击距离内失去平衡"这类关键状态极其罕见，
靠 rollout 从 2m 外开局碰运气到达效率极低。直接从 buffer 里重置到这些状态附近，
是数量级的效率差异。PPO 也能用这招，但 PPO 没有 buffer 来提供状态源。

---

## 2. 实验契约（`ExperimentSAC`，不与 `ExperimentV2` 共享）

差异不是"多了个 `sac_params`"，而是**多了一整层数据分布控制**：

```python
class ExperimentSAC(ABC):

    # ---- 配置 ----
    def reward_channels() -> Tuple[SACRewardChannel, ...]
        # name, gamma, n_step, n_critics, in_target_min,
        # actor_weight_share(是否参与梯度占比归一化)

    def sac_params() -> SACParams
        # utd_ratio, batch_size, warmup_steps, tau,
        # target_entropy(可为 schedule 或 per-tag),
        # q_arch(trunk 分组策略 / dropout / layernorm)

    def common_params() -> CommonParams    # 复用，但语义改为按 env_step 计数

    # ---- 数据摄入（新增的核心层）----
    @abstractmethod
    def data_sources() -> Tuple[DataSource, ...]
        """声明所有数据来源，而非只有"当前策略的 rollout"。
        - SelfRollout(agent="learner")
        - SelfRollout(agent="opponent")      ← 白捡的 2x
        - PoolRollout(pool_config)            ← 对手池自对弈
        - ScriptedRollout(policy_bp)          ← StandingPolicy / RandomMove
        - RecordedEpisodes(path)              ← episode_recorder 的产物
        每个 source 带 sampling_share，控制它在 buffer 中的目标占比。
        """

    @abstractmethod
    def build_slices(episodes, source) -> List[TrajectorySlice]
        """≈ 原 build_trajectories，但每条 slice 额外携带：
        - reward_features: Dict[str, np.ndarray]   ← 重标注的原料（见下）
        - tags: Dict[str, np.ndarray]              ← 分层/采样/诊断的依据
        - core_states: Optional[...]               ← buffer-based reset 用
        """

    def relabel(features, tags, ctx) -> (rewards, actor_weights)
        """可选。从存储的原始特征重新计算 reward 和 actor_weight。
        课程推进 / 系数调整时，整个 buffer 立刻与新定义一致 ——
        这是 actor_weight 陈旧性问题的正解。
        """

    def replay_plan() -> ReplayPlan
        """声明分层保留与采样策略：
        - strata: 按 tag 定义的分层 + 每层容量下限/上限
        - per_channel_sampling: 每通道的 tag 过滤器或优先级
        - freshness: 新旧数据的采样偏好
        """

    @abstractmethod
    def on_eval(episodes, step) -> Dict   # 语义不变，复用
```

### 关键取舍说明

- **`tags` 是最便宜、最通用的新抽象。** 一个 per-transition 的标签字典
  （`phase`、`in_strike_range`、`level`、`source`、`fell_within_20`），
  同时驱动分层保留、分层采样、per-channel 采样、per-tag 诊断四件事。
  实验侧写它的成本几乎为零（相位 mask 本来就在算），收益覆盖了原方案里
  D1/D7/D8 三个未决问题。

- **`reward_features` + `relabel` 取代"冻结 actor_weight"。** 存原料而不是存成品。
  `follow_v2` 的 13 级课程升级时，buffer 不需要清空，也不需要接受陈旧权重 ——
  直接按新课程重标注。代价是内存（多存几个标量数组）和一次重标注的计算，都很便宜。
  **让"课程学习 + off-policy"从冲突变成协同。**

- **`data_sources` 让"用什么数据训练"成为一等公民**，而不是隐含在 `build_jobs` 里。

---

## 3. 功能分层与核心应用场景

### 第 1 层：SAC 内核（`baseline/framework/sac/`，独立 package）

| 特性 | 核心应用场景 |
|---|---|
| **`TaggedReplay`** —— 轨迹连续存储（支持 n-step）、per-channel reward/done、tags、reward_features、可选 core_state | 全部上层能力的载体。轨迹连续性是 per-channel n_step 的前提 |
| **分层保留（stratified retention）** | 学会站立后仍保有 STANDUP 数据 → 不遗忘起立；替代 `RandomFallenStatePlugin` 的机制 |
| **per-channel 采样器** | `r_damage_*` 只在打击距离内的 transition 上训练 Q，稀疏通道样本效率 ×10 量级 |
| **`relabel` 通道** | 课程推进 / reward 系数调整后 buffer 立刻自洽，无需清空重 warmup |
| **动作梯度归一化的 actor loss** | `actor_weight` 成为可测量的梯度占比；LR 与通道数解耦；输出"实际梯度占比"诊断 |
| **per-channel 悲观配置**（n_critics / in-target min / LayerNorm+Dropout(DroQ)） | 稀疏通道（damage）配强悲观防高估，密集通道（foot）配弱悲观省算力。SAC 独有的、与 reward 稀疏性直接对应的旋钮 |
| **按 γ/n_step 分组的 Q trunk + 多头** | `fight` 9 通道从 36 网络降到 ~4 trunk × 9 head。分组依据是"时间感受野"，语义上正确 |
| **auto-α，支持 per-tag / schedule 的 target_entropy** | 起立相位需要大探索，打击相位需要精确动作。PPO 的 `entropy_coef` 是全局标量，做不到这个 |

### 第 2 层：训练循环（`sac_loop.py`）

| 特性 | 核心应用场景 |
|---|---|
| **异步采集 + 持续梯度** —— 采集 worker 常驻，主进程按 UTD 持续更新 | off-policy 的吞吐红利。小批量采集会放大"导出 policy + 重启 worker"的固定开销，异步化正好摊薄 |
| **以 env_step 为主时钟**（而非 update 计数） | eval / checkpoint / 课程推进 / α schedule 的节奏必须锚定在环境交互量上，否则和 PPO 的日志无法比较、UTD 一改全乱 |
| **多源采集调度** | 按 `data_sources` 的 share 分配采集预算（自己 / 对手池 / 脚本策略） |
| **buffer-based reset（二期）** | 定向攻克 `fight` 的罕见关键状态 |
| **发散护栏** —— Q 幅度、TD error、target-online 偏离、α 触底的自动检测与早停 | SAC 在 21-DoF + 9 通道 shaped reward 上高估发散是高频失败模式；静默跑几天的成本不可接受 |

### 第 3 层：可观测性（与内核同等优先级）

SAC 的失败模式比 PPO 隐蔽得多，且这套框架引入了不少新自由度。日志必须能直接回答：

| 诊断 | 回答的问题 |
|---|---|
| **per-channel 实际策略梯度占比** | 我设的 `actor_weight` 真的生效了吗？哪个通道在实际主导策略？ |
| **per-channel Q / TD error / target 偏离 / 高估指标** | 哪个通道的 Q 在发散？悲观配置够不够？ |
| **buffer 组成**：per-tag 占比、per-source 占比、数据年龄分布、策略陈旧度 | 训练分布是我设计的那个吗？STANDUP 数据是不是已经被挤空了？ |
| **per-tag 的 Q / TD / 梯度占比** | 相位切换处是不是有断层？打击距离内的 Q 是不是根本没学？ |
| **α / entropy / target_entropy 三条线** | 探索是不是崩了？是被 `log_std_min` 夹住还是 α 自己降的？ |

---

## 4. 配套的实验设计（针对 SAC，不复用 V2 实验）

不移植现有实验，而是设计三个**专门用来兑现上述能力**的实验，构成一条验证链：

### `sac_balance`（2 通道，200 步）— 验证内核

最小配置。目标不是打过 PPO，而是验证：
- 动作梯度归一化是否让 `w=3:1` 真的实现 3:1 占比
- UTD 能推到多高不发散
- 异步采集的吞吐比

这是所有后续结论的地基。

### `sac_standup_recover`（4 通道，起立+平衡+踏步）— 验证分层 replay

核心设计：**移除 `RandomFallenStatePlugin`**，改用分层保留保证 STANDUP transition 占比 ≥ 20%。
这是一个干净的对照实验，直接回答"buffer 分层能不能替代环境侧的重置分布 hack"。

如果成立，这条结论对后面所有实验（包括未来的 PPO 实验）都有价值；
如果不成立，也是一个明确的负结果。

副产物：`standup_step_v3` 那套 plateau 检测 + 硬相位切换有多少可以被 tag 分层替代。

### `sac_fight`（多通道 + 自博弈）— 验证多源数据与定向重置

核心设计：
- 同时摄入学习者与对手双方视角
- 对手池自对弈数据
- 稀疏 damage 通道配 per-channel 采样 + 强悲观 + 大 n_step
- 二期加入 buffer-based reset 到打击距离内的关键状态

最能体现"SAC-native"价值的一个，但也是风险最高的（自博弈非平稳 + replay 是最坏组合）——
所以必须排在验证链最后。

---

## 5. 需要深入考虑并决策的问题

原方案里的 D1（权重陈旧）、D2（尺度均衡）、D8（门控帧稀释）已经被上面的设计回答掉了
（分别是 `relabel`、动作梯度归一化、per-channel 采样）。剩下的是新的、更本质的问题：

### 阻塞性（必须先定，直接决定代码结构）

**N1. 内存预算，进而决定 buffer 能存什么。**

`fight` 一条 transition 要存：obs(96) + action(21) + 9 通道 reward/done/aw + tags +
reward_features + 可选 core_state。按 100 万 transition 估算，不含 core_state 约 1.5~2 GB，
含 core_state（qpos+qvel ≈ 60~80 维 float32，每 4 帧存一个）再加 ~300 MB。
机器有 1TB 内存所以物理上不是问题，但它决定了：buffer 容量上限、是否落盘、
`relabel` 是全量重扫还是采样时惰性计算。**这个数字定下来，`TaggedReplay` 的结构才能定。**

**N2. `relabel` 是全量批处理还是采样时惰性计算？**

- 全量：课程推进时扫一遍全 buffer，之后采样零开销，但一次几秒到几十秒的停顿。
- 惰性：采样时对 batch 重算，开销分摊但每步都有，且要求 relabel 是纯函数
  （相位 mask 的滑窗依赖历史 → 需要把 mask 本身作为 feature 存下来，而不是重算）。

倾向全量 + 版本号标记，但这直接影响 `relabel` 的接口形状。

**N3. Q 网络的 trunk 分组：按 γ/n_step 自动分组，还是让实验显式声明？**

自动分组更省心，但实验可能有语义上的分组意图（比如"打击相关的三个通道共享表征"）。
显式声明更灵活但增加实验侧负担。也可能需要"允许单通道独占 trunk"作为逃生舱
（给最重要的 `r_fall` 用）。

**N4. 异步采集要不要做在一期？**

它是 SAC 的核心吞吐红利，但引入并发写 buffer、策略版本追踪、陈旧度可观测性 ——
复杂度不小。倾向：一期做同步小批量但把 `TaggedReplay` 的写入接口设计成线程安全的，
先测出同步版的 rollout/train 耗时比，再决定异步的优先级。

### 重要（影响效果，但不阻塞结构）

**N5. `log_std_min` 硬夹 vs auto-α 的冲突。**

现有实验用 `log_std_min=-1.8~-2.5` 强行维持探索。SAC 里这会和 α 打架
（α 想降熵时降不下去，于是一路滑向 0）。SAC 侧建议完全放开 `log_std` 范围交给 α ——
但这是对现有调参经验的一次切断。同时 `target_entropy` 的取值
（教科书 `-action_dim = -21` 对 tanh-squashed 21 维是否合适）需要实测。

**N6. per-channel 采样与多头 Q 冲突。**

每通道独立采 batch，就享受不到多头 Q 的一次前向；共享 batch，per-channel 采样就退化成
重要性权重。折中方案（共享大 batch + per-channel 加权/子集掩码）在效果上打几折，
需要在 `sac_balance` 上实测。

**N7. 动作梯度归一化是有创新性的做法**（与多任务学习里的 GradNorm 同源），不是现成方案。

它必须在 `sac_balance` 上被验证（对照组：源端 reward 归一化 + 朴素 `Σ w_c Q_c`），
而不能假定成立。认为它对，但要为它不成立准备退路。

**N8. 自博弈非平稳性 + replay。**

`sac_fight` 的根本风险。对手池分布漂移 + 高 UTD + 老数据，是 off-policy 的最坏情形。
缓解手段（对手 id 入 tag、新鲜度偏好采样、对手池冻结期）需要设计，但效果不确定。
可能的结论是 `fight` 这一档 SAC 不如 PPO —— 这个负结果也有价值，但要提前接受这个可能性。

### 可延后

**N9.** buffer-based reset 需要 `core_state` 的存储和 env 侧配合
（`episode_options` 传入初始 state），env blueprint 要加一个 plugin。二期。

**N10.** obs 归一化：Q 网络吃 `concat(obs, action)`，action 已在 [-1,1]，
obs 若量级差异大会主导。SAC 对此比 PPO 敏感。`baseline/common/normalize` 已存在，
接入成本低，但会引入"归一化统计量也要进 checkpoint"的复杂度。

**N11.** eval 用确定性动作还是采样：SAC 的最优策略本身是随机策略，
确定性 eval 会系统性低估。建议两个都记录。

---

## 6. 建议的推进顺序

1. **定 N1 / N2 / N3**（内存预算 → 存储结构 → Q 架构），这三个定了才能开始写代码
2. `TaggedReplay` + `sac_update` 内核 + 单元测试
   （重点：n-step 目标、per-channel done、relabel 幂等性）
3. `sac_balance` 跑通 → **在这里验证 N7（动作梯度归一化）和 UTD 上限**，
   这是关键决策点
4. `sac_loop` 同步版 + 完整诊断层 + 发散护栏
5. `sac_standup_recover` → 验证分层 replay 能否替代 `RandomFallenStatePlugin`
6. 依据步骤 4 的耗时测量决定异步采集（N4）
7. `sac_fight` → 多源摄入 + per-channel 采样；buffer-based reset 作为二期

---

## 7. 与 PPO V2 的关系

- **不共用 `ExperimentV2` 接口。** SAC 有自己的 `ExperimentSAC`，多出一整层数据分布控制。
- **不共用实验。** 三个 SAC 实验是专门设计的，不复用 V2 的 `exp_*.py`。
- **共用底层基础设施**：`TanhGaussianMLPPolicy`、`ParallelRollouter`、
  `Episode` / `EpisodeCollection`、`PolicyBlueprint` 导出、checkpoint/视频/日志格式、
  `__RAW_STATS__` 协议、git code snapshot、`--background` 机制。
- **共用设计取向**：experiment 拥有语义、framework 拥有机械；reward channel 是一等公民；
  curriculum 不藏在 framework hack 里；eval 和 state 持久化由实验定义。
- **CLI 对称**：`train.py --algo sac`，与 `--algo ppo` 体验一致
  （`--smoke` / `--background` / `--set` / git snapshot 全部复用）。
- **V1 SAC（`sac_loop.py` / `sac_trainer.py`）作为 prior art 参考**，
  但不作为 V2 的基础。V1 用的是 legacy `Experiment` 接口。
