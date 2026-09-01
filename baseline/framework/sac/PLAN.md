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
