# SAC Dump 原子诊断切口顶层设计

> 类型：设计
>
> 状态：讨论稿 v1（待批准执行）
> 关联：`DESIGN_debug_system.md`（R3 UI 骨架与路由）、`ARTIFACTS.md`（工件契约）、
> `DEBUG_PLAYBOOK.md`（诊断操作手册）

---

## 0. 问题陈述：从“事后叙事”到“过程可证伪”

此前我们对 SAC 实验的分析停留在**结果 → 经验叙事**闭环：

- 看到 `u_floor` 比 `shannon` 快 → 解释为“不确定性底线防止了早期探索坍缩”；
- 看到 `replay 2M` 消除振荡 → 解释为“记忆窗口覆盖了遗忘周期”；
- 看到 `batch 1024` 没省 tick → 解释为“不动点迭代次数守恒”。

这些解释听起来都合理，但**经验叙事可以把任何结果解释成合理的**。当训练不工作或出反常结果时，我们缺少机制去**质疑和证伪数据产生的每一个原子过程本身是否正确、是否符合它声称的数学与工程契约**。

典型例子（本次对既有 dump `reg_balance_ufloor_s42/critic_tick_00212684` 摸底时直接暴露的事实）：
`sac_standup` 和 `sac_balance` 在 `build_slices()` 中把 `versions.regularizer_mode` 硬编码成了 `"entropy"` / `"shannon"`——即使整个训练实际跑在 `u_floor` 下，每個样本携带的 provenance 元数据依然声称自己是 `shannon`。如果没有原子切口去逐项核对“配置声称的语义 vs 样本实际携带的语义 vs 算子实际执行的公式”，这类静默错位永远不会被发现。

### 设计目标

对照 PPO `dumpkit` 的九个原子诊断切口（⓪ Rollout ε → ① Episode→Traj → ② Reward→ADV → ③ ADV Norm → ④ Channel Merge → ⑤ Grad Analyze → ⑥ Update Process → ⑦ Post Update → ⑧ Delta Analyze），为 SAC 定义一套**完全贴合 SAC off-policy 数据流与计算图**的原子诊断切口体系。

每一刀必须满足四条刚性标准：
1. **切在语义接缝上**：对应 SAC 数据流中一次明确的数学变换或状态跃迁；
2. **有明确的可证伪命题（Falsifiable Claim）**：回答“这一步产出的张量，是否严格等于输入经声称公式变换的结果？分布与不变量是否在合理域内？”；
3. **同源重算 / 独立核验（Recompute & Verify）**：凡涉及公式变换的切口，必须用 trainer/experiment 同源函数或独立代数恒等式做逐样本核验，给出 `max_abs_err` 与违约样本定位；
4. **物理画面锚定（Visual Grounding）**：凡涉及样本/帧的切口，必须能通过确定性回放把抽象数字钉回仿真画面（与 PPO 视频编辑器骨架同构）。

---

## 1. PPO vs SAC：为什么切口不能照搬，应该切在哪里？

PPO 是**同步同策略（on-policy）单次轨迹消费**：一轮采集的 episodes 直接转成 trajectories，算一次 GAE/Norm/Merge，跑几个 epoch 就丢弃。因此 PPO 的切口沿着“单批轨迹的生命周期”线性展开。

SAC 是**异步跨代（off-policy）双循环架构**，存在一条**慢循环（Collection → Slice → Replay）**和一条**快循环（Sample → Target → Critic TD → Actor/Reg → Alpha/Target Update）**，两者通过 Replay Buffer 解耦：

```text
【慢循环：每 collection_round 一次】
  Env 交互 (Episode)
     │  ── 切口 ⓪ 行为策略与实现探索 (Realized Exploration)
     ▼
  CollectedEpisode (raw observer_outputs, pre_action_facts, policy_action)
     │  experiment.build_slices()
     │  ── 切口 ① Episode → Slice 转换与奖励/终止契约
     ▼
  SACTransitionSlice (rewards, actor_gate, bootstrap, task_facts, source_keys)
     │  replay.add_slices()
     ▼
  Replay Buffer (FIFO 环，跨百代策略混杂，带 draw_counts / age)
     │  replay.sample(B)
     │  ── 切口 ② Replay → Batch 采样人口学与重要性偏置
     ▼
【快循环：每 critic_tick 一次（Dump 主截面）】
  Sampled Batch (B=256/1024, 携带各自诞生时的 collection/behavior/versions 元数据)
     │
     ├──► [Target 支路: next_obs, actor(θ_pre), critic_target(ψ_pre)]
     │       ── 切口 ③ Next-State 候选枚举与 Twin-Min 合成 (F_next)
     │       ── 切口 ④ Bellman Target 组装与续接正则 (r·scale + γ·b·(SoftQ + B(s')))
     │
     ├──► [Critic 支路: obs, actions, critic(ϕ_pre), target]
     │       ── 切口 ⑤ Critic TD 拟合、掩码与通道梯度
     │
     ├──► [Actor & Reg 支路: obs, actor(θ_pre), critic(ϕ_post!)]
     │       ── 切口 ⑥ Actor 候选评估、通道门控合成与正则项分解 (Q vs α·logπ vs Bθ(s))
     │       ── 切口 ⑦ 逐样本梯度信号诊断 (GradSig: g_i · Ĝ，谁在主导/抵消更新)
     │
     └──► [参数更新与跨步演化: θ_pre→θ_post, ϕ_pre→ϕ_post, α_pre→α_post, ψ_pre→ψ_post]
             ── 切口 ⑧ 优化器步进、Alpha/Target 软更新与 Round 内 Tick 轨迹
             ── 切口 ⑨ Post-Update 同批复评与跨代策略漂移 (Drift & Replay)
```

> **关键语序差异（SAC 独有陷阱）**：在 `sac_update_v2` 中，Critic 先完成 `backward() + optimizer.step()`（变为 $\phi_{\text{post}}$），随后 Actor 在**同一个 tick 内**对 $\phi_{\text{post}}$（而非 $\phi_{\text{pre}}$）求梯度更新。任何离线复算、分解或梯度归因若搞错 $\phi_{\text{pre}}$ 与 $\phi_{\text{post}}$ 的时点，结果全错。

---

## 2. 十个原子诊断切口（⓪–⑨）完整定义

下表定义 SAC Dump 的十个原子切口。每个切口规定：**输入 → 变换 → 输出**、**可证伪断言（Automated Invariants）**、**人机检验手段（Interactive Inspection）**、以及**当前缺口**。

---

### 切口 ⓪：Rollout 行为策略与实现探索（Realized Exploration）

- **对标 PPO**：⓪ Rollout (`exploration`)
- **解剖过程**：采集轮 $r$ 中，行为策略 $\pi_{\text{beh}}$（由 `r{r:05d}` 蓝图 + `behavior_explore` + `random_start` 决定）在环境状态 $s_t$ 上采样动作 $a_t^{\text{env}}$，同时在 `action_extras` 中记录确定性参考动作 `policy_action` ($a_t^{\text{det}}$)。
- **核心问题**：“采集这个样本时，实际送进物理引擎的探索噪声 $\varepsilon_t = a_t^{\text{env}} - a_t^{\text{det}}$ 有多大？是否饱和（卡在 $\pm 1$ 边界）？`behavior_explore` / `random_start` 是否按预期生效？”
- **可证伪断言**：
  1. **直通核验**：SAC 的 `batch.pt` 已逐行保存 `actions` ($a_t^{\text{env}}$) 与 `policy_action` ($a_t^{\text{det}}$)！无需像 PPO 那样必须跑子进程才能拿到基础 $\varepsilon$——直接算 $\varepsilon_i = \text{actions}_i - \text{policy\_action}_i$；
  2. **重放核验（可选 Job）**：若 `policy_exports/r{round:05d}` 仍在盘（或 dump 时固化了行为策略），对存储的 `obs` 重跑 `det_act(obs)`，核验 $\|a^{\text{det}}_{\text{replayed}} - \text{policy\_action}\|_{\infty} < 10^{-5}$，证伪“导出蓝图与实际采集策略不一致”；
  3. **模式一致性**：当 `metadata.behavior.random_start == True` 时，$a_t^{\text{env}}$ 应服从 $\text{Uniform}[-1,1]^D$；当 `explore_factor == 0` 且确定性采集时 $\|\varepsilon\| = 0$。
- **检验视图**：
  - Batch 级：$\|\varepsilon\|_2$ 分布、各关节维度 $\varepsilon_d$ 方差、边界饱和率 $\text{sat\_frac} = P(|a_{t,d}| > 1 - 10^{-6})$、按 `collection_round` / `explore_factor` 分组的实现探索强度；
  - Episode/Slice 级（配画面）：沿帧轴画 $a_t^{\text{env}}$ vs $a_t^{\text{det}}$ 与 $\varepsilon_t$ 曲线 + 频谱/自相关（ACF）。

---

### 切口 ①：Episode → Slice 转换与奖励/终止契约（Slice Fidelity）

- **对标 PPO**：① Episode → Trajectory (`ep2traj`)
- **解剖过程**：`experiment.build_slices(episodes)` 从 `CollectedEpisode`（含 `observer_outputs`、`pre_action_facts`、`agent_termination_reason`、`physics_steps`）提取出 `SACTransitionSlice`（含 `rewards`、`actor_gate`、`actor_weight`、`actor_gate_next`、`actor_weight_next`、`terminated`、`truncated`、`bootstrap`、`task_facts`、`reward_features`）。
- **核心问题**：“实验代码算出的 reward、门控权重和 bootstrap 掩码，是否忠实于物理仿真器产出的原始 observer 数据？终止边界有无疑似 silent bug？”
- **可证伪断言（ハード不变量）**：
  1. **Bootstrap 铁律**：
     - `terminated == True` $\implies$ `bootstrap == 0.0` 且 `termination_reason != "timeout"`；
     - `truncated == True` $\implies$ `bootstrap == 1.0` 且 `termination_reason == "timeout"`；
     - 非末帧（$t < T-1$）$\implies$ `terminated == False` 且 `truncated == False` 且 `bootstrap == 1.0`；
  2. **门控归一化恒等式**：每一行 $\sum_c \text{actor\_weight}_{i,c} = 1$（由 `actor_gate / sum(actor_gate)` 生成），且对 `sac_balance`：
     $$\frac{\text{actor\_gate}_{i,\text{r\_cross}}}{\text{actor\_gate}_{i,\text{r\_fall}}} \stackrel{?}{=} \frac{w_{\text{cross}}}{w_{\text{fall}}} \cdot \text{clip}(\phi_{\text{pre}}, 0, 1)^2$$
     $$\frac{\text{actor\_gate\_next}_{i,\text{r\_cross}}}{\text{actor\_gate\_next}_{i,\text{r\_fall}}} \stackrel{?}{=} \frac{w_{\text{cross}}}{w_{\text{fall}}} \cdot \text{clip}(\phi_{\text{post}}, 0, 1)^2$$
     ——**直接用 `batch.pt` 里的 `task_facts["phi_pre"]` / `reward_features` 逐行代数复算 `rewards` 和 `actor_weight`**，最大误差必须 $< 10^{-6}$！
  3. **奖励公式恒等式**：
     - `sac_standup`：$r_{\text{potential},i} \stackrel{?}{=} (1 - \gamma) \cdot \text{clip}(\text{potential}_i, 0, 1)$；
     - `sac_balance`：$r_{\text{fall},i} \stackrel{?}{=} c_\phi \cdot \text{clip}(\phi_{\text{post},i}, 0, 1)$，$r_{\text{cross},i} \stackrel{?}{=} \text{reward\_features}[\text{"cross\_support"}]_i$；
  4. **物理步进非退化**：`physics_delta > 0`；
  5. **元数据诚实性**：`metadata.versions.regularizer_mode` 与 `spec.sac_params.regularizer_mode` 比对（直接捕获前文发现的硬编码 bug）。
- **检验视图**：
  - 契约核验表：列出上述 5 条恒等式的 `max_abs_err` 与违约行数（0 = 绿灯，>0 = 红灯并置顶违约 `sample_id`）；
  - 逐样本/逐帧对照表：`task_facts` / `reward_features` 原始值 $\to$ 公式复算值 $\to$ `batch` 实存值；
  - 视频编辑器视图（对选中样本所属 episode 进行确定性重放渲染后）：画面 + `phi_pre` / `phi_post` / `height` / `cross_support` 轨道 + 各通道 `reward` 与 `actor_weight` 轨道。

---

### 切口 ②：Replay → Batch 采样人口学与覆盖诊断（Replay & Sampling）

- **对标 PPO**：SAC 独有切口（PPO 无 Replay Buffer）
- **解剖过程**：从容量 $N$（500K / 2M）、当前大小 $S$ 的 FIFO Replay Buffer 中均匀（或按 `ReplayPlan`）抽样 $B$ 条 transition 组成当前 tick 的训练 batch。
- **核心问题**：“当前 tick 喂给网络的 $B$ 个样本到底代表什么数据分布？是不是被某几轮旧数据垄断？有没有样本被抽了上百次、或者大量样本从未被抽中？跨代策略（policy age）跨度多大？”
- **可证伪断言**：
  1. **均匀性检验（Chi-square / KS 检验）**：在 uniform 采样下，batch 的 `collection_round` 分布与 Replay Buffer 当前存活样本的 `collection_round` 分布应统计一致；`sample_id` 在存活区间 `[min_live_id, max_live_id]` 内应近似均匀；
  2. **存活边界一致性**：batch 中所有 `sample_id` 必须属于采样时刻存活区间，无覆盖写坏（corrupted slot）或重复 `sample_id`（除非有放回采样撞车，记录唯一率）；
  3. **年龄与抽样次数理论值对照**：给定 `utd_ratio`、$B$、$S$，理论平均被抽次数 $\mathbb{E}[\text{draws}] \approx \text{UTD} \cdot B \cdot (\text{age} / \text{transitions\_per\_round})$，实测 `draw_counts` 是否偏离理论曲线。
- **检验视图**：
  - Batch vs Replay 人口学双层直方图：样本年龄 `age = current_round - sample_round` 分布、`draw_counts` 分布、`policy_fingerprint` 世代构成；
  - 终止帧/超时帧在 Batch 中的稀疏度（如 256 个样本里只有 2 个 `terminated=True`，解释为何终态价值传播慢）。

---

### 切口 ③：Next-State 候选枚举与 Twin-Min 顺序核验（Candidate & Twin Selection）

- **对标 PPO**：SAC 独有切口（对应 `compute_critic_targets` 前半段）
- **解剖过程**：对 `next_obs` ($s'$)，用冻结的随机源 `u_next` $\in [B, K, M, D]$ 经 `actor.expectation_samples(next_obs, u_next)` 生成 $K \times M$ 个候选动作 $a'_{b,k,m}$、对数密度 $\log \pi(a'_{b,k,m}|s'_b)$ 和积分权重 $w^{\text{integ}}_{b,k,m}$；再对每个通道 $c$ 的两个 target Q 网络求值 $Q^{\text{targ}}_{c,1}(s', a')$, $Q^{\text{targ}}_{c,2}(s', a')$，先按 `actor_weight_next` 加权合成 $F_j = \sum_c w^{\text{next}}_c Q^{\text{targ}}_{c,j}$，取 $j^* = \arg\min_{j \in \{1,2\}} F_j$，最后从各通道取出 $Q^{\text{targ}}_{c, j^*}$。
- **核心问题**：“候选动作积分权重和为 1 吗？Twin-min 选的是哪一个网络（pair 0 还是 pair 1）？两套 Q 网络有没有发生严重分歧（twin gap）或某一个塌陷？**先合成 $F_j$ 再 argmin、最后求候选期望**的顺序不变量是否严格成立？”
- **可证伪断言**：
  1. **积分权重归一恒等式**：$\forall b, \sum_{k,m} w^{\text{integ}}_{b,k,m} = 1.0$（误差 $< 10^{-6}$）；
  2. **候选重放确定性**：用 `trainer_pre` 的 actor + 捕获的 `u_next` 重跑 `expectation_samples`，与 `forward["next_actions"]` / `forward["next_log_probs"]` 逐元素误差 $< 10^{-5}$；
  3. **Twin-Min 联合选择恒等式**：独立重算 $F_1, F_2$ 并验证 `target_pair_index` $\stackrel{?}{=} \mathbf{1}[F_2 < F_1]$；同时对比“先联合 argmin 再通道取值”与“各通道独立 $\min(Q_{c,1}, Q_{c,2})$”的差值（量化跨通道耦合选择的实际影响）；
  4. **Twin 偏置告警**：`target_pair1_frac` 若长期极端接近 0 或 1（如 $>0.98$ 或 $<0.02$），说明两个 Q 网络出现系统性高低偏置，twin-min 退化为单网络。
- **检验视图**：
  - 逐通道 $(Q^{\text{targ}}_{c,1}, Q^{\text{targ}}_{c,2})$ 散点图与 Twin Gap $|Q_1 - Q_2|$ 分布；
  - `pair1_frac` 占比与跨通道不一致率（即通道 $c_1$ 觉得 $Q_1 < Q_2$ 但联合 $F$ 选了 $Q_2$ 的样本比例）。

---

### 切口 ④：Bellman Target 组装与续接正则重算（Reward → Target）

- **对标 PPO**：② Reward → ADV (`gae`，带 $\gamma/\lambda$ 滑杆重算)
- **解剖过程**：将切口 ③ 选出的 $Q^{\text{targ}}_{c, j^*}$ 与续接正则项结合，组装出每个通道的 1-step Bellman Target：
  - `shannon` 模式：$V_c(s') = \sum_{k,m} w^{\text{integ}}_{k,m} \big(Q^{\text{targ}}_{c,j^*}(s', a'_{k,m}) - \alpha \log \pi(a'_{k,m}|s')\big)$；
  - `u_bonus` / `u_floor` 模式：$V_c(s') = \big(\sum_{k,m} w^{\text{integ}}_{k,m} Q^{\text{targ}}_{c,j^*}(s', a'_{k,m})\big) + B_\theta(s')$，其中 $B_\theta(s') = \lambda U(s')$ 或 $-\lambda \,\text{relu}(f - U(s'))^2$；
  - 最终：$y_{b,c} = r_{b,c} \cdot \text{reward\_scale} + \gamma_c \cdot \text{bootstrap}_b \cdot V_c(s'_b)$。
- **核心问题**：“Target 里到底是当前步 reward 占主导，还是未来 bootstrap 价值占主导，还是正则项把 target 顶歪了？如果改变 $\gamma_c$、`reward_scale`、$\alpha$、`u_floor` ($f$)、`reg_lambda` ($\lambda$)，Target 会怎么变？”
- **可证伪断言**：
  1. **三段闭合恒等式**：$\forall b, c$，`reward_term + softq_term + reg_term` 与 `forward["targets"][c]` 的最大绝对误差 $< 10^{-5}$；
  2. **U-正则精确核验**：在 `u_floor` / `u_bonus` 模式下，用 `trainer_pre` 的 actor 在 `next_obs` 上直接调用 `actor.uncertainty(next_obs)` 得到 $U(s'_b)$，独立算 $-\lambda\,\text{relu}(f - U(s'_b))^2$，核验与 `forward["reg_next"]` 逐样本误差 $< 10^{-6}$；并统计**触发惩罚率** $P(U(s'_b) < f)$（直接证伪“u_floor 设了但其实 $U(s')$ 全在 $f$ 之上、惩罚恒为 0”的假象！）；
  3. **终态归零铁律**：所有 `bootstrap == 0` 的行，`softq_term == 0` 且 `reg_term == 0`，Target 严格等于 $r \cdot \text{scale}$。
- **检验视图**：
  - 逐样本三段分解瀑布图/堆叠条（`reward_term` / `softq_term` / `reg_term`）+ $U(s')$ vs `u_floor` 分布直方图（标出 $f$ 阈值红线与激活占比）；
  - **交互式反事实重算器**（后端走 trainer 同源逻辑）：前端提供 $\gamma_c$、`reward_scale`、$\alpha$、`u_floor`、`reg_lambda` 滑杆，实时重算并叠加对比新旧 Target 分布与 TD Error 变化。

---

### 切口 ⑤：Critic TD 拟合、有效掩码与通道更新（TD → Critic）

- **对标 PPO**：③ ADV Norm / Critic Value Loss 诊断
- **解剖过程**：对每个通道 $c$，用 `critic(ϕ_pre)` 前向算 $Q_{c,1}(s,a), Q_{c,2}(s,a)$，与切口 ④ 的 $y_c$ 算残差 $\delta_{c,1} = Q_{c,1} - y_c$, $\delta_{c,2} = Q_{c,2} - y_c$，乘以有效掩码 $m_{b,c} = \text{channel\_valid}_{b,c} \cdot \text{sample\_weight}_b$，求加权 MSE，各自独立 backward + `clip_grad_norm_` + Fused Adam step，得到 $\phi_{\text{post}}$。
- **核心问题**：“Critic 预测跟得上 Target 吗（Explained Variance 是多少）？TD error 是全 batch 均匀偏大，还是被少数几个极端离群样本（比如刚摔倒的终止帧）拉爆了？梯度裁剪是否每步都在触发？”
- **可证伪断言**：
  1. **Loss 闭合恒等式**：由捕获的 `q1_pred`, `q2_pred`, `targets`, `channel_valid`, `sample_weight` 独立算出的 MSE 与 `update_stats["q1_loss_<ch>"]` 误差 $< 10^{-5}$；
  2. **SAC Critic Explained Variance（新增核心指标）**：补齐 PPO 有的而 SAC 缺失的通道可信度指标：
     $$\text{EV}_{c,j} = 1 - \frac{\text{Var}_m(y_c - Q_{c,j})}{\text{Var}_m(y_c)}$$
     若 $\text{EV} \le 0$，证明 Critic 的预测还不如直接猜常数均值，此时它给 Actor 提供的梯度纯属噪声！
  3. **离群样本集中度（Top-K Share）**：Top 5% 样本占了多少比例的 $\sum \delta^2$？若 $>50\%$，说明 Critic 更新被极少数异常转换绑架。
- **检验视图**：
  - 逐通道 $Q_{\text{pred}}$ vs $y_{\text{target}}$ 对角散点图（理想应贴近 $y=x$ 线，偏离立即可见）+ $\text{EV}_{c,1}, \text{EV}_{c,2}$ 读数；
  - TD Error $\delta$ 分布（区分正过估计 $\delta > 0$ 与负低估计 $\delta < 0$）+ Top $|\delta|$ 样本排名表（点击直跳该样本的切口 ① 画面或切口 ④ Target 分解）；
  - 各通道 `grad_norm_pre_clip` vs `grad_clip_norm` 阈值条。

---

### 切口 ⑥：Actor 通道门控合成与目标三项分解（Channel Merge & Actor Objective）

- **对标 PPO**：④ Channel Merge (`merge`)
- **解剖过程**：用**更新后的 Critic $\phi_{\text{post}}$** 对 `actor(θ_pre)` 在 `obs` ($s$) 上枚举的候选动作 `new_actions` 求值 $Q^{\text{online}}_{c,1}(s, \tilde{a}), Q^{\text{online}}_{c,2}(s, \tilde{a})$；乘以门控权重 $w^{\text{actor}}_{b,c} = \text{actor\_weight}_{b,c} \cdot \mathbf{1}[\text{share}_c]$ 合成联合价值 $F^{\text{actor}}_j = \sum_c w^{\text{actor}}_c Q^{\text{online}}_{c,j}$，取 $j^* = \arg\min_j F^{\text{actor}}_j$；最后加上熵项或状态不确定性正则项，构成逐样本 Actor Loss：
  $$\ell^{\text{actor}}_b = \underbrace{\sum_c \Big(\sum_{k,m} w^{\text{integ}}_{k,m} \big(-w^{\text{actor}}_{b,c} Q^{\text{online}}_{c,j^*}(s_b, \tilde{a}_{b,k,m})\big)\Big)}_{\text{per-channel Q terms}} + \underbrace{\alpha \sum_{k,m} w^{\text{integ}}_{k,m} \log \pi(\tilde{a}_{b,k,m}|s_b)}_{\text{entropy\_term (shannon only)}} + \underbrace{\big(-B_\theta(s_b)\big)}_{\text{reg\_term (U modes only)}}$$
- **核心问题**：“多通道合并时，是 `r_fall` 说了算还是 `r_cross` 说了算？门控权重 $\phi_{\text{pre}}^2$ 有没有把某个通道彻底关死？Actor 总目标里，Q 项、熵项、$U$-floor 惩罚项各自的量级和梯度拉力是多少？”
- **可证伪断言**：
  1. **Actor 目标闭合恒等式**：$\sum_c \text{per\_channel\_q}_c + \text{entropy\_term} + \text{reg\_term} \stackrel{?}{=} \text{recorded\_obj}$（当前 `objective_decomp` 已验 `check.objective_max_abs_err` $< 10^{-5}$）；
  2. **$\phi_{\text{post}}$ 语义一致性**：验证 `forward["actor_cand_q"]` 确实等于 `critic(ϕ_post)`（而非 $\phi_{\text{pre}}$）在 `new_actions` 上的前向输出；
  3. **当前状态 $U(s)$ 正则精确核验**：在 `u_floor` 下，逐样本核算 $U(s_b) = \text{actor.uncertainty}(\text{obs}_b)$ 与 $-\text{reg\_term}_b = -\lambda\,\text{relu}(f - U(s_b))^2$，给出：
     - $U(s_b)$ 的分布（均值、分位数、最小值）；
     - **越界比例（Floor Violation Rate）** $P(U(s_b) < f)$：只有越界的样本才会给 Actor 产生把不确定性往上推的梯度！如果越界率为 0，说明 `u_floor` 此时在 Actor 侧完全休眠；如果越界率 100%，说明策略整体过窄。
  4. **通道有效贡献占比**：不仅看 `actor_weight` 均值，还要看 `|w_c · Q_c|` 的方差与梯度范数占比（防止某个通道虽然权重 0.5，但因为 $Q_c$ 常数无梯度而实际贡献为 0）。
- **检验视图**：
  - 通道门控散点图：`phi_pre` $\to$ `actor_weight[r_cross]` $\to$ 各通道加权 Q 贡献；
  - $U(s)$ 分布直方图 + `u_floor` 阈值线 + 越界样本高亮表；
  - 词项级梯度分解（调用 `grad_attribution` 的 `actor.terms`）：展示 `q_term`、`entropy_term`、`reg_term` 三个梯度向量的范数 $\|g_{\text{term}}\|$ 与对总梯度的余弦 $\cos(g_{\text{term}}, G_{\text{actor}})$——一眼看清“正则项是在辅助 Q 方向，还是在与 Q 方向正面对抗（$\cos < 0$）”。

---

### 切口 ⑦：逐样本梯度信号诊断（Grad Analyze / Per-Sample GradSig）

- **对标 PPO**：⑤ Grad Analyze (`gradsig` under $\theta_{\text{old}}$)
- **解剖过程**：在更新前的参数 $(\theta_{\text{pre}}, \phi_{\text{pre}})$ 上，Batch 内每个样本 $i \in \{0, \dots, B-1\}$ 产生自己的 Actor 梯度 $g^{\text{actor}}_i = \nabla_\theta \ell^{\text{actor}}_i$ 和各通道 Critic 梯度 $g^{\text{critic}}_{c,i} = \nabla_{\phi_c} \ell^{\text{critic}}_{c,i}$，全 batch 平均得到聚合梯度 $G = \frac{1}{B}\sum_i g_i$。计算每样本的梯度范数 $\|g_i\|$、带符号投影 $p_i = g_i \cdot \hat{G}$、方向余弦 $c_i = \cos(g_i, G)$。
- **核心问题**：“这 256 个样本里，是谁在主导这次参数更新？有没有几个离群样本一己之力扭转了整个 Batch 的梯度方向？不同样本之间是在合力推进（高 coherence），还是在互相内耗抵消（高负投影占比）？”
- **为什么 SAC 比 PPO 更容易且更精确**：
  PPO 的 buffer 有 $40\text{万}$ 帧，只能抽样 2000 帧近似；而 **SAC 的 batch 只有 $B=256$（或 $1024$）行**！离线分析时可以**对全 Batch 100% 样本做精确逐样本梯度投影**（零抽样误差），完美满足恒等式：
  $$\frac{1}{B} \sum_{i=1}^B p_i \equiv \|G\|_2$$
- **可证伪断言**：
  1. **投影均值恒等式**：$\big|\frac{1}{B}\sum_i p_i - \|G\|_2\big| < 10^{-5}$（自证逐样本梯度计算与实际反传梯度完全一致）；
  2. **梯度相干性（Coherence）**：$\text{coherence} = \|G\|_2 / \text{mean}(\|g_i\|_2) \in [0, 1]$。若 coherence 极低（如 $<0.05$）且 `frac_neg_proj` 接近 $50\%$，证明 batch 内样本信号严重冲突，大部分梯度在相互抵消；
  3. **通道间 Actor 梯度冲突矩阵（Multi-Channel Grad Conflict）**：分别求出每个通道 $c$ 对 Actor 的梯度 $G^{\text{actor}}_c = \nabla_\theta \mathbb{E}[-w_c Q_c(s, \pi(s))]$，计算通道间余弦矩阵 $\cos(G^{\text{actor}}_{c_1}, G^{\text{actor}}_{c_2})$——直接回答 `sac_balance` 中“`r_fall`（别摔倒）和 `r_cross`（双脚交叉支撑）在当前状态下是不是在往相反方向拽机器人”！
- **检验视图**：
  - $\cos(g_i, G) \times \|g_i\|_2$ 二维热力图/散点图（对标 PPO Scene 5）；
  - 正投影 Top-K（推动者）与负投影 Top-K（被牺牲/对抗者）样本排名表，点击任意样本直接下钻到 Trace / 视频画面；
  - 多通道 Actor 梯度余弦矩阵 + 来源分组归因（按 `collection_round` / `explore_factor` 聚合投影贡献）。

---

### 切口 ⑧：Update 过程、Alpha 动态与 Round 内 Tick 时间线（Update Process）

- **对标 PPO**：⑥ Update Process (`timeline`)
- **解剖过程**：
  - 单 Tick 内：梯度裁剪 $\to$ Adam 步进（$\theta_{\text{pre}} \to \theta_{\text{post}}$, $\phi_{\text{pre}} \to \phi_{\text{post}}$）$\to$ 温度更新 $\mathcal{L}(\alpha) = -\alpha \mathbb{E}[\log \pi + \bar{H}]$（若 `auto_alpha`）$\to$ Target 软更新 $\psi_{\text{post}} = (1-\tau)\psi_{\text{pre}} + \tau \phi_{\text{post}}$；
  - Round 内宏观尺度：本轮 `n_grad_steps`（如 3200 或 12800）个 critic tick 的连续演化轨迹，以及当前 dump tick 在其中的坐标。
- **核心问题**：“优化器这一步迈了多大？哪一层参数动得最多？Target 软更新 $\tau$ 是否严格执行？在当前 round 的上万次 tick 中，这个 dump 抓在平稳段还是尖峰段？”
- **可证伪断言**：
  1. **Target Polyak 软更新恒等式**：由于 `trainer_pre.pt` 和 `trainer_post.pt` 同时保存了 online critic ($\phi$) 与 target critic ($\psi$)，可直接逐参数核验：
     $$\big\|\psi_{\text{post}} - \big((1-\tau)\psi_{\text{pre}} + \tau \phi_{\text{post}}\big)\big\|_{\infty} < 10^{-6}$$
     （一步证伪 target 更新漏跑、跑了两次或用了 $\phi_{\text{pre}}$ 的 bug！）；
  2. **Alpha 不动点/冻结检验**：
     - `shannon` + `auto_alpha=True`：核验 $\alpha$ 梯度符号与 $(\bar{\log \pi} + \text{target\_entropy})$ 一致；
     - `u_floor` / `u_bonus` 或 `auto_alpha=False`：核验 $\|\alpha_{\text{post}} - \alpha_{\text{pre}}\| \equiv 0$；
  3. **参数位移分层审计（`param_delta`）**：Actor trunk / mu_head / std_head 与各通道 Critic 各层的 $\|\Delta \theta\|_2$ 和相对步长 $\|\Delta \theta\|_2 / \|\theta\|_2$。
- **检验视图**：
  - Round 内 Tick 时间线（`tickline`）：Loss / Q / TD / $\alpha$ / grad_norm 曲线 + 当前 Dump Tick 竖线锚点；
  - Polyak 恒等式核验绿灯 + 分层参数位移条形图。

---

### 切口 ⑨：Post-Update 同批复评与跨代策略漂移（Post-Update & Cross-Gen Drift）

- **对标 PPO**：⑦ Post Update (`postupdate`, $\theta_{\text{new}}$ on buffer) + ⑧ Delta Analyze (`delta`, cross-generation)
- **解剖过程**：
  - **同批即时效应（Immediate Post-Update）**：在当前 dump 的**同一个 Batch $(s, a, s')$** 上，对比更新前后 $(\theta_{\text{pre}}, \phi_{\text{pre}})$ vs $(\theta_{\text{post}}, \phi_{\text{post}})$ 的行为与价值变化；
  - **跨代演化效应（Cross-Round Delta）**：固定一批观测状态（来自当前 dump batch 或指定 episode），重放历代策略（`policy_exports/r*` 或 `checkpoints/checkpoint_s*` 中的 actor）的确定性动作 $\pi^{\text{det}}_k(s)$、不确定性 $U_k(s)$ 与价值估计 $Q_k(s, \pi^{\text{det}}_k(s))$。
- **核心问题**：
  - 即时：“这一步梯度更新下去后，Critic 在这批样本上的 TD Error 真的变小了吗？Actor 在这批状态上的动作往高 Q 方向移了多少？有没有一步更新把 log_prob 推爆？”
  - 跨代：“过去几十轮（几十万 env_step）里，策略在同样这批状态上的动作是在稳步收敛，还是来回拉锯振荡（如 balance 500K replay 的记忆→遗忘循环）？”
- **可证伪断言**：
  1. **单步 TD 下降检验**：在当前 batch 上计算 $\text{MSE}(\phi_{\text{post}}, y_{\text{pre}})$ vs $\text{MSE}(\phi_{\text{pre}}, y_{\text{pre}})$——走了一步梯度后，在当前 batch 和固定 target 下 Critic loss 应当下降（若上升说明学习率/动量冲过头）；
  2. **单步 Actor 改进检验**：在固定 $\phi_{\text{post}}$ 和固定候选 `u_actor` 下，$\ell^{\text{actor}}(\theta_{\text{post}}) < \ell^{\text{actor}}(\theta_{\text{pre}})$ 是否成立；
  3. **跨代振荡量化**：跨 $G$ 个历史 checkpoint 在同一批 $s$ 上计算动作位移矩阵 $D_{g_1, g_2} = \mathbb{E}_s\|\pi_{g_1}(s) - \pi_{g_2}(s)\|_2$——若净位移 $\| \pi_{g_K} - \pi_{g_0} \| \ll \sum_k \| \pi_{g_k} - \pi_{g_{k-1}} \|$，定量坐实策略在状态空间原地画圈振荡！
- **检验视图**：
  - 同批 Pre vs Post 逐样本散点/差值图：$\Delta Q(s,a)$, $\Delta \text{TD}^2$, $\Delta \pi_{\text{det}}(s)$, $\Delta \log \pi(a|s)$, $\Delta U(s)$；
  - 跨代漂移热力图与轨迹曲线（支持基于 `checkpoints/` 的跨代回放，解决 `policy_exports/` 只保留最近 8 轮的限制）。

---

## 3. 物理画面锚定：Episode 确定性重放与视频编辑器骨架

PPO 工具页之所以“可检验”，很大程度是因为 ①②③④⑦⑧⓪ 全部建立在**同一个视频编辑器舞台（大图 + 缩略图条 + 帧滑杆 + 下方多轨曲线）**上——人不仅看到某帧 `r_cross = -0.32` 或 `proj = -1.8`，还能直接看到那一帧机器人是不是正处于双腿交叉快要摔倒的姿态。

### SAC 的特殊挑战与解法

PPO 在 dump 时手头就握着刚采完的 `episodes.npz`；而 SAC dump 发生在 `critic_tick`，Batch 里的 256 个样本来自过去不同轮次的最多 ~256 个不同 episode，不可能在常规训练中把所有历史 episode 的全帧数组常驻内存。

**好消息：SAC 的每个样本在 `batch.pt["metadata"][i]` 中已经完整保存了重现该 Episode 所需的全部确定性凭据！**

```json
{
  "frame_index": 167,
  "behavior": {"mode": "stochastic", "explore_factor": 0.0, ...},
  "collection": {
    "collection_round": 106,
    "agent_id": "robot_b",
    "episode_seed": 27331,
    "env_blueprint_hash": "279af497...",
    "policy_fingerprint": "a5b47d5d..."
  }
}
```

### 按需 Episode 重建与渲染管线（On-Demand Episode Reconstruction Job）

当用户在任何切口选中一个 `sample_id`（或 `slice_id`）并点击 **“重建 Episode 与画面 (Render & Verify Episode)”** 时，后台单 Job 槽执行：

1. **定位策略蓝图**：
   - 优先查找 `policy_exports/r{collection_round:05d}`；
   - 若已被 prune，则在 dump 捕获时**自动把 batch 中涉及的、尚存于 `policy_exports/` 的蓝图硬链/拷贝进 `<dump_dir>/behavior_policies/`**（每个蓝图仅 ~350KB，最多 8 个 $\approx 2.8\text{MB}$），或回退使用 `trainer_pre` 的策略并显著标注 `policy_source: "current_actor_fallback (original pruned)"`；
2. **重跑单局仿真 + 录帧**：
   - 用实验的 `env_blueprint` + `episode_seed` + 行为策略跑一遍 `SACEpisodeRunner` + `BaseFrameRecorder`，产出：
     - 逐帧 PNG：`<dump_dir>/episodes/e{seed}_r{round}/frames/frame_NNNN.png`
     - 重建的完整 `CollectedEpisode` 与 `SACTransitionSlice`：`<dump_dir>/episodes/e{seed}_r{round}/episode_slice.npz`；
3. **一致性自校验（Bitwise / Float Verification）**：
   - 将重放出的第 `frame_index` 帧的 `obs`, `action`, `reward` 与 `batch.pt` 中该 `sample_id` 的实存值比对：
     - 若策略来自原版 `r{round}` 且 RNG 对齐，验证 $\| \text{obs}_{\text{replay}} - \text{obs}_{\text{batch}} \|_{\infty}$；
     - 同时把该 slice 在 `replay.pt` 中仍存活的所有帧（来自 `slice_context`）全部与重放帧比对，输出 `verification_report.json`；
4. **解锁视频编辑器舞台**：
   - 一旦某样本所属 Episode 完成重建，切口 ⓪（Rollout $\varepsilon$）、切口 ①（Episode→Slice 逐帧核验）、切口 ④/⑥（沿该 Episode 时间轴用当前 $\theta_{\text{pre}}, \phi_{\text{pre}}$ 扫描 $Q(s_t, a_t)$、Target $y_t$、不确定性 $U(s_t)$ 曲线）全部获得与 PPO 完全一致的**大图 + 缩略图条 + 帧滑杆（自动定位到 `frame_index` 高亮帧）+ 多轨曲线**体验！

---

## 4. 统一验证报告：契约审计引擎（Contract Ledger）

为了让“质疑数据产生过程”不靠人眼大海捞针，每个 Dump 顶部（Overview 卡与 `debugkit verify <dump>` 命令）新增一张**自动化契约审计表（Contract Ledger）**。

它对十个切口运行所有可证伪断言，每条断言给出明确状态：
- `PASS`（误差在浮点容差内，如 `max_abs_err = 1.2e-7 <= 2e-5`）
- `WARN`（数学上没算错，但落入病态统计域，如 `critic EV < 0`、`u_floor 激活率 = 0%`、`twin pair1_frac = 0.996`、`grad coherence < 0.05`）
- `FAIL`（契约违背或静默 Bug，如 `terminated_with_bootstrap > 0`、`versions.regularizer_mode != spec.sac_params.regularizer_mode`、Polyak 更新残差超标、奖励重算不一致）

| 编号 | 切口 | 断言名称 | 检验公式 / 判据 | 状态 |
|---|---|---|---|---|
| `C0.1` | ⓪ Rollout | `action_bounds` | $\max |a^{\text{env}}| \le 1 + 10^{-6}$ | PASS/FAIL |
| `C0.2` | ⓪ Rollout | `realized_eps_present` | `policy_action` 存在且非全 NaN；统计 $\|\varepsilon\|_2$ 与饱和率 | PASS/WARN |
| `C1.1` | ① Ep→Slice | `bootstrap_terminal_exclusivity` | `terminated` $\implies b=0$；`truncated` $\implies b=1$ | PASS/FAIL |
| `C1.2` | ① Ep→Slice | `reward_formula_recompute` | 由 `task_facts`/`reward_features` 复算 `rewards` 误差 $<10^{-5}$ | PASS/FAIL |
| `C1.3` | ① Ep→Slice | `actor_weight_gate_recompute` | 由 `phi_pre`/`phi_post` 复算 `actor_weight`/`_next` 误差 $<10^{-5}$ | PASS/FAIL |
| `C1.4` | ① Ep→Slice | `provenance_version_match` | `batch.metadata.versions` 与 `spec.sac_params` 一致 | PASS/FAIL |
| `C2.1` | ② Replay | `live_slot_integrity` | 无 `sample_id < 0`、无 NaN/Inf、`physics_delta > 0` | PASS/FAIL |
| `C2.2` | ② Replay | `sampling_coverage` | 唯一 `sample_id` 占比、终止帧占比告警（如全 batch 0 终止帧） | PASS/WARN |
| `C3.1` | ③ Twin-Min | `integration_weights_sum_one` | $\max_b |\sum_{k,m} w^{\text{integ}}_{b,k,m} - 1| < 10^{-5}$ | PASS/FAIL |
| `C3.2` | ③ Twin-Min | `candidate_replay_exact` | 冻结 `u_next`/`u_actor` 重放动作误差 $< 2\times 10^{-5}$ | PASS/FAIL |
| `C3.3` | ③ Twin-Min | `twin_balance` | `target_pair1_frac` 与 `actor_pair1_frac` $\in [0.05, 0.95]$ | PASS/WARN |
| `C4.1` | ④ Target | `bellman_decomp_closure` | `reward_term + softq_term + reg_term == target` ($<10^{-5}$) | PASS/FAIL |
| `C4.2` | ④ Target | `u_reg_next_exact` | `actor.uncertainty(next_obs)` 复算 `reg_next` 误差 $<10^{-5}$ | PASS/FAIL |
| `C5.1` | ⑤ Critic | `critic_mse_exact` | 复算 `q1_loss`/`q2_loss` 与记录值误差 $<2\times 10^{-5}$ | PASS/FAIL |
| `C5.2` | ⑤ Critic | `critic_explained_variance` | 各通道 $\text{EV}_{c,1}, \text{EV}_{c,2} > 0.1$ | PASS/WARN |
| `C6.1` | ⑥ Actor | `actor_decomp_closure` | $\sum_c Q_c + \text{ent} + \text{reg} == \text{obj}$ ($<10^{-5}$) | PASS/FAIL |
| `C6.2` | ⑥ Actor | `u_floor_activity` | `u_floor` 模式下统计 $U(s)$ 分布与越界率 $P(U(s)<f)$ | PASS/WARN |
| `C7.1` | ⑦ GradSig | `per_sample_proj_identity` | $|\text{mean}(p_i) - \|G\|_2| < 10^{-4}$ 且统计 coherence / 通道冲突 | PASS/WARN |
| `C8.1` | ⑧ Update | `polyak_soft_update_exact` | $\|\psi_{\text{post}} - ((1-\tau)\psi_{\text{pre}} + \tau\phi_{\text{post}})\|_{\infty} < 10^{-5}$ | PASS/FAIL |
| `C8.2` | ⑧ Update | `alpha_mode_invariant` | 非 `auto_alpha` 或 U 模式下 $\Delta \alpha \equiv 0$ | PASS/FAIL |
| `C9.1` | ⑨ Post | `single_step_loss_descent` | 同 batch 固定 target 下 $\mathcal{L}_{\text{critic}}(\phi_{\text{post}}) \le \mathcal{L}_{\text{critic}}(\phi_{\text{pre}})$ | PASS/WARN |

---

## 5. 现有能力盘点与增量建设地图

我们并不是从零开始——`sac_dump_v3` 已经抓全了最核心的原始张量（`batch.pt`, `forward.pt`, `trainer_pre.pt`, `trainer_post.pt`, `spec.json`）。差的是把“散装材料”变成“十个闭合的原子切口”。

| 切口 | 现有基础 (`analysis.py` / `debugkit.py`) | 核心缺口（需补齐） |
|---|---|---|
| **全局** | `recompute_dump`、` _batch_consistency` | **Contract Ledger 引擎**（21 条断言自动核验 + 违约样本定位）+ 修复 `exp_sac_*` 中硬编码的 `versions.regularizer_mode` |
| **⓪ Rollout** | `batch.pt` 已存 `actions` 和 `policy_action`！但未分析 | 新增 `rollout_exploration(dump)`：直接从 `actions - policy_action` 算逐样本 $\varepsilon$、维分布、饱和率；dump 时保留当轮行为策略蓝图以支持重放 |
| **① Ep→Slice** | 仅验 `terminated/bootstrap` 互斥 | 新增 `slice_verification(dump)`：基于 `task_facts`/`reward_features` 逐行复算 `rewards` 和 `actor_weight`/`_next`；新增按需 Episode 重放渲染 Job（生成完整帧序列 + 画面） |
| **② Replay** | `batch_attribution`、`replay_stats`、`slice_context` | 补齐 Batch vs Replay 存活分布对比（年龄/抽样次数偏置检验）、终态样本稀疏度统计 |
| **③ Twin-Min** | `forward.pt` 已存 `next_cand_q`, `target_pair_index`, `integ_w` | 新增 `candidate_twin_analysis(dump)`：$(Q_1, Q_2)$ 散点、twin gap $|Q_1 - Q_2|$、联合选择 vs 独立选择差异、积分权重和校验 |
| **④ Target** | `_target_parts` 已有三段分解；前端有简易 $\gamma$ 滑杆 | 补齐 $U(s')$ 逐样本显式提取与 `u_floor` 越界率；后端新增通用反事实重算 API `/api/dump/<d>/recompute-target?gamma=...&scale=...&u_floor=...&reg_lambda=...&alpha=...` |
| **⑤ Critic** | 有 `td_q1/q2`、`q1/q2_pred` | **新增逐通道 Explained Variance ($\text{EV}_c$)**、$Q_{\text{pred}}$ vs $y_{\text{target}}$ 散点数据、TD 平方误差 Top-K 集中度 |
| **⑥ Actor** | `_actor_parts` 已有通道 + 熵 + reg 分解 | **新增逐样本 $U(s)$ 分布与 `u_floor` 激活率 $P(U(s)<f)$**、门控曲线 $\phi_{\text{pre}} \to w_c \to w_c Q_c$、词项梯度余弦（无需等重 Job 的快速版或合并进切口 ⑦） |
| **⑦ GradSig** | `grad_attribution` 只按 provenance 分组算梯度，**无逐样本梯度** | **新增 `per_sample_gradsig(dump)`**：对 $B=256$ 逐样本算 $g_i$、投影 $p_i = g_i \cdot \hat{G}$、$\cos(g_i, G)$、二维直方图、正负 Top 样本、以及**多通道 Actor 梯度冲突矩阵 $\cos(G^{\text{actor}}_{c_1}, G^{\text{actor}}_{c_2})$** |
| **⑧ Update** | `param_delta`、`round_tick_series` | 新增 **Polyak 软更新精确核验** $\|\psi_{\text{post}} - ((1-\tau)\psi_{\text{pre}} + \tau\phi_{\text{post}})\|$、分层相对步长 $\|\Delta\theta\|/\|\theta\|$ |
| **⑨ Post/Delta** | `policy_drift` 已算 $\Delta a_{\text{det}}, \Delta \log\pi, \Delta Q$ 标量均值 | 扩展 `policy_drift` 输出**逐样本数组** + 单步 Loss 下降核验 + $\Delta U(s)$；新增**跨 Checkpoint 世代漂移 Job (`cross_gen_delta`)**，观察同一批 $s$ 在训练全程的动作/Q 演化与振荡 |

---

## 6. 统一数据访问与 API / CLI / 前端对齐

遵循 PPO `dumpkit` 的单一信息源原则：

```text
sac_dump_v3 目录 (batch.pt, forward.pt, trainer_pre/post.pt, spec.json)
   + run 目录 (checkpoints/, policy_exports/, metrics/events.jsonl)
        │
        ▼
  baseline/framework/sac/analysis.py  （唯一计算与契约核验源）
        ├──► debugkit.py CLI          （离线 JSON，AI 与脚本直接消费）
        └──► debugserver.py HTTP API  （Viewer 前端消费，逐字节同源）
```

### 6.1 Dump 首页：十切口流水线卡片（⓪–⑨ + 顶部 Contract Ledger）

将现有 Dump 首页的八张静态模块卡升级为**带契约绿/黄/红灯徽章的十切口流水线（⓪–⑨）**：
- 顶部 **Contract Ledger** 一眼告知“本次 update 的 21 项过程契约是否全部通过”；
- 每张切口卡展示 4–6 个核心诊断标量 + 微型分布图 + 切口专属判定灯 + `open cut →` 进入对应切口工具页；
- 底部固定 **共享样本游标栏（Sample / Slice / Episode / Frame）**：在任意切口点选一个异常样本（如切口 ⑤ 中 TD Error 最大的样本，或切口 ⑦ 中负梯度投影最大的样本），游标跨所有切口保持——切到切口 ① 看它的原始 reward 和画面，切到切口 ④ 看它的 Target 三段分解，切到切口 ⑥ 看它的 $U(s)$ 是否跌破 `u_floor`。

---

## 7. 分阶段实施路线图（Work Packages）

为保证每一步都扎实可验，拆分为四个工作包：

### WP1：契约审计引擎（Contract Ledger）+ 切口 ①③④⑤⑥ 数学闭合（无重计算）
1. 修复 `exp_sac_standup.py` / `exp_sac_balance.py` 中 `versions.regularizer_mode` 硬编码为 `"entropy"` 的失真问题，使其动态反映实验真实配置；
2. 在 `analysis.py` 实现 `verify_dump_contracts(dump_dir)`（21 项断言中的静态/前向项）与增强版切口分析函数：
   - 切口 ⓪：`rollout_exploration`（基于 `actions - policy_action` 的即时 $\varepsilon$ 分布与饱和率）；
   - 切口 ①：`slice_verification`（基于 `task_facts`/`reward_features` 逐行复算 `rewards` 与 `actor_weight`）；
   - 切口 ③：`candidate_twin_analysis`（Twin $(Q_1, Q_2)$ 散点、twin gap、联合 vs 独立 min 对比）；
   - 切口 ④：反事实 Target 重算（支持自定义 $\gamma, \text{scale}, \alpha, f, \lambda$）+ $U(s')$ 分布与越界率；
   - 切口 ⑤：Critic Explained Variance ($\text{EV}_c$) + $Q_{\text{pred}}$ vs $y$ 散点 + TD 集中度；
   - 切口 ⑥：Actor $U(s)$ 显式分布 + `u_floor` 越界率 + 门控权重链；
   - 切口 ⑧：Polyak 软更新精确恒等式核验 + 分层相对步长；
   - 切口 ⑨：逐样本 Pre→Post 漂移数组 + 单步 TD/Actor Loss 下降核验。
3. CLI 暴露 `debugkit verify <dump>` 与各切口子命令，编写单测并在既有真实 dump（`shannon` 与 `u_floor`）上跑通验证。

### WP2：切口 ⑦ 逐样本梯度信号（SAC GradSig）+ 通道梯度冲突矩阵
1. 在 `analysis.py` 实现 `per_sample_gradsig(dump_dir)`：
   - 利用 $B=256$ 规模优势，精确计算全 batch 逐样本 Actor 梯度与各通道 Critic 梯度的 $\|g_i\|, p_i, \cos(g_i, G)$；
   - 验证 $\text{mean}(p_i) \equiv \|G\|_2$ 恒等式；
   - 计算多通道 Actor 梯度余弦矩阵 $\cos(G^{\text{actor}}_{c_1}, G^{\text{actor}}_{c_2})$ 与各正则项（Q / Entropy / $U$-reg）梯度夹角；
2. 挂入单 Job 槽与缓存（`<dump_dir>/gradsig.json`），CLI 与 HTTP 端点同源暴露。

### WP3：切口 ⓪①⑨ 的回放与画面锚定（Episode 重建渲染 + 跨代 Delta）
1. Dump 捕获增强：`finish_critic_tick_dump` 时将当前轮行为策略蓝图快照（或索引）保留，确保采集策略不因 `prune_policy_exports` 丢失；
2. 实现 `reconstruct_episode(dump_dir, sample_id)`：由样本 `metadata` 中的 `(episode_seed, collection_round, agent_id)` 重新运行单局仿真 + 渲染 PNG 帧 + 产出整条 Episode 的 `SACTransitionSlice` 与逐帧验证报告；
3. 实现 `cross_gen_delta(dump_dir)`：遍历 run 的 `checkpoints/`，在当前 dump batch 的 `obs` 上重放历代策略的 $\pi_{\text{det}}(s), U(s), Q(s, \pi_{\text{det}}(s))$，量化跨代收敛与振荡。

### WP4：Viewer 前端十切口工作台与视频编辑器集成 + 实战复盘
1. 升级 `sac_viewer.html` 与 `debugserver.py`：
   - Dump 首页顶部 Contract Ledger + ⓪–⑨ 十张切口卡；
   - 各切口专属工具页（含 Target 反事实滑杆、GradSig 热力图与通道冲突表、Pre/Post 散点、Episode 视频编辑器舞台）；
2. **用新工具回头审视我们的 4 个 Base 实验**：在 `standup` 和 `balance`（`shannon` vs `u_floor`）的真实 dump 上跑完整十切口诊断，用实打实的切口证据（而非经验叙事）回答之前留下的核心问题（例如：`u_floor` 在突破期到底有多少比例的样本触发了底线惩罚？`r_fall` 和 `r_cross` 在 Actor 上的梯度夹角是多少？）。
