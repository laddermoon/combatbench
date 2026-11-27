# 梯度通道绕行记录：从 RewardEstimator 到 DPG 的二分定理

> 记录一次完整的设计探索循环：每一个转向的动机、每一条路的死法、
> 以及最终收敛出的理论图景。目的不是记录失败——是记录**为什么这些
> 看起来成立的方案在数学上必然坍缩**，以及它们各自映射到文献中的
> 哪个已知位置。

背景：Dyna-BPTT 主实验（rep-space WM + margin 奖励）在跑，
`core_main_bal2`（u1000+, margin 0.03→0.09, eval_surv→0.88）。
本文档记录的是并行进行的架构层面探索：能否绕开动力学展开，
用"策略无关的奖励估计器"直接当梯度通道。结论：**不能，且原因
是可证明的**。但绕行本身产出了清晰的 dichotomy 和相关文献地图。

---

## 一、绕行起点：为什么想绕开 WM

经验驱动（均有实测数据支撑，见 metrics.jsonl / 实验记录）：

1. **WM 拟合存在混沌地板**：关节速度通道对接触微冲量的响应
   rmse≈std≈1.0，离线地板实验（/tmp/fit_floor.py，256 分支 lr=1e-3
   60 epochs）证明 1% nRMSE 物理不可达 → 判据改 Bryson 归一化
2. **φ 奖励驱动病态策略**：φ = uprightness×z 奖励"瞬时直立"，
   最优解是挺胸踮脚=零稳定裕度 → "主动蹲"失败模式（/tmp/policy_u149.mp4）
3. **前沿冻结**：严格 s₁ 准入 + 确定性 rollout → StartSet φ 分布
   恒冻结在 0.96-0.99，分支永不死（branch_surv=1.00）→ WM/策略
   对下沉走廊全盲 → eval_surv 平台期

**动机**：WM 侧同时承压于"拟合地板"和"覆盖缺口"。如果能扔掉
动力学、直接学一个"策略无关的可微奖励估计器"当梯度通道，
两个问题一起消失——不需要拟合混沌转移，想象链不再存在，
覆盖问题变形为"哪些状态上有 label"。

## 二、候选方案与各自的死法

### 方案 v1：RewardEstimator（即时奖励估计器）

**形式化**：`R̂(s,a) ← 回归 r_true(s,a,s')`，冻结后在 S5 对 D 中
真实轨迹逐状态评估 `R̂(s_t, π_θ(s_t))`，loss = -Σ γᵗ·r̂_t 反传。

**卖点**：r 不依赖 π（真·策略无关），训练/部署输入分布一致，
评估可用排序一致性代替精度。

**死法——固定状态坍缩定理**：轨迹状态 s_t 是常量（来自数据），
`π(s_{t+k})` 的梯度只推它自己状态的即时 r̂，`π(s_t)` 对 `s_{t+1}`
没有因果链。GAE/求和只是给每个贪心项加了正权重：

```
∇_θ L = Σ_k w_k · ∇_aR̂(s_k,·)·∇_θπ(s_k)   ← 逐状态贪心项加权
```

外观像多步回报，数学上就是逐状态 H=1 贪心。深蹲反例原样适用
（蹲下即时 φ 最高）。**多步信用是幻觉。**

**文献位置**：SVG(0) 退化端点（Heess et al. 2015 的 taxonomy
里这就是 DPG/greedy 端）。政策无关奖励估计器本身是真实部件
（SuperMimic task_heads、GAIL discriminator），但在所有能跑
的版本里梯度都还要穿动力学或穿价值，不能只穿奖励。

### 方案 v2：RTG + Critic 尾巴的可微 Loss

**形式化**：`Ĝ_t = Σ_k γ^k·r̂_{t+k}(θ) + γ^H·Q̂(s_H, π_θ(s_H))`，
每项都带 θ——"整体上对 Policy 可导"。

**死法——可导 ≠ 信用分配对**：可微性成立（每项经 π 带 θ），
但展开后仍是逐状态独立项的加权和。`Q̂` 尾巴只给 H 状态自己
的梯度，中间 H-1 步因果断着。**同一个公式放对位置才对：**

```
放 S3 当 label（实测 reward、记录动作——全常量）：
    Q̂(s_t,a_t) ← Σ γ^k·r_true(s_{t+k}) + γ^H·Q̂(s_H,a_H)  = n-step Q 学习 ✓
放 S5 当 loss（反事实 reward、π 的动作）：
    坍缩成贪心项之和 ✗
```

**教训**：RTG+bootstrap 属于 label 侧，不属于 loss 侧。

### 方案 v3：那就拟合 Q —— 发现回到了 DPG

**形式化**：`Q̂(s,a) ← RTG label`；`Loss = -Σ_t Q̂(s_t, π_θ(s_t))`；
`∇_θL = ∇_aQ̂·∇_θπ`。

**判决**：机制上这就是 **DPG 定理**（Silver 2014）= DDPG/SAC 的
actor update。不是偏方，是连续控制当前主干路线（TD3/SAC 即
MuJoCo SOTA）。区别只剩 label 来源：实测 n-step return vs
TD bootstrap = "n-step DDPG"变体。

**代价**：Q^π 语义绑死在 π 上——策略每更新，旧数据的 return
语义漂移，critic 要永远追（off-policy 税）。

### 方案 v4：让 Critic 变成 Policy 的函数 —— V(s;θ)

**动机**：既然问题是"标量场绑死策略"，那把策略显式放进
估计器的输入里——`V(s;θ)` 或 `V(s; policy-embedding)`，
对 θ 求导量的就是"策略动一点、这个状态的价值变多少"。

**理论观察**：`∇_θ V^{π_θ}(s)` 按定义就是策略梯度——如果这个
回归能学出来，就是"回归出来的策略梯度"。你的 replay buffer
恰好有拟合它所需的全部数据（每条轨迹都是某历史策略产生的）。

**死法（半死）**：RL 的困难没有消失，是整体搬进估计器——
V 作为 θ 的函数要把"策略→回报"全映射学进回归。且它给的是
**策略级信用**（"什么策略配置值钱"）不是**动作级因果信用**
（"这个动作导致什么"）。meta-RL 级难度，不是今天能写的代码。

**文献位置**：EPG（Houthooft 2018）/ LPG（Oh 2020）meta-learn
loss 场；UVFA 条件化 value；ES/NES/PGPE 是其零阶对应物。

## 三、收敛出的二分定理

绕完一圈后浮现的结构性事实：

```
多步信用必然包含"未来的动作"，而未来动作是 π 的函数。

  策略无关场（R̂ 即时奖励）→ 只能携带"这一步"→ 无 bootstrap → 贪心 → 不成立
  策略绑定场（Q̂/V̂）      → 能携带"之后"   → 绑死 π      → 退回 actor-critic

未来信息只有两条通道：
  通道A: 动力学模型 —— 把 a 对 s_{t+1..t+H} 的影响机械展开（BPTT）
  通道B: 价值函数   —— 把后果压缩记忆进 Q(s,a)          （DPG/TD）
没有第三条通道。
```

**推论（对本项目的具体判决）**：

1. R̂ 在本设置里彻底失业——我们的奖励本来就是解析可微的
   （margin/φ 是 rep 的解析函数），R̂ 唯一存在理由
   （提供 ∇_a 通道）已证伪。
2. Q 的策略绑定是**税不是死刑**——SAC/DDPG 一直在交
   （critic 持续重训）。半例外：`Q*`（max-bootstrap，理论上
   策略无关，实践不稳）；真例外：policy-conditioned critic。
3. 活路恰为两条通道，且**不互斥**——SHAC/APG 的最优形态：
   `J = Σ_{t≤H} γᵗ r(s_t,a_t) + γ^H·V(s_H)`（短程 BPTT +
   critic 尾部）。官方 SuperMimic 实测就是这个结构
   （MotionCritic 只做 tail bootstrap，imagination.py:157-165）。

## 四、"WM 梯度效率是 PPO 的 100 倍"的准确出处

出自 SHAC/APG 系列（Xu et al. 2021-2022）及 SuperMimic 引用链。
量的不是"可微奖励 vs 不可微"，是**轨迹级解析梯度 vs 逐点
score-function 梯度**的样本效率：

- PPO：`A·∇logπ`，每样本信息带宽 = 1 个标量
- BPTT：梯度沿展开轨迹逐维回流，动作影响在 H 步的每个
  状态维度上都留梯度信号

**增益来自"展开"，不是"奖励可微"。** R̂ 只让梯度有东西可传；
让它值 100 倍的是 a_t→s_{t+1}→…→s_{t+H} 这条链本身。

## 五、相关文献地图（按主题，供后续阅读）

### 梯度通道 taxonomy（核心必读）
- **Heess et al., 2015, "Learning Continuous Control Policies by
  Stochastic Value Gradients" (SVG)**——把学习奖励模型+学习动力学
  +horizon-k展开+critic尾参数化为 SVG(k)；k=0≈DDPG，k=∞≈纯BPTT。
  本文档所有讨论的文献原点。
- **Silver et al., 2014, DPG theorem**；**Lillicrap et al., 2015,
  DDPG**；**Fujimoto et al., 2018, TD3**；**Haarnoja et al., 2018,
  SAC**——通道B的完整谱系与三个标准补丁（双Q/熵/target net，
  分别治套利/单点查询/bootstrap漂移）。

### 可微/解析策略梯度（通道A的工作）
- **Xu et al., 2021, "Accelerated Policy Learning with Parallel
  Differentiable Simulation" (APG)** 及 **2022 SHAC**——短程BPTT
  +critic尾的工程形态；"梯度效率"数字的出处。
- **Degraeve et al. / PILCO**（解析 GP 动力学上的策略梯度）；
  **Ha & Schmidhuber 2018 World Models**；**DreamerV1-V3**
  （Hafner）——想象展开+critic的另一条实现线。
- **SuperMimic 官方仓库**（本地 /tmp/supermimic_mjlab）——
  task_heads = 策略无关奖励头（活的R̂），MotionCritic = 尾部
  bootstrap，sensor noise 在观测端不在动作端。

### "把标量场当梯度通道"的元学习线（v4 的前身）
- **Houthooft et al., 2018, "Evolved Policy Gradient" (EPG)**——
  meta-learn 一个loss场使其梯度产生策略改进。
- **Oh et al., 2020, "Discovering RL Algorithms" (LPG)**——
  可微meta学习整个更新规则。
- **Schaul et al., 2015, UVFA**——条件化 value function。
- ES/NES/PGPE——"策略→回报"响应面的零阶梯度估计。

### 奖励塑形与贪心何时合法
- **Ng et al., 1999, potential-based reward shaping**——若奖励
  本身是势函数（margin 奖励正是手工势函数），贪心攀升≈价值
  攀升，v1 方案在此特例下部分成立。

### Q label 的构造（S3 侧）
- **Schulman et al., 2016, GAE**——λ-return 的位置在 label
  构造不在可微 loss。
- **Feinberg et al., 2018, MVE**；**MBPO/MOPO**——模型生成数据
  喂 critic 的谱系。
- Offline RL（BCQ, TD3+BC, IQL, n-step return critics）——
  "固定数据集上按学习场改进策略"的完整文献。

## 六、当前实验决策

通道A（BPTT through rep-WM + margin 解析奖励）继续跑
`core_main_bal2` 至收敛拿结论；通道B（Q̂-RTG-DPG，实测return
label + StartSet数据引擎）作为对照实验，回答"展开贡献了多少"——
若 DPG 版同达，BPTT 是过度设计；若 DPG 卡而 BPTT 突破，
展开的增量被实证。两边共享 S2 数据引擎，边际成本主要在
Q̂ 训练循环 + label 构造。
