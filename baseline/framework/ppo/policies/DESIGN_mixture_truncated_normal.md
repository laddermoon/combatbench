# Design: MixtureTruncatedNormalPolicy — 混合截断正态策略

## 1. 状态、目的与边界

本文是已确认的第一版数学设计，**不是已实现或已通过训练验收的声明**。
目标类名为 `MixtureTruncatedNormalPolicy`，拟放在
`mixture_truncated_normal_mlp.py`，与现有两种截断正态策略并列。

目标：在一个状态下表达多组可选的完整动作模式，同时保持动作在
`[-1, 1]^D` 内、PPO 评分准确、训练侧 uncertainty 可微且无动作依赖。

核心决策：

| 项目 | 第一版定义 |
|---|---|
| 分布 | K 个对角截断正态的联合混合，默认 K=3 |
| 分量选择 | 一整组 D 维动作共享一个分量编号 |
| explore_factor | 仅将各分量的 σ 乘以 `3^e`，位置与权重不变 |
| log_prob | 先在分量内对维度求和，再对分量做 logsumexp |
| uncertainty | 各动作边缘分布的 Rényi-2 有效宽度，除以区间宽度后取维度平均 |
| U 的分布 | 原始策略分布，不含 explore_factor |
| U 的计算 | 分量两两密度乘积的闭式积分；不采样、不网格寻峰 |
| act | 最大权重分量的中心向量，不对模式求加权平均 |
| 初始化 | 权重均匀、σ 初始为 e^-1、位置分支打破分量对称 |
| 接口 | 现有 Policy / StochasticPolicy / TrainablePolicy，不改 PPO 框架 |

非目标：本轮不证明优于单分量策略，不引入相关协方差、动态 K、权重探索温度、
分量使用率正则、时序模式保持或完整联合分布 uncertainty。
旧 `todo/DESIGN_mixture_gaussian.md` 是 tanh-squashed / entropy 旧接口设计，
不是本策略的实现规范；不能照搬其 entropy regularizer、业务 σ bounds 或温度状态。

## 2. 接口约束与符号

接口来源：`../stochastic_policy.py`、`../experiment.py` 中的
`StochasticPolicy`、`ActorEval` 和 `TrainablePolicy`。

- `sample(obs, *, explore_factor=0.0, want_extra=False)`：训练 rollout 采样。
- `act(obs, *, want_extra=False)`：确定性部署 / 评估。
- `evaluate_actions(obs, actions, explore_factor, *, want_stats=False)`：
  返回 `ActorEval(log_prob, uncertainty, stats)`。
- `log_prob` 与 `uncertainty` 均为 `(B,)`，训练中必须可微。
- U 只依赖 obs 和参数，范围 `[0,1]`；不能使用传入的 actions 或被抽中的分量。
- explore_factor 是每步 / 每帧数据，不是可变策略状态。
- 导出策略同时实现 act 与 sample，训练侧与导出侧分布一致。
- 已移除的 `exploration_grad_diagnostics` hook 不得重新引入。

记 B 为 batch 大小、D 为动作维数（humanoid21 为 21）、K 为固定分量数。
以下数学式通常省略状态 s 和 batch 下标。φ、Φ 分别为标准正态 PDF、CDF。
动作区间固定为 `[-1,1]`，宽度 W=2；π_k 表示混合权重，常数 π 表示圆周率。

## 3. 网络与分布定义

### 3.1 参数化

共享 trunk 沿用两层 MLP：

```text
trunk: Linear(obs_dim, hidden_dim) -> Tanh
       -> Linear(hidden_dim, hidden_dim) -> Tanh
head:  Linear(hidden_dim, K + 2*K*D)

logits:       (B, K)
raw_mean:     (B, K, D)
raw_log_std:  (B, K, D)

log_pi = log_softmax(logits, dim=-1)
pi     = exp(log_pi)
mu     = tanh(raw_mean)
sigma  = exp(clamp(raw_log_std, -20, 20))
```

head 输出布局固定为 `logits | flatten(raw_mean) | flatten(raw_log_std)`，
两个展平块均按 component-major `(K,D)` 排列，训练 / 导出必须相同。
K 必须为正整数，是架构参数，不在训练中变化。

不引入旧 todo 的 `[-4,0]` log_std 业务边界。
`[-20,20]` 只限制数值尺度，**不保证分布计算在整个范围内已经准确**，见 §10。
σ 是截断前正态的尺度，不是截断后动作的标准差；μ 是位置参数和该分量的众数，
通常不等于截断后期望。μ 数学上在 `(-1,1)`，浮点 tanh 可能取到端点。

### 3.2 一维截断正态

对 x∈[-1,1]：

\[
Z(\mu,\sigma)=\Phi\!\left(\frac{1-\mu}{\sigma}\right)
-\Phi\!\left(\frac{-1-\mu}{\sigma}\right),
\qquad
t(x;\mu,\sigma)=\frac{\phi((x-\mu)/\sigma)}{\sigma Z(\mu,\sigma)}.
\]

区间外密度为零，每个 t 在动作区间上的积分为 1。
这是真正的截断并重新归一化，**不是对普通高斯样本做 clip**。

### 3.3 联合混合

\[
\boxed{
p(\mathbf a\mid s)=\sum_{k=1}^{K}\pi_k(s)
\prod_{d=1}^{D}t(a_d;\mu_{kd}(s),\sigma_{kd}(s))
}
\]

一次采样只选一个 k，用该 k 生成整组动作。不能对每个维度独立选 k；后者对应
`product_d(sum_k(pi_k*t_kd))`，不是这里的 `sum_k(pi_k*product_d(t_kd))`。

分量内部条件独立；混合后通常不独立。共享分量选择能表达跨关节的模式相关性，
但不等于在每个分量内部建模协方差。

## 4. explore_factor 的精确定义

令 e∈[-1,1]：

\[
c(e)=\exp(e\ln3)=3^e,
\qquad \sigma^{(e)}_{kd}=c(e)\sigma_{kd},
\qquad \mu^{(e)}_{kd}=\mu_{kd},\quad \pi^{(e)}_k=\pi_k.
\]

实际 rollout / log_prob 分布：

\[
\boxed{
p_e(\mathbf a\mid s)=\sum_k\pi_k\prod_d
 t(a_d;\mu_{kd},3^e\sigma_{kd})
}
\]

- e=-1：σ 除以 3；e=0：原始策略；e=+1：σ 乘以 3。
- 缩放施加在 policy σ 之后，不能再按业务 bounds 截回去吞掉探索信号。
- batch e 为 `(B,)`，广播成 `(B,1,1)`，禁止误用 `(B,1)` 广播。
- 同步重算有效 σ 对应的 Z，不能只改 Normal 项而沿用原 Z。
- 不改变 logits / mixture weights，不改变 μ。

语义是“分量内部的尺度”，不是一个同时控制所有随机性的全局温度。
模式选择仍按 π 随机：e=-1 不是确定性，e=+1 也不保证混合分布的每一种
uncertainty 指标单调上升（扩宽会改变分量重叠）。只有 K=1 时直接退化成
现有 state-dependent 策略的同一探索映射。

第一版不调权重温度，以免一个旋钮同时改变模式偏好与模式内部噪声。
若未来需要模式选择探索，应独立设计，不能悄悄改变 e 的本版含义。

## 5. 采样、评分与 PPO 一致性

### 5.1 sample

先采样 `k ~ Categorical(pi)`，再对选定分量的各维独立作 inverse-CDF：

\[
\alpha_d=\frac{-1-\mu_{kd}}{\sigma^{(e)}_{kd}},\quad
\beta_d=\frac{1-\mu_{kd}}{\sigma^{(e)}_{kd}},\quad u_d\sim U(0,1),
\]
\[
a_d=\mu_{kd}+\sigma^{(e)}_{kd}\Phi^{-1}
\left(\Phi(\alpha_d)+u_d[\Phi(\beta_d)-\Phi(\alpha_d)]\right).
\]

这是实数域定义；浮点实现要求见 §10。禁止用“各分量分别采样后加权平均”代替。
PPO 不需要穿过离散分量选择求 pathwise gradient；权重通过完整 log_prob 学习。

### 5.2 log_prob

令 `Z_kd_e = Z(mu_kd, sigma_kd_e)`，对所有分量评分：

\[
L_k=\sum_d\left[-\tfrac12
\left(\frac{a_d-\mu_{kd}}{\sigma^{(e)}_{kd}}\right)^2
-\log\sigma^{(e)}_{kd}-\tfrac12\log(2\pi)-\log Z_{kd}^{(e)}\right],
\]
\[
\boxed{\log p_e(\mathbf a\mid s)=
\operatorname{logsumexp}_k(\log\pi_k+L_k).}
\]

张量路径：逐维 log density `(B,K,D)` -> sum D 得 `(B,K)` -> 加 log_pi
-> logsumexp K 得 `(B,)`。直接使用 log_softmax / logsumexp，不先算微小概率再取 log。

`sample(..., want_extra=True)` 的 `extra['log_prob']` 必须为上述完整混合密度。
不能返回选中分量的条件密度，也不能返回带隐藏标签的联合密度 `log_pi_k + L_k`。
`evaluate_actions` 不需要知道当时抽中了哪个 k；使用记录的每帧 e 即可重建分布。

合法支持外的动作密度为零（log_prob=-∞）；不能把任意越界动作静默 clip 后声称
这是原动作的密度。对合法浮点动作的边界约定须在采样、评分和导出侧一致。

## 6. uncertainty：归一化边缘有效宽度

### 6.1 定义

先取**中性策略**的第 d 维边缘密度：

\[
p_d(x\mid s)=\sum_k\pi_k(s)t(x;\mu_{kd}(s),\sigma_{kd}(s)).
\]

定义：

\[
R_d(s)=\int_{-1}^{1}p_d(x\mid s)^2\,dx,
\qquad U_d(s)=\frac1{2R_d(s)},
\qquad \boxed{U(s)=\frac1D\sum_dU_d(s).}
\]

Rényi-2 differential entropy 为 `H2=-log(R_d)`；`exp(H2)=1/R_d` 是有效支持长度。
除以动作区间长度 2 后得到无量纲有效宽度。
**必须逐维取倒数再平均，不能先把 R_d 平均再取倒数。**

### 6.2 范围与直觉

Cauchy–Schwarz 给出：

\[
1=\left(\int_{-1}^{1}p_d(x)\,dx\right)^2
\le2\int_{-1}^{1}p_d(x)^2\,dx=2R_d,
\]

所以 `0<U_d<=1`。均匀边缘密度 1/2 对应 U_d=1；趋于尖峰时 U_d→0。
对于固定有限 K，多处离散尖峰即使彼此很远，在宽度趋零时 U 仍趋零。
有限 σ 的截断正态通常只能逼近均匀极限，不能承诺任意 floor（特别是 1）可精确满足。

两个相同分量拆分权重后密度不变，U 不变；K 本身不会获得奖励。
两个等权、同宽且几乎不重叠的窄峰，若单峰平方积分为 I，则混合 R≈I/2，
U≈单峰的 2 倍。峰已分离后继续拉远不会持续显著增加 U：它度量有效覆盖宽度，
不是模式距离，也不是动作方差。

### 6.3 action-independence、可微性与局限

U 只使用 obs 产生的 π、μ、σ，不读取传入 action，不读取随机采样结果，不使用 e。
它对 logits、μ、σ 可微，但不要求所有参数在所有对称情形都有非零梯度。
特别是分量完全相同时改变 π 不改变密度，logits 的 U 梯度为零是正确现象。

U 仅度量边缘密度，不描述完整 D 维联合分布的相关结构。
同样边缘、不同关节联合模式的两个策略可以具有相同 U。
U=1 表示所有边缘均匀，不能一般性地解释成联合分布完全均匀 / 独立。
此外，维度平均不保证每一个关节都达到同样的探索宽度。

U 对 μ、σ、π 都有梯度，因此 floor loss 可以扩宽分量、移动位置或重新分配权重。
不保证 σ、分量间距或权重熵在训练中单调增加，也不保证每个分量都被使用。

### 6.4 为什么不用另外三个定义

- `sum_k(pi_k*U_k)`：仅度量分量内部宽度，忽略混合重叠，不作主 U。
- `1/(2*max_k p_d(mu_kd))`：真实混合峰值可能在分量中心之间，只是峰值近似。
- rollout actions 上的 `-log_prob.mean()`：不是本方案的无动作依赖解析 uncertainty。

本版不引入权重熵加成，也不把 categorical entropy 与连续宽度任意相加。

## 7. 闭式计算：分量两两重叠积分

展开平方：

\[
R_d=\sum_{i=1}^K\sum_{j=1}^K\pi_i\pi_j I_{ij,d},
\qquad I_{ij,d}=\int_{-1}^{1}t_{id}(x)t_{jd}(x)\,dx.
\]

下面省略 d。令 `v_i=sigma_i²`、`v_j=sigma_j²`：

\[
v_* = \frac{v_i v_j}{v_i+v_j},\qquad
m_* = \frac{\mu_i v_j+\mu_j v_i}{v_i+v_j},
\]
\[
C_{ij}=\frac{\exp[-(\mu_i-\mu_j)^2/(2(v_i+v_j))]}
{\sqrt{2\pi(v_i+v_j)}},
\qquad Z_* = Z(m_*,\sqrt{v_*}).
\]

使用高斯乘积恒等式：

\[
N(x;\mu_i,v_i)N(x;\mu_j,v_j)=C_{ij}N(x;m_*,v_*),
\]

这里 `N(x; μ, v)` 表示均值为 μ、方差为 v 的普通高斯密度。积分得：

\[
\boxed{I_{ij}=\frac{C_{ij}Z_*}{Z_iZ_j}.}
\]

因此不需要采样估计、数值积分或全局寻峰来计算训练侧 U。
计算同时包含 i=j 的自重叠与 i≠j 的交叉项，不能遗漏交叉项。
如只算上三角，非对角项必须乘 2。

实现使用 log 空间：

```text
log_I_ij = log_C_ij + log_Z_star - log_Z_i - log_Z_j
log_R_d  = logsumexp_ij(log_pi_i + log_pi_j + log_I_ij)
U_d      = exp(-log(2) - log_R_d)
U        = mean_d(U_d)
```

pairwise 张量可组织为 `(B,K,K,D)`，对两个 K 轴归约，保留 B、D。
复杂度为 O(B*K²*D)。默认 K=3 只有 9 个有序分量对，但全 rollout buffer
可能有约 20 万帧，不能只测 minibatch：须评估临时张量和 autograd 峰值内存。
必要时策略内部沿 B 分块计算，保持同样结果和梯度，不改 trainer 接口。

## 8. 与旧 uncertainty 的关系和 floor 语义

旧策略逐维使用：

\[
U_{\mathrm{peak}}=\frac1{2\max_xp_d(x)}.
\]

本版使用 `U_L2=1/(2*integral(p_d²))`。由于
`integral(p_d²)<=max(p_d)*integral(p_d)=max(p_d)`，有：

\[
U_{\mathrm{L2}}\ge U_{\mathrm{peak}}.
\]

对远离边界的窄单高斯：

\[
U_{\mathrm{peak}}\approx\sigma\sqrt{\pi/2},\qquad
U_{\mathrm{L2}}\approx\sigma\sqrt\pi.
\]

二者相差约 √2；这不是整个参数区间的固定换算系数。
因此 K=1 的验收应拆成：

1. 分布 / log_prob / 分量采样 / act：退化为同参数的 state-dependent 截断正态。
2. U：退化为**单分量 L2 公式**，不等于旧 peak U，不能要求与旧 U bit-identical。

当前 trainer 实际使用逐帧平方 hinge：

\[
L_{\mathrm{floor}}=c\,\operatorname{mean}_b
\left[w_b\max(f-U(s_b),0)^2\right],
\]

其中 f 为 floor、c 为 coef、w_b 为框架 floor_weight。
是先逐帧惩罚再平均，不是 `relu(f-mean(U))²`；平均 U 高于 f 也不能证明所有帧无惩罚。
floor 是软约束，不保证 U>=f，也不保证防止分量塌缩。

旧实验的 floor=0.4 不可无解释地当成相同强度的对照。
本策略相对旧策略同时改变“混合表达力”和“U 度量”；严格能力 A/B 应统一 U
（例如另设单分量 L2 对照）或单独控制正则差异。本设计不顺带修改旧策略的 U。

### 8.1 实测数值（standup_floor04_mixture run）

`train_standup_floor04_mixture_ppo_20260922_235158` 前 40 updates 观测：

- 初始化 σ≡e⁻¹≈0.368 时 U_L2≈0.65，训练中 σ 分化到 [0.16, 0.76]
  后 U_L2≈0.63；`floor_loss` 恒为 0——U 从未跌破 floor=0.4。
- 等效触发阈值换算：窄单高斯下 floor=0.4 对 L2 指标要求
  π加权 σ 缩到 ~0.23 以下才激活；旧 peak 指标在 σ≈0.32 即触发。
  即同一 floor 值对本策略是更弱的探索下限，σ 可以更自由地收窄。
- 如需与基线等强度对照（在 σ≈0.32 触发），floor 应设
  ~0.32·√π≈0.57 而非 0.4；该换算只对窄单峰近似成立，
  混合 / 边界效应下会漂移。
- 基线 run `train_standup_floor04_ppo_20260920_164819` 中 floor
  是持续激活的（U_peak≈0.33<0.4）；本 run 中它完全闲置——
  比较两条曲线时这是一个独立的混杂变量。

## 9. 确定性动作与初始化

### 9.1 act

\[
k^*=\arg\max_k\pi_k(s),\qquad \operatorname{act}(s)=\boldsymbol\mu_{k^*}(s).
\]

权重并列时选择最小分量编号，训练侧 / 导出侧一致。可直接对 logits 做 argmax。
这是“最大权重分量的中心”规则，不是完整混合分布的全局众数，也不是混合期望。
禁止默认返回分量中心的加权平均，以免不同动作模式的平均落在低密度区域。

此选择在模式权重交叉时可能不连续，第一版不增加记忆、迟滞或动作平滑。
评估时 act 不采样；不能将评估中的动作波动直接解释成 sample 的随机噪声。

### 9.2 打破训练初始化对称性

- logits 分支 weight/bias 置零，初始 π_k=1/K。
- log_std 分支 weight 置零、bias=-1，初始各 σ=e^-1。
- 位置分支在一个公共初始映射上加入分量不同的小幅扰动，使分量不完全相同。
  扰动分布与幅度属于实现时须明确记录并测试的初始化超参数，不能声称已被实验验证。
- K=1 不需要分量对称破缺。

若全部分量完全相同、权重相同，完整混合 log_prob 的对称梯度可能使分量一直共同移动。
离散采样本身不是可靠的破对称机制，因为 PPO 评分会对所有分量求和。
“复制相同分量”仅用于密度不变性测试，不作默认训练初始化。

## 10. 数值正确性要求

### 10.1 Z 的稳定形式

由于 μ∈[-1,1]，可用两个非负 erf 值相加：

\[
Z(\mu,\sigma)=\tfrac12\left[
\operatorname{erf}\!\left(\frac{1-\mu}{\sqrt2\sigma}\right)
+\operatorname{erf}\!\left(\frac{1+\mu}{\sqrt2\sigma}\right)\right].
\]

大 σ 时两个原始 CDF 都接近 0.5，直接相减会丢失有效位；此形式避免该消减。
m_* 是区间内两位置的凸组合，Z_* 同样可用此式。
不能简单沿用 `Z.clamp(min=1e-8)`：合法 Z 可更小，硬抬 Z 会改变归一化密度。
log 空间和最终 U clamp 也不能自动修复错误的 Z。

### 10.2 inverse-CDF 的浮点实现

§5 是数学定义，不要求直接以两个 CDF 相减实现。一个等价的中心化形式为：

\[
A=\operatorname{erf}(\alpha/\sqrt2),\quad
B=\operatorname{erf}(\beta/\sqrt2),\quad
q=(1-u)A+uB,\quad
a=\mu+\sqrt2\sigma\operatorname{erf}^{-1}(q).
\]

它避免大 σ 时向接近 0.5 的数叠加微小 CDF 增量；但仍需处理 u 的机器端点、
erfinv 的 ±1、窄分布的位置相消及输出 dtype 舍入。
精度提升、边界处理及误差阈值必须在实现中明确并测试，不能以大范围动作 clip
制造边界堆积，再按连续 TN 评分。正常浮点舍入与人为改变分布必须区分。

### 10.3 验收原则

- 数值测试覆盖 raw_log_std 的安全范围及 e=±1 后的有效尺度，不只测常用 σ。
- 覆盖 μ 靠近两端、尺度极不相同、分量完全重合 / 分离、极端 logits。
- 训练默认 dtype、float64 参考、训练 / 导出实现分别验证。
- 必要时对分布数学使用更高精度，但必须保持 autograd 路径并评估成本。
- U 的最终 `[0,1]` clamp 仅容许抹平已量化的舍入误差；明显越界必须暴露。
- 验收不止“无 NaN”：还包括积分归一化、CDF / 样本统计、log_prob 和梯度误差。
- 极窄且远离的分量交叉积分可数值下溢到零；不能因此让可微主路径出现 NaN。
- 若不能通过声明范围内的精度验收，应明确调整数值方案或声明限制，不能静默
  降级成均匀采样、恢复旧业务 bounds 或以任意 epsilon 掩盖问题。

数值正确性优先于与旧实现偶然相同的浮点舍入；退化等价采用明确容差，
不能把“使用同一 RNG seed”直接等同于不同采样路径必须 bit-identical。

## 11. 计算路径、诊断与导出

`evaluate_actions` 一次 trunk+head 得到 log_pi、μ、policy σ：

```text
obs -> shared forward -> log_pi, mu, sigma_policy
  -> sigma_effective = sigma_policy * 3^e -> mixture log_prob(actions)
  -> pairwise integrals with sigma_policy -> U
  -> optional detached stats, only when want_stats=True
```

不重复网络前向，不在 minibatch 热路径 `.item()` 收集诊断。
推荐诊断及精确定义：

| 指标 | 定义 / 注意 |
|---|---|
| uncertainty | batch mean U，与 ActorEval 返回值相同 |
| std_mean | `mean_(b,d)(sum_k pi_bk*sigma_bkd)`，分量尺度加权均值，非动作标准差 |
| eff_std_mean | 同上但使用有效 σ |
| std_min / std_max | 所有 `(b,k,d)` 分量尺度的极值，含低权重分量，不作覆盖度结论 |
| mixture_weight_entropy | `mean_b(-sum_k pi_bk*log_pi_bk)`，范围 [0,log K] |
| effective_components | `mean_b(exp(-sum_k pi_bk*log_pi_bk))`，范围 [1,K] |
| max_component_weight | `mean_b(max_k pi_bk)` |
| component_weight_k | `mean_b(pi_bk)`，每个分量单独报告 |
| sigma_state_std | `mean_(k,d)(std_b(sigma_bkd, correction=0))`，只沿状态轴求标准差 |
| component_overlap | K>1 时 `mean_(b,d,i<j)(I_ij/sqrt(I_ii*I_jj))`，相同密度为 1 |

`effective_components` 只反映权重，不保证存在同样数量的不同模式；须结合 overlap。
K=1 时 overlap 指标省略，不能对空集合求均值。若记录采样使用率，应逐分量统计
频率并与 mean π_k 比较；不能再对 k 求平均，那恒为 1/K，没有诊断意义。
不能仅凭 σ 极值或聚合方差断言“困难状态探索更多”，需按状态条件分析。

导出沿用 `model.pt + policy.py + MANIFEST.json` 的自包含结构，拟用
`ExportedMixtureTruncNormPolicy`；policy.py 不依赖 baseline / envs import。
架构元数据必须含 obs_dim、action_dim、hidden_dim、K，并校验 format_version、
policy_class 和 strict state_dict；记录 `uncertainty_kind=marginal_renyi2_width_v1`
用于解释训练配置。导出推理无需实现训练 U，但必须保留一致的采样、评分、act、
分量布局及 reset(seed) 行为。类名 / 模板名在实现时固定后不可静默混用。

## 12. 实现验收清单

### 数学与分布

- [ ] 一维 / 二维混合归一化，与独立数值积分对照。
- [ ] 共享分量测试：两组分离的多维模式不产生逐关节独立选模式的错误联合分布。
- [ ] sample 返回值与 evaluate_actions 的完整混合 log_prob 一致。
- [ ] K=1 分布退化测试；相同分量拆分权重后密度与 L2 U 不变。
- [ ] 单一权重趋于 1 时退化到对应分量；分量置换不改变密度与 U。
- [ ] 采样经验 CDF / 矩与独立参考吻合，分量频率吻合 π；不能只检验两条代码路径彼此一致。
- [ ] e=-1/0/+1 尺度比例精确符合定义；每帧 e 的 batch 广播正确；π 与 μ 不变。

### uncertainty 与梯度

- [ ] I_ij 闭式值与高精度数值积分一致，包含 i=j、边界、不同尺度和重叠程度。
- [ ] U 的闭式值与一维混合密度平方积分一致，逐维聚合顺序正确。
- [ ] 同 obs 不同 actions / e 得到相同 U；无随机采样引入 U 方差。
- [ ] 窄峰极限趋零、宽分布趋一、复制分量不变、分离等权窄峰近似加倍。
- [ ] logits、μ、log_std 的 log_prob / U 梯度以非对称样例做有限差分或 gradcheck。
- [ ] 完全重合分量的 logits U 梯度为零；不强迫对称情形出现非零梯度。
- [ ] floor 对逐帧 U 计算，保留 π、μ、σ 的梯度，不将 U detach 后用于损失。

### 工程

- [ ] act 使用最大权重分量，确定性 tie-break 与导出一致。
- [ ] 不同 K、空配置错误、错误 K payload、缺失 / 多余 state_dict keys 的 strict 测试。
- [ ] 自包含导出能在无 repo PYTHONPATH 子进程加载；act / sample / reset parity。
- [ ] 完整数值范围和边界误差验收，不以 clamp 后的表面有界代替正确性。
- [ ] minibatch 与整 rollout buffer 的速度 / 内存测试，必要时分块结果与梯度一致。
- [ ] 默认初始化确实破对称，σ 初始 e^-1，初始化随机性可复现。
- [ ] 端到端 smoke 后验证真实 PPO 更新、有限梯度、评估 / 导出 / 恢复链路。

本轮文档不启动训练。后续“可用性验收”与“是否更优 / 有何特性”的实验分开：
前者验证接口与数值实现，后者需要一致的 U、受控初始化 / 容量和多 seed 证据。
不得把单次训练或通过有限测试称为所有状态下实现正确的数学证明。

## 13. 实现记录（已落地）

实现于 `mixture_truncated_normal_mlp.py`，导出模板
`_export_template_mixture_truncnorm.py`（`ExportedMixtureTruncNormPolicy`），
blueprint `init_policy_mixture_truncated_normal.yaml`，测试
`test_mixture_truncated_normal.py`（42 项），对照实验
`exp_standup_floor04_mixture.py`。

实现细节，相对于前文规范：

- **Head 布局**：`[logits (K) | raw_mean (K·D) | raw_log_std (K·D)]`，
  分量优先（component-major），单个 `nn.Linear`。
- **初始化参数**：`num_components`（默认 3）、`component_init_noise`
  （默认 0.02）为构造参数；mean 块 = 分量 0 默认初始化 + 各分量独立
  N(0, noise) 扰动，σ 块 w=0/b=−1，logits 全零。
- **Z 的浮点形式**：按 §10.1 使用 erf 求和形式
  `Z = 0.5·[erf((1−μ)/√2σ) + erf((1+μ)/√2σ)]`，μ∈[−1,1] 保证两项非负，
  无大 σ 相消；`clamp_min(tiny)` 仅为下溢兜底。
- **采样**：`torch.multinomial(exp(log_pi))` 选分量后逐维 erf 空间
  inverse-CDF。**K=1 时跳过多项分布采样**（分量恒为 0，不消耗 RNG），
  使 K=1 的随机数流与单分量策略一致，退化等价测试可用同 seed 对拍。
- **U 计算**：全部 log 空间，`(B,K,K,D)`  pairwise 张量后
  `logsumexp` 收缩 K×K；`clamp(0,1)` 仅吸收 sub-ulp 舍入。
  buffer 整批调用（B=204800, K=3, D=21）在 `no_grad` 下约产生
  ~1.5GB 瞬态张量，实测通过，暂不分块；如换大 K/D 或 GPU 内存紧张
  再按 batch 分块。
- **诊断**：实现了 §11 表中的全部指标（K=1 时省略 overlap）。
- **已验证**：§12 数学与分布、uncertainty 与梯度、工程验收中除
  「端到端真实 PPO 更新」外的全部条目；其中 U 与 log_prob 的解析
  梯度通过 float64 gradcheck，采样矩与分量频率与 scipy 参考吻合。
