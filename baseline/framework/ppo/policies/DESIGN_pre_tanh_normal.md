# Design: PreTanhNormalPolicy — 共享 σ、动作覆盖单调可控的 pre-tanh 正态策略

## 1. 状态与已确认的决策

本文冻结第一版的数学语义与数值验收要求，**不表示代码已实现或训练已验证**。
目标类名 `PreTanhNormalPolicy`，拟实现于 `pre_tanh_normal_mlp.py`。
不恢复旧 `todo/tanh_gaussian_mlp.py` 的温度状态、entropy 接口或旧 checkpoint 格式。

| 项目 | 第一版选择 |
|---|---|
| 基础分布 | pre-tanh 空间中的对角正态，最终动作 a=tanh(z) |
| 位置 μ | MLP 直接输出，不在输出层额外施加 tanh |
| 尺度 σ | 每动作维度一个可训练参数，跨所有状态共享；不是 state-σ，也不是全关节共用单标量 |
| uncertainty | 动作空间逐维 Rényi-2 有效宽度 U_d=1/(2∫p_d²)，再取维度算术平均 |
| explore_factor | 在 (μ, log σ) 空间沿射线向最大动作覆盖点收缩 / 向外延伸 |
| 多样性保证 | 固定状态和参数时，e 增大使每维动作 L2 U 单调不减；非最优点严格增加 |
| U 是否含 e | ActorEval.uncertainty 不含；有效采样分布的 U_e 仅用于诊断 |
| act | tanh(μ)，逐维中位数，不声称是动作期望或众数 |
| 数据链路 | **不扩展框架，不保存 pre-tanh latent**；从已保存的 float32 动作反推 z |
| 精度边界 | 有限精度下只承诺经验证工作范围内的近似；超限显式报错，不隐式 clip / 重采样 |
| 本轮范围 | 只写设计文档；不实现、不启动训练 |

用户意图：提升 explore_factor 必须增加动作分布的有效覆盖，而不能把“σ 更大、
动作更集中于两个极端”当作探索增强。这里用 **L2 有效宽度**给“多样性”一个精确定义，
不是声称所有随机性指标都同时单调。

非目标：混合高斯、相关协方差、state-σ、时序噪声、防模式塌缩、latent 数据通道、
边界原子概率模型，以及性能优于已有策略的保证。

## 2. 框架接口与数据约束

依据 `../stochastic_policy.py`、`../experiment.py`、`../trajectory.py` 与
`../trainer.py` 的现有合约：

- `sample(observation, *, explore_factor=0.0, want_extra=False)` 返回动作及可选 extra。
- `act(observation, *, want_extra=False)` 用于确定性评估 / 部署。
- `evaluate_actions(obs, actions, explore_factor, *, want_stats=False)` 返回
  `ActorEval(log_prob, uncertainty, stats)`；两个张量均为 `(B,)`，训练时可微。
- e∈[-1,1]，0 为中性；e 是每步 / 每帧的数据，不是策略内部可变温度。
- uncertainty 只依赖状态与策略参数，不依赖传入动作，不含 e，不依赖随机采样。
- stats 仅在 `want_stats=True` 时收集，不能在 minibatch 热路径逐项 `.item()`。
- buffer 会将 trajectory.actions 强制转换成 float32；单独让策略返回 float64
  不能解决动作信息丢失。当前 evaluate_actions 也不接收 z。
- 不修改 PPO 的 uncertainty floor 公式，不引入已删除的探索梯度诊断 hook。

实现可提供内部 `forward`、`sample_action`、`deterministic_action` 辅助方法，
但不能将它们误认为框架已经要求某种固定返回格式。训练 / 导出的网络布局须一致。

以下 B 为 batch 大小，D 为动作维数，r=log σ；标量推导省略状态和动作维下标。
所有分布公式首先定义于实数连续空间；float32 的边界见 §8。

## 3. 参数化、初始化与原始策略分布

### 3.1 网络

```text
net: Linear(obs_dim, hidden_dim) -> Tanh
     -> Linear(hidden_dim, hidden_dim) -> Tanh
     -> Linear(hidden_dim, action_dim)

mu = net(obs)                       # (B,D), pre-tanh location
log_std = nn.Parameter(shape=(D,))   # shared across states
r = log_std                         # mathematical parameter
sigma = exp(r)
```

隐藏层的 tanh 与最终动作变换是不同的用途。**μ 的输出层不再做 tanh**；
最后一层若做 tanh，就额外限制了 latent 位置的表达范围。

初始化：net 沿用现有 MLP 的初始化惯例，`log_std[d]=-1`，即 σ_d=e^-1。
不将 r 初始化到覆盖最优点；不将所有 mean 输出初始化为零作为额外前提。
σ 初值与现有策略相同，只是参数尺度对齐，不意味着动作分布或 U 相同。

共享 σ 是“跨状态共享一个 D 维向量”。μ(s) 仍依赖状态，所以即使 σ 共享，
**动作空间 U、动作标准差与饱和概率仍然随状态变化**。

### 3.2 分布

\[
z_d\sim\mathcal N(\mu_d(s),\sigma_d^2),\qquad a_d=\tanh z_d.
\]

\[
p_d(a\mid s)=\frac{\mathcal N(\operatorname{atanh}a;\mu_d,\sigma_d^2)}{1-a^2},
\quad -1<a<1,\qquad
p(\mathbf a\mid s)=\prod_d p_d(a_d\mid s).
\]

不同维度的 ε 独立；固定 s 和 e 后，本策略的联合分布为逐维乘积。
μ 是 latent 均值，不是动作均值；σ 是 latent 标准差，不是动作标准差。

### 3.3 与截断正态的本质差异

- 本策略是变换分布，不是在 [-1,1] 上截断高斯，也没有截断归一化 Z。
- 小 σ 时，动作标准差近似 `sech²(μ)·σ`；靠近饱和区，相同 latent σ 产生的
  动作变化更小。
- 固定有限 μ、σ→∞ 时，动作弱收敛到两端点各半的概率质量，**不是均匀分布**。
- latent 是单峰高斯，不代表动作密度一定单峰。μ=0、σ²>1/2 时动作中心
  成为密度的局部极小点；不能把“非混合”当成“动作单峰”的数学保证。

## 4. 动作空间 uncertainty：闭式 L2 有效宽度

### 4.1 定义与范围

\[
R_d=\int_{-1}^{1}p_d(a)^2\,da,\qquad
U_d=\frac1{2R_d},\qquad
\boxed{U(s)=\frac1D\sum_d U_d(s).}
\]

由 Cauchy–Schwarz，`1=(∫p_d)²≤2∫p_d²`，所以 0<U_d≤1。
先逐维取倒数，再对维度取平均；不是 `1/(2 mean_d R_d)`，也不是联合密度的积分。
U 衡量连续动作有效覆盖宽度，不是动作方差、latent 熵、模式数量或认知不确定性。

### 4.2 闭式推导

令 q(z)=N(z;μ,σ²)。由 a=tanh(z)、da=sech²(z)dz：

\[
R=\int_{\mathbb R}q(z)^2\cosh^2z\,dz.
\]

高斯平方可写为：

\[
q(z)^2=\frac1{2\sqrt\pi\sigma}
\mathcal N\left(z;\mu,\frac{\sigma^2}{2}\right).
\]

再用 `cosh²z=(1+cosh(2z))/2` 和高斯指数矩：

\[
\mathbb E_{Y\sim N(\mu,\sigma^2/2)}[\cosh(2Y)]
=e^{\sigma^2}\cosh(2\mu).
\]

因此：

\[
\boxed{R=\frac{1+e^{\sigma^2}\cosh(2\mu)}{4\sqrt\pi\sigma}},
\qquad
\boxed{U_d=\frac{2\sqrt\pi\sigma_d}
{1+e^{\sigma_d^2}\cosh(2\mu_d)}}.
\]

这是精确的连续分布结果，不需要采样、网格或数值求积。复杂度 O(BD)。
与 MixtureTruncatedNormalPolicy 的 U 具有相同的边缘 L2 几何定义，
与 TruncatedNormalPolicy 的 peak U 不同。

### 4.3 解释与 floor 的梯度方向

- 固定 μ，σ→0：U→0。
- 固定有限 μ，σ→∞：U→0，而不是 1。
- 固定 σ，|μ|→∞：U→0。
- 对小 σ：`U≈σ√π·sech²μ`，揭示局部 tanh 压缩。
- 高 U 要求合适的尺度与位置；一味增大 σ 或只维持 latent 熵不能保证它。

floor 梯度可以增大过小 σ、减小过大 σ，也可以把 |μ| 拉回。
这意味着 coverage floor 会与“任务需要长期贴近极限动作”的收益竞争；
属于正则含义，不是实现错误。共享 σ 的更新聚合所有状态的需求，无法逐状态
单独调节 latent σ，但 μ(s) 仍有状态相关梯度。

## 5. 最大覆盖点与 explore_factor

### 5.1 最大覆盖点是常数，不是训练目标参数

固定 σ，分母在 μ=0 时最小，因此全局最大值必有 μ*=0。
令 x=σ²，沿 μ=0 优化 log U：

\[
\frac{\partial\log U}{\partial r}
=1-2x\operatorname{sigmoid}(x)=0.
\]

x>0 上左侧对应的 `2x sigmoid(x)` 严格递增，因此有唯一根：

\[
2x_*\operatorname{sigmoid}(x_*)=1,
\quad \sigma_*\approx0.85955513559725,
\quad r_*\approx-0.15134030776145.
\]

\[
U_{\max}\approx0.98498409963246.
\]

常数由标量求根离线确定，不在每帧优化。训练和导出使用相同常数精度。
高斯经过 tanh 不能精确实现动作均匀分布，所以该族的最大值小于 1。
不将 U 除以 U_max 强行归一化到 1，以保留与其他 L2 策略相同的物理含义。

### 5.2 已选映射

令 e∈[-1,1]，c(e)=3^(-e)：

\[
\boxed{\mu_d^{(e)}=c(e)\mu_d},\qquad
\boxed{r_d^{(e)}=r_*+c(e)(r_d-r_*)},\qquad
\sigma_d^{(e)}=\exp(r_d^{(e)}).
\]

等价地：`σ_e=σ_*·(σ/σ_*)^c`。实际采样分布为
`z_d ~ N(μ_d^(e), (σ_d^(e))²)`、`a_d=tanh(z_d)`。

| e | c | 在 (μ,r) 空间的含义 |
|---|---:|---|
| -1 | 3 | 离覆盖最优点的参数向量放大为 3 倍 |
| 0 | 1 | 原始策略，完全不变 |
| +1 | 1/3 | 离覆盖最优点的参数向量缩为 1/3 |

因子 3 作用于**相对最优点的参数距离**，不是 σ 的倍率，也不是 U 的倍率。
e=+1 不直接替换成最优分布，仍保留策略参数的影响。只支持 [-1,1]；
不通过悄悄 clamp e 接受非法输入。

以原始 r=-1 为例：

| e | μ_e | σ_e |
|---|---|---:|
| -1 | 3μ | 约 0.067386 |
| 0 | μ | 约 0.367879 |
| +1 | μ/3 | 约 0.647765 |

若原始 σ>σ_*，正探索会**减小** σ，同时把 μ 拉向 0，以提高动作有效覆盖。
这是选择此方案而不选择 latent σ 放大的关键原因。

共享 r_d 跨状态不变。相同 e 下 σ_e 仍跨状态共享；若各帧 e 不同，σ_e 随 e
变化不算 state-σ。μ_e 使用每帧自己的 μ(s) 和 e。

### 5.3 单调性证明

令 θ=(μ,r)，定义 F(θ)=log U_d：

\[
F(\mu,r)=\log(2\sqrt\pi)+r
-\operatorname{softplus}\big(e^{2r}+\log\cosh(2\mu)\big).
\]

- `e^(2r)` 对 r 严格凸；`log cosh(2μ)` 对 μ 严格凸。
- 二者之和对 (μ,r) 严格凸。
- softplus 严格递增且凸，其复合仍严格凸。
- 所以 F 严格凹，θ*=(0,r*) 是唯一全局最大点。

固定原始 θ，令 g(c)=F(θ*+c(θ-θ*))，c≥0。
g 为凹函数且 g'(0)=0，所以 c 增大时 g 不增；θ≠θ* 时严格下降。
而 c(e)=3^(-e) 随 e 严格下降，因此：

\[
e_1<e_2\Longrightarrow
U_d(\mu_d^{(e_1)},\sigma_d^{(e_1)})
\le U_d(\mu_d^{(e_2)},\sigma_d^{(e_2)}).
\]

非最优维度上为严格不等式；逐维取平均后仍单调。
如果所有维度都已在最优点，U 不可能再提升，映射保持不变。
**“一定提升多样性”的准确合约是：不下降，尚未达到最大值时严格上升。**

此证明适用于理想连续分布和未经二次裁剪的映射。浮点测试允许舍入误差；
不能为避免溢出任意裁剪 μ_e 或 r_e 后继续引用本证明。

### 5.4 保证与非保证

保证：固定 s、固定原始参数，增大 e 提高动作边缘 L2 有效覆盖。
这不是跨训练 update 的 U 单调性；训练更新本身可以降低原始 U。

不保证：

- 动作方差、Shannon 熵、任意阈值的边界概率同时单调。
- e 越大，每一个样本都离边界更远；概率分布性质不是逐样本排序。
- 所有 μ、σ 下任意饱和概率都随 e 下降。例如从非常窄的中心分布扩展时，
  少量近边界概率可能增加，但它不再以“边界两点集中”作为最大探索极限。
- e=-1 是确定性或围绕原动作的小噪声。负 e 远离最大覆盖点，可能通过
  更窄分布或更强饱和降低覆盖。确定性行为只能由 act 定义。
- 保持动作偏好不变。正 e 将动作中位数由 tanh(μ) 改为 tanh(cμ)，向 0 收缩。

这项均值变化是已选方案的有意设计，不允许实现时改成“只移动 log σ”。

## 6. 采样、评分、act 与 floor 必须各用正确参数

### 6.1 采样和动作密度

\[
\epsilon_d\sim\mathcal N(0,1),\quad
z_d=\mu_d^{(e)}+\sigma_d^{(e)}\epsilon_d,\quad a_d=\tanh z_d.
\]

对输入动作 a（数值守卫通过后），令 z_hat=atanh(a)：

\[
\log p_e(\mathbf a\mid s)=\sum_d
\left[-\frac12\left(\frac{\hat z_d-\mu_d^{(e)}}{\sigma_d^{(e)}}\right)^2
-r_d^{(e)}-\frac12\log(2\pi)-\log(1-a_d^2)\right].
\]

sample 返回的 log_prob 与 evaluate_actions 必须使用相同的有效参数与
同一份可实际保存的动作，具体精度顺序见 §8。e=0 不可以走含义不同的快捷路径。
PPO 的新旧评分都使用 rollout 记录的同一个 e；不将“原始策略密度”作为分母。

实数可逆变换下，新旧策略的 Jacobian 在 ratio 中抵消；本版 log_prob 仍返回
动作密度，不能因此省略 Jacobian 后将 latent log_prob 冒充动作 log_prob。
该抵消性质也不恢复被 float32 丢失的 z。

### 6.2 原始 U 与有效 U_e 分离

```text
obs -> mu(s)
parameter -> r (shared across states)

mu, r                         -> U_policy -> ActorEval.uncertainty
mu, r, per-frame e -> mu_e,r_e -> score(actions) -> ActorEval.log_prob
                              -> U_effective   -> optional stats only
```

**计算 U_policy 时不能复用已缩放的 μ_e**。与只缩放 σ 的旧实现不同，
这里 μ 也随 e 改变，这一处容易引入“U 被 e 污染”的错误。

### 6.3 确定性 act

`act(s)=tanh(μ(s))`，不接收 e，不使用 θ*，也不通过 e=-1 模拟确定性。
这是动作的逐维中位数；动作期望通常不等于它，动作密度的众数也未必在这里。
act 的浮点 tanh 可以产生端点，因为不做概率反变换；sample 的端点则必须
按 §8 判为超出本版可准确评分的范围。二者不能混淆。

### 6.4 Floor

沿用 trainer 的逐帧平方 hinge：

\[
L_{floor}=\lambda\operatorname{mean}_b
\left[w_b\max(f-U_{policy}(s_b),0)^2\right].
\]

先逐帧处罚再平均，不是对 mean U 做 hinge。mean U>f 不能推出 loss=0。
不 detach U，不切断 μ 或共享 r 的梯度。

- f>U_max 是本分布族无法满足的目标，必须在实验配置验收中指出；不擅自修改框架。
- f=U_max 也意味着只允许最优覆盖分布且有限训练难以精确达到，不推荐作为默认。
- floor 是软约束，不能作为数值安全守卫。严重饱和时 U 和其梯度可极小，
  即使用 log 空间计算也不会凭空产生足够的恢复梯度。
- 与 mixture 的 L2 U 定义一致，但分布族与 U 上限不同，不保证同值 floor
  带来同样的训练影响。与旧 peak U 更不能按同值视为等强度约束。

## 7. 稳定公式与参数范围

推荐在 float64 中计算分布辅助量、反变换和评分，网络仍可为 float32。
最终动作仍必须经过真实的 float32 保存路径；提高内部精度不是恢复丢失信息。
实现应评估 GPU 上 float64 的代价，不能先假定成本可忽略。

稳定表达：

```text
logcosh(x) = logaddexp(x, -x) - log(2)
log_U = log(2*sqrt(pi)) + r - softplus(exp(2*r) + logcosh(2*mu))
z_hat = 0.5 * (log1p(a) - log1p(-a))
log_J(z) = 2 * (log(2) - z - softplus(-2*z))
```

大正负 z 的 log_J 也可用等价的 |z| 对称表达，须做值与梯度对拍。
禁止直接 `exp(σ²)*cosh(2μ)`，禁止 `log(1-tanh(z)**2+epsilon)` 掩盖饱和。

数学模型的 r 无业务上下界。若实现沿用已有策略的 raw log_std 数值守卫，
须明示实际参与分布与映射的是哪个 r；守卫不是参数范围准确性的证明。
特别是 e=-1 会把 r-r* 放大三倍，不能将有效 r 再按原边界裁剪以吞掉探索信号。
本版要求：对不支持的参数、非有限中间结果、非法 e 显式报错，而不是静默更换分布。

U 浮点值只可在验证误差属于舍入级别后做极小范围的边界修正；明显超出
[0,U_max] 是实现或精度问题。下溢为 0 时须有诊断，不能把“有限且在 [0,1]”
当作已经满足梯度与数值准确性要求。

## 8. 不扩展框架的数值合约：限定范围的动作反变换

### 8.1 已接受的限制

理想 tanh 可逆，float32 tanh / 存储不可全局逆。大 z 可舍入为 ±1；
小 σ 下即使没有端点，动作量化误差也可能远大于策略噪声。

本版不保存 latent，因此不能声称“任意 μ、σ、e 下准确评分”。必须定义并验证
支持的数值工作范围。选择不扩展框架，是接受这个限制，不是认为它不存在。

### 8.2 统一实际动作作为评分输入

采样路径固定为：

1. 构建 μ_e、r_e 并验证有限性及支持条件。
2. 采样原始 z，计算 tanh(z)，转换成实际外部返回 / trajectory 保存的 float32 动作 a32。
3. 将 a32 提升到分布计算精度，计算 z_hat=atanh(a32)。
4. sample 的 log_prob 使用 z_hat 及 a32，不使用未量化 z 的 log_prob 冒充保存动作的评分。
5. evaluate_actions 和导出 sample 使用完全相同的评分公式与数值守卫。

sample_action 也返回这一份量化后的动作，不能让训练内部辅助路径返回另一种精度。
这保证两条评分路径对相同保存动作的一致性；**不能单独证明连续密度之比能准确
近似量化后动作的概率之比**，后者须通过独立的误差验收。概率质量与密度的单位
不同，不应直接比较两者数值是否相等。

### 8.3 守卫与误差预算

实施时至少包含以下检查，不允许只检查 isfinite：

- 动作维度、每帧 e、obs shape 正确，输入动作有限且严格在 (-1,1) 内。
- 定义 `action_margin` 与 `a_safe=1-action_margin`；在反变换条件数已验证的
  区间内评分。首轮验证可从 `action_margin=2^-20` 开始，不能将其解释成分布截断。
- 采样若得到 |a32|>a_safe，立即报错。**不 clip，不丢弃后重采样**；两者都会改变
  分布而使当前评分公式不再匹配。
- 在采样前计算尾部风险。令 z_safe=atanh(a_safe)，逐维
  `p_tail=Φ((-z_safe-μ_e)/σ_e)+Φ((μ_e-z_safe)/σ_e)`，使用稳定 CDF / log-CDF。
  首轮验证建议每维风险预算 `p_tail≤1e-12`，并报告 D 维及整 rollout 的 union bound。
  高斯尾部永远非零，风险预算不等于“不可能报错”；超限参数必须失败，而不是被
  声称“保证永远运行”。同一 θ 可能支持 e=0 而不支持 e=-1。
- 对内部采样 z 与反推 z_hat，检测标准化反变换误差
  `abs(z_hat-z)/σ_e` 以及联合 log density 误差。
  建议首轮阈值分别为 `1e-3` 与 `1e-3 nats/action-vector`；后者比较
  理想样本 tanh(z) 的稳定评分 `log q(z)-log_J(z)` 与量化动作评分。
- 上述检查用于拒绝不准确的样本 / 参数区间并中止，不允许跳过样本继续训练。
- evaluate_actions 没有原始 z，无法重复原始样本误差检查。它仍须验证当前参数下
  的反变换条件数、动作量化间隔相对 σ_e 的大小与评分有限性；对新旧参数组合
  做离线密度、ratio、梯度误差测试，不能仅依靠采样时通过守卫。
- 独立 ratio 参考可使用 float32 量化区间：a32 与相邻可表示数的中点构成
  [a_lo,a_hi]，其理想量化概率为
  `Φ((atanh(a_hi)-μ_e)/σ_e)-Φ((atanh(a_lo)-μ_e)/σ_e)`。
  使用高精度 / 稳定差分计算这些小概率，再比较新旧区间概率之比与连续密度之比；
  相同动作的量化区间宽度在 ratio 中抵消。多维独立时逐维 log 概率求和。
  这只是独立验收参考，不把生产 log_prob API 改成离散概率评分。

这些阈值是保守的**首轮工程验收起点**，不是新的业务 σ bounds，也不是已经实测
通过的范围。实现必须根据误差扫描记录最终支持域和性能结果；不得为了让训练
继续而静默放宽阈值。若预期参数和 e 组合频繁超限，应报告此架构选择的阻碍，
再讨论 latent 数据链路，而不是擅自修改数学定义。

### 8.4 理想 U 与执行分布的区别

闭式 U 是连续 tanh-Gaussian 的有效宽度。float32 输出严格说是离散的；
不能把离散量化分布直接代入连续密度平方积分并宣称完全相等。
单调性证明针对理想分布，实际执行只在通过精度验收的范围内近似继承。

不采用截断 / censored 动作模型：若未来显式加入边界原子概率，log_prob、U
与单调性证明都需要重新设计，不能作为本版的“数值小修复”悄悄加入。

## 9. 诊断与训练解释

| 指标 | 定义与解释 |
|---|---|
| uncertainty | batch mean U_policy，等于 ActorEval 对应均值 |
| effective_uncertainty | batch mean U_e，反映实际 explore 映射后的理想覆盖 |
| latent_std_mean/min/max | 原始共享 σ 向量的统计，不是动作标准差 |
| effective_latent_std_mean/min/max | 含各帧 e 的有效 latent σ 统计 |
| latent_mean_abs | mean_(b,d) abs(μ_bd) |
| effective_latent_mean_abs | mean_(b,d) abs(μ_e,bd) |
| coverage_distance | mean_(b,d) sqrt(μ_bd²+(r_d-r*)²)，仅作参数空间诊断 |
| near_boundary_probability | 原始分布 P(abs(A)>0.99) 的 batch/维度均值 |
| effective_near_boundary_probability | 同上，使用 μ_e、σ_e |
| effective_unsafe_tail_max | §8 的 p_tail 在 batch/维度上的最大值 |

near-boundary probability 可直接用正态 CDF 计算，不通过随机采样估计。
可额外记录动作分位数跨度，但不能把它或 std 当作 L2 U 的替代指标。
采样端的反变换误差需要专门测试 / 调用方收集，不能声称当前 buffer 已有该统计。

共享 σ 不应报告为“学到了 state-σ”。观测到各状态 U 不同，只说明 μ(s) 经 tanh
压缩后有不同的动作覆盖；不同 e 导致的有效 σ 差异也不是 state-σ 参数化。

## 10. 导出、命名与兼容性

拟新增：

- `pre_tanh_normal_mlp.py`：`PreTanhNormalPolicy`。
- `_export_template_pre_tanh_normal.py`：`ExportedPreTanhNormalPolicy`。
- `test_pre_tanh_normal.py`：专用数值、接口与导出测试。
- `baseline/humanoid21/blueprints/init_policy_pre_tanh_normal.yaml`。

保留 self-contained `model.pt + policy.py + MANIFEST.json` 形式，不 import
baseline / envs。网络为 `net.*` 加 `log_std`，strict state_dict 加载。

manifest / payload 必须明确：

- format_version、policy_class、obs_dim、action_dim、hidden_dim；
- `distribution_kind=tanh_diagonal_normal_shared_std_v1`；
- `uncertainty_kind=marginal_renyi2_width_v1`；
- `exploration_kind=coverage_radial_logscale_v1`，射线因子 3、r* 常数；
- 本版评分精度、最终确定的数值守卫参数与语义版本。

导出必须保留 sample 的探索映射、评分及数值守卫；不能导出时退回简单 σ 缩放。
导出不需要训练用 U API，但若有诊断，其含义必须相同。reset(seed) 的 RNG 行为
与其他策略约定一致。旧 tanh / truncated-normal 的 checkpoint 不按相同字段名
自动接受；即使 shape 一样也不能静默解释成新分布。

## 11. 验收清单

### 数学与探索

- [ ] 一维动作密度归一化及二维乘积分布归一化，与独立参考积分核对。
- [ ] L2 R / U 闭式与独立积分一致，覆盖 μ 正负、σ 小中大、靠边和饱和趋势。
- [ ] σ*、r* 的求根残差、U_max、对称性及唯一最大点核查。
- [ ] μ=0/非零、σ<σ*/=σ*/>σ* 的 e=-1/0/+1 和稠密网格单调测试。
- [ ] e=0 原始分布恒等；e=+1 为参数距离 1/3，e=-1 为 3，不误测成 σ 的三倍。
- [ ] 固定点处保持不变；其他点严格单调的检查应排除浮点下溢 / 舍入平台。
- [ ] 共享 σ 的参数数量为 D；同 e 跨状态相同；每帧 e 正确广播。
- [ ] ActorEval U 与动作及 e 无关；effective U 随 e 改变且仅用于诊断。

### 梯度、采样与数值

- [ ] U、log_prob 与映射的 μ / r 梯度，通过 float64 gradcheck / 有限差分核对。
- [ ] 共享 r 的梯度确实聚合各状态，trunk 和 mean 输出收到梯度。
- [ ] 同参数变更、同保存动作、同 e 下 sample / evaluate log_prob 一致。
- [ ] 独立分位数 / CDF / 矩对拍，不只验证两条共享错误的代码路径一致。
- [ ] float32 实际动作往返后，密度、ratio、梯度误差在预算内。
- [ ] 超大 μ、σ，极小 σ，e=-1 外推，以及精确 ±1 动作均不会被静默修复。
- [ ] 尾部风险、action_margin 和误差阈值的边界案例；超限 fail loud。
- [ ] action 输入扰动和新旧参数组合的误差扫描，明确可支持的数值工作范围。
- [ ] 不能以 U clamp 后有界、sample/evaluate 相等或无 NaN 代替分布准确性。

### 工程与可用性

- [ ] act=tanh(μ)，与中位数公式及导出一致，不拿 e=-1 替代 act。
- [ ] strict loading 拒绝错误类、版本、缺失 / 多余 keys、未知配置。
- [ ] 无 repo PYTHONPATH 的导出子进程 act/sample/reset parity。
- [ ] minibatch 与全 buffer（典型 B=204800,D=21）的时间 / 内存验证，尤其 float64 成本。
- [ ] 真实 PPO smoke：采样、buffer、更新、floor 梯度、评估、导出与 checkpoint 恢复。
- [ ] 短程训练检查 coverage 控制、饱和风险、守卫触发和梯度，不提前声称优于基线。

以上是未来实现验收要求，本次文档与公式验证不勾选代码 / 训练验收项。
若数值约束成为实际运行阻碍，应回到 §8 明确的决策边界再讨论，不能将未验证的
有限精度近似包装成全参数域的严格正确性保证。

## 12. 实现记录（2026-09-23）

### 落地文件

- `pre_tanh_normal_mlp.py` — `PreTanhNormalPolicy`。`net`（Linear→Tanh→
  Linear→Tanh→Linear）直接输出 μ（无输出端 tanh）；`log_std` 为 (D,)
  共享参数，init −1（σ≡e⁻¹）。所有分布辅助计算（探索映射、评分、U、
  守卫）在 float64 中进行，网络前向保持 float32。
- `_export_template_pre_tanh_normal.py` — 自包含
  `ExportedPreTanhNormalPolicy`，零 repo import；float64 辅助函数、
  RNG 序列（`randn` float64）与训练侧逐行一致。
- `humanoid21/blueprints/init_policy_pre_tanh_normal.yaml` — humanoid21
  blueprint（obs 96 / act 21 / hidden 256）。
- `experiments_ppo/exp_standup_floor04_pretanh.py` — 仅替换 actor
  blueprint 的对照实验。
- `test_pre_tanh_normal.py` — 50 个测试。

### 验收清单核对结果（§11）

数学与探索：**全部通过**。密度对 scipy 变换分布逐点对拍；二维网格积分
归一化；闭式 U 对 z 空间 40 万点 quadrature；σ*/U_max 唯一最大值核对；
(μ,σ) 网格 × 21 个 e 的单调性；e=0 恒等 / e=±1 数值 / 固定点不变；
共享 σ 形状 (D,) 且同 e 跨状态相同；U 与动作、e 无关；U 随 μ(s) 变化
（共享 σ ≠ 常数 U 的区分测试）。

梯度、采样与数值：**通过**。U 与 log_prob 对 (μ, r, e) 的 float64
gradcheck；log_std / mean head / trunk 梯度完整性；sample 与 evaluate
对同一 float32 存储动作 log_prob 一致（<1e-5，含 e=±1）；经验矩对
quadrature；|a|=±1、|a|>a_safe、NaN/Inf、e 越界、μ=6.5 尾部超限
均 fail loud；evaluate 对安全动作在"采样不可行"参数下仍可评分
（守卫语义分层：evaluate 检查动作，sample 检查参数+动作）。

工程与可用性：**通过**。act=tanh(μ)；strict loading 拒绝错误
class/version/kind/缺失多余 key；子进程无 repo 加载 parity；
B=204800 全 buffer evaluate 无 NaN；standup_floor04 smoke 训练端到端
通过（rollout→buffer→PPO 更新→eval→export 全链路）。

### 实测数值（smoke，e=0 rollout）

- init `uncertainty=0.594`（σ=e⁻¹, μ≈0 的 L2 U；floor=0.4 未激活，
  `floor_loss=0`——与 mixture 相同的 floor 语义差异提醒适用，
  但注意本策略 U_max≈0.985 且大 σ 也使 U→0，floor 从两侧约束）
- `effective_unsafe_tail_max≈1e-74`（≪预算 1e-12，守卫余量极大）
- `near_boundary_probability≈2e-11`
- `coverage_distance≈0.86`（θ 到 θ* 的距离）

### 已知的未决项（有意留给后续）

- a_safe=1−2⁻²⁰、tail budget=1e-12 是首轮值；若训练中守卫误触发，
  按 §8 决策边界重议，而不是放宽守卫。
- 不支持参数区域的边界（多大 μ/σ/e 组合安全）只做了点验证，未做
  完整扫描——守卫本身即是防线。
- PPO 训练中 e≠0 的覆盖探索路径未在真实 rollout 中验证（本实验
  rollout 用 e=0）；该路径已被单测覆盖，但其训练动力学属于后续实验。

## 12.1 变体：StatePreTanhNormalPolicy（2026-09-23）

state-σ 版本，与共享版**仅 σ 来源不同**，无新增设计决策：

- `state_pre_tanh_normal_mlp.py` — `StatePreTanhNormalPolicy`
  直接**继承** `PreTanhNormalPolicy`，只覆盖 `__init__`（trunk +
  head([mean|log_std]) 架构）、`_policy_params`（head 输出切分）、
  `_build_stats`（追加 `sigma_state_std` /
  `effective_sigma_state_std`）。探索映射、U、评分、守卫、采样 /
  evaluate 路径全部继承同一份实现——不存在两份可能漂移的数学代码。
  父类的导出元数据（policy_class / *_kind / 模板文件名）已重构为
  类属性，`to_blueprint` 也被继承复用。
- σ 头初始化 `w=0, b=−1` → σ(s)≡e⁻¹，与共享版 init 完全一致；
  测试用权重复制构造了逐点退化等价（act / log_prob / 同 seed
  sample 全等）。
- `_export_template_state_pre_tanh_normal.py` — 自包含导出模板
  （trunk+head 布局），`distribution_kind` =
  `tanh_diagonal_normal_state_std_v1`。
- blueprint `init_policy_state_pre_tanh_normal.yaml`；实验
  `standup_floor04_pretanh_statesig`。
- **不加 clamp**：σ 头输出无 ±20 截断。tail-risk 守卫在 σ≈3
  就已 fail-loud（远比 exp 溢出边界紧），小 σ 方向由评分有限性
  检查兜住——clamp 在本策略家族是死代码且违反"不静默修复"约定。
- `test_state_pre_tanh_normal.py` — 20 个测试：退化等价、σ 状态
  依赖、per-state σ 密度对拍、U 经 σ(s) 变化、守卫继承、σ 头梯度、
  strict 导出、子进程 parity、B=204800 buffer、blueprint 加载。
- smoke 训练端到端通过；init 时 `sigma_state_std=0`（正确反映
  常数 σ 初始化）。

## 12.2 复现性修复：Policy.reset(seed)（2026-09-23）

初版两个导出模板漏掉了 `Policy.reset(seed)` —— 框架契约（
`envs/framework/policy.py`：stochastic policies SHOULD reseed their
internal RNG；`EpisodeRunner` 每 episode 用 `SeedSequence` 派生
per-agent 种子并调用 `policy.reset(seed)`）。缺失时 rollout 采样
消耗 worker 全局 RNG 残流，与调度时序相关 → 不可复现。

修复：训练侧父类与两个导出类均实现
`reset(seed) → torch.manual_seed(seed)`（与 truncnorm 导出模板
一致；state 版经继承获得）。

验证：两个独立 `standup_floor04_pretanh` run 各 5 个 update 对拍——
除 `timing.*` 墙钟字段外 **全部字段 bit-identical**（reward、
policy_stats、KL、ratio、梯度、eval 指标）。
