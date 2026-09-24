# Design: BoundedStdTruncatedNormal — 有界标准差的截断正态策略族

日期：2026-09-24。状态：设计阶段，尚未实现或训练验收。

## 1. 决策摘要与范围

本族包含共享 σ 和 State σ 两个版本，动作分布均直接定义在 `[-1,1]`。
**截断之后不再接 tanh，不使用 pre-tanh latent，不需要 atanh 或 Jacobian。**
位置参数 μ 仍由网络输出经过 tanh 得到；标准差改为有界 log σ 参数化。
探索采用已经确定的加法：`v_effective = v + alpha * explore_factor`。

| 项目 | 决策 |
|---|---|
| 分布 | 对角高斯按每个动作维度截断到 `[-1,1]` 并归一化 |
| μ | `tanh(raw_mean)`，是未截断高斯的位置参数 |
| σ | `exp(r_min + (r_max-r_min) * sigmoid(v))` |
| 共享版 | 每个动作维度一个可训练 v，跨状态共享，不是所有关节共用一个标量 |
| State 版 | 网络逐状态输出 v(s)，其余数学与共享版一致 |
| explore_factor | `v_e = v + alpha*e`；不改变 μ；e∈[-1,1] |
| U | 沿用 TruncatedNormal 的 `1/(2*peak)`，逐动作维度算术平均 |
| 确定性 act | 输出 μ；它是该截断分布的众数，不宣称是实际均值或中位数 |
| 数据路径 | 保留现有 action-only rollout / trajectory / buffer / trainer |
| 实现原则 | 容易出错的数学共用；State 子类只改变网络和原始 v 的来源 |
| 本次交付 | 只新增设计文档；不替换旧策略、不删除 pre-tanh 文件、不启动训练 |

本设计是新的 TruncatedNormal 变体，不是对旧 PreTanhNormal checkpoint 的数值修复。
此前关于“截断 latent 后再缩放 tanh”的 K、L2 U 和复杂探索路径不属于本族。
旧的无业务 σ 边界策略保留用于对照；其设计决策不应被新类静默覆盖。

### 1.1 候选默认配置，而非已验证最优配置

| 参数 | 首轮候选 | 状态 |
|---|---:|---|
| sigma_min | 0.05 | 实验起点；约束精细控制能力 |
| sigma_max | 2.0 | 实验起点；约束最大覆盖能力 |
| init_std | exp(-1) ≈ 0.3678794412 | 与旧 TruncatedNormal 初始 σ 对齐 |
| explore_alpha | 初始化点局部标定，约 1.1993389105 | 标定规则见 §4，不是恒定三倍 σ |
| uncertainty_floor | 对照实验先保留 0.4 | 实验配置，不写死在策略中 |

已确定的是分布、log σ 的 Sigmoid 参数化、加法探索及 U 定义。
上下界和探索强度仍需训练对照；不把“有界”当作性能更好的证明。
第一版采用全维共用的标量上下界及标量 alpha，不引入可学习边界或逐关节范围。

## 2. 动机与非目标

现有 TruncatedNormal 使用 `exp(clamp(log_std, -20, 20))`，e 通过 `3^e` 乘 σ。
±20 是很宽的指数安全保护，不能有效限制分布的业务范围。
本族有意引入以下取舍：

- σ 下界避免无限锐化，限制小 σ 下的评分敏感性及动作量化相对误差。
- σ 上界避免极大 σ 下几乎均匀、参数变化却很少改变动作分布的冗余区域。
- 探索前后始终处于同一 σ 范围，避免外部 e 把参数推离验证域。
- μ 保持在截断区间内，避免只保留高斯极远单侧尾部的病态条件分布。

直接 TruncatedNormal 的大 σ 极限是均匀动作分布，并非 tanh-Gaussian 的双端点集中。
因此限制 σ_max 的目的主要是数值范围和优化参数化控制，而不是修复 tanh 饱和。

不承诺：所有任务性能提升、KL 自动变小、Sigmoid 永不饱和、梯度永不消失、
上下界任取都数值安全，或仅通过单测便证明全程训练稳定。

## 3. 原始参数与两个版本

令 D 为动作维度，B 为 batch 大小。固定：

\[
r_{\min}=\log\sigma_{\min},\quad
r_{\max}=\log\sigma_{\max},\quad
\Delta r=r_{\max}-r_{\min}>0.
\]

对网络原始位置输出 m 和原始尺度控制量 v：

\[
\mu=\tanh m,\quad p=\operatorname{sigmoid}(v),\quad
r=r_{\min}+\Delta r\,p,\quad \sigma=\exp r.
\]

这里 v **不是 log_std**。建议参数名 `raw_std`，不要继续使用具有旧含义的
`log_std` 名称存储 v。理想实数下 σ 严格位于开区间；浮点 Sigmoid 可能达到
0 或 1，实际验证按闭区间 `[sigma_min, sigma_max]` 加舍入容差进行。
有限 v 的数值饱和不报成非法分布，但必须可诊断；NaN/Inf 输入必须报错。

### 3.1 共享版

- 均值网络沿用 `net: obs -> hidden -> hidden -> D`，隐藏层 Tanh。
- `raw_std` 是 `(D,)` 的 `nn.Parameter`。
- 原始 μ 是 `(B,D)`，原始 v 是 `(D,)`，显式按 batch 广播。
- 相同 e 下 σ 跨状态相同；不同帧 e 可产生不同的有效 σ，这不等于 State σ。

### 3.2 State 版

- `trunk: obs -> hidden -> hidden`，隐藏层 Tanh。
- `head: hidden -> 2D`，输出 `[raw_mean | raw_std]`。
- μ、v 都为 `(B,D)`；只将 v 送入共用有界映射。
- σ 半侧 head 的 weight=0、bias=v_init，初始为常数 σ。
- 使用共享版同样的 sigma_min、sigma_max、init_std、alpha。

两版同 seed 构造并不自动保证均值权重相同，因为 head 形状不同会影响 RNG 消耗。
退化等价测试必须显式复制均值网络权重与 v，而不是仅依赖相同初始化种子。

### 3.3 配置约束

要求所有配置有限，`0 < sigma_min < init_std < sigma_max`，`alpha > 0`。
配置范围还必须通过 §8 的数值验收；仅满足代数不等式不代表任意范围均受支持。
上下界是策略语义，不得在训练中随意调节或从 checkpoint 加载时静默回退。
不在 v 上额外叠加 ±20 clamp，不在有效 σ 上再乘 `3^e` 或硬 clip。

## 4. explore_factor：Sigmoid 之前的加法

\[
\boxed{
v_e=v+\alpha e,\quad p_e=\operatorname{sigmoid}(v_e),\quad
r_e=r_{\min}+\Delta r\,p_e,\quad \sigma_e=\exp r_e.
}
\]

μ_e=μ，不改变位置参数。e=0 严格使用同一条数学路径恢复原始分布。
标量 e 用于 sample；逐帧 `(B,)` e 在批量评分时扩展为 `(B,1)`。
必须检查 e 有限且在 `[-1,1]`；不得静默裁剪越界请求。

### 4.1 为什么不是 v * 3^e

v 可正可负。乘以大于 1 的系数会使负 v 更负，从而降低 σ；v=0 则完全不变。
这控制的是“远离 Sigmoid 中点”，不满足正探索统一增加覆盖的要求。
加法满足：

\[
\frac{\partial\log\sigma_e}{\partial e}
=\alpha\Delta r\,p_e(1-p_e)>0
\]

（有限参数的理想实数计算）。浮点饱和时允许退化为非严格单调。

若仅取 alpha=log(3)，则 Sigmoid odds `p/(1-p)` 在 e=1 时乘 3；
不意味着 σ 乘 3。本设计的候选 alpha 采用下述局部响应标定，不混淆这两种定义。

### 4.2 初始点与 alpha 标定

\[
p_0=\frac{\log\sigma_0-r_{\min}}{\Delta r},\quad
v_0=\log\frac{p_0}{1-p_0},\quad \sigma_0=\text{init_std}.
\]

让初始化点 e=0 的相对 σ 响应与旧映射 `log sigma_e = log sigma + e*log(3)` 相同：

\[
\boxed{\alpha=\frac{\log3}{\Delta r\,p_0(1-p_0)}}.
\]

候选默认配置得到：

- `v_init ≈ 0.1644220033`，不是 -1。
- `explore_alpha ≈ 1.1993389105`。
- e=-1、0、+1 时初始 σ 分别约为 `0.1314986052、0.3678794412、0.9436326169`。

这是**局部一阶匹配**，不是 e=±1 时的三倍/三分之一，也不匹配旧参数化下的
优化器步长。alpha 在构造时解析确定并冻结，绝不按每帧当前 v 重算；否则会改变
探索语义，甚至抵消想要的饱和抑制。允许显式配置固定 alpha，但必须序列化最终值。
init_std 过于贴近边界时标定 alpha 会很大，不能未经测试宣称这种配置可用。

### 4.3 探索语义边界

- 正 e 增加 σ 和 §6 定义的动作 peak U；不是所有随机性指标的共同单调保证。
- e=1 是接口允许的最大正探索，不代表到达 sigma_max 或均匀分布。
- e=-1 仍是随机分布，不等于 act。
- μ 保持不变；截断后分布的实际期望和中位数可能随 σ 改变，不能宣称它们不变。
- v 接近饱和区时，训练和 e 的作用都会减弱。这是代价，需记录而非隐藏。

## 5. 分布、采样、评分与确定性动作

令：

\[
\ell=\frac{-1-\mu}{\sigma_e},\quad
h=\frac{1-\mu}{\sigma_e},\quad
C_e=\Phi(h)-\Phi(\ell).
\]

单维动作密度：

\[
p_e(a)=\frac{\phi((a-\mu)/\sigma_e)}{\sigma_e C_e},\quad a\in[-1,1].
\]

区间外概率为零。各动作维度条件独立。
μ 与 σ 是未截断正态的参数，不是最终动作分布的均值和标准差。

### 5.1 采样

\[
u\sim\operatorname{Uniform}(0,1),\quad
q=\Phi(\ell)+uC_e,\quad
a=\mu+\sigma_e\Phi^{-1}(q).
\]

这是截断分布的 inverse-CDF 采样，不是抽无界高斯后把动作压到端点。
保存实际返回的 float32 动作；sample 的 log_prob 对同一份动作重新评分。
有限精度采样约定见 §8，不把 inverse-CDF 中的 clipping 说成精确分布操作。

### 5.2 evaluate_actions

\[
\boxed{
\log p_e(\mathbf a\mid s)=
\sum_d\left[-\frac12\left(\frac{a_d-\mu_d}{\sigma_{e,d}}\right)^2
-r_{e,d}-\frac12\log(2\pi)-\log C_{e,d}\right].
}
\]

返回动作空间的连续 log density；没有 latent 反变换和 tanh Jacobian。
PPO 新旧策略都必须使用 rollout 记录的同一个 e，不能把原始 σ 当作评分 σ。
归一化项依赖当前 μ、σ_e，必须参与梯度；不能只修改采样而忘记重新归一化。

### 5.3 act 与 reset

`act(obs)=mu(obs)`，沿用基线确定性行为，是截断分布的众数。
它不依赖 σ、e，也不是截断分布的真实期望。不要继承旧注释中的均值误称。

训练类、共享/State 导出类都必须实现 `reset(seed)`：非 None 时按框架约定
重置 Torch RNG；None 时不自行选随机种子。训练侧和导出侧均测试 reset/replay。
共享代码必须避免为了构造子类而先构造、丢弃父类网络，防止多消耗随机数。

## 6. Uncertainty：沿用 peak U，含探索单调性证明

使用**原始** σ（e=0）：

\[
C=\Phi((1-\mu)/\sigma)-\Phi((-1-\mu)/\sigma),\quad
\boxed{U_d=\frac1{2\,\mathrm{peak}}=\frac{\sigma\sqrt{2\pi}C}{2}}.
\]

由于 μ∈[-1,1]，peak 位于 μ。等价形式为：

\[
U_d(\mu,\sigma)=\frac12\int_{-1}^{1}
\exp\left[-\frac{(a-\mu)^2}{2\sigma^2}\right]da.
\]

由此可见：

\[
\frac{\partial U_d}{\partial\sigma}
=\frac12\int_{-1}^{1}
\exp\left[-\frac{(a-\mu)^2}{2\sigma^2}\right]
\frac{(a-\mu)^2}{\sigma^3}da>0.
\]

结合 §4，固定状态/原始参数，逐维 `dU_effective/de > 0`；算术平均仍然单调。
理想实数下严格增加，浮点精度和 Sigmoid 饱和允许相等。该证明不适用于
训练更新之间的 U 变化，也不证明真实 rollout 回报随 e 增大。

位置导数为：

\[
\frac{\partial U_d}{\partial\mu}
=\frac12\left[
\exp\left(-\frac{(-1-\mu)^2}{2\sigma^2}\right)
-\exp\left(-\frac{(1-\mu)^2}{2\sigma^2}\right)
\right].
\]

所以 U 关于 μ 对称，并随 |μ| 增大而下降；μ=0 时位置导数为零。
旧 `DESIGN_truncated_normal.md` §3.4 的导数写法不能作为本族验收依据，应以
这里的推导和独立 gradcheck 为准。多维明确用算术平均，不采用旧文档早期讨论的乘积。

最终 `ActorEval.uncertainty = mean_d(U_d)`，形状 `(B,)`；不依赖输入动作、不 detach、
不混入 e。`effective_uncertainty` 可用于诊断，但不送入 floor loss。
这不是 L2/Renyi-2 U，也不是 Shannon entropy；不要与 pre-tanh/mixture 的 floor
强度作完全等价解释。U 不再除以本族可达最大值，保持原有几何标尺。

### 6.1 有界 σ 对 U 可达性的影响

固定 σ，μ=0 时 U 最大，|μ|=1 时最小。因此理论闭包范围由
`U(1,sigma_min)` 与 `U(0,sigma_max)` 给出，极限值不要求有限 v 真正达到。

| σ | U(μ=0) | U(|μ|=1) |
|---:|---:|---:|
| 0.05 | 0.062666 | 0.031333 |
| 1 | 0.855624 | 0.598144 |
| 2 | 0.959850 | 0.855624 |
| 3 | 0.981786 | 0.930614 |

候选上界 2 下，全局 U 上确界约 0.959850，不可能达到 1；固定边界 μ 的
上确界约 0.855624。floor 超过前者全局不可达，超过后者则在部分 μ 下无法
仅靠增加 σ 满足，需要改变 μ。floor=0.4 在候选范围内可达，但不保证训练必达。

floor 沿用 trainer 的逐帧平方 hinge 和 floor_weight；不修改算法：
`coef * mean(floor_weight * relu(floor - U)^2)`。
mean U 高于 floor 不保证所有帧 loss 为零；维度平均也不能保证每个关节 U 都高。
σ_min 与 floor 不等价：前者是逐维硬表达范围，后者是有梯度但可能未满足的软约束。

## 7. 代码复用与接口接入

拟命名（实现阶段可统一调整，但两种旧参数不能混名）：

- `BoundedStdTruncatedNormalPolicy`：`bounded_std_truncated_normal_mlp.py`。
- `StateBoundedStdTruncatedNormalPolicy`：`state_bounded_std_truncated_normal_mlp.py`。
- 两个初始 blueprint、两个独立测试文件；State 类继承共享类。

数学关系建议沿用 `TruncatedNormalPolicy -> BoundedStdTruncatedNormalPolicy
-> StateBoundedStdTruncatedNormalPolicy`，前提是建立正确的参数入口，而非强行套用旧接口。

### 7.1 必须保留原始 v 的统一参数入口

当前父类 `_policy_params(obs)` 返回 `(mu, sigma_policy)`，再调用
`effective_sigma(sigma_policy, e)`。这不足以稳定实现新映射：从 sigma 逆推
logit(v) 在 Sigmoid 饱和处不可行，也会增加重复计算和舍入误差。

建议增加一个内部入口：

`_distribution_params(obs, explore_factor) -> (mu, sigma_policy, sigma_effective)`。

- 旧父类实现委托现有参数/探索函数，保留原运算顺序、形状、统计和导出语义。
- 新共享类从一次网络前向获取 `(mu, v)`，直接计算原始及有效 σ。
- 新 State 子类只覆盖网络构造、`(mu,v)` 提取、必要的 State 统计/导出架构标识。
- `forward`、`sample_action`、`evaluate_actions` 走同一个入口；不重复网络前向。
- 密度、归一化、采样、U 及动作协议共用；不复制两份容易漂移的数学实现。
- 不把父类 `_policy_params` 的第二返回值从 σ 悄悄改成 v。

导出需要额外配置入口：现有父类 to_blueprint 只保存架构和权重，不能直接
继承后丢失 σ 边界及 alpha。通过共用 payload/manifest 构造 hook 增加配置。
若公共父类需最小重构，必须用旧类回归与历史运行对拍证明无行为变化。
本设计不授权为了代码复用而改变旧策略的数值实现、分布或 checkpoint 语义。

### 7.2 公共契约

- `act`: 单观测 -> `(D,)` float32 动作和 extra；默认确定性。
- `sample`: 标量 e -> 动作及可选 log_prob；与导出路径一致。
- `evaluate_actions`: `(B,obs_dim)`、`(B,D)`、`(B,) e` -> `(B,) log_prob/U`。
- `want_stats=False` 不做诊断用 CPU 同步；stats 仅在显式请求时计算。
- device 从参数动态推导，`.to(device)` 后输入与分布计算不得使用旧设备快照。
- 不修改 TrainablePolicy、Trajectory、PPOBuffer 或 trainer，不新增 latent 字段。

## 8. 数值合约：有界不等于免验证

默认 μ∈[-1,1]、σ∈[0.05,2]，归一化质量满足：

\[
C\ge\Phi(1)-\tfrac12\approx0.3413447461.
\]

最小值位于边界 μ 和最大 σ，因此没有极小截断归一化常数。
可独立参考以下非负 erf 求和形式检验 C：

\[
C=\tfrac12\left[
\operatorname{erf}\frac{1-\mu}{\sqrt2\sigma}
+\operatorname{erf}\frac{1+\mu}{\sqrt2\sigma}
\right].
\]

但“C 不小”不保证 inverse-CDF 的端点计算完美，也不保证浮点动作密度等于
量化动作概率。实现阶段须遵守：

1. 优先复用现有 truncnorm 数值内核和精度约定，以便隔离新参数化的影响；
   float64 作为独立参考。若需升级内核，单独评估其性能、旧类兼容性与对照影响。
2. 现有内核有 inverse-CDF 概率裁剪 `1e-6`、动作内缩 `1e-6` 和 C 下限 `1e-8`。
   这些是有限精度近似，不是理想截断分布的一部分；尤其概率裁剪会改变尾部样本。
   复用前必须量化影响，不能把它们当作“采样与真实密度精确匹配”的证明。
3. 新族的 sigma 映射不依赖任何事后 clip；在候选域内 C 下限不应触发。
   如果触发，应视为范围/实现异常，而非默默依赖保护项继续训练。
4. 对实际返回的 float32 动作统一评分，检查 sample/evaluate parity。
   非有限输入和明显越界动作必须报错，不能被继承的动作 clamp 掩盖。
5. 对有效域边缘做归一化、分位数、密度、ratio 和梯度误差扫描；检查被裁剪尾部
   的概率质量。必要时用量化区间 CDF 概率比作参考，不能只比较两条相同公式的输出。
6. Sigmoid 饱和时 σ 落在端点是合法数值退化，梯度可能为零；检查并统计，
   不以“所有 v 都有有效梯度”作为承诺。网络 m 的 tanh 饱和也仍然存在。
7. U 的 [0,1] clamp 仅作为已有接口的舍入保护；候选域距 1 有明显余量，
   显著越界意味着计算有误，不能用 clamp 掩盖。任意自定义边界必须重新验收。

理想 σ 映射及 U 单调性是解析保证；实际 float32 采样近似的误差和最终支持范围
仍待实现验收。若现有内核不满足误差预算，必须报告并选择稳定内核方案后再训练，
不能只因 0.05/2 看起来温和就跳过验证。

## 9. 诊断

保留既有 keys：`uncertainty`、`std_mean/min/max`、`eff_std_mean`、`mean_abs`。
std 在这里是未截断高斯 σ，不是最终动作标准差。

建议新增（只在 want_stats 时计算）：

| 指标 | 定义 |
|---|---|
| effective_uncertainty | 使用 μ、σ_e 的逐维 peak U，再平均；不参与 floor |
| eff_std_min/max | 实际评分/采样 σ 的范围 |
| raw_std_min/max | v 范围，定位 Sigmoid 饱和 |
| std_position_mean | mean(sigmoid(v))，log σ 区间内的位置 |
| std_lower/upper_saturation_frac | p<1e-3 / p>1-1e-3 的比例，阈值仅用于诊断 |
| log_std_sensitivity | mean(Delta_r * p * (1-p))，训练尺度响应 |
| exploration_sensitivity | mean(alpha * Delta_r * p_e * (1-p_e))，e 的局部响应 |
| sigma_state_std | `sigma.std(dim=0, correction=0).mean()`，只统计跨状态变化 |

共享版 sigma_state_std 理论为 0。不要用 `sigma.std()` 同时混合动作维度差异和
状态差异，也不要把有界 σ 解释为动作分布已经接近确定性或均匀。
这些 whole-batch stats 在框架 buffer 的更新前策略上采集，不是优化后的指标。

## 10. 导出、版本与兼容性

继续提供 self-contained `policy.py + model.pt + MANIFEST.json`；不依赖
`baseline.*` / `envs.*`，独立进程可加载。共享/State 导出复用同一份有界映射和
分布内核，通过架构分支加载不同网络，不维护两份手工复制的概率公式。

payload/manifest 至少包含：

- format_version、独立 policy_class / exported_class、架构维度、精确 state_dict keys。
- `distribution_kind = bounded_std_diagonal_truncated_normal_v1`。
- `std_source = shared | state`；`std_parameterization = sigmoid_log_std_v1`。
- `uncertainty_kind = marginal_peak_width_v1`。
- `exploration_kind = raw_std_additive_shift_v1`。
- sigma_min、sigma_max、init_std、**已解析的 explore_alpha**、动作边界。
- 数值实现版本及实际采用的精度、inverse-CDF/动作裁剪约定。

导出加载必须校验语义字段、配置合法性和严格权重键；缺失配置不能使用当前源码
默认值静默补齐。训练 checkpoint/resume 也必须校验配置一致性，不能只匹配张量形状。

旧共享 log_std 不能直接作为新 raw_std；在范围内可数学转换为
`logit((log_std-r_min)/Delta_r)`，但本版不提供隐式迁移，且边界外不可这样转换。
旧 State 的线性 head 输出 log_std，经过逆 logit 后通常无法用同一线性 head 精确表示。
旧 optimizer 状态也不适合直接复用；即使 e=0 分布匹配，新的探索和参数梯度仍不同。
因此默认从头训练，不把新 run 视为旧 run 的 bit-identical 续训。

## 11. 验证与对照实验计划

### 11.1 数学/单元验收（尚未执行实现级验收）

- [ ] σ 原始/有效范围、e=0 恒等、scalar 与 `(B,)` 广播、非法配置和非有限值。
- [ ] 正负 v、饱和区、不同 e 下的 σ/U 单调性；专门覆盖 v<0 防乘法回归。
- [ ] e 的均值位置不变；U 原始值不含 e；floor 对 μ、v 可微。
- [ ] 截断密度与 SciPy/独立积分对拍，含边界 μ、上下界 σ。
- [ ] U 对 quadrature，μ 对称性、位置导数符号、σ 导数正号与 gradcheck。
- [ ] 采样统计、实际动作评分一致、尾部裁剪质量及量化 ratio 误差。
- [ ] 共享/State 显式复制权重退化等价；State v 对状态有响应且梯度完整。
- [ ] 不重复网络前向、不将 σ 反推 v、不将 raw_std 当 log_std 导出。

### 11.2 接口/导出/集成验收

- [ ] 训练类与独立导出：act/sample/log_prob、scalar e、同 seed parity。
- [ ] 训练及导出 reset(seed) 重播；缺失/错误 metadata 与 state_dict 严格拒绝。
- [ ] CPU 与目标 GPU 测试、`.to(device)`、float32 buffer 规模测试。
- [ ] 两个 blueprint/experiment 能被 registry 发现，端到端 smoke。
- [ ] 若重构旧父类：现有全部正式 policy 测试回归，旧配置前后数值对拍。
- [ ] 每个新变体独立启动两次短 run，比对全部非 timing 指标；模型张量若声称
  bit-identical 也需直接比较，不能从日志相同推导 artifact 字节相同。
- [ ] 新类与旧类的长期性能比较不要求 bit-identical；性能和实现正确性分开验收。

### 11.3 最小对照矩阵

在同一任务、同一奖励、网络宽度、seed 和训练超参数下比较：

1. 旧共享 TruncatedNormal vs 新共享 bounded-std。
2. 旧 StateTruncatedNormal vs 新 State bounded-std。
3. 新共享 vs 新 State：仅原始 v 的来源不同。

注意新旧之间同时改变了 σ 范围、优化参数化和非零 e 的映射，不能把收益全部
归因于某一项。必要时用 e=0 对照和后续消融分离因素；不在同一首轮试验中再更换 U。
同初始 σ 不代表同有效学习率；应监控 KL/early-stop、ratio、梯度、饱和比例、
探索响应和任务成功率，并用多 seed 评估训练性能。

## 12. 本文设计阶段的公式核查记录

2026-09-24，使用独立内存脚本（未调用拟实现类、未改训练代码）核查候选配置：

- 80 组 μ/σ，U 闭式对数值积分，最大绝对误差 `3.33e-16`。
- 2,000 组 μ/v × 201 个 e，σ 映射对应的 U 单调性网格检查通过。
- e=0 的原始/有效映射数组完全相同。
- 对 μ/v/e 的 float64 U 与截断 log density 联合 gradcheck 通过。
- v_init、alpha、e=±1 初始 σ、U 可达界和 C 下界数值与本文一致。

这些核查支持公式和候选参数解释，不等于策略实现、导出、有限精度采样或训练已经
通过验收。§11 清单保持未勾选，实施时不得把上述数学检查替代代码与训练验证。

## 13. 实现记录（共享版，2026-09-25）

### 13.1 落地结构

- `bounded_std_truncated_normal_mlp.py::BoundedStdTruncatedNormalPolicy`：继承
  `TruncatedNormalPolicy`，仅覆盖 `_distribution_params` / `_build_stats` /
  `_export_extra` 与 `__init__`/`reset`；采样、评分、U、act/sample、导出骨架
  全部复用父类。
- 父类接缝（`truncated_normal_mlp.py`）：`_DistParams(mean, std_control,
  policy_sigma, eff_sigma)` + `_distribution_params(obs, e)`；基类由
  `_policy_params` + `effective_sigma` 构造，`std_control=log σ`。
  `_build_stats(uncertainty, params)`、`to_blueprint` 的 `_export_extra()`
  合并点（基类返回 `{}`，payload/manifest 不变）。
- 导出模板 `_export_template_bounded_std_truncnorm.py`：自包含，带
  `v_e = v + αe`、`σ(v)=exp(r_min+Δr·sigmoid(v_e))`，strict metadata 校验，
  `reset(seed)`。
- 接线：`init_policy_bounded_std_truncated_normal.yaml`、
  `exp_standup_floor04_boundedstd.py`、`policies/__init__.py` 注册。
- 实测：v_init=0.1644、α=1.1993，σ(e=−1/0/+1)=0.1315/0.3679/0.9436。

### 13.2 验收结果

- 父类重构 run 级 bit-identical：同一 session 内旧代码控制 run 与重构 run
  的 update 指标逐位一致（`kl_mean=0.05109779164195061`）。
  注意：与 2026-09-16~23 的历史 run（`kl_mean=0.05421725160704227`）存在
  run 级差异，但旧代码控制 run 同样得到 0.0511——是期间的环境变化
  （线程/库版本类），非本次重构引入。
- 测试：policies 全套 234 通过（bounded 43 + truncnorm + state 77 + 其他）；
  `policies/todo/` 目录有遗留 stale 测试（import 不存在的模块），与本次无关。
- 导出 parity：训练类与导出类 act/sample bit-identical，reset 重播一致。
- `standup_floor04_boundedstd` smoke 端到端通过，诊断统计符合设计
  （`std_mean=0.368≈e⁻¹`、`raw_std≈0.164`、`e=0` 时 `eff_std=std`、
  `uncertainty=0.455 > floor 0.4`）。

### 13.3 已知边界

- v 深饱和区（如 |v|>30）float32 下 sigmoid 输出 0/1，`exp(r_min)` 有
  ~1e-9 舍入，允许 σ 轻微越界；不做硬 clamp 以保留光滑参数化。
- e=0 时 `eff_sigma==policy_sigma` 逐元素成立（映射独立计算，同值）。
- State 版未实现；落地时只换 `_distribution_params` 的 v 来源，其余复用。
