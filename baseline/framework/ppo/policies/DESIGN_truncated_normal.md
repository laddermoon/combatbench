# Design: TruncatedNormalPolicy — 动作空间上的截断正态策略

## 1. 动机

当前 `TanhGaussianMLPPolicy` 在 pre-tanh 空间定义正态分布，再通过 tanh 映射到动作空间 `[-1, 1]`。这带来两个问题：

1. **熵不直观**：pre-tanh 熵是高斯解析熵，和动作空间的实际熵不一致；σ 大时高估策略不确定性。
2. **边界堆积**：mean 靠近 ±1 时，tanh 压缩导致概率密度在边界附近堆积，动作分辨率下降。

新策略直接在动作空间 `[-1, 1]` 上定义截断正态分布，消除这两个问题。

## 2. 分布定义

### 2.1 截断正态分布

策略网络输出 mean（经 tanh 映射到 `[-1, 1]`），σ 是全局可训练参数。

```
mean = tanh(net(obs))          ∈ (-1, 1)
σ    = exp(log_std)            > 0
```

动作分布是 `Normal(mean, σ)` 截断到 `[-1, 1]` 并重新归一化：

```
p(x) = φ((x - mean) / σ) / (σ × Z),   x ∈ [-1, 1]
Z = Φ((1 - mean) / σ) - Φ((-1 - mean) / σ)
```

其中 φ 是标准正态 PDF，Φ 是标准正态 CDF，Z 是截断归一化项。

`p(x)` 在 `[-1, 1]` 上的积分 = 1。`[-1, 1]` 之外概率为零。

### 2.2 采样（重参数化）

```
a = Φ((-1 - mean) / σ)
b = Φ(( 1 - mean) / σ)
u ~ Uniform(a, b)
ε = Φ⁻¹(u)              # 逆 CDF，可微
action = mean + σ × ε   ∈ [-1, 1]
```

梯度通过 Φ⁻¹ 流到 mean 和 σ，重参数化成立。

### 2.3 log_prob

```
log p(action) = Normal.log_prob(action) - log(Z)
```

`Normal.log_prob` 是标准高斯 log 密度，`log(Z)` 是截断归一化项。两者都可微。

## 3. 不确定度定义

### 3.1 几何直觉

在动作空间 `[-1, 1]` 上画一个矩形：
- 底边：y = 0（x 轴）
- 左边：x = -1
- 右边：x = +1
- 上边：y = peak（分布 PDF 的最大值）

**不确定度 U = 分布曲线下面积 / 矩形面积 = 1 / (2 × peak)**

因为分布曲线下面积 = 1（概率归一化），矩形面积 = 2 × peak。

### 3.2 性质

| 分布 | peak | U | 含义 |
|---|---|---|---|
| 均匀分布 U[-1,1] | 0.5 | 1.0 | 完全不确定 |
| delta 分布 | ∞ | 0.0 | 完全确定 |
| 窄高斯 | 大 | 接近 0 | 较确定 |
| 宽高斯 | 小 | 接近 1 | 较不确定 |

- 范围天然 [0, 1]，不需要人为归一化参考点
- 0 = 完全确定，1 = 完全不确定

### 3.3 解析公式

peak 的位置取决于 mean 相对于 `[-1, 1]` 的位置：

**mean ∈ [-1, 1]**（peak 在 x = mean 处）：
```
peak = 1 / (σ × √(2π) × Z)
U = σ × √(2π) × Z / 2
```

**mean < -1**（peak 在 x = -1 处）：
```
peak = φ(a) / (σ × Z)
U = σ × Z / (2 × φ(a))
```

**mean > 1**（peak 在 x = +1 处）：
```
peak = φ(b) / (σ × Z)
U = σ × Z / (2 × φ(b))
```

其中 `a = (-1 - mean) / σ`, `b = (1 - mean) / σ`, `Z = Φ(b) - Φ(a)`。

### 3.4 可导性

以 mean ∈ [-1, 1] 为例：
```
U = σ × √(2π) × Z / 2

∂U/∂mean = √(2π) / 2 × (φ(b) + φ(a))
```

对 σ 也可导（链式法则）。梯度能流到 mean 和 σ。

**注意**：mean 跨过 ±1 时 peak 位置切换，函数连续但导数不连续。实践中 mean 很少恰好在 ±1，通常不构成问题。

### 3.5 多维扩展

action_dim 维（各维独立，共享 σ）：
```
U_total = 1 / (2^d × ∏_i peak_i)
```

或等价地用几何平均：
```
U_total = ∏_i U_i = ∏_i 1 / (2 × peak_i)
```

当各维同分布时：`U_total = U_per_dim^action_dim`

## 4. 与 TanhGaussianMLPPolicy 的对比

| 特性 | TanhGaussianMLPPolicy | TruncatedNormalPolicy |
|---|---|---|
| 分布定义空间 | pre-tanh（实数轴） | 动作空间 [-1,1] |
| 分布类型 | 正态 → tanh 变换（非正态） | 截断正态（在 [-1,1] 上是正态形状） |
| 采样 | Normal + tanh | 截断正态重参数化 |
| log_prob | Normal.lp - log(1-tanh²) | Normal.lp - log(Z) |
| 熵 | pre-tanh 解析熵（不反映动作空间实际熵） | 不用熵，用不确定度 U |
| 不确定度 | H_norm（人为归一化，[0,1] 是参考点非极值） | U（天然 [0,1]，0=确定，1=均匀） |
| per-obs | 否（H 只依赖 σ） | 是（U 依赖 mean 和 σ） |
| 边界处理 | tanh 压缩，概率堆积 | 精确截断，无堆积 |
| entropy_floor 目标 | H_norm（人为标尺） | U（几何意义明确） |

## 5. 网络结构

与 TanhGaussianMLPPolicy 相同：

```
输入: obs (obs_dim维)
  │
  ├──→ net (mean 网络)
  │     Linear(obs_dim, hidden_dim)
  │     Tanh
  │     Linear(hidden_dim, hidden_dim)
  │     Tanh
  │     Linear(hidden_dim, action_dim)
  │     → raw_mean
  │
  │     tanh(raw_mean)  → mean ∈ (-1, 1)
  │
  ├──→ log_std (全局参数)
  │     (action_dim,) 初始值 -1.0
  │     → σ = exp(log_std)
  │
  └──→ 截断正态分布 TruncNormal(mean, σ, -1, 1)
         │
         ▼
       action ∈ [-1, 1]
```

mean 经过 tanh 确保在 `(-1, 1)` 内，这样 peak 总是在 mean 处（不需要处理 mean 在边界外的情况）。但为了数值稳定性，实现时仍处理 mean 接近边界的情况。

## 6. 设计决策

### 6.1 多维聚合：算术平均

```
U_total = mean(U_i),  i = 1..action_dim
```

算术平均对单个维度的确定度不敏感——一个维度确定不会过度拉低整体。
几何平均会对低维度过度敏感，不适合。

### 6.2 loss 形式：不在策略关心范围

策略只负责输出 per-obs 不确定度 U。loss 的形式（单向 MSE、单向 L1、
双向等）由框架（trainer）决定。策略文档不定义 loss。

### 6.3 explore_intensity：保留，线性缩放 σ

explore_intensity 是框架要求的探索控制接口，必须保留。语义改为
直接缩放 σ（不再用 log_std offset）：

```
# 分段线性插值，三个锚点：0→1/3, 0.5→1, 1→3
if ei <= 0.5:
    scale = 1/3 + (ei / 0.5) × (1 - 1/3)
else:
    scale = 1 + ((ei - 0.5) / 0.5) × (3 - 1)
σ_effective = σ × scale
```

| explore_intensity | scale | 含义 |
|---:|---:|---|
| 0.0 | 1/3 ≈ 0.333 | 最大压缩：σ 除以 3 |
| 0.25 | ≈ 0.667 | 中度压缩 |
| 0.5 | 1.0 | 中性：不改变 σ |
| 0.75 | 2.0 | 中度扩张 |
| 1.0 | 3.0 | 最大扩张：σ 乘以 3 |
| 中间 | 分段线性插值 | 平滑过渡 |

这和 TanhGaussianMLPPolicy 的 offset 语义不同——之前是在 log_std 空间
加偏移，现在是直接乘 σ。因为截断正态的 σ 已经在动作空间，直接缩放更自然。

采样和 log_prob 用 `σ_effective`，不确定度 U 用策略原始 σ（不含
explore_intensity 缩放），反映策略自身的确定度。

### 6.4 checkpoint：不兼容

全新策略，不从 TanhGaussianMLPPolicy 的 checkpoint 加载。网络结构
相同（mean 网络 + log_std 参数），但分布定义、采样、log_prob、
不确定度都不同，是独立的策略类。
