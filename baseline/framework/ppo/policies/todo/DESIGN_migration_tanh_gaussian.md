# Design: TanhGaussianMLPPolicy 迁移到新接口

Reads `DESIGN_unified_exploration_control.md` (框架层新接口设计) as prerequisite.

## 0. 为什么选这个策略作为第一个

`TanhGaussianMLPPolicy` 是迁移到新接口的最简单策略：

- **不继承基类**：独立实现 `nn.Module + Policy`，改动不波及其他策略。
- **无 OU 探索**：不需要处理 `noise_shift`、`noise_scale`、`noise_tau_steps`。
- **有解析熵**：`Normal.entropy()` 是 closed-form，不需要采样估计。
- **σ 状态无关**：`log_std` 是全局 `(action_dim,)` 参数，归一化只需常量 `H_max`/`H_min`。
- **是 Baseline**：81 个已导出的 policy 文件依赖它的参数格式，必须保证 checkpoint 兼容。

## 1. 当前接口 vs 新接口

| 维度 | 当前 | 新接口 |
|---|---|---|
| `ActorEval` 字段 | `log_prob`, `regularizer`, `stats` | `log_prob`, `entropy`, `stats` |
| `regularizer` | 策略算 `-entropy_coef * H.mean()`，标量 loss | 移除，框架从 `entropy` 自己算 hinge loss |
| `entropy` | 不存在 | `(B,)` per-obs，可导，归一化到 [0,1] |
| `set_exploration` | `(spec: ExplorationSpec) -> Dict[str, float]` | `(explore_intensity: float) -> None` |
| `set_exploration` 接收 | `temperature`, `entropy_coef` | 只接收 `explore_intensity` |
| `entropy_coef` | 策略持有 | 框架持有 |
| `temperature` | 策略持有，`log_std += log(temperature)` | 移除，被 `explore_intensity` 偏移量映射替代 |
| `_log_std_offset` | 由 `temperature` 设置 | 由 `explore_intensity` 偏移量映射设置 |

## 2. explore_intensity 和 entropy 的语义

### explore_intensity：对称 temperature 控制

`explore_intensity` 是一个对称的 temperature-like 控制，以 0.5 为中性点：

- **0.5 = 中性**：offset=0，σ 就是策略自己学的值。策略完全自由表达。
- **→ 0 = 挤压**：offset < 0，σ 变小。ei=0 时 offset = -EXPLORE_SPAN/2，
  σ 缩放为 exp(-1) ≈ 0.37x。
- **→ 1 = 扩平**：offset > 0，σ 变大。ei=1 时 offset = +EXPLORE_SPAN/2，
  σ 缩放为 exp(+1) ≈ 2.72x。

```python
EXPLORE_SPAN = 2.0  # offset 范围 ±1.0，σ 缩放 0.37x ~ 2.72x

self._log_std_offset = (explore_intensity - 0.5) * EXPLORE_SPAN
effective_log_std = self.log_std + self._log_std_offset
```

#### EXPLORE_SPAN 的含义和选择

`EXPLORE_SPAN` 控制 offset 的最大幅度，也就是 σ 能被缩放多少倍。
它决定了 explore_intensity 两端的极端效果：

| EXPLORE_SPAN | ei=0 (压缩极) | ei=0.5 (中性) | ei=1 (扩平极) | 有用 ei 范围 |
|---|---|---|---|---|
| 2.0（当前） | σ × 0.37 | σ × 1.0 | σ × 2.72 | 全范围 [0, 1] |
| 3.0 | σ × 0.22 | σ × 1.0 | σ × 4.48 | 全范围 [0, 1] |
| 5.0（旧设计） | — | σ × 1.0 | σ × 148x | 仅 [0, 0.2] |

**为什么选 2.0**：

以策略初始 log_std=-1（σ≈0.37）为例，21 维 tanh 动作空间：

- **ei=0（压缩极）**：σ ≈ 0.37 × 0.37 = 0.14。pre-tanh 采样集中在 mean 附近 ±0.14，
  tanh 后动作变化量约 ±0.14。接近确定性执行，但保留微弱噪声避免 PPO ratio 退化。
- **ei=0.5（中性）**：σ ≈ 0.37。策略完全自由表达，这是默认训练状态。
- **ei=1（扩平极）**：σ ≈ 0.37 × 2.72 = 1.0。pre-tanh 采样在 mean 附近 ±1.0，
  tanh 后动作覆盖大部分 [-1, 1] 范围。强探索但不是完全随机——策略 mean 仍有影响。

如果 span=5.0，ei=1 时 σ ≈ 0.37 × 148 = 54.6，pre-tanh 采样范围 ±54.6，
tanh 后动作几乎完全均匀分布，策略 mean 被彻底淹没。这在 PPO 中没有意义——
on-policy 假设要求采样分布和策略分布有足够重叠，σ 放大 148 倍后重叠几乎为零，
PPO ratio 爆炸，训练崩溃。

span=2.0 让整个 [0, 1] 区间都有实际意义，调参时不需要在 [0, 0.2] 的窄区间里微调。

**如何判断 span 是否合适**：观察训练日志中的 `eff_std_mean`。
如果 ei=1 时 `eff_std_mean` 超过 2.0（pre-tanh σ > 2.0 意味着 tanh 接近饱和），
说明 span 偏大；如果 ei=1 时探索效果不够（episode 长度没有变化），可以尝试加大 span。

#### 为什么是 log 空间加法而不是 σ 空间乘法

两者数学等价（`log_std + offset` ≡ `σ × exp(offset)`），但 log 空间加法有三个优势：
1. `log_std` 本身就是网络参数，offset 和梯度在同一坐标系
2. entropy 在 log 空间是线性的（`H = 0.5×log(2πe) + log_std`），offset 对熵的影响可预测
3. 数值更稳定，clamp ±20 就够了

#### 警告

`ei=0` 会将 σ 压缩到 ~0.37x，接近确定性采样。只在需要时使用，
安全默认值是 0.5。如果误设 ei=0，探索几乎消失，PPO 可能因采样过于集中而卡住。

#### explore_intensity 数值速查表

> **目标**：给一个数字就能有体感，想深入时往下看公式。
> σ_scale = exp((ei - 0.5) × EXPLORE_SPAN)，EXPLORE_SPAN=2.0。
> "体感"列描述 rollout 时的采样随机性相对于策略自身 σ 的倍率。

| ei | offset | σ_scale | 体感 | 典型用途 |
|---:|---:|---:|---|---|
| 0.00 | -1.00 | 0.37× | **近确定性**：动作几乎完全跟随 mean，仅保留微弱噪声避免 PPO ratio 退化 | 评估/部署；训练中慎用，易卡住 |
| 0.05 | -0.90 | 0.41× | 极低噪声：比 native 稍安静，动作可预测性很高 | 精细控制任务收敛后期 |
| 0.10 | -0.80 | 0.45× | 低噪声：明显比 native 收敛，但仍有可辨识的随机性 | 已收敛策略的微调 |
| 0.15 | -0.70 | 0.50× | 半倍 σ：采样范围缩窄一半，动作比较确定 | 精细控制任务中期 |
| 0.20 | -0.60 | 0.55× | 偏安静：比 native 略收敛，探索性减弱 | 需要稳定但不卡住时 |
| 0.25 | -0.50 | 0.61× | 轻度压缩：采样范围约为 native 的 60% | 收敛后期的温和探索 |
| 0.30 | -0.40 | 0.67× | 轻度压缩：接近 native 但略安静 | — |
| 0.35 | -0.30 | 0.74× | 微压缩：几乎和 native 一样，略收一点 | — |
| 0.40 | -0.20 | 0.82× | 极微压缩：肉眼几乎不可辨差异 | — |
| 0.45 | -0.10 | 0.90× | 几乎中性：和 native 非常接近 | — |
| **0.50** | **0.00** | **1.00×** | **中性（native）**：策略完全自由表达，σ 就是它自己学的值 | **默认训练值** |
| 0.55 | +0.10 | 1.11× | 几乎中性：和 native 非常接近，略多一点噪声 | — |
| 0.60 | +0.20 | 1.22× | 轻度扩张：比 native 略吵，探索性微增 | 需要打破局部最优时 |
| 0.65 | +0.30 | 1.35× | 温和扩张：采样范围约为 native 的 1.35 倍 | 已收敛策略学新技能 |
| 0.70 | +0.40 | 1.49× | 半倍扩张：探索性明显增强，动作变化幅度约 native 的 1.5 倍 | 强探索学习 |
| 0.75 | +0.50 | 1.65× | 显著扩张：σ 放大 1.65 倍，动作明显更随机 | 强探索学习 |
| 0.80 | +0.60 | 1.82× | 强扩张：接近 2 倍 σ，动作随机性显著 | 打破顽固局部最优 |
| 0.85 | +0.70 | 2.01× | 强扩张：σ 翻倍，pre-tanh 采样范围约 ±2σ | 冷启动强探索 |
| 0.90 | +0.80 | 2.23× | 很强扩张：σ 超过 2 倍，tanh 开始有饱和风险 | 冷启动；注意 tanh_sat_frac |
| 0.95 | +0.90 | 2.46× | 极强扩张：接近最大探索，tanh 饱和概率上升 | 极端冷启动 |
| 1.00 | +1.00 | 2.72× | **最大探索**：σ 放大 e 倍，pre-tanh 采样范围 ±2.7σ，tanh 后动作覆盖大部分 [-1,1] | 最大探索；注意 tanh_sat_frac |

**快速记忆**：
- `0.5` = native，`0.0` ≈ 1/3 σ，`1.0` ≈ 3× σ
- 每 `0.1` 的 ei 变化 ≈ σ 乘除 ~1.22（即 ±22%）
- `ei < 0.3` 或 `ei > 0.8` 属于"明显偏离 native"，需要明确理由

**诊断指标**：训练日志中的 `eff_std_mean` 是含探索偏移的 σ。
如果 `eff_std_mean > 2.0`，说明 tanh 接近饱和，explore_intensity 偏高或 span 偏大。

### log_std_min / log_std_max 的含义

- **log_std_min = -4.0**（σ ≈ 0.018）：熵归一化的下界参考点。策略 σ 接近这里
  意味着近确定性。
- **log_std_max = 1.0**（σ ≈ 2.7）：熵归一化的上界参考点。

  > 保持默认 1.0 不变。`log_std_max` 不再需要代表"tanh 输出接近均匀"的 σ，
  > 它只是归一化参考点。explore_intensity 的缩放范围由 `EXPLORE_SPAN` 控制，
  > 不再依赖 `log_std_max - log_std_min`。

### entropy：策略自身分布的确定性

entropy 用**不含探索偏移**的 σ 计算，反映策略自身的确定性判断：

```python
# entropy 用策略原始 σ（不含 explore offset）
policy_log_std = self.log_std  # 不加 _log_std_offset
entropy_raw = Normal(mean, policy_log_std.exp()).entropy().sum(dim=-1)
```

这样 entropy 不受 `explore_intensity` 影响——不管实验叠加了多少探索噪声，
entropy 始终反映策略自己认为的确定性。entropy_floor 约束的是策略内在分布，
不是采样分布。

### explore_intensity 和 entropy_floor 的关系

两者不在同一个坐标系，这是**有意为之**：

- `explore_intensity`：相对偏移，在策略 σ 上叠加。控制采样时的额外随机性。
- `entropy_floor`：绝对下界，约束策略自身分布的熵。控制训练时的防坍缩。

- `explore_intensity=0.5, entropy_floor=0.3`：采样用策略自然分布，但训练时不允许
  策略 σ 的熵低于 30%——hinge loss 防止策略坍缩。
- `explore_intensity=0.8, entropy_floor=0.1`：采样时扩平 σ 去探索，但允许
  策略自身快速收敛到低熵——hinge 只在策略熵极低时干预。

## 3. 不 clamp：防坍缩由 entropy floor 接管

### 为什么去掉业务 clamp

旧设计用 `torch.clamp(log_std, log_std_min, log_std_max)` 限制 σ 范围。但
`Normal.entropy()` per-dim = `0.5 * log(2πe) + log_std`，当 `log_std` 被 clamp
到边界时，熵变成常数，**梯度为零**。

这正好是防坍缩最需要起作用的时刻——策略把 σ 推到下界，hinge loss 想把 σ 拉回来，
但梯度死了，拉不动。

新接口下，防坍缩已经由 entropy floor（`relu(entropy_floor - H_norm)`）接管。
业务 clamp 是多余的——它只会制造梯度死区，阻止 entropy floor 工作。

### 数值安全边界

不 clamp 不意味着完全不限制。`exp(log_std)` 在 `log_std` 极大时会溢出（`exp(100)=inf`），
极小时会归零（`exp(-100)=0`），导致 `Normal` 的 `log_prob` / `entropy` 产生 NaN。

保留一个极宽松的数值安全边界，实际训练中永远不会触及：

```python
_LOG_STD_SAFE_MIN = -20.0  # exp(-20) ≈ 2e-9
_LOG_STD_SAFE_MAX = 20.0   # exp(20) ≈ 5e8

def effective_log_std(self) -> torch.Tensor:
    return torch.clamp(
        self.log_std + self._log_std_offset,
        _LOG_STD_SAFE_MIN,
        _LOG_STD_SAFE_MAX,
    )
```

这个 clamp 只在数值异常时激活，正常训练中 `log_std` 不会接近 `±20`。
`log_std_min` / `log_std_max` 不再用于 clamp，只用于 explore_intensity 映射
和熵归一化（见 §4）。

## 4. 熵归一化

### 原始熵

entropy 用**策略原始 σ**（不含 explore offset）计算：

```python
policy_log_std = self.log_std  # 不加 _log_std_offset
H_raw = Normal(mean, policy_log_std.exp()).entropy().sum(dim=-1)  # (B,) in nats
```

### H_max 和 H_min

```
H_max = action_dim × (0.5 × log(2πe) + log_std_max)
H_min = action_dim × (0.5 × log(2πe) + log_std_min)
```

`H_norm=0` 对应策略 σ 在 `log_std_min`（近确定性），`H_norm=1` 对应策略 σ 在
`log_std_max`。0 和 1 是策略 σ 预期工作范围的标尺。

### 归一化

```python
H_norm = (H_raw - H_min) / (H_max - H_min)
```

- 策略 σ 在 `log_std_max` → `H_norm = 1.0`
- 策略 σ 在 `log_std_min` → `H_norm = 0.0`
- 初始 `log_std=-1.0`，`[-4, 1]` 范围 → `H_norm ≈ 0.6`
- 策略 σ 试图低于 `log_std_min` → `H_norm < 0`，hinge loss 激活，梯度把 σ 推回

`H_norm` 可以超出 `[0, 1]`——这是有用信号，表示策略正在试图突破预期工作范围。
hinge loss 的梯度正好能把它推回来。

**注意：H_norm 不受 `explore_intensity` 影响**。不管实验叠加了多少探索噪声，
entropy 始终反映策略自身分布的熵。这是 entropy_floor 能有效防坍缩的前提——
它约束的是策略内在属性，不是采样时的临时噪声。

### entropy_floor 数值速查表

> **目标**：给一个 floor 值就能有体感，想深入时往下看公式。
> `entropy_floor` 约束的是 `H_norm`（归一化熵），不是 `entropy_raw`（nats）。
> H_norm 的物理含义：策略 σ 在 `[log_std_min, log_std_max]` 范围内的相对位置。
> H_norm=0 → σ=log_std_min（最确定），H_norm=1 → σ=log_std_max（最随机）。
>
> 下表以 **standup 配置**（`log_std_min=-2.5, log_std_max=0.0, action_dim=21`）
> 为例。其他配置的 H_norm → σ 映射不同，但 H_norm 本身的语义不变。

| floor | 对应 σ | 对应 log_std | 体感 | 典型用途 |
|---:|---:|---:|---|---|
| 0.00 | 0.082 | -2.50 | **无约束**：策略可以收敛到任意窄的分布，完全确定性也允许 | 已知最优行为不需要探索时（如 standup baseline） |
| 0.05 | 0.093 | -2.38 | 极低容忍：允许策略几乎完全确定，仅在极端坍缩时托底 | 精细控制任务，需要极低 σ |
| 0.10 | 0.105 | -2.25 | 很低容忍：策略可以非常确定，但不会到极限 | 收敛后期的防坍缩 |
| 0.15 | 0.119 | -2.13 | 低容忍：策略确定性强，σ 约 0.12 | — |
| 0.20 | 0.135 | -2.00 | 偏低容忍：σ 约 0.14，动作比较可预测 | 精细控制 + 轻度防坍缩 |
| 0.25 | 0.153 | -1.88 | 中低容忍：σ 约 0.15，确定性和探索性的平衡偏确定性 | — |
| 0.30 | 0.174 | -1.75 | **中等偏低**：σ 约 0.17，接近 standup 收敛值 | standup 收敛后的维持 |
| 0.35 | 0.197 | -1.63 | 中等：σ 约 0.20，确定性和探索性较平衡 | 通用防坍缩默认值 |
| 0.40 | 0.223 | -1.50 | 中等偏高：σ 约 0.22，策略被要求保持一定随机性 | 需要持续探索的任务 |
| 0.45 | 0.253 | -1.38 | 偏高：σ 约 0.25，策略不能太确定 | 多技能学习，防止锁定单一行为 |
| 0.50 | 0.287 | -1.25 | **中等偏高**：σ 约 0.29，策略被要求保持中等随机性 | 需要较强探索的任务 |
| 0.55 | 0.325 | -1.13 | 偏高：σ 约 0.32，策略随机性较强 | — |
| 0.60 | 0.368 | -1.00 | 高：σ 约 0.37，策略被强制保持较高随机性 | 强探索任务 |
| 0.65 | 0.417 | -0.88 | 很高：σ 约 0.42，策略几乎不能收敛 | — |
| 0.70 | 0.472 | -0.75 | 很高：σ 约 0.47，策略被强制保持高随机性 | 极端防坍缩 |
| 0.75 | 0.535 | -0.63 | 极高：σ 约 0.54，策略随机性很强 | — |
| 0.80 | 0.607 | -0.50 | 极高：σ 约 0.61，策略几乎无法确定 | — |
| 0.85 | 0.687 | -0.38 | 接近上限：σ 约 0.69 | — |
| 0.90 | 0.779 | -0.25 | 接近上限：σ 约 0.78，策略被强制接近最大随机性 | — |
| 0.95 | 0.882 | -0.13 | 几乎上限：σ 约 0.88 | — |
| 1.00 | 1.000 | 0.00 | **上限**：σ=log_std_max，策略被强制保持最大随机性 | 理论值，实际不使用 |

**快速记忆**：
- `0.0` = 无约束（策略自由收敛），`0.3` = 中等偏低（standup 收敛值附近），`0.5` = 中等
- `floor` 直接对应 σ 在 `[log_std_min, log_std_max]` 范围内的相对位置
- standup 收敛后 `H_norm ≈ 0.29`，所以 `floor=0.30` 刚好在收敛值附近托底
- `floor > 0.5` 属于"强制高随机性"，需要明确理由

**诊断指标**：训练日志中的 `entropy_raw`（nats）可以换算成 `H_norm`：
```
H_norm = (entropy_raw - H_min) / (H_max - H_min)
```
如果 `H_norm` 持续低于 `floor`，hinge loss 应该在推它回来；
如果推不动，说明 `entropy_coef` 太小或 PPO 梯度太强。

**注意：不同配置的 H_norm → σ 映射不同**。上表以 standup 配置
（`log_std_min=-2.5, log_std_max=0.0`）为例。默认配置
（`log_std_min=-4.0, log_std_max=1.0`）下，同样的 `H_norm=0.3`
对应 `σ=0.082`（更确定），因为归一化范围更大。

### 4.1 σ 的物理含义：概率覆盖与动作范围

> **目的**：防止对 σ 数值产生"体感偏差"。σ=0.17 不是"已经很确定"，
> 而是"较确定但仍有明显随机性"。下面给出量化标尺。

策略的 pre-tanh 分布是 `Normal(mean, σ)`，post-tanh 输出在 `[-1, 1]`。
动作空间总跨度为 2。高斯分布的概率覆盖：

| 区间 | 覆盖概率 | 含义 |
|---|---|---|
| mean ± 1σ | **68.3%** | 大部分采样落在这里 |
| mean ± 2σ | **95.4%** | 绝大部分采样落在这里 |
| mean ± 3σ | **99.7%** | 几乎所有采样落在这里 |

tanh 变换在小 σ 时近似线性（`tanh(x) ≈ x`），所以 post-tanh 的动作范围
≈ pre-tanh 的 σ 范围。σ 越大 tanh 压缩越明显，post-tanh 范围会小于 pre-tanh。

**σ → 概率覆盖 → 动作范围** 完整对照表（standup 配置，action_dim=21）：

> 动作空间为 `[-1, 1]`，总跨度 2。表中"范围"为 post-tanh（实际动作）的边界，
> "占动作空间"为 `2 × 范围 / 2.0 × 100%`。覆盖概率对 pre-tanh 和 post-tanh
> 相同（tanh 单调，不改变概率质量分配）。

| σ | log_std | H_norm | 1σ 范围 | 1σ 概率 | 1σ 占动作空间 | 2σ 范围 | 2σ 概率 | 2σ 占动作空间 | 体感 |
|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| 0.018 | -4.00 | -0.61 | ±0.018 | 68.3% | 1.8% | ±0.036 | 95.4% | 3.6% | 近确定性 |
| 0.050 | -3.00 | -0.20 | ±0.050 | 68.3% | 5.0% | ±0.100 | 95.4% | 10.0% | 很确定 |
| 0.082 | -2.50 | 0.00 | ±0.082 | 68.3% | 8.2% | ±0.163 | 95.4% | 16.3% | 确定 |
| 0.120 | -2.12 | 0.15 | ±0.119 | 68.3% | 11.9% | ±0.236 | 95.4% | 23.5% | 较确定 |
| **0.175** | **-1.74** | **0.30** | **±0.173** | **68.3%** | **17.3%** | **±0.336** | **95.4%** | **33.6%** | **较确定（standup 收敛值）** |
| 0.250 | -1.39 | 0.45 | ±0.245 | 68.3% | 24.5% | ±0.462 | 95.4% | 46.2% | 中等 |
| 0.368 | -1.00 | 0.60 | ±0.352 | 68.3% | 35.2% | ±0.627 | 95.4% | 62.7% | 中等（初始 log_std=-1.0） |
| 0.500 | -0.69 | 0.72 | ±0.462 | 68.3% | 46.2% | ±0.762 | 95.4% | 76.2% | 较随机 |
| 0.607 | -0.50 | 0.80 | ±0.542 | 68.3% | 54.2% | ±0.838 | 95.4% | 83.8% | 随机 |
| 0.800 | -0.22 | 0.91 | ±0.664 | 68.3% | 66.4% | ±0.922 | 95.4% | 92.2% | 随机 |
| 1.000 | 0.00 | 1.00 | ±0.762 | 68.3% | 76.2% | ±0.964 | 95.4% | 96.4% | 很随机 |

**如何读这张表**：

以 standup 收敛值 σ=0.175 为例：
- **1σ**：68.3% 的采样落在 `mean ± 0.173` 内，占动作空间 17.3%
- **2σ**：95.4% 的采样落在 `mean ± 0.336` 内，占动作空间 33.6%
- 含义：每个关节的动作在 mean 附近 ±0.17 波动（1σ），偶尔到 ±0.34（2σ）
- **这不是"已经确定"——三分之一的动作空间仍在 2σ 范围内被探索**

**关键解读**：

- **σ=0.175（standup 收敛值）**：1σ 覆盖 17%、2σ 覆盖 34% 的动作空间。
  每个关节的动作有明显的随机波动。归一化熵 0.30 处于 [0,1] 的 30% 位置，
  偏确定但远未到极端。

- **"近确定性"的门槛**：σ < 0.05（1σ < 5%，2σ < 10% 动作空间），
  对应 H_norm < -0.2（standup 配置下）。这才是策略真正"几乎确定"的区域。
  standup 收敛的 σ=0.175 离这个门槛还有 3.5 倍的差距。

- **σ=0.368（初始值）**：1σ 覆盖 35%、2σ 覆盖 63% 的动作空间。
  这是策略的默认起点，随机性中等，大部分动作空间都在探索范围内。

- **σ=1.0（log_std_max）**：1σ 覆盖 76%、2σ 覆盖 96% 的动作空间。
  tanh 后接近均匀分布，策略几乎完全随机。

**为什么 standup 能收敛到 σ=0.175 而不是更低**：

PPO 的 policy gradient 对 log_std 的推力是 `E[adv × (z² - 1)]`，
其中 `z = (action - mean) / σ`。理论上 `E[z²] = 1`（因为 action 从
`Normal(mean, σ)` 采的），正负抵消。但实践中 advantage 和 z 不是独立的，
收敛后高 advantage 动作的 adv 很小（value function 已经预测到了），
使得净梯度略偏正（推 σ↓），但推力很小——σ 在 0.175 附近达到平衡，
而不是继续降到 0。这是 PPO 的固有特性，不是 bug。

## 5. 具体改动

### 5.1 `__init__`

```python
def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    log_std_min: float = DEFAULT_LOG_STD_MIN,
    log_std_max: float = DEFAULT_LOG_STD_MAX,
    device: torch.device | str = "cpu",
    deterministic: bool = False,
    model_path: Optional[str] = None,
):
```

移除：`log_std_offset`, `entropy_coef`。
`_log_std_offset` 仍保留为内部状态，初始值 `0.0`，由 `set_exploration` 设置。
`log_std_min` / `log_std_max` 保留，语义从"硬截断边界"变为"偏移量上限 + 熵归一化参考点"。

### 5.2 `effective_log_std`（改为数值安全 clamp）

```python
_LOG_STD_SAFE_MIN = -20.0
_LOG_STD_SAFE_MAX = 20.0

def effective_log_std(self) -> torch.Tensor:
    return torch.clamp(
        self.log_std + self._log_std_offset,
        _LOG_STD_SAFE_MIN,
        _LOG_STD_SAFE_MAX,
    )
```

从"业务边界 clamp"改为"数值安全 clamp"。正常训练中不会触及 `±20`。

### 5.3 `evaluate_actions`

```python
def evaluate_actions(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    noise_shift: Optional[torch.Tensor] = None,
    want_stats: bool = False,
) -> ActorEval:
    # log_prob 用 effective σ（含 explore offset），PPO ratio 正确
    clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
    raw_actions = torch.atanh(clipped_actions)
    mean, eff_log_std = self.forward(obs)  # eff_log_std 含 offset
    dist = Normal(mean, eff_log_std.exp())
    log_prob = (dist.log_prob(raw_actions) - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)).sum(dim=-1)

    # entropy 用策略原始 σ（不含 explore offset），反映策略自身确定性
    policy_log_std = self.log_std  # 不加 _log_std_offset
    entropy_raw = Normal(mean, policy_log_std.exp()).entropy().sum(dim=-1)  # (B,)
    H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_max)
    H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_min)
    entropy_norm = (entropy_raw - H_min) / (H_max - H_min)

    # stats
    stats = None
    if want_stats:
        with torch.no_grad():
            eff_std = eff_log_std.exp()
            policy_std = policy_log_std.exp()
            stats = {
                "entropy_raw": float(entropy_raw.mean().item()),
                "std_mean": float(policy_std.mean().item()),       # 策略自身 σ
                "eff_std_mean": float(eff_std.mean().item()),      # 含探索偏移的 σ
                "std_min": float(policy_std.min().item()),
                "std_max": float(policy_std.max().item()),
                "tanh_sat_frac": float((mean.abs() > 2.0).float().mean().item()),
            }

    return ActorEval(log_prob=log_prob, entropy=entropy_norm, stats=stats)
```

log_prob 和 entropy 用同一个 `dist`（同一个 σ），不再分两路计算。
移除：`regularizer` 构造、`_entropy_coef` 判断。`frame_modes` 保留（协议级参数，本策略忽略）。

### 5.4 `set_exploration`

```python
EXPLORE_SPAN = 2.0  # offset 范围 ±1.0

def set_exploration(self, explore_intensity: float) -> None:
    # 对称映射：0.5 = 中性 (offset=0), 0 = 挤压, 1 = 扩平
    self._log_std_offset = (explore_intensity - 0.5) * self.EXPLORE_SPAN
```

移除：`spec` 参数、`Dict` 返回值、`temperature`/`entropy_coef` 分支。

### 5.5 移除的代码

- `_entropy_coef` 字段及 `evaluate_actions` 中的 regularizer 构造
- `set_exploration` 中的 `spec.temperature` / `spec.entropy_coef` 分支
- `evaluate_actions` 的 `frame_modes` 参数（当前已忽略，新接口移除）
- `from baseline.framework.ppo import ExplorationSpec` 导入（不再需要）

## 6. Checkpoint 兼容性

### 网络参数

`net.*` 和 `log_std` 不变，`strict=True` / `strict=False` 加载都正常。

### 移除的字段

`_entropy_coef`, `_log_std_offset` 是 plain floats，不在 `state_dict` 中。

### 新增的字段

无新增 `nn.Parameter` 或 `buffer`。

### 结论

**checkpoint 完全兼容**。81 个已导出的 policy 文件不受影响——它们加载的是
`state_dict`（网络权重），不涉及探索状态或 `ActorEval` 结构。

## 7. 迁移检查清单

- [ ] `__init__` 移除 `log_std_offset` / `entropy_coef` 参数
- [ ] `effective_log_std` 改为数值安全 clamp（`±20`），移除业务 clamp
- [ ] `evaluate_actions` 返回 `entropy`（per-obs 归一化）替代 `regularizer`
- [ ] `evaluate_actions` 中 log_prob 用 effective σ，entropy 用策略原始 σ
- [ ] `evaluate_actions` 保留 `frame_modes` 参数（协议级，本策略忽略）
- [ ] `set_exploration` 改签名为 `(explore_intensity: float) -> None`，偏移量映射
- [ ] 移除 `_entropy_coef` 字段
- [ ] 移除 `ExplorationSpec` 导入
- [ ] stats 中 `entropy` → `entropy_raw`（nats），新增 `eff_std_mean`（含探索偏移的 σ）
- [ ] 更新测试
- [ ] 验证 checkpoint 兼容性
- [ ] 验证 `explore_intensity=0` 时 entropy 反映策略原始 σ，不受 offset 影响
- [ ] 验证 `log_std` 突破 `log_std_min` 时 `H_norm < 0` 且 hinge loss 梯度非零

## 8. 风险

1. **`set_exploration` 签名变化**：当前 `loop.py` 调用 `actor.set_exploration(spec)` 并
   使用返回的 `Dict` 做 logging。需要同步更新 `loop.py`。但 `loop.py` 的改动属于
   框架侧迁移，不在本文档范围——在所有策略迁移完成前，`loop.py` 需要同时支持
   新旧接口（或分阶段迁移）。

2. **`evaluate_actions` 签名变化**：`trainer.py` 当前从 `ActorEval.regularizer` 取
   loss 项。需要同步更新 `trainer.py` 改为从 `ActorEval.entropy` 算 hinge loss。
   同样属于框架侧迁移。

3. **`explore_intensity` 语义与旧 `temperature` 不同**：旧 `temperature` 是乘法缩放
   （`σ *= temperature`），新 `explore_intensity` 是以 0.5 为中心的对称偏移
   （0.5=中性，0=压缩，1=扩平）。数学上等价于 `σ × exp(offset)`，但在 log 空间
   操作。实验的 exploration schedule 需要重新校准，数值不一一对应。

4. **log_prob 和 entropy 用不同 σ**：log_prob 用 effective σ（含 explore offset），
   entropy 用策略原始 σ（不含 offset）。这是有意为之——entropy 反映策略自身确定性，
   不受探索噪声影响。但需要确认 PPO ratio 计算正确：old_log_prob 和 new_log_prob
   都应该用各自时刻的 effective σ（含当时的 offset），而不是策略原始 σ。
