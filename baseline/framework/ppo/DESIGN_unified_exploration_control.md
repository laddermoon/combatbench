# 探索控制统一设计（Unified Exploration Control）

**状态**：设计定稿，待实现
**日期**：2026-09-03
**关联文档**：`TODO_temporally_correlated_exploration.md`（OU 实现）、`experiment.py`（当前接口）、`trainer.py`（当前 PPO update）

---

## 1. 问题

当前框架的探索控制散落在 6+ 个参数上，彼此耦合，心智负担重：

| 参数 | 位置 | 作用 |
|---|---|---|
| `entropy_coef` | 实验类 | 熵正则系数（防坍缩） |
| `temperature` | ExplorationSpec | 采样温度（放大 σ） |
| `log_std_min/max` | 实验类 | σ 硬边界 |
| `noise_tau_steps` | 实验类 | OU 相关时间 |
| `noise_scale` | 实验类 | OU 平移幅度 |
| `clip_eps` / `target_kl` | PPOParams | 信任域 |

用户要调探索行为，需要在多个地方同时改参数，且参数之间的交互不透明：
- `entropy_coef` 太大 → 策略不收敛；太小 → 坍缩
- `temperature` 放大 σ 但受 `log_std_max` 截断
- `noise_scale` 和 `temperature` 都在放大噪声，但机制不同，叠加效果不可预期

**目标**：用两个旋钮 `explore_intensity ∈ [0, 1]` 和 `entropy_floor ∈ [0, 1]` 统一控制探索行为，降低用户心智负担，同时保持底层灵活性。

---

## 2. 设计哲学

### 2.1 三层解耦

探索控制分为三个正交层面，每层有独立的职责：

| 层 | 职责 | 时机 | 旋钮 |
|---|---|---|---|
| **Rollout 层** | 采什么样的动作 | rollout 时 | `explore_intensity` → 噪声幅度 |
| **Training 层** | 策略分布能窄到什么程度 | training 时 | `entropy_floor` → 熵下界 |
| **诊断层** | 策略当前有多坍缩 | 观测时 | `stats` 里的标准化指标 |

三层解耦的核心 insight：**探索（rollout）和防坍缩（training）是两件不同的事**。
- 探索是"采多敢冒险的动作"——改变数据分布。
- 防坍缩是"策略分布不能太窄"——约束策略参数。

当前 `entropy_coef` 把两件事耦合在一起。本设计将它们拆开为两个独立旋钮。
大多数场景下两者同步退火就够（设成相同值），少数场景需要独立控制（见 §3.2）。

### 2.2 和 PPO Clip 的哲学对齐

PPO clip 的核心思想是"只在出问题时干预"：
- ratio 在 [1±ε] 内：梯度正常通过，策略自由学习。
- ratio 超出 [1±ε]：clip 截断梯度，阻止策略跑飞。

本设计的熵下界采用相同哲学：
- 熵高于下界：梯度为零，策略自由收敛。
- 熵低于下界：hinge 产生梯度，阻止策略坍缩。

不是"时刻拉向目标"，而是"只在危险时托住"。这让策略在安全区内完全由 advantage 信号驱动，不被熵项干扰。

---

## 3. 核心设计

### 3.1 双旋钮

探索控制有两个正交旋钮：

```
explore_intensity ∈ [0, 1]   → rollout 侧：对称 temperature 控制（0.5=中性）
entropy_floor     ∈ [0, 1]   → training 侧：策略熵下界
```

**`explore_intensity`**（Rollout 路径）：
```
log_std_offset = (explore_intensity - 0.5) × EXPLORE_SPAN
```
以 0.5 为中性点（offset=0，策略用自身 σ），→ 0 压缩 σ，→ 1 扩平 σ。
每个策略族自己定义 `EXPLORE_SPAN` 和映射方式。Gaussian 用 log 空间加法
（等价于 σ 乘法缩放），MoG 是 component σ 的缩放，RealNVP 是 base σ 的缩放。

**`entropy_floor`**（Training 路径）：
```
熵下界 = entropy_floor × H_max
```
每个策略族自己定义 `H_max`（策略在当前 obs 下的最大可能熵）。策略自己负责
把 `entropy` 归一化到 [0, 1]，使 `entropy_floor` 可以直接作为归一化目标。

```python
# 中性（最常见默认）
set_exploration(explore_intensity=0.5, entropy_floor=0.3)

# 独立控制
set_exploration(explore_intensity=0.5, entropy_floor=0.3)  # on-policy + 防坍缩
set_exploration(explore_intensity=0.8, entropy_floor=0.1)  # 强探索 + 快收敛
```

### 3.2 为什么需要双旋钮

如果只有一个旋钮强制 rollout 探索和 training 防坍缩同步退火，以下三个场景无法表达：

**场景 1：on-policy + 防坍缩（最常见的不同步需求）**

```
explore_intensity = 0.5   (中性，纯 on-policy，策略用自身 σ)
entropy_floor = 0.3       (但策略不能坍缩到中等以下)
```

这是当前所有不用 OU 的实验的默认运行模式——纯 on-policy + entropy_coef 防坍缩。
单旋钮在 explore_intensity=0.5 时熵下界也是 0.5，防坍缩过强，策略无法收敛。

**场景 2：强探索 + 快收敛**

```
explore_intensity = 0.8   (扩平 σ 去发现新行为)
entropy_floor = 0.1       (但允许策略在发现好行为后快速锁定)
```

用强探索去撞步态，一旦撞到了希望策略快速收敛到确定性策略。
单旋钮在 explore_intensity=0.8 时熵下界也是 0.8，策略被强制保持高熵，无法快速收敛。

**场景 3：异步退火**

```
update  500: explore=0.3, floor=0.7   (压缩探索，但防坍缩还在)
update 1000: explore=0.5, floor=0.3   (回到中性，但防坍缩还在)
update 2000: explore=0.5, floor=0.0   (完全自由收敛)
```

探索先退、防坍缩后退。理由：探索噪声在后期是纯干扰（策略已经知道往哪走），
但防坍缩在后期仍然需要（策略可能在 advantage 驱动下过度收缩）。

### 3.3 旋钮解析

```python
@dataclass(frozen=True)
class ExplorationSpec:
    explore_intensity: Optional[float] = None
    entropy_floor: Optional[float] = None
    entropy_coef: Optional[float] = None

    def resolve(self) -> tuple[float, float]:
        """返回 (explore_intensity, entropy_floor)。
        explore_intensity 默认 0.5（中性），entropy_floor 默认 0.0。"""
        return (
            self.explore_intensity if self.explore_intensity is not None else 0.5,
            self.entropy_floor if self.entropy_floor is not None else 0.0,
        )
```

### 3.4 策略接口：evaluate_actions

```python
@dataclass
class ActorEval:
    log_prob: torch.Tensor                              # (B,) 可微，和 action 有关
    entropy: torch.Tensor                               # (B,) 可微，和 action 无关
    stats: Optional[Dict[str, float]] = None            # 诊断，no_grad
```

**`log_prob`**：`log π(a|s)`，PPO importance ratio 的分子。和当前定义一致。

**`entropy`**：`H(π(·|s))`，策略分布本身的熵。关键属性：
- **和 action 无关**——只依赖 obs 和策略参数。
- **可微**——梯度可以回传到策略参数。
- **per-obs**——shape `(B,)`，因为 state-dependent 策略的熵随 obs 变化。
- **归一化到 [0, 1]**——策略自己负责归一化，0 = 完全确定，1 = 策略最大熵。

**`stats`**：诊断字典，no_grad。用于层 3 的标准化诊断。框架不解释 key，只透传。

**不再有 `regularizer` 字段。** 防坍缩不再由策略自己算好系数返回，而是由框架用
`entropy` 和 `entropy_floor` 统一处理。

### 3.5 训练侧：熵下界损失

```python
# ppo_update 里
explore_int, floor = exploration.resolve()       # 双旋钮解析
target = floor                                    # 归一化熵下界 ∈ [0, 1]
entropy_floor_loss = entropy_coef * relu(target - entropy_normalized).mean()
loss = ppo_clip_loss + entropy_floor_loss
```

**单向 hinge**：`relu(target - H_norm) = max(0, target - H_norm)`

- `H_norm < target`（策略太确定）：有梯度，推熵上升。
- `H_norm ≥ target`（策略足够随机）：梯度为零，不干预。

这和 PPO clip 的"只在出问题时干预"哲学一致。策略在安全区内完全由 advantage 驱动，不被熵项干扰。

**梯度性质**：
```
∇_θ L_floor = -entropy_coef × 1[H_norm < target] × ∇_θ H
```
- 只在 H 低于 target 时非零。
- 不依赖采样了哪个 action（H 是分布属性，不是样本属性）。
- on-policy 和 off-policy 下行为一致（不像 `-log_prob.mean()` 在 on-policy 时梯度为零）。

**`entropy_coef` 的来源**：
- 默认联动：`entropy_coef = 0.01 * explore_int`（探索越强，防坍缩也越强）。
- 可被 `ExplorationSpec.entropy_coef` 覆盖。
- 联动是启发式默认值，不是硬绑定——用户可以 `entropy_coef=0.05, explore_int=0`
  实现纯 on-policy + 强防坍缩。

### 3.6 策略归一化职责

每个策略族必须提供两个东西：

1. **`entropy_normalized`**：当前策略在给定 obs 下的归一化熵 ∈ [0, 1]。
2. **`max_noise` 的定义**：`explore_intensity=1` 时噪声幅度的物理含义。

策略族自己知道"什么样的熵是高的，什么样的熵是低的"，所以归一化由策略负责，框架不需要理解。

各策略族的归一化方案：

| 策略族 | H_max 定义 | 归一化方式 |
|---|---|---|
| Gaussian (FixedSigma, StateGaussian) | `0.5 × log(2πe × σ_max²) × D` | `H_norm = (H - H_min) / (H_max - H_min)` |
| LowRankGaussian | `0.5 × log((2πe)^D × det Σ_max)` | 同上，Σ_max = diag(σ_max²) + U_max U_maxᵀ |
| MoG | mixture 熵 + Σ w_k H_k，component σ = σ_max | `H_norm = H / H_max` |
| RealNVP | `H(base_max) + E[log\|det J\|]`（采样估计） | `H_norm = H / H_max` |

`H_min` 对 Gaussian 是 `log_std_min` 对应的熵（接近 0）。对 MoG/RealNVP 是单 component / 单 mode 的熵。

### 3.7 退火

`explore_intensity` 和 `entropy_floor` 各自可以独立退火：

```python
def exploration(self, update: int) -> ExplorationSpec:
    u = update / self.max_updates
    # 探索先退（线性），防坍缩后退（cosine，慢退）
    explore = max(0.0, 1.0 - 2.0 * u)           # u=0.5 时 explore=0
    floor = 0.5 * (1.0 + math.cos(math.pi * u))  # u=1.0 时 floor=0
    return ExplorationSpec(
        explore_intensity=explore,
        entropy_floor=floor,
    )
```

退火过程中，策略的熵会**平滑地跟随下界下降**，因为：
- 熵高于下界时：无约束，advantage 驱动自然收敛。
- 熵碰到下界时：被托住，不会继续下降。
- 下界继续降低：策略跟着降。

物体在缓慢下降的桌子上的类比：
- 桌面 = 熵下界。
- 物体 = 策略熵。
- 物体跟着桌子下降，但永远不会穿过桌子。
- 双向追踪（SAC 式）会试图把物体"钉"在桌面上，产生振荡——单向 hinge 不会。

---

## 4. 为什么不是其他方案

### 4.1 为什么不用 `-log_prob.mean()` 做防坍缩

```
L = coef × (-log π(a|s)).mean()    # a 是 rollout 采的 action
```

**致命缺陷**：on-policy 时梯度恒为零。

```
∇_θ E_{a~π}[log π(a|s)] = ∫ π(a|s) ∇_θ log π(a|s) da
                        = ∫ ∇_θ π(a|s) da
                        = ∇_θ 1 = 0
```

这是 score function gradient 的经典结论。on-policy（explore_intensity=0）时防坍缩完全不工作。explore_intensity 退火到 0 的那一刻，防坍缩消失，策略可以自由坍缩。

而 `H(π(·|s))` 是分布属性，不依赖采样了哪个 action，梯度在任何情况下都非零。

### 4.2 为什么不用双向熵目标

```
L = coef × (H - H*)²    # 双向追踪
```

**问题**：熵高于目标时往下压，阻止"过度探索"——但过度探索不是问题。

PPO 的 advantage 信号本身就是"该往哪走"的指导。当策略熵高（探索充分）时，advantage 可以自由地引导策略收敛。双向追踪会在这种时候和 advantage 打架——"压低熵"的力度和 advantage 的力度无关，可能太猛（策略突然坍缩）或太弱。

单向下界不参与这个方向，只在坍缩危险时介入。策略在安全区内完全由 advantage 驱动。

### 4.3 为什么不只用 σ_min 物理防坍缩

```
effective_log_std = log_std + (explore_intensity - 0.5) × EXPLORE_SPAN
```

**优点**：简单直接，对 Gaussian 完美工作。

**局限**：
- 只对 Gaussian 有 σ 概念。MoG 的"最小熵"和 RealNVP 的"最小熵"不好这样控制。
- 是硬截断，梯度在截断点不连续。
- 不提供 per-obs 的自适应——所有 obs 用同一个 σ_min。

熵下界损失是软约束（梯度连续），策略族无关（只要能算 H），per-obs（H 依赖 obs）。对 Gaussian 族可以和 σ_min 叠加使用，但对其他族是更通用的方案。

---

## 5. 完整 API

### 5.1 同步退火路径（大多数用户）

```python
# 实验类只需设一个参数
class MyExperiment(ExperimentPPO):
    exploration_intensity_schedule = "linear"  # 或 "constant", "cosine"
    exploration_intensity_start = 1.0
    exploration_intensity_end = 0.0

# 或手动控制：两个旋钮设成相同值
def exploration(self, update: int) -> ExplorationSpec:
    v = max(0.0, 1.0 - update / self.max_updates)
    return ExplorationSpec(explore_intensity=v, entropy_floor=v)
```

用户不需要理解 `entropy_coef`、`temperature`、`log_std_min/max`、`noise_scale`、
`noise_tau_steps`。两个旋钮设成相同值即可。

### 5.2 独立控制路径（需要探索和防坍缩不同步）

```python
# on-policy + 防坍缩（场景 1）
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec(
        explore_intensity=0.5,      # 中性，纯 on-policy
        entropy_floor=0.3,          # 但策略不能坍缩到中等以下
    )

# 强探索 + 快收敛（场景 2）
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec(
        explore_intensity=0.8,      # 扩平 σ 去发现新行为
        entropy_floor=0.1,          # 但允许策略快速锁定
    )

# 异步退火（场景 3）
def exploration(self, update: int) -> ExplorationSpec:
    u = update / self.max_updates
    return ExplorationSpec(
        explore_intensity=max(0.5, 1.0 - 0.5 * u),         # 探索从 1.0 退到 0.5
        entropy_floor=0.5 * (1.0 + math.cos(math.pi * u)),  # 防坍缩后退
    )
```

### 5.3 覆盖 entropy_coef

```python
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec(
        explore_intensity=0.7,      # 扩平 σ
        entropy_floor=0.5,          # 熵下界
        entropy_coef=0.01,          # 覆盖默认联动值 (0.01 * explore_intensity)
    )
```

`ExplorationSpec` 只有三个字段。PPO 信任域参数 (`clip_eps`, `target_kl`) 在
`PPOParams` 中配置，不随 update 变化。OU 时间相关性 (`noise_tau_steps`) 是
策略 init 时配置。策略族特定的缩放由 `explore_intensity` 统一处理。

### 5.4 策略接口

```python
class TrainablePolicy(Protocol):
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        *, noise_shift: Optional[torch.Tensor] = None,
        want_stats: bool = False,
    ) -> ActorEval:
        """返回 log_prob + entropy + stats。

        log_prob: (B,) 可微，log π(a|s)，PPO ratio 用。
        entropy:  (B,) 可微，H(π(·|s)) 归一化到 [0,1]，防坍缩用。
        stats:    诊断字典，no_grad。
        """
        ...

    def set_exploration(self, explore_intensity: float) -> None:
        """接收探索指令。

        explore_intensity 是对称 temperature 控制（0.5=中性）：
            → 0 压缩 σ，→ 1 扩平 σ
        entropy_floor 控制策略熵下界（由框架在 ppo_update 里使用，
            策略不需要处理）。

        策略自己负责把 explore_intensity 映射到内部参数
        （log_std offset, noise_scale 等）。
        """
        ...
```

### 5.5 训练侧

```python
# ppo_update 里
explore_int, floor = exploration.resolve()  # 双旋钮解析
entropy_coef = exploration.entropy_coef if exploration and exploration.entropy_coef is not None \
    else default_entropy_coef(explore_int)  # 默认联动: 0.01 * explore_int

actor_eval = actor.evaluate_actions(obs, actions, **eval_kwargs)
new_lp = actor_eval.log_prob
entropy_norm = actor_eval.entropy  # (B,) 归一化到 [0,1]

# PPO clip loss
ratio = exp(new_lp - old_lp)
ppo_loss = -mean(min(ratio * adv, clip(ratio) * adv))

# 熵下界损失（单向 hinge，target 来自 entropy_floor 而非 explore_int）
entropy_floor_loss = entropy_coef * relu(floor - entropy_norm).mean()

loss = ppo_loss + entropy_floor_loss
```

---

## 6. 诊断层：标准化指标

`stats` 字典提供策略族无关的标准化诊断：

| 指标 | 含义 | 策略族无关？ |
|---|---|---|
| `entropy_normalized` | 归一化熵 ∈ [0, 1] | ✅ 策略自己归一化 |
| `entropy_raw` | 原始熵（nats） | ✅ 但量纲依赖策略族 |
| `entropy_floor` | 当前熵下界（= `entropy_floor` 旋钮值） | ✅ |
| `floor_active_frac` | batch 中触发 hinge 的帧比例 `mean(H_norm < floor)` | ✅ |
| `explore_intensity` | 当前 rollout 探索强度 | ✅ |
| `clip_frac` | PPO clip 比例 | ✅ 框架计算 |
| `approx_kl` | 新旧策略 KL | ✅ 框架计算 |
| `grad_ppo_norm` | PPO clip loss 的梯度 L2 范数 | ✅ 框架计算 |
| `grad_floor_norm` | 熵下界 loss 的梯度 L2 范数 | ✅ 框架计算 |
| `grad_floor_ratio` | `grad_floor_norm / (grad_ppo_norm + grad_floor_norm)` | ✅ |

### 6.1 熵分布诊断

`entropy_normalized` 和 `entropy_floor` 的关系是核心诊断：
- `entropy_normalized >> entropy_floor`：策略远高于下界，防坍缩未激活，策略自由学习。
- `entropy_normalized ≈ entropy_floor`：策略贴着下界，防坍缩在托底。
- `entropy_normalized < entropy_floor`：不应该持续发生（hinge 会推回），偶尔出现说明学习率太大或 `entropy_floor` 变化太快。

`floor_active_frac` 量化 hinge 的激活程度：
- ≈ 0：策略远高于 floor，floor 未激活。
- 0.3–0.7：约一半帧在 floor 附近，正常防坍缩状态。
- ≈ 1：几乎所有帧都在 floor 以下——策略在剧烈坍缩，floor 在全力托底。

### 6.2 梯度占比诊断

`grad_floor_ratio` 是核心调参指标——它直接告诉你 PPO loss 和 entropy floor loss
谁在主导参数更新。**注意：必须用梯度范数占比，不能用 loss 数值占比**，
因为两个 loss 项的量纲不同（PPO loss 是 advantage 加权的 ratio，entropy floor loss
是 [0,1] 区间的 hinge），数值大小和梯度大小之间没有线性关系。

| `grad_floor_ratio` | 含义 | 调参方向 |
|---|---|---|
| ≈ 0 | floor 几乎没参与梯度 | floor 没用，调低 floor 或调高 explore_intensity |
| 0.2–0.5 | 两者势均力敌，floor 在积极防坍缩 | 健康，无需调整 |
| > 0.5 | floor 主导，PPO advantage 信号被淹没 | coef 太大或 floor 太高，调低其一 |
| ≈ 1 | floor 完全主导 | 严重失衡，coef 或 floor 必须大幅降低 |

`floor_active_frac` 和 `grad_floor_ratio` 一起判断调参方向：

| `floor_active_frac` | `grad_floor_ratio` | 诊断 | 行动 |
|---|---|---|---|
| ≈ 0 | ≈ 0 | 策略远高于 floor，floor 未激活 | 调低 floor 或调高 explore_intensity |
| > 0 | 0.2–0.5 | floor 在正常防坍缩 | 健康 |
| > 0 | > 0.5 | floor 太强 | 调低 coef 或 floor |
| ≈ 1 | > 0.5 | 策略在剧烈坍缩，floor 在全力托底 | 调高 explore_intensity（根因是探索不够，不是 coef 不够） |

### 6.3 实现方式

梯度占比统计需要对两个 loss 分别求梯度范数。两种方案：

**方案 A：每 N 个 update 统计一次（推荐）**

在 `ppo_update` 的最后一个 epoch 的最后一个 minibatch 上，用
`torch.autograd.grad` 分别对 `L_ppo` 和 `L_floor` 求梯度，取 L2 范数。
代价是每 N 个 update 多两次 backward，开销可忽略。

```python
if update % grad_diag_interval == 0:
    grads_ppo = torch.autograd.grad(L_ppo, actor_params, retain_graph=True)
    grads_floor = torch.autograd.grad(L_floor, actor_params)
    grad_ppo_norm = sum(g.norm()**2 for g in grads_ppo).sqrt().item()
    grad_floor_norm = sum(g.norm()**2 for g in grads_floor).sqrt().item()
    grad_floor_ratio = grad_floor_norm / (grad_ppo_norm + grad_floor_norm + 1e-8)
```

**方案 B：每个 update 都统计**

在正常 backward 之后、optimizer.step 之前，从 `param.grad` 读取总梯度范数，
再单独 backward 一次 `L_floor` 取其梯度范数。代价是每个 update 多一次 backward。

方案 A 足够——梯度占比是趋势性指标，不需要每个 update 都看。

跨实验、跨策略族比较时，看 `entropy_normalized`、`floor_active_frac` 和
`grad_floor_ratio` 即可——不需要理解每个策略族的原始熵量纲。

---

## 7. 实现计划

### Stage 1：ActorEval 重构

- [ ] `ActorEval` 增加 `entropy: torch.Tensor` 字段（per-obs，可导，归一化到 [0,1]）。
- [ ] `ActorEval` 移除 `regularizer` 字段。
- [ ] 各策略族实现 `_normalized_entropy(obs) → (B,)` 方法。
- [ ] `evaluate_actions` 返回新的 `ActorEval`。

### Stage 2：双旋钮

- [ ] `ExplorationSpec` 增加 `explore_intensity`、`entropy_floor` 字段。
- [ ] 实现 `ExplorationSpec.resolve() → (explore_intensity, entropy_floor)` 方法。
- [ ] 各策略族实现 `set_exploration` 对 `explore_intensity` 的映射：
  - Gaussian: `log_std_offset = (explore_intensity - 0.5) × EXPLORE_SPAN`，`noise_scale = (explore_intensity - 0.5) × 2 × noise_scale_max`。
  - MoG / RealNVP: 类似映射。
- [ ] `noise_tau_steps` 保留为 init 时配置，`explore_intensity` 不控制时间结构。

### Stage 3：训练侧改造

- [ ] `ppo_update` 用 `actor_eval.entropy` 和 `entropy_floor` 计算熵下界损失。
- [ ] 移除 `loss += actor_eval.regularizer` 路径。
- [ ] `entropy_coef` 默认联动 `explore_intensity`（如 `0.01 × explore_int`），可被 `ExplorationSpec.entropy_coef` 覆盖。

### Stage 4：诊断标准化

- [ ] 各策略族在 `stats` 里返回 `entropy_normalized`、`entropy_raw`。
- [ ] 框架在 stats 里添加 `entropy_floor`、`floor_active_frac`、`explore_intensity`。
- [ ] 框架每 N 个 update 统计梯度占比：`grad_ppo_norm`、`grad_floor_norm`、`grad_floor_ratio`。
- [ ] 更新日志格式，输出标准化诊断。

### Stage 5：测试

- [ ] 各策略族：`entropy_normalized ∈ [0, 1]`，`explore_intensity=0.5` 时反映策略原始 σ。
- [ ] 熵下界损失：H > floor 时梯度为零，H < floor 时梯度非零。
- [ ] on-policy（`explore_intensity=0.5`）+ `entropy_floor>0` 时防坍缩仍然工作（不像 `-log_prob.mean()` 那样失效）。
- [ ] 同步退火：`explore_intensity` 和 `entropy_floor` 设成相同值，`entropy_normalized` 平滑跟随下界下降。
- [ ] 异步退火：`explore_intensity` 和 `entropy_floor` 独立 schedule，互不干扰。
- [ ] 向后兼容：旧实验（不用 explore/floor）行为不变。

### Stage 6：文档与迁移

- [ ] 更新 `GUIDE.md` 探索章节。
- [ ] 更新 `experiment.py` 的 `ExplorationSpec` 文档。
- [ ] 提供迁移指南：旧参数 → `explore_intensity` / `entropy_floor` 的映射表。

---

## 8. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-03 | 采用双旋钮 `explore_intensity` + `entropy_floor`，不设单旋钮语法糖 | 单旋钮无法表达 on-policy+防坍缩、强探索+快收敛、异步退火三个场景；单旋钮语法糖带来的优先级混乱大于收益，直接用双旋钮设成相同值即可实现同步退火 |
| 2026-09-03 | `evaluate_actions` 返回 `entropy` 而非 `regularizer` | 策略只返回原始熵，系数由框架控制；API 更干净 |
| 2026-09-03 | 熵损失用单向下界 `relu(floor - H)` 而非双向目标 `(H - H*)²` | 和 PPO clip 哲学一致：只在出问题时干预；策略在安全区内由 advantage 自由驱动 |
| 2026-09-03 | 熵损失用解析熵 `H(π(·\|s))` 而非样本确定度 `-log_prob.mean()` | 解析熵不依赖采样分布，on-policy 时梯度非零；样本确定度在 on-policy 时梯度恒为零 |
| 2026-09-03 | 策略自己负责熵归一化 | 策略知道自己的 H_max 和 H_min，框架不需要理解策略族细节 |
| 2026-09-03 | `explore_intensity` 控制 rollout 噪声，`entropy_floor` 控制 training 熵下界 | 两者可同步（设成相同值）或独立退火，覆盖同步和异步需求 |
| 2026-09-03 | `noise_tau_steps` 保留为 init 时配置，不纳入旋钮 | 时间相关性是结构选择（像优化器类型），不是强度参数（像学习率）；init 时决定即可 |
| 2026-09-03 | `ExplorationSpec` 只保留三个字段：`explore_intensity`、`entropy_floor`、`entropy_coef` | 控制复杂度；`temperature`/`noise_scale` 被 `explore_intensity` 替代，`entropy_target` 被 `entropy_floor` 替代，`clip_eps`/`target_kl` 不是 per-update 旋钮（留在 `PPOParams`），`policy_extras` 在三字段设计下无必要 |
| 2026-09-03 | `entropy_coef` 默认联动 `explore_intensity` | 探索越强防坍缩也越强是合理默认；可被 `ExplorationSpec.entropy_coef` 覆盖 |
| 2026-09-03 | `explore_intensity` 改为以 0.5 为中心的对称控制（0=压缩, 0.5=中性, 1=扩平） | 原单向设计（0=不加噪声, 1=最大噪声）只能加不能减；对称设计更直观（像 temperature），支持压缩场景。EXPLORE_SPAN=2.0 限制两端到 σ×0.37~2.72x，避免极端值。默认值从 0.0 改为 0.5 |

---

## 9. 未来扩展：Per-frame 探索控制（V2）

### 9.1 动机

当前双旋钮设计（`explore_intensity` + `entropy_floor`）是 **per-update** 粒度——
整个 update 内所有 rollout 帧使用同一个探索强度。这对大多数场景足够，但有一个
重要的场景需要 **per-frame** 粒度：

**长序列动作的分段探索**：机器人在学一个长序列动作（如"站立→平衡→迈步→转向"），
前序已学会的阶段只需要微探索甚至不探索（否则摔了根本走不到后面），真正需要学习的
阶段才加强探索。

```
帧  1-30:  站立平衡（已学会）→ explore_intensity=0.4（轻微压缩，保命）
帧 30-60:  迈步（正在学）   → explore_intensity=0.8（扩平 σ，发现步态）
```

如果统一探索强度，会出现两难：
- 探索强 → 前半段站不稳，摔了，后半段根本到不了，学不到迈步。
- 探索弱 → 到了后半段但没探索，学不到迈步。

这正是当前 A/B 实验 CTRL arm 遇到的问题——episode length 从 26 涨到 66 就卡住，
策略学会了"不摔"但学不会"迈步"，部分原因就是探索在前半段（已掌握的平衡）上浪费了，
到后半段（需要探索的迈步）时 entropy 已被压低。

### 9.2 PPO 上没有硬障碍

逐条检查 PPO 的要求：

**Importance ratio**：PPO 需要 `old_log_prob = log π_old(a|s)`，即采样时的分布。
如果每帧探索强度不同，每帧的采样分布不同，但只要每帧的 `old_log_prob` 是在
**该帧实际采样分布**下算的，ratio 就是正确的。这和 `noise_shift` 的原理完全一样——
框架已经做到了 per-frame 记录采样时的分布信息。

**GAE**：per-frame 计算，`A_t = δ_t + (γλ)δ_{t+1} + ...`，不要求探索强度一致。

**PPO loss**：逐帧计算，不要求探索强度一致。

**熵下界损失**：`relu(floor - H(π(·|s_t)))`，per-frame，不同帧可以有不同的 floor。

**唯一要求**：per-frame 记录采样时的 `old_log_prob`（和 `noise_shift` 如果有 OU）。
当前框架已经做到了。

### 9.3 一个微妙但不是硬障碍的点

高探索帧的动作偏离 `π` 更远，`old_log_prob` 更低，ratio 方差更大，更容易被 PPO
clip 截断。这意味着**高探索帧的梯度贡献天然被 clip 抑制**。

这其实不是 bug，是 feature：
- 高探索帧是"探索数据"——PPO 自然地给它们更小的权重。
- 低探索帧是"on-policy 数据"——PPO 给它们更大的权重。
- 恰好符合意图：已学会阶段的数据权重高（巩固已学行为），学习前沿的数据权重低但
  提供新信号。

如果高探索帧被 clip 抑制得太厉害（学不到新东西），可以调大 `clip_eps`、降低探索
强度、或用 `target_kl` 早停替代 clip（KL 早停是全局的，不抑制单帧）。

### 9.4 先例

1. **步态相位门控探索（robotics locomotion）**：四足机器人 RL 在 swing phase 注入
   探索噪声，在 stance phase 保持确定性。stance 需要精确力控制（摔了就完了），
   swing 可以自由探索（脚在空中，摔不了）。这就是 per-frame 探索控制。

2. **Curiosity-driven exploration（ICM/RND）**：内在好奇心模块根据预测误差调制探索
   强度——高误差（新颖状态）→ 多探索，低误差（已知状态）→ 少探索。隐式的 per-frame
   探索控制，探索强度是 obs 的函数。

3. **Safety-constrained exploration**：安全 RL 在危险状态关闭探索，在安全状态开启
   探索。per-frame 的。

4. **Options framework / 分层 RL**：已学会的子策略是确定性 option，新学的 option
   是探索性的。高层 controller 在已掌握的区域调用确定性 option，在需要学习的区域
   调用探索 option。概念上和本场景一致。

5. **Scheduled noise in robotics RL**：训练早期对整个轨迹注入噪声，后期只在特定
   关节或特定相位注入噪声。per-joint + per-frame 的探索控制。

### 9.5 设计方案

不改当前 `set_exploration` 的接口（它仍然是 per-update 的默认值），而是给 `act()`
加一个 per-frame 覆盖参数：

```python
class TrainablePolicy(Protocol):
    def act(self, obs, *,
            exploration_intensity: Optional[float] = None,
            deterministic: bool = False) -> tuple[Action, ActionExtras]:
        """采样动作。

        exploration_intensity: per-frame 覆盖。None 时用 set_exploration
            设的默认值。指定时这一帧用这个强度，不影响其他帧。
        """
        ...
```

实验层提供可选的 `frame_intensity_fn`：

```python
def frame_intensity_fn(obs, frame_idx, episode_step) -> Optional[float]:
    """Per-frame 探索强度。返回 None 时用 set_exploration 的默认值。"""
    if episode_step < self.mastered_horizon:
        return 0.1  # 已学会的阶段，低探索保命
    return None     # 学习阶段，用默认探索强度
```

这个 `frame_intensity_fn` 是可选的——不提供时行为和 V1 完全一致。

### 9.6 和 OU 状态的关系

OU 的 AR(1) 状态是 per-trajectory 的。Per-frame 探索强度不需要影响 OU 状态——
它只控制**当前帧注入多少噪声**，OU 状态可以继续演化：

```python
# OU 状态继续演化（per-trajectory）
ou_state = theta * ou_state + sigma * randn()

# 但注入多少取决于 per-frame intensity
noise = frame_intensity * ou_state
```

`frame_intensity` 是 per-frame 的，`ou_state` 是 per-trajectory 的。两者正交。

### 9.7 记录与一致性

Per-frame intensity 必须被记录到 trajectory extras 中，和 `noise_shift` 一样：
- Rollout 时记录每帧实际使用的 `explore_intensity`。
- PPO update 时用记录的值重算 `old_log_prob`（如果 intensity 影响了采样分布）。
- `entropy_floor` 仍然 per-update（它是 training 侧的约束，不需要 per-frame）。

### 9.8 V1 → V2 的兼容性

V1 的 `act()` 接口预留 `exploration_intensity` 参数（默认 None），V2 只需要实验层
提供 `frame_intensity_fn`。策略层和训练层不需要改动——V1 实现的 per-update
`set_exploration` 是 V2 的 fallback 默认值。

### 9.9 决策

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-09-03 | Per-frame 探索控制作为 V2 扩展，不纳入 V1 | V1 的 per-update 双旋钮覆盖大多数场景；per-frame 是长序列分段探索的专项需求，作为可选扩展更合适 |
| 2026-09-03 | V1 的 `act()` 预留 `exploration_intensity` 参数 | V2 只需实验层提供 `frame_intensity_fn`，策略层和训练层不需要改动 |
| 2026-09-03 | `frame_intensity_fn` 由实验层提供，不由策略层 | "哪些帧已学会"是课程进度知识，属于实验职责，策略不应知道自己在课程的哪个阶段 |
| 2026-09-03 | `entropy_floor` 保持 per-update，不做 per-frame | 熵下界是 training 侧的分布约束，不是 rollout 侧的采样控制；per-update 粒度足够 |
