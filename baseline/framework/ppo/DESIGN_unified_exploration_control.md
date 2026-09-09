# 探索控制设计（Unified Exploration Control）

---

## 1. 核心概念

探索控制分为两个正交旋钮：

| 旋钮 | 范围 | 作用层 | 含义 |
|---|---|---|---|
| `explore_factor` | `[-1, 1]` | Rollout | 附加探索强度。`0` = 不变，`+1` = 最大附加探索，`-1` = 最大探索压制 |
| `uncertainty_floor` | `[0, 1]` | Training | 策略不确定性的下界。`0` = 不限制。1 的含义由策略定义 |

两者独立，可同步退火（设成相关联的 schedule）或异步退火。

---

## 2. explore_factor

### 2.1 语义

`explore_factor` 是**附加在策略已学分布之上的探索强度**：

- `0`：不改变策略分布，纯 on-policy
- `+1`：最大附加探索
- `-1`：最大探索压制

**每个值的具体含义由策略自己定义。** 框架只规定 `[-1, 1]` 的范围和中性点 `0`，不规定 `+1` 或 `-1` 对应什么分布参数的变化。策略自己负责把 `explore_factor` 映射到内部参数（如 σ 缩放、log_std 偏移、温度等）。

### 2.2 数据流

```
experiment.build_jobs(...) → Job.explore_factor_a / explore_factor_b = ef
  → ParallelRollouter: stochastic=True → ExploratoryPolicy(policy, ef)
  → EpisodeRunner → ExploratoryPolicy.act() → inner.sample(obs, explore_factor=ef)
  → action_extras["explore_factor"] 记录每帧值
  → extract_explore_factor(episode, agent_id, T)
  → Trajectory.explore_factor  (T,) float32
  → PPOBuffer 拼接 → evaluate_actions(obs, acts, ef_tensor)
  → ppo_update 每 minibatch 切片传入

experiment.build_trajectories(...) → Trajectory.floor_weight  (T,) float32
  → PPOBuffer 拼接（None → ones，向后兼容）
  → ppo_update 每 minibatch 切片，作为 floor loss 的 per-frame 权重

experiment.exploration(u) → ExplorationSpec
  → (uncertainty_floor, uncertainty_coef)
  → ppo_update 计算 uncertainty_floor_loss = coef × (relu(floor - U)² × fw).mean()
```

关键不变量：**rollout 采样和 PPO log_prob 重算用同一个 explore_factor**，保证 importance ratio 正确。

> **注意**：``explore_factor`` 和 ``uncertainty_floor`` 是两个独立的旋钮。
> ``explore_factor`` 在 ``build_jobs`` 中决定（通常从 ``self.explore_factor``
> 读取），不经过 ``ExplorationSpec``。``ExplorationSpec`` 只管训练侧的
> ``uncertainty_floor`` 和 ``uncertainty_coef``。

### 2.3 策略接口

```python
def evaluate_actions(
    self, obs, actions,
    explore_factor: torch.Tensor,  # (B,) per-frame
    *, want_stats: bool = False,
) -> ActorEval
```

`explore_factor` 是必传参数（无默认值），因为 PPO 要求 log_prob 在采样分布下计算。

---

## 3. uncertainty_floor

### 3.1 语义

策略返回不确定性 `U ∈ [0, 1]`。**0 和 1 的具体含义由策略自己定义**，框架只限定数值范围。`uncertainty_floor` 是这个不确定性的下界。

### 3.2 损失函数

```python
uncertainty_floor_loss = uncertainty_coef × relu(floor - U)².mean()
```

**单向二次 hinge**：只在 `U < floor` 时产生梯度，推不确定性上升。`U ≥ floor` 时梯度为零，策略由 advantage 自由驱动。二次形式使远离 floor 时推力更强，接近 floor 时平滑减弱，避免过冲。

这和 PPO clip 的哲学一致："只在出问题时干预"。

### 3.3 为什么用解析不确定性而非 `-log_prob.mean()`

`-log_prob.mean()` 在 on-policy 时梯度恒为零（score function gradient 的经典结论），无法防坍缩。解析不确定性 `U(π(·|s))` 是分布属性，不依赖采样了哪个 action，梯度在任何情况下都非零。

### 3.4 uncertainty_coef

- 默认值：`0.0`（不产生 floor loss）
- 由 `ExplorationSpec.uncertainty_coef` 显式设置，`None` = 使用默认 0.0

---

## 4. ExplorationSpec

```python
@dataclass(frozen=True)
class ExplorationSpec:
    uncertainty_floor: Optional[float] = None   # 默认 0.0（不限制）
    uncertainty_coef: Optional[float] = None    # 默认 0.0
```

实验类通过 `exploration(update)` 方法返回 `ExplorationSpec`，实现 per-update 退火。

> **注意**：``explore_factor`` 不在 ``ExplorationSpec`` 里。它在 ``build_jobs``
> 中决定，写入 ``Job.explore_factor_a`` / ``explore_factor_b``。这样
> ``build_jobs`` 可以按 per-job / per-agent / per-frame 设置不同的探索强度。

---

## 5. 使用示例

### 5.1 中性默认（最常见）

```python
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec()  # floor=0, coef=0
```

### 5.2 防坍缩退火

```python
def exploration(self, update: int) -> ExplorationSpec:
    u = update / self.max_updates
    floor = 0.3 * (1.0 - u)  # 从 0.3 线性退到 0.0
    return ExplorationSpec(uncertainty_floor=floor, uncertainty_coef=0.01)
```

### 5.3 on-policy + 防坍缩

```python
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec(
        uncertainty_floor=0.3,   # 策略不能坍缩
        uncertainty_coef=0.01,
    )
```

> **rollout 侧的 ``explore_factor`` 调度**：如果需要同步退火 rollout 探索
> 强度，在 ``on_update`` 里修改 ``self.explore_factor``，``build_jobs``
> 会自动读取它。这和 ``exploration()`` 返回的 ``ExplorationSpec`` 是两个
> 独立的旋钮，可以同步也可以异步。

---

## 6. 诊断指标

`ActorEval.stats` 提供策略族无关的标准化诊断：

| 指标 | 含义 |
|---|---|
| `uncertainty` | 归一化不确定性 ∈ [0, 1]（策略自身不确定度，不含 explore scale） |
| `std_mean` | 策略原始 σ 均值 |
| `eff_std_mean` | 有效 σ 均值（含 explore scale） |
| `std_min` / `std_max` | σ 范围 |
| `mean_abs` | 策略均值绝对值均值 |

框架侧诊断（PPO stats）：

| 指标 | 含义 |
|---|---|
| `approx_kl` | 新旧策略 KL |
| `clip_frac` | PPO clip 比例 |
| `policy_loss` | PPO clip loss |
| `value_loss` | critic loss |
| `ev` | explained variance |

---

## 7. 设计原则

1. **探索和防坍缩是两件不同的事**：探索改变数据分布（rollout），防坍缩约束策略参数（training）。两个独立旋钮。

2. **只在出问题时干预**：不确定性下界用单向 hinge，策略在安全区内由 advantage 自由驱动。和 PPO clip 哲学一致。

3. **策略自己负责归一化**：每个策略族知道自己的 H_max 和 σ 语义，框架不需要理解策略族细节。

4. **per-frame 一致性**：rollout 采样和 PPO log_prob 重算用同一个 explore_factor，保证 importance ratio 正确。
