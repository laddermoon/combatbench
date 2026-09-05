# 探索控制设计（Unified Exploration Control）

---

## 1. 核心概念

探索控制分为两个正交旋钮：

| 旋钮 | 范围 | 作用层 | 含义 |
|---|---|---|---|
| `explore_intensity` | `[-1, 1]` | Rollout | 附加探索强度。`0` = 不变，`+1` = 最大附加探索，`-1` = 最大探索压制 |
| `entropy_floor` | `[0, 1]` | Training | 策略归一化熵的下界。`0` = 不限制。1 的含义由策略定义 |

两者独立，可同步退火（设成相关联的 schedule）或异步退火。

---

## 2. explore_intensity

### 2.1 语义

`explore_intensity` 是**附加在策略已学分布之上的探索强度**：

- `0`：不改变策略分布，纯 on-policy
- `+1`：最大附加探索
- `-1`：最大探索压制

**每个值的具体含义由策略自己定义。** 框架只规定 `[-1, 1]` 的范围和中性点 `0`，不规定 `+1` 或 `-1` 对应什么分布参数的变化。策略自己负责把 `explore_intensity` 映射到内部参数（如 σ 缩放、log_std 偏移、温度等）。

### 2.2 数据流

```
experiment.exploration(u) → ExplorationSpec
  → resolve() → (explore_intensity, entropy_floor)
  → build_jobs(explore_intensity=ei)
  → job options["explore_intensity"] = ei
  → EpisodeRunner → policy.act(obs, explore_intensity=ei)
  → action_extras["explore_intensity"] 记录每帧值
  → extract_explore_intensity(episode, agent_id, T)
  → Trajectory.explore_intensity  (T,) float32
  → PPOBuffer 拼接 → evaluate_actions(obs, acts, ei_tensor)
  → ppo_update 每 minibatch 切片传入
```

关键不变量：**rollout 采样和 PPO log_prob 重算用同一个 explore_intensity**，保证 importance ratio 正确。

### 2.3 策略接口

```python
def evaluate_actions(
    self, obs, actions,
    explore_intensity: torch.Tensor,  # (B,) per-frame
    *, want_stats: bool = False,
) -> ActorEval
```

`explore_intensity` 是必传参数（无默认值），因为 PPO 要求 log_prob 在采样分布下计算。

---

## 3. entropy_floor

### 3.1 语义

策略返回归一化熵 `H_norm ∈ [0, 1]`。**0 和 1 的具体含义由策略自己定义**，框架只限定数值范围。`entropy_floor` 是这个归一化熵的下界。

### 3.2 损失函数

```python
entropy_floor_loss = entropy_coef × relu(floor - H_norm).mean()
```

**单向 hinge**：只在 `H_norm < floor` 时产生梯度，推熵上升。`H_norm ≥ floor` 时梯度为零，策略由 advantage 自由驱动。

这和 PPO clip 的哲学一致："只在出问题时干预"。

### 3.3 为什么用解析熵而非 `-log_prob.mean()`

`-log_prob.mean()` 在 on-policy 时梯度恒为零（score function gradient 的经典结论），无法防坍缩。解析熵 `H(π(·|s))` 是分布属性，不依赖采样了哪个 action，梯度在任何情况下都非零。

### 3.4 entropy_coef

- 默认联动：`entropy_coef = 0.01 × max(explore_intensity, 0)`
- 可被 `ExplorationSpec.entropy_coef` 覆盖

---

## 4. ExplorationSpec

```python
@dataclass(frozen=True)
class ExplorationSpec:
    explore_intensity: Optional[float] = None   # 默认 0.0（中性）
    entropy_floor: Optional[float] = None       # 默认 0.0（不限制）
    entropy_coef: Optional[float] = None        # 默认联动 explore_intensity

    def resolve(self) -> tuple[float, float]:
        return (
            self.explore_intensity if self.explore_intensity is not None else 0.0,
            self.entropy_floor if self.entropy_floor is not None else 0.0,
        )
```

实验类通过 `exploration(update)` 方法返回 `ExplorationSpec`，实现 per-update 退火。

---

## 5. 使用示例

### 5.1 中性默认（最常见）

```python
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec()  # ei=0, floor=0
```

### 5. 同步退火

```python
def exploration(self, update: int) -> ExplorationSpec:
    u = update / self.max_updates
    v = 1.0 - u  # 从 1.0 线性退到 0.0
    return ExplorationSpec(explore_intensity=v, entropy_floor=0.3 * v)
```

### 5. 异步退火（探索先退，防坍缩后退）

```python
def exploration(self, update: int) -> ExplorationSpec:
    u = update / self.max_updates
    explore = max(0.0, 1.0 - 2.0 * u)           # u=0.5 时退到 0
    floor = 0.5 * (1.0 + math.cos(math.pi * u))  # u=1.0 时退到 0
    return ExplorationSpec(explore_intensity=explore, entropy_floor=floor)
```

### 5. on-policy + 防坍缩

```python
def exploration(self, update: int) -> ExplorationSpec:
    return ExplorationSpec(
        explore_intensity=0.0,   # 纯 on-policy
        entropy_floor=0.3,       # 但策略不能坍缩
    )
```

---

## 6. 诊断指标

`ActorEval.stats` 提供策略族无关的标准化诊断：

| 指标 | 含义 |
|---|---|
| `uncertainty` | 归一化熵 ∈ [0, 1]（策略自身确定度，不含 explore scale） |
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

2. **只在出问题时干预**：熵下界用单向 hinge，策略在安全区内由 advantage 自由驱动。和 PPO clip 哲学一致。

3. **策略自己负责归一化**：每个策略族知道自己的 H_max 和 σ 语义，框架不需要理解策略族细节。

4. **per-frame 一致性**：rollout 采样和 PPO log_prob 重算用同一个 explore_intensity，保证 importance ratio 正确。
