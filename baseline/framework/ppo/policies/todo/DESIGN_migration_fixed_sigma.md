# Design: FixedSigmaGaussianMLPPolicy 迁移到新接口

Reads `DESIGN_unified_exploration_control.md` (框架层新接口设计) as prerequisite.

## 0. 为什么选这个策略作为第一个

`FixedSigmaGaussianMLPPolicy` 是迁移到新接口的最简单策略：

- **有解析熵**：`Normal.entropy()` 是 closed-form，不需要采样估计。
- **σ 状态无关**：`log_std` 是全局参数，归一化只需一个 `H_max` 常量。
- **继承基类**：改基类 hook 签名后，其他 4 个继承基类的策略可以复用同样的模式。
- **checkpoint 兼容**：参数名和结构与 baseline 完全一致，迁移不涉及网络结构变化。

## 1. 当前接口 vs 新接口

| 维度 | 当前 | 新接口 |
|---|---|---|
| `ActorEval` 字段 | `log_prob`, `regularizer`, `stats` | `log_prob`, `entropy`, `stats` |
| `regularizer` 语义 | 策略计算 `-entropy_coef * H.mean()`，标量 loss | 移除。框架从 `entropy` 自己算 hinge loss |
| `entropy` 语义 | 不存在 | `(B,)` per-obs，可导，归一化到 [0,1] |
| `set_exploration` 签名 | `(spec: ExplorationSpec) -> Dict[str, float]` | `(explore_intensity: float) -> None` |
| `set_exploration` 接收 | `temperature`, `entropy_coef`, `noise_tau_steps`, `noise_scale` | 只接收 `explore_intensity` |
| `entropy_coef` 归属 | 策略持有，用于构造 `regularizer` | 框架持有，用于 hinge loss |
| `temperature` 归属 | 策略持有，`log_std += log(temperature)` | 移除。`explore_intensity` 统一映射 |

## 2. explore_intensity → 内部参数映射

当前策略有两个影响采样噪声的内部参数：

- `_temperature`：乘法缩放 σ，`effective_log_std = clamp(log_std + log(temperature), min, max)`
- `_noise_scale`：OU 探索的稳态 std（raw space）

新接口只给一个 `explore_intensity ∈ [0, 1]`，策略自己决定如何分配。

### 映射方案

```
explore_intensity = 0 → 确定性执行（temperature=1, noise_scale=0）
explore_intensity = 1 → 最大探索（temperature=max_temp, noise_scale=max_noise）
```

**temperature 映射**：

```python
# explore_intensity 线性映射到 log_std 偏移量
# 0 → offset=0（不改变 σ），1 → offset=log_std_max - log_std_min（最大偏移）
log_std_offset = explore_intensity * (log_std_max - log_std_min)
effective_log_std = clamp(log_std + log_std_offset, log_std_min, log_std_max)
```

这比当前的 `log(temperature)` 更直观——`explore_intensity=0` 时 σ 不变，`explore_intensity=1` 时 σ 被推到 `log_std_max`。

**noise_scale 映射**：

```python
# explore_intensity 线性映射到 noise_scale
# 0 → noise_scale=0（无 OU），1 → noise_scale=max_noise_scale
noise_scale = explore_intensity * max_noise_scale
```

`max_noise_scale` 是 init 时配置的参数（如 0.3），不随 update 变化。

### 与旧参数的关系

| 旧参数 | 新映射 | init 时配置 |
|---|---|---|
| `temperature` | 移除，被 `log_std_offset = explore_intensity × (log_std_max - log_std_min)` 替代 | `log_std_min`, `log_std_max` 保留 |
| `noise_scale` | `noise_scale = explore_intensity × max_noise_scale` | `max_noise_scale` 新增，`noise_tau_steps` 保留为 init 配置 |
| `entropy_coef` | 移除，框架持有 | 不再是策略参数 |

## 3. 熵归一化

### 原始熵

对角高斯的 per-obs 熵：

```python
H_raw = Normal(mean, σ).entropy().sum(dim=-1)  # (B,) in nats
```

### H_max 和 H_min

- `H_max = action_dim × 0.5 × log(2πe) + action_dim × log_std_max`
  （σ = `exp(log_std_max)` 时的熵，是策略能达到的最大熵）
- `H_min = action_dim × 0.5 × log(2πe) + action_dim × log_std_min`
  （σ = `exp(log_std_min)` 时的熵，是策略能达到的最小熵）

### 归一化

```python
H_norm = (H_raw - H_min) / (H_max - H_min)  # ∈ [0, 1]
```

当 `explore_intensity=0` 且 `log_std` 在初始值 `-1.0` 时：
- `H_raw ≈ action_dim × 0.5 × log(2πe × exp(-2.0))`
- `H_norm ≈ (−1.0 − log_std_min) / (log_std_max − log_std_min) ≈ 0.75`（在 `[-4, 0]` 范围内）

当 `explore_intensity=1` 时 σ 被推到 `log_std_max`，`H_norm → 1.0`。
当策略坍缩到 `log_std_min` 时，`H_norm → 0.0`。

### 为什么用 log_std_max/min 而不是理论最大熵

理论上一个 21 维高斯的最大熵是无穷大（σ → ∞）。但策略的 σ 被 clamp 在
`[log_std_min, log_std_max]`，所以实际可达的熵范围是有限的。用这个范围归一化
使得 `H_norm=0` 和 `H_norm=1` 都对应策略实际可达的状态，而不是理论极限。

## 4. 基类 hook 改动

当前基类 hook：

```python
def _regularizer_and_stats(
    self, obs, raw_action, raw_log_prob, want_stats,
    sample_extras, score_extras,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    """返回 (regularizer, stats)，regularizer 是已签名已缩放的标量 loss。"""
```

新基类 hook：

```python
def _entropy_and_stats(
    self, obs, want_stats,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """返回 (entropy, stats)。
    
    entropy: (B,) per-obs 归一化熵 ∈ [0,1]，可导。
    stats: 可选诊断字典。
    """
```

关键变化：
- **不再接收 `raw_action` / `raw_log_prob`**：熵只依赖 obs 和策略参数，不依赖采样动作。
- **不再返回标量 regularizer**：返回 per-obs 熵张量，框架自己算 hinge loss。
- **不再需要 `entropy_coef`**：策略不参与 loss 构造。
- **stats 里的 `entropy` 字段改为 `entropy_raw`**（nats），`entropy_normalized` 由框架从 `entropy` 张量算。

## 5. FixedSigmaGaussianMLPPolicy 的具体改动

### 5.1 `__init__`

```python
def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    log_std_min: float = DEFAULT_LOG_STD_MIN,
    log_std_max: float = DEFAULT_LOG_STD_MAX,
    *,
    device: torch.device | str = "cpu",
    deterministic: bool = False,
    noise_tau_steps: float = 0.0,
    max_noise_scale: float = 0.0,      # 替代 noise_scale
    model_path: Optional[str] = None,
):
```

移除：`entropy_coef`, `temperature`, `noise_scale`。
新增：`max_noise_scale`（OU 幅度上限，init 时配置）。

### 5.2 `_effective_log_std`

```python
def _effective_log_std(self) -> torch.Tensor:
    # explore_intensity=0 → offset=0（σ 不变）
    # explore_intensity=1 → offset=log_std_max - log_std_min（σ 推到 max）
    offset = self._explore_intensity * (self.log_std_max - self.log_std_min)
    return torch.clamp(
        self.log_std + offset,
        self.log_std_min,
        self.log_std_max,
    )
```

### 5.3 `_entropy_and_stats`（替代 `_regularizer_and_stats`）

```python
def _entropy_and_stats(
    self, obs, want_stats,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    mean, log_std = self._forward(obs)
    entropy_raw = Normal(mean, log_std.exp()).entropy().sum(-1)  # (B,) nats

    # 归一化到 [0, 1]
    H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_max)
    H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + self.log_std_min)
    entropy_norm = (entropy_raw - H_min) / (H_max - H_min)  # (B,) ∈ [0, 1]

    stats = None
    if want_stats:
        with torch.no_grad():
            eff_std = self._effective_log_std().exp()
            stats = {
                "entropy_raw": float(entropy_raw.mean().item()),
                "std_mean": float(eff_std.mean().item()),
                "std_min": float(eff_std.min().item()),
                "std_max": float(eff_std.max().item()),
                "tanh_sat_frac": float((mean.abs() > 2.0).float().mean().item()),
            }
    return entropy_norm, stats
```

### 5.4 `set_exploration`

```python
def set_exploration(self, explore_intensity: float) -> None:
    self._explore_intensity = float(explore_intensity)
    self._noise_scale = explore_intensity * self._max_noise_scale
```

基类持有 `_explore_intensity`，子类的 `_effective_log_std` 读它。
基类持有 `_noise_scale`（OU 用），子类在 `set_exploration` 里更新它。

### 5.5 移除的代码

- `_regularizer_and_stats` → 被 `_entropy_and_stats` 替代
- `_entropy_coef` 字段及相关逻辑
- `_temperature` 字段及相关逻辑
- `set_exploration` 中的 `spec.temperature` / `spec.entropy_coef` / `spec.noise_scale` 分支

## 6. 基类 TanhSquashedPolicyBase 的改动

### 6.1 `__init__`

```python
def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    *,
    device: torch.device | str = "cpu",
    deterministic: bool = False,
    noise_tau_steps: float = 0.0,
    max_noise_scale: float = 0.0,
):
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.action_dim = int(action_dim)
    self.device = torch.device(device)
    self._deterministic = bool(deterministic)
    self._explore_intensity = 0.0  # 由 set_exploration 设置
    # OU 探索
    self._noise_tau_steps = float(noise_tau_steps)
    self._max_noise_scale = float(max_noise_scale)
    self._noise_scale = 0.0  # = explore_intensity * max_noise_scale
    ...
```

移除：`_entropy_coef`, `_temperature`。
新增：`_explore_intensity`, `_max_noise_scale`。

### 6.2 `evaluate_actions`

```python
def evaluate_actions(self, obs, actions, *, noise_shift=None, want_stats=False) -> ActorEval:
    # log_prob 计算（不变）
    ...
    
    # 熵 + stats（新路径）
    entropy, stats = self._entropy_and_stats(obs, want_stats)
    
    return ActorEval(log_prob=log_prob, entropy=entropy, stats=stats)
```

移除整个 regularizer 构造逻辑（closed-form fallback、score-function fallback、
`_compute_stats` wrapper）。熵直接从 `_entropy_and_stats` 获取。

### 6.3 `set_exploration`

```python
def set_exploration(self, explore_intensity: float) -> None:
    self._explore_intensity = float(explore_intensity)
    self._noise_scale = explore_intensity * self._max_noise_scale
```

### 6.4 移除 `_compute_stats`

stats 现在直接从 `_entropy_and_stats` 返回，不需要 wrapper。

### 6.5 移除 `frame_modes` 参数

`evaluate_actions` 的 `frame_modes` 参数当前没有被任何策略实现使用，
新接口中移除。如果 V2 per-frame 探索需要它，届时再加回。

## 7. Checkpoint 兼容性

### 网络参数

`net.*` 和 `log_std` 参数名不变，`strict=True` 加载仍然有效。

### 移除的字段

`_entropy_coef`, `_temperature`, `_noise_scale` 是 plain floats（不是 buffers），
不在 `state_dict` 中，所以 checkpoint 不包含它们，加载不受影响。

### 新增的字段

`_explore_intensity`, `_max_noise_scale` 同样是 plain floats，不影响 `state_dict`。

### 结论

**checkpoint 完全兼容**，旧 checkpoint 可以直接加载到新接口的策略上。

## 8. 迁移检查清单

- [ ] `__init__` 参数更新（移除 `entropy_coef`/`temperature`/`noise_scale`，新增 `max_noise_scale`）
- [ ] `_effective_log_std` 改用 `_explore_intensity` 偏移
- [ ] `_regularizer_and_stats` → `_entropy_and_stats`（返回 per-obs 归一化熵）
- [ ] `set_exploration` 改签名为 `(float) -> None`
- [ ] 基类 `evaluate_actions` 移除 regularizer 逻辑，改用 `_entropy_and_stats`
- [ ] 基类 `set_exploration` 改签名
- [ ] 基类 `__init__` 移除 `_entropy_coef`/`_temperature`，新增 `_explore_intensity`/`_max_noise_scale`
- [ ] 基类移除 `_compute_stats`
- [ ] `evaluate_actions` 移除 `frame_modes` 参数
- [ ] 更新 `__init__.py` 导出（如有变化）
- [ ] 更新测试
- [ ] 验证 checkpoint 兼容性
- [ ] 验证 entropy ∈ [0, 1] 且 explore_intensity=0/1 时行为正确

## 9. 风险

1. **基类改动影响其他 4 个策略**：基类 hook 签名变化会导致 StateGaussian、LowRank、MoG、RealNVP 全部需要适配。但这是必要的——先改基类建立模式，后续策略按同样模式迁移。在所有策略迁移完成之前，未迁移的策略会 import error，这是预期的。

2. **`explore_intensity` 映射的语义变化**：旧 `temperature=2.0` 对应 `log(2)≈0.69` 的 log_std 偏移；新 `explore_intensity` 映射到 `(log_std_max - log_std_min)` 范围。实验需要重新校准探索强度的 schedule。

3. **`entropy_coef` 移除后框架侧需要接管**：trainer 需要从 `ActorEval.entropy` 和 `ExplorationSpec.entropy_coef` 计算 hinge loss。这是框架侧的改动，不在本文档范围，但需要在实现前确认框架侧已就绪。
