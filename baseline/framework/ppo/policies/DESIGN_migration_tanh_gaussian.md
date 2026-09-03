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

### explore_intensity：额外探索噪声

`explore_intensity` 控制的是**在策略学到的 σ 基础上叠加多少额外噪声**，不是替换策略 σ。

- **0 = 不叠加**：offset=0，σ 就是策略自己学的值。策略完全自由表达。
- **1 = 叠加最大预期噪声**：offset = log_std_max - log_std_min，σ 被推到
  `log_std + (log_std_max - log_std_min)`。策略的判断被最大噪声淹没。

```python
self._log_std_offset = explore_intensity * (self.log_std_max - self.log_std_min)
effective_log_std = self.log_std + self._log_std_offset
```

为什么是偏移量而不是绝对位置：策略通过 PPO 梯力学到的 σ 编码了"这个状态下该多确定"
的判断。退火到 0 时应该回到策略自然分布，让策略自由表达——不应该被推到一个
固定的确定性位置。确定性执行由 `deterministic=True` 控制，不是 `explore_intensity=0`。

### log_std_min / log_std_max 的含义

- **log_std_min = -4.0**（σ ≈ 0.018）：熵归一化的下界参考点。策略 σ 接近这里
  意味着近确定性。
- **log_std_max = 1.0**（σ ≈ 2.7）：熵归一化的上界参考点，也是 `explore_intensity=1`
  时叠加的偏移量上限。

  > 保持默认 1.0 不变。`log_std_max` 不再需要代表"tanh 输出接近均匀"的 σ，
  > 它只是偏移量上限和归一化参考点。

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

- `explore_intensity=0, entropy_floor=0.3`：采样用策略自然分布，但训练时不允许
  策略 σ 的熵低于 30%——hinge loss 防止策略坍缩。
- `explore_intensity=0.8, entropy_floor=0.1`：采样时叠加大量噪声去探索，但允许
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
def set_exploration(self, explore_intensity: float) -> None:
    # 偏移量映射：在策略 σ 基础上叠加额外噪声
    self._log_std_offset = explore_intensity * (self.log_std_max - self.log_std_min)
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
   （`σ *= temperature`），新 `explore_intensity` 是加性偏移（0=不叠加，1=叠加最大预期噪声）。
   实验的 exploration schedule 需要重新校准，数值不一一对应。

4. **log_prob 和 entropy 用不同 σ**：log_prob 用 effective σ（含 explore offset），
   entropy 用策略原始 σ（不含 offset）。这是有意为之——entropy 反映策略自身确定性，
   不受探索噪声影响。但需要确认 PPO ratio 计算正确：old_log_prob 和 new_log_prob
   都应该用各自时刻的 effective σ（含当时的 offset），而不是策略原始 σ。
