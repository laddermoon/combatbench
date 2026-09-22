# Design: StateTruncatedNormalPolicy — 状态相关 σ 的截断正态策略

## 1. 动机

`TruncatedNormalPolicy` 的 σ 是全局 `nn.Parameter`：所有状态共享同一个
探索宽度。训练只能学出"平均而言该多探索"，floor loss 调整 σ 时所有
状态联动。

本策略把 σ 变成状态的函数 σ = f(obs)：已掌握的状态可以收窄 σ（精确
执行），新奇/困难状态可以放大 σ（加强探索）；uncertainty floor 的
梯度也只作用于对应的状态区域。

**核心约束：与基线对比时隔离单一变量。** 除 σ 的参数化方式外，
分布数学、采样、log_prob、U 定义、explore_factor 语义、导出结构
与 `TruncatedNormalPolicy` 完全一致。

## 2. 网络

```
trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
head:  Linear(hidden, 2 * action_dim)
raw_mean, raw_log_std = head(trunk(obs)).split(action_dim, dim=-1)

mean = tanh(raw_mean)                    ∈ (-1, 1)
σ    = exp(clamp(raw_log_std, ±20))      > 0, per-state per-dim
```

trunk 形状与基线 net 的前两层相同，参数增量（96k → 101k，+5.4k）
全部来自 head 变宽，容量差异可做对照实验（baseline hidden_dim≈300
匹配参数量）。

**初始化**：head 的 σ 半侧 weight=0、bias=-1.0 → 初始 σ ≡ e⁻¹ ≈ 0.368，
与基线 `log_std` init=-1.0 逐点对齐。

## 3. 关键决策：σ 无业务边界

旧 todo 版（`todo/state_gaussian_mlp.py`）把 log_std 平滑 tanh squash
到 `[-4, 0]`（σ ∈ [0.018, 1]）。本设计**不沿用**该业务边界，只用与
基线相同的 ±20 数值安全 clamp。理由：

1. **单变量隔离**：基线 σ 实际无界，若新策略额外加窄边界，A/B 就
   混入第二个变量（σ 范围），无法归因于"状态相关性"。
2. **U 可达性**：U 对 σ 单调递增，U→1（均匀分布）是 σ→∞ 的渐近
   极限。σ≤1 时 U_max ≈ 0.856（mean=0 处），floor > 0.86 永不可达；
   且 mean 靠边界时同一 σ 下 U 更低（σ=1、mean→±1 时 U≈0.598）。
   无业务上界后不存在 floor 不可达问题。
3. **clamp 语义**：hard clamp 在 ±20 是纯数值保护（σ=e²⁰≈5e8 远超
   任何合理运行区间），不构成行为约束。

## 4. explore_factor 语义

与基线完全一致：乘性缩放，施加在 state-dependent σ 之上。

```
σ_eff(obs) = σ_policy(obs) × exp(ei × ln3)
```

ei=±1 恒等于 σ×3 / σ÷3，无论当前状态的 σ_policy 是多少。不做
squash 内偏移（旧接口的做法会在 σ 接近上界时吞掉探索信号）。

## 5. uncertainty U

与基线相同定义：`U = 1/(2×peak)`，mean ∈ (-1,1) 保证 peak 恒在
x=mean 处：

```
U_per_dim = σ_policy · √(2π) · Z / 2      (Z = Φ(b) − Φ(a))
U = mean(U_per_dim)                        # (B,)
```

- 用 policy σ（不含 explore 缩放）；log_prob 用 effective σ
- 可微、action-independent、∈(0,1) 渐近
- 与基线的区别：σ 随 obs 变化，U 是真正逐状态的不确定度
- `evaluate_actions` 一次 trunk+head 前向同时得到 mean、policy σ、
  effective σ —— 天然满足 P1-8（不重复 forward）

## 6. 正确性保证：退化等价测试

`test_state_truncated_normal.py::TestDegenerateEquivalence`：
将基线 `net[0]`/`net[2]` 拷入 trunk、`net[4]` 拷入 head 的 mean
半侧、σ 半侧保持 w=0/b=-1（基线 log_std=-1）→ 两策略的
forward / sample_action / evaluate_actions 输出 **bit-identical**
（atol=0）。

这是证明"移植只改了 σ 参数化、未改变计算路径"的归约证明——数学
助手（Φ、Φ⁻¹、log_Z）直接从 `truncated_normal_mlp` import，计算
顺序逐行一致。

## 7. stats 增量

沿用基线 keys（uncertainty/std_mean/eff_std_mean/std_min/std_max/
mean_abs），新增 `std_std`：batch 内 σ 的空间方差。≈0 表示 σ head
接近常数（行为退化为全局 σ 基线）；>0 表示状态依赖性被实际使用。

## 8. 接入方式

```yaml
# baseline/humanoid21/blueprints/init_policy_state_truncated_normal.yaml
cls: "baseline.framework.ppo.policies.state_truncated_normal_mlp:StateTruncatedNormalPolicy"
```

实验侧一行接入：`actor_blueprint = "init_policy_state_truncated_normal.yaml"`。
零框架改动（experiment.py / trainer.py / loop.py / trajectory.py
均不动）——这是 TrainablePolicy 接口设计的验收点。
