# Design: TruncatedNormal 策略族总览 — 2×2×2 八格体系

> **STATUS: 体系设计冻结（2026-09-25）**
> 本文定义 truncated-normal 策略族的分类体系、三个正交维度的换装规则、
> 每格的核心语义与已确认的决策。各格的详细数学与验收标准见各自
> 的 DESIGN_*.md；本文只记录"格与格之间的关系"和跨格决策。

## 1. 体系：三个正交维度

每个策略 = 三个维度各取一值，共 8 格：

| 维度 | 取值 A | 取值 B |
|---|---|---|
| 分布结构 | 单分量 truncated normal | MoG（K 分量对角混合） |
| σ 来源 | shared：全局可训练参数 | state：trunk+head 输出 σ(obs) |
| σ 域 | unbounded：`σ = exp(log_std)` | bounded：`σ(v) = exp(r_min + Δr·sigmoid(v))` |

## 2. 八格总表

| 格 (MoG, state, bound) | 类名 | 状态 |
|---|---|---|
| (no, no, no) | `TruncatedNormalPolicy` | ✅ 基线，完整验证 |
| (no, no, yes) | `BoundedStdTruncatedNormalPolicy` | ✅ 1500u 训练达标 |
| (no, yes, no) | `StateTruncatedNormalPolicy` | ✅ 实现+测试 |
| (no, yes, yes) | `StateBoundedStdTruncatedNormalPolicy` | ✅ 3-seed 训练验证 |
| (yes, yes, no) | `MixtureTruncatedNormalPolicy` | ✅ 实现+测试（类名保留，不改名） |
| (yes, no, no) | `SharedMixtureTruncatedNormalPolicy` | ⬜ 待实现 |
| (yes, no, yes) | `SharedMixtureBoundedStdTruncatedNormalPolicy` | ⬜ 待实现 |
| (yes, yes, yes) | `StateMixtureBoundedStdTruncatedNormalPolicy` | ⬜ 待实现 |

注意命名约定：单个词 `Mixture`/`State`/`BoundedStd` 各标记一个轴的 B 值；
`MixtureTruncatedNormalPolicy` 历史遗留名字，实际占 (yes,yes,no) 格。

## 3. 三条换装规则（用户确认的决策逻辑）

族内任意两格如果只差一个维度，它们的差异被严格限定为：

### 3.1 state ↔ shared：**只有 σ 的计算方法不同**

- 其它完全一样：网络结构（trunk+head vs 参数）、ef 作用方式、
  uncertainty 计算、数据链路、导出结构全部相同。
- state 版的 σ head 约定：weight 零初始化、bias 置为对应 init 值
  （unbounded → log σ_init = −1；bounded → v_init ≈ 0.1644），
  使 state 版初始化时与 shared 版退化等价。
- MoG 格中 π(s) 与 μ(s) 在 shared-σ 版里仍然是 state-dependent——
  "shared"轴只约束 σ，不约束 π/μ。

### 3.2 MoG ↔ 单分量：**只有 uncertainty 的计算方法不同**

- MoG 的 U 以 `mixture_truncated_normal_mlp.py` 中的实现为基准：
  边际分布逐维 Rényi-2 有效宽度 `1/(2∫p_d²)`，分量间 overlap
  积分闭式解，维度取均值。
- ef 按与非 MoG 版**相同的逻辑应用于每个分量头**——逐元素、
  同一份语义，不因为多了分量而改变。
- 采样/评分/act/act 语义继承现有 MoG：每步采一个分量索引作用于
  整条动作向量；act() 返回最大权重分量的 μ；log_prob 为边际
  混合密度。
- **声明（非决策）**：单分量下已证明的 `e↑ ⇒ U↑` 单调性在 MoG
  上不成立（分量 σ 同步增大可能抹平分离模态，边际 U 可下降）。
  ef 在 MoG 格的承诺降级为"逐分量覆盖扩大"，不承诺边际 U 单调。

### 3.3 bound ↔ unbound：**只有 ef 的作用方式不同**（+ σ 映射）

- unbounded：`σ = exp(log_std)`；ef 乘性 `σ_eff = σ·3^e`。
- bounded：`σ(v) = exp(r_min + Δr·sigmoid(v))`，`v` 取代 `log_std`
  作为控制坐标；ef 加性 `v_e = v + αe`。基准实现见
  `state_bounded_std_truncated_normal_mlp.py`。
- bounded 参数统一沿用：`σ_min=0.05, σ_max=2.0, init_σ=e⁻¹,
  v_init≈0.164422, α≈1.19933891045474`。
  α 在初始化点按旧乘性方案的局部斜率标定；逐元素作用，对
  K·D 个分量 σ 同样成立，不重标定。
- U、floor、数据链路与 unbound 对应格完全相同（U 按 e=0 的
  策略 σ 计算，含 truncation 归一化项）。
- bounded 下 e=−1 不退化为确定性（σ ≥ σ_min > 0），与
  unbounded 下 e=−1 的非退化语义一致。

## 4. 跨格通用约定（所有 8 格共享）

- **接口**：`Policy.act/sample`、`StochasticPolicy`、
  `TrainablePolicy.evaluate_actions`、`reset(seed)` 重放，全部按
  `TruncatedNormalPolicy` 家族契约。
- **policy RNG（2026-09-25 统一）**：每个策略实例持有私有
  `torch.Generator`，`reset(seed)` → `gen.manual_seed`；所有采样
  点（rand/multinomial）走该 generator，**不碰全局 torch RNG**。
  由此：① 同一进程内两个 agent 的 policy 各自持有独立流，
  `seeds.policies[agent]` 真正生效；② episode 级重放成立——
  给定 episode 派生 seed 可单独重放该 episode 的采样序列；
  ③ worker 调度/worker 数变化不影响各 episode 采样。
  **退化语义**：rollout 在 policy_a_bp==policy_b_bp 时共享同一
  实例，reset 被调用两次，**后调用的 seed 生效**（单流，确定性，
  与旧 manual_seed 行为等价）。
  **谱系断裂声明**：采样 RNG 来源从"worker 全局流"换成
  "per-episode generator"后，所有格与此前 run（含基线
  `train_standup_floor04_ppo_20260920_164819`）不再 bit-identical——
  这是有意的语义升级；bit-identical 验证以改动后的新参考 run 为准。
  训练侧与导出侧（模板内联 Generator）语义一致。
- **数据链路**：动作空间只存 float32 动作；评分在截断动作空间进行；
  不引入 latent 通道。
- **U 与 floor**：`ActorEval.uncertainty` 始终是 e=0 的策略分布的 U；
  ef 下的有效宽度只进诊断（`effective_uncertainty`）。
- **初始化**：所有格 init σ ≡ e⁻¹（shared 参数直接置 −1/v_init；
  state head 置 zero-weight + bias）。
- **导出**：每格一个自包含模板，strict metadata 校验
  （`distribution_kind/std_source/std_parameterization`），
  训练↔导出 act/sample 须逐位一致。
- **state/shared 退化等价**：同格两版在显式复制共享权重后，所有
  e ∈ {−1,−0.5,0,0.5,1} 下 forward/evaluate/sample 逐位一致。

## 5. 已确认的决策清单

| # | 决策 | 结论 |
|---|---|---|
| D1 | shared-σ MoG 的 σ 粒度 | `(K·D,)` 参数：每分量×每维独立 |
| D2 | MoG 中 π/μ 是否随 σ 轴变化 | 不随：π(s)、μ(s) 恒 state-dependent |
| D3 | bounded MoG 边界参数 | 沿用 0.05/2.0/e⁻¹/α=1.1993，逐元素 |
| D4 | 现有 `MixtureTruncatedNormalPolicy` | 归属 (yes,yes,no)，不改名不迁移 |
| D5 | ef 在 MoG 的语义承诺 | 逐分量覆盖扩大；不承诺边际 U 单调 |
| D6 | unbounded 格 ef | 一律乘性 `σ·3^e`（含 MoG 格） |
| D7 | 代码组织 | 接缝复用：先给 `MixtureTruncatedNormalPolicy`
     开 σ 来源接缝（bit-identical 验证），变体只覆盖接缝 |
| D8 | 实施顺序 | (yes,no,no) → (yes,no,yes) → (yes,yes,yes)，
     每步只换一个维度 |
| D9 | 验收标准 | 每格全套单元测试 + 导出 parity + smoke；
     正式训练按需启动 |
| D10 | pre-tanh 族 | 不属于本体系；已标记 on hold，见
     DESIGN_pre_tanh_normal.md |
| D11 | policy RNG 机制 | 每实例 `torch.Generator` + `reset(seed)`
     reseed；替代旧的"ABC no-op / torch.manual_seed"混合状态。
     修复双 agent 共享全局流时后 reset 覆盖先 reset 的缺陷；
     接受与历史 run 的 RNG 谱系断裂（§4 已声明） |

## 6. 已完成格的验证记录（摘要）

- **(no,no,no) 基线**：`train_standup_floor04_ppo_20260920_164819`，
  1500u，终值 s≈1.0 / h≈1.28 / pot≈0.99。多次验证 bit-identical。
- **(no,no,yes)**：`train_standup_floor04_boundedstd_ppo_20260924_125534`，
  1500u，终值 s=1.0 / h=1.294 / pot=0.994，≥基线。σ 无饱和。
- **(no,yes,yes)**：e=0 一版 + ef=0.5 三 seed（42/43/44）。ef=0.5 下
  success 拐点 u165–270，为全部变体中最快；`sigma_state_std` 终值
  0.17–0.24 证明状态依赖真实学到；σ 上界贴界为瞬态现象。
- **父类接缝重构**：`verify_gpu_rerun` vs `verify_resume_A`
  5 updates 逐位一致（**bit-identical 要求 actor 设备一致，
  不能加 `CUDA_VISIBLE_DEVICES=""`**——历史教训，见
  DESIGN_bounded_std_truncated_normal.md §13.2）。

## 7. 待实施项（无新决策，纯执行）

1. `MixtureTruncatedNormalPolicy` σ 来源接缝重构（u1 对拍 bit-identical）
2. `SharedMixtureTruncatedNormalPolicy`：(yes,no,no)
3. `SharedMixtureBoundedStdTruncatedNormalPolicy`：(yes,no,yes)
4. `StateMixtureBoundedStdTruncatedNormalPolicy`：(yes,yes,yes)
5. 每格：导出模板、blueprint、`__init__` 注册、测试、实验文件
