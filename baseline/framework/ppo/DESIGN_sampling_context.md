# SamplingContext 设计（采样条件与参考差分探索）

> 面向使用者的对外契约文档。记录框架暴露什么、各层语义是什么、
> 策略实现要满足什么不变量。实施细节见代码与
> `TODO_reference_policy_delta_exploration.md`。

---

## 1. 定位

本机制解决一个问题：**让 rollout 采样分布成为"策略参数 + 外部条件"
的函数，并保证 PPO 训练回放时两边的条件逐位相同。**

```
π_θ(a | s, z)    z = (explore_factor, reference_action, delta_factor, delta_mix)
```

`z` 由框架构造、记录、原样回放；策略决定**如何解释** z。
PPO 比值 `π_θ(a|s,z) / π_θold(a|s,z)` 的正确性只要求两边 `z` 相同——
z 里包含一个向量（参考动作）不影响 importance sampling 恒等式成立。

---

## 2. 三层结构

| 层 | 类型 | 位置 | 生命周期 |
|---|---|---|---|
| 意图配置 | `SamplingSpec`（+`ReferenceSpec`） | `Job.sampling_a/b` | 实验构造一次，随 Job pickle 进 rollout worker |
| 逐帧容器 | `SamplingContext` | `ppo/sampling_context.py` | wrapper 每帧构造 → extras 记录 → buffer → minibatch 原样重建 |
| 策略机制 | ef/Δ → σ_eff | 各 policy 内部 | 不跨层暴露 |

### 2.1 SamplingSpec（实验 → 框架的意图）

```python
SamplingSpec(
    explore_factor: EfSpec = 0.0,       # float 或 (obs, step) -> float
    reference: Optional[ReferenceSpec], # 历史策略加权参考
    delta_factor: float = 0.0,          # c：Δ→σ 标定系数（静态）
    delta_mix: float = 0.0,             # λ ∈ [0,1]：原尺度/Δ尺度混合权重
    delta_mode: str = "dynamic",        # "dynamic" | "frozen"，互斥
)
```

`ReferenceSpec(policies, weights)`：K 个历史 policy blueprint +
归一化非负权重。框架**不维护**跨 update 的历史策略管理——
选哪几代、权重多少，完全是实验侧职责。

`delta_mode` 选择 Δ 载荷的记录形态，二者互斥：

- `"dynamic"`（默认）：ctx 携带 `reference_action`，Δ =
  `m_θ − a_ref` 由策略按当前 θ 重算。
- `"frozen"`：ctx 携带 `delta`——**动作级 Δ** =
  `det_action(当前策略) − a_ref`，由采样层（SamplingPolicy /
  推理 server）在 rollout 时逐帧算好；回放原值不随 θ 变。

### 2.2 SamplingContext（框架 → 策略的逐帧输入）

```python
SamplingContext(
    explore_factor: float | Tensor,      # 本帧 ef
    reference_action: Array | None,      # Σw·act_k(obs)，wrapper 已加权
    delta_factor: float | Tensor,
    delta_mix: float | Tensor,
    delta: Array | None,                 # frozen 模式的动作级 Δ 输入
)
```

- `reference_action`：当 spec 携带 reference 时，wrapper 每帧用
  各历史策略对**当前 observation** 做确定性 `act()` 加权得到；
  无 reference 时为 `None`。frozen 模式下不进 ctx（与 `delta`
  互斥）。
- `delta`：frozen 模式的逐帧 Δ 输入 `(D,)`——由采样层用
  `det_action(当前策略) − a_ref` 算好放入，σ-mix 原样消费。
  **载荷即模式**：`sctx__delta` 存在即 frozen，`sctx__reference_action`
  存在即 dynamic；`delta_mix ≠ 0` 但两个载荷都缺失 = 畸形 ctx
  （载荷丢失），σ-mix 报错而非静默退化——两模式同一判据。
- 字段可以是标量（逐帧构造）或批量张量（训练回放时切片得到），
  策略内部必须同时兼容两者。

### 2.3 策略内部机制（策略自己的解释权）

框架不规定 ctx 字段如何映射到分布参数，但约定既有语义：

- `explore_factor`：各族已有的 ef→σ 机制（unbounded 乘性
  `σ·3^e`；bounded raw 域 `v+αe`），语义不变。
- `reference_action` + `delta_factor` + `delta_mix`：
  参考差分探索（§3）。

---

## 3. 参考差分探索（Reference-Delta Exploration）

### 3.1 公式

```
Δ       = m_θ(s) − a_ref                         # 当前参数下重算（动态）
σ_delta = c · |Δ|
σ_eff   = clip( √((1−λ)·σ_ef² + λ·max(σ_delta², ε²)), σ_min, σ_max )
```

- `m_θ(s)`：策略的代表性确定性动作（single：μ；MoG：每个分量的
  μ_k，见 §3.3）
- `σ_ef`：各族 ef 机制处理完之后的策略自身尺度
- `ε`：固定下界（模块常量 `_DELTA_EPS`），防 Δ→0 时尺度归零
- `clip`：bounded 族收敛到 [σ_min, σ_max]；unbounded 族无 clamp

**梯度**：`a_ref` 是冻结数据；`m_θ` 一侧完整求导（不 detach）。
这意味着 Δ 路径让"均值与尺度耦合"进入优化目标——这是有意的
算法语义，detach 版本留作对照变体（未实现）。

### 3.2 与 ef 的复合顺序

族内先处理 ef 得到 `σ_ef`，再在 σ² 域混入 Δ 项。**λ=0 时
`σ_eff ≡ σ_ef`，行为与无 reference 时逐位相同**——这是硬不变量。

### 3.3 各族适配表

| cell | Δ | 混合位置 |
|---|---|---|
| single ×（shared|state）× unbounded | `μ−a_ref`，(D,) | σ² 域 |
| single ×（shared|state）× bounded | 同上 | σ² 域 + clamp |
| MoG ×（shared|state）× unbounded | `μ_k−a_ref`，(K,D) | σ² 域 |
| MoG ×（shared|state）× bounded | 同上 | σ² 域 + clamp |

注意：MoG shared-σ 变体的 σ_eff 仍会因 Δ 变成 state-dependent——
"shared" 指 σ_policy **参数**共享，不表示有效尺度恒定。

---

## 4. 数据流（框架保证）

```
Experiment.build_jobs → Job(sampling_a/b=SamplingSpec)
  → worker: SamplingPolicy(inner, spec)
      每帧: ctx = SamplingContext(ef, a_ref=Σw·act(obs), c, λ)
            action = inner.sample(obs, ctx=ctx)
            extras["sctx__<field>"] = ctx 字段原值
  → Episode.sampling_contexts（extras 派生 property）
  → Trajectory.sampling_ctx {field: (T,...)}
  → PPOBuffer.ctx_fields 拼接（跨 traj schema 不一致 → raise）
  → minibatch: ctx_sl = SamplingContext.from_batch(fields, sl)
  → evaluate_actions(obs_b, act_b, ctx=ctx_sl)
```

- **记录即输入**：记录的字段值 == 传给 `sample()` 的值，中间层
  不解释、不加工。
- **回放语义按 `delta_mode` 分两种**：dynamic 模式训练侧拿到
  同样的 `a_ref`，Δ 用当前 θ 的 `m_θ(s)` 重算——"相同外部规则、
  不同参数"；frozen 模式 `delta` 本身就是要回放的字段值，σ_eff
  在 update 内对 θ 完全静止（修掉 `σ=c·|m_θ−a_ref|` 的值级耦合）。
- 旧 dump 键 `explore_factor`/`ef__<agent>` 不变；新字段一律
  `sctx__<field>` 扁平键（npz 兼容，无 object array）。

---

## 5. 策略实现契约

实现一族策略时，以下必须成立：

1. **λ=0 / 无 reference 短路**：delta 谓词为假时直接返回
   `σ_ef`，不做任何多余浮点运算（λ=0 bit-identical 基准）。
2. **广播兼容**：ctx 字段接受 scalar 或 (B,)；`a_ref` 接受 (D,) 或
   (B,D)；内部统一升维到与 σ 相同形状。
3. **广播一致性**：`sample()`（rollout，逐帧）与
   `evaluate_actions()`（训练，批量）走同一公式同一精度。
4. **统计正确性**：`eff_sigma` 进入 `_DistParams` 后，框架已有的
   `eff_std_mean` 等统计自动反映混合后尺度——策略不用额外上报。
5. **导出等价**：export 模板内置同款 `_delta_sigma*` 实现，
   导出策略数值与源策略逐位一致（rollout 的 inner 就是导出件）。

### 5.1 跨进程边界契约

导出 `policy.py` 在 rollout worker 内 exec 加载，而 worker 在 spawn
时冻结了自己的 import——因此**模板内嵌代码只能读 ctx 的字段**
（`explore_factor`/`reference_action`/`delta_factor`/`delta_mix`/
`delta`），不得调用 ctx 的方法：ctx 对象可能由
旧版 `SamplingContext` 构造，字段语义稳定但方法可能不存在。

反之方向由显式能力握手负责，**不允许静默降级**：

- 策略类声明 `SUPPORTS_REFERENCE_DELTA`（8 个 delta cell 为
  `True`；pre_tanh 族为 `False`；旧导出物缺省即不支持）。
- `SamplingPolicy.__init__` 在 wrap 时校验：`spec.reference` 非空
  且 `delta_mix != 0` 要求 inner 声明该能力，否则 `TypeError`。
- `SamplingSpec` 构造时校验：`delta_mix != 0` 必须配
  `reference`——没有参考系的 λ>0 永远无法激活，属于非法配置。
- 三层校验（spec 构造 → wrap → policy 内部谓词）逐级把
  "意图 vs 能力"的错配提前到最早可报错的边界。

---

## 6. 边界与非目标

- `act()`（确定性路径）不消费 ctx；`stochastic=False` 不包
  wrapper、不构建参考策略。
- dump 导出的 `ExportedExploratoryPolicy` 只支持烘焙 ef 调度；
  spec 携带 reference/delta 时 `NotImplementedError`（fail-loud）。
- c/λ 当前为 spec 级静态值；如需逐帧调度，按 EfSpec 同型升级。
- frozen-Δ 已实现（`delta_mode="frozen"`）：Δ 定义为**动作级**
  `det_action(当前策略) − a_ref`，由采样层算好作为 ctx 输入——
  mixture 单元同样共享该 (D,) 载荷广播到各分量。
- ReferenceSpec 的选取策略（哪几代、什么权重）是实验设计，
  不在框架内。

---

## 7. 验收基线

- **单元**：λ=0 短路、ε 下界、bounded clamp、MoG per-component
  独立性、μ/σ 双路梯度连通、export 模板数值一致。
- **端到端**：默认 `SamplingSpec()` 下 8 策略 × 5 updates 与
  ef05 基线 `__RAW_STATS__` 逐位一致；λ>0 smoke 检查
  `eff_std_mean` 随 ‖Δ‖ 单调、首 minibatch ratio 守门、无 NaN。
