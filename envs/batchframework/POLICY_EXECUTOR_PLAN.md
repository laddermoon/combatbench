# 设备端策略 executor 全族支持计划

> 类型：记录

> 目标：device collector 支持全部 PPO 策略导出类，消除
> "只有 TruncatedNormalExecutor" 的单点限制——`step_mbs`（MoG actor）
> 等实验因此可完整迁移。日期：2026-10-09。

## 现状（源码核查）

- `policy_executor.py` 有 `register_executor(kind, factory)` 注册机制，
  但 `_default_factory` 是 stub——**无视 payload 的 `policy_class`，
  一律构造 `TruncatedNormalExecutor`**。喂非 TN 导出会在
  `load_state_dict` 时崩（shape 不匹配），错误信息指向权重而非
  真实原因（策略类不对）——必须先修分发正确性。
- `model.pt` payload 自带分发键：`policy_class`、`distribution_kind`、
  `arch`（构造参数全集）、`state_dict`、各 `*_kind` 版本标签。
- 全族 API 统一：`sample_action(obs, ctx=, u=)` /
  `deterministic_action(obs)` / `evaluate_actions`——单一泛型
  executor + 按类构造即可覆盖，无需逐族重写采样数学。

## 覆盖面（8 个 truncnorm 族导出类）

| policy_class | 族特征 |
|---|---|
| `TruncatedNormalPolicy` | 全局共享 σ（已支持） |
| `BoundedStdTruncatedNormalPolicy` | 有界 σ |
| `StateTruncatedNormalPolicy` | σ=f(obs) |
| `StateBoundedStdTruncatedNormalPolicy` | 有界 σ=f(obs) |
| `MixtureTruncatedNormalPolicy` | MoG，全局 σ |
| `SharedMixtureTruncatedNormalPolicy` | MoG 共享 |
| `SharedMixtureBoundedStdTruncatedNormalPolicy` | MoG 共享+有界 |
| `StateMixtureBoundedStdTruncatedNormalPolicy` | step_mbs 用：MoG+有界+状态σ |

**明确不支持（显式拒绝，非静默错载）：**

| policy_class | 理由 |
|---|---|
| `PreTanhNormalPolicy` | tanh-squashed normal，不同 distribution_kind；不支持 |
| `StatePreTanhNormalPolicy` | 同上 |

未注册类（含上述两个 pre_tanh 族及任何未来新增类）走 W1 的
显式拒绝路径：错误信息列出 policy_class + 已注册清单。

## 工作包

### W1 分发正确性（正确性 bug，先行）

- `_default_factory` → 读 payload `policy_class` 查 `register_executor`
  注册表分发；未注册类**显式拒绝**（`ValueError` 指明 policy_class
  + 已注册清单），不再静默错载。
- cache 键仍是 `model.pt` 路径+stat；kind 从 payload 读（torch.load
  反正要发生，不引入额外 I/O）。

### W2 GenericTorchPolicyExecutor

- `policy_class` → `baseline.framework.ppo.policies.<module>:<Class>`
  映射表（8 条，模块名按惯例 `*_mlp`）；
- `cls(**arch)` 构造 + `load_state_dict` + `eval()`——arch 参数完备
  性由 golden 测试兜住；
- `capabilities`/`ctx_fields` 按族声明：先核对各族对
  `explore_factor`/`delta_factor`/reference 的支持差异（参考
  `test_reference_delta.py` 与各 `DESIGN_*` 文档），不支持 delta 的
  族声明更小的 ctx_fields，让 `check_spec` 自然拒绝；
- `act()` 统一委托 `sample_action/deterministic_action`，`u=` 噪声
  注入路径全族一致（E6-W1 job-keyed 语义不变）。

### W3 golden 验证（每族）

- 随机权重 → `to_blueprint(tmp)` 导出 → executor 加载：
  - `deterministic_action` 与导出 `policy.py` 的 CPU `act()` 对拍；
  - 同一 `u` 注入下 `sample_action` 对拍（fp32 设备容差，不追求
    bit-identical）；
- 未知 `policy_class` → 显式拒绝的契约测试；
- `capabilities.ctx_fields` 声明 vs sampling spec 的 `check_spec`
  交叉测试。

### W4 接线与文档

- `E8_SUPPORT_MATRIX.md` 策略面行更新；
- `PUBLIC_INTERFACE.md` executor 扩展点登记；
- `MIGRATION_GUIDE.md` §7 参考实现索引补 executor 扩展条目；
- `R3_TASK_BRIEF.md`：MoG executor 从"已知缺口"移出（step_mbs 的
  剩余缺口只剩 binding/observer/逐帧 ef）。

## 风险与判定口径

- **arch 完备性**：某族 ctor 需要 payload `arch` 之外的参数时，
  golden 测试会在加载期暴露——按"补 arch 字段优先于 executor 特例"
  处理（arch 是导出口径的一部分，缺参数是导出的 bug）；
- **ctx_fields 差异**：若某族不支持 delta_factor，如实声明并让
  `check_spec` 拒绝，不为兼容放宽；
- **pre_tanh 两族**：distribution_kind 不同（tanh normal），
  **不做支持**——显式拒绝路径由 W1 覆盖；golden 测试里专门加一条
  pre_tanh 导出必须被拒而非错载的回归用例；
- **obs_dim**：executor 只读 `arch`，99 维 gait-clock obs 无特殊处理。

## 验收

- 8 族 golden 全绿 + pre_tanh/未知类显式拒绝测试通过；
- device collect 用 `step_mbs` 真实 MoG 导出（若 checkpoint 存在）
  或任意 mixture 族导出跑通确定性 eval wave；
- 全部既有 executor/rollout 契约测试不回归。
