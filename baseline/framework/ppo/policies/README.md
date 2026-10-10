# 预置策略族（TruncNorm 八格）

> 类型：指南

PPO 训练框架的**预置动作分布库**：8 个正交设计、全部实现并验证过的
`StochasticPolicy` 策略族，覆盖探索分布设计空间的主要自由度。
通过 `actor_blueprint` 参数换装即可切换，无需改训练代码。

## 🏆 这项成果的价值

- **探索分布不用自己设计**：动作分布是 PPO 在格斗任务上最容易踩坑
  的部件（σ 失控、动作越界、探索塌陷）。本目录提供 2×2×2 正交八格
  的完整实现——**每一格都有单元测试、导出 parity 测试、RNG replay
  测试与 smoke 验证**，不是草稿。
- **选型有实测依据而非玄学**：48 run 对照 sweep（ef=0/0.5 × 8格 ×
  3seed，standup_floor04 600 updates）产出了按证据分级的结论与
  推荐格，见 `POLICY_SELECTION.md` + `RESULTS_truncnorm_sweeps.md`。
- **附赠一套可复用的判断方法论**：选型文档记录了"先看失败模式再
  看速度""归因到格不归因到因子""支配关系可判死"的决策框架——
  做同类策略对比实验时可直接借用（`POLICY_SELECTION.md` §判断逻辑）。
- **是探索机制的策略侧实现面**：本族全部实现
  `StochasticPolicy.sample(obs, ctx)`/`evaluate_actions()` 契约——
  框架层三通道探索干预（ef/uncertainty floor/reference-delta，
  见 `baseline/framework/README.md`）经由这些策略的 ctx 解释生效。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| 2×2×2 正交八格 | 每个轴的效应可独立消融——实测 MoG 加速收敛是三轮配对差中唯一排 0 的轴效应（`RESULTS_truncnorm_sweeps.md` A2） |
| TruncatedNormal 而非 Gaussian | 动作域天然贴合 `[-1,1]` 控制规范，无越界重采 |
| bounded-σ（`exp(r_min+Δr·sigmoid)`） | 修掉 unbounded state-σ 的崩溃模式（ef 乘性放大无兜底，sweep 中仅有的 2 个失败都在该格） |
| MoG 混合分量 | 多模态动作分布 + 对 unbounded σ 的冗余吸收（机制假设，见选型待定项） |
| 统一 `StochasticPolicy` 契约 | 八格与框架探索机制正交——换装分布不需要动 rollout/训练链路 |

## 📐 八格总览

| 格 (MoG, state-σ, bound-σ) | 类名 | 定位 |
|---|---|---|
| (no,no,no) | `TruncatedNormalPolicy` | 空白基线（永不淘汰的对照系） |
| (no,no,yes) | `BoundedStdTruncatedNormalPolicy` | 最简安全选项 |
| (no,yes,no) | `StateTruncatedNormalPolicy` | ⚠️ 已判淘汰（崩溃模式） |
| (no,yes,yes) | `StateBoundedStdTruncatedNormalPolicy` | ef05 单分量黑马 |
| (yes,yes,no) | `MixtureTruncatedNormalPolicy` | MoG 无界基线，AUSC 全场最顶 |
| (yes,no,no) | `SharedMixtureTruncatedNormalPolicy` | ⚠️ 支配性淘汰 |
| (yes,no,yes) | `SharedMixtureBoundedStdTruncatedNormalPolicy` | ef0 milestone 冠军 |
| (yes,yes,yes) | `StateMixtureBoundedStdTruncatedNormalPolicy` | ef05 最强均衡格 |

另有八格外补充变体 `PreTanhNormalPolicy`（设计见各自 DESIGN_*.md）。

## 🧭 怎么选

`POLICY_SELECTION.md` 是当前权威的选型指南（含 ef 前提标注的
推荐格与淘汰理由）。快速原则：**默认从 4 个 MVP 格起步**
（`mixture`/`state_mix_bnd`/`shared_mix_bnd`/`state_bnd`），
需要空白对照用 `base`，已判淘汰的两格勿用。

## 🔑 关键点与边界

- **选型结论是任务+ef 条件的**：ef 强度会重排优劣——引用结论必须
  带 ef 前提；换任务后 state-σ 价值需重评（本任务未证明净收益）。
- **4 个 MVP 格在 n=3 下排名不可分辨**，补 seed 至 5-6 是待定项——
  别把格间排名当成定论。
- **测试矩阵**：每格含 unit/导出 parity/退化等价/RNG replay 测试；
  `todo/` 内是归档的未维护探索性测试（OU 探索等，仅供参考）。
- 深入阅读：`DESIGN_truncnorm_family.md`（八格体系与换装规则）、
  各格 `DESIGN_*.md`（数学与验收）、`POLICY_SELECTION.md`、
  `RESULTS_*.md`（sweep 全量数据）。
