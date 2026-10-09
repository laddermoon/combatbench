# TruncNorm 家族策略选型记录

> 类型：指南

日期：2026-09-27
依据：`RESULTS_truncnorm_sweeps.md`（48 run = 8格 × 3seed × {ef=0, ef=0.5}，
standup_floor04，floor=0，600 updates）

## 8 格的三级分流

```
                                            ┌─ 主推候选（MVP，4格，继续投入验证）
                                            │
8 格 (MoG × state-σ × bounded-σ) ──┬───┬───┼── mixture / mix_shared_bnd / mix_bnd_statesig / bnd_statesig
                                   │   └─── 基础对照（2格，留作参照系）── base / boundedstd
                                   └─────── 淘汰（2格）──────────────── statesig / mix_shared
```

## 主推候选（MVP candidates）

| 策略 | 定位 | 入选理由 |
|---|---|---|
| `MixtureTruncatedNormalPolicy` | MoG 无界基线 | 两轮 AUSC 均全场最顶（ef0 中位 0.471、ef05 单点最高 0.618）；零失败 |
| `StateMixtureBoundedStdTruncatedNormalPolicy` | 三轴全开 | ef05 最强均衡格：u100 中位 370，3/3 seed 配对 Δu100 全负（唯一） |
| `SharedMixtureBoundedStdTruncatedNormalPolicy` | MoG+shared+bounded | ef0 milestone 冠军（u100 中位 345）；ef05 收益变小是天花板效应 |
| `StateBoundedStdTruncatedNormalPolicy` | 单分量黑马 | ef05 AUSC 中位 0.562 超所有 MoG 格；ef+bounded 使 state-σ 由负债变资产 |

四格内部排名 n=3 下不可分辨，这是下一轮要解决的问题。

## 基础对照（baselines，不作主推但保留）

| 策略 | 定位 |
|---|---|
| `TruncatedNormalPolicy` | 全因子关闭的空白对照——任何新变体都比照它，永不淘汰 |
| `BoundedStdTruncatedNormalPolicy` | 最简安全选项：ef 增益真实（ef05 u100 中位 395），但上限已证明低于带 state-σ/MoG 的格 |

## 淘汰（不再投入）

| 策略 | 判决类型 | 理由 |
|---|---|---|
| `StateTruncatedNormalPolicy` | **危险淘汰（硬）** | 两轮仅有的 2 个失败都在此格：ef0_s44 未达 100%（best 0.898）、ef05_s42 彻底崩溃（600u 全程 ≈0）。单头无界 σ 被 ef 乘性放大后无冗余兜底 |
| `SharedMixtureTruncatedNormalPolicy` | **支配性淘汰（软）** | 无失败但无存在理由：同 unbounded 被 `mixture` 全面快于它，同 σ 源被 `mix_shared_bnd` 更快更稳——任何场景都有严格更优的兄弟格 |

## 判断逻辑（可复用的决策框架）

1. **先看失败模式，再看速度**：崩溃/未达标是定性事实，不依赖 n 的大小——`statesig` 仅凭失败记录即可判死；速度排名才需要统计显著性
2. **归因要落到格，不要落到因子**：`statesig` 的脆弱属于"单分量+state-σ+无界"这个三元组合，不能泛化为"state-σ 危险"——`mixture` 共享同两因子却是顶级格（MoG 冗余吸收风险，待验证假设）
3. **支配关系可以判死**：一格若在所有场景下都有严格更优的兄弟格，即使没失败也可淘汰（`mix_shared`）
4. **对照组的价值独立于性能**：`base`/`boundedstd` 留存理由是参照系价值，不是竞争资格
5. **ef 轴会重排优劣**：ef0 的最优（mix_shared_bnd）与 ef05 的前沿格不完全重合——"最优格"依赖 ef 强度，选型结论必须标注 ef 前提

## 待定事项

- 4 个 MVP 格补 seed 至 5-6/格（或扫 ef 强度 0.3/0.7/1.0 on bounded 格）以分胜负
- "MoG 冗余吸收 unbounded state-σ 风险"这一机制假设可单独验证
- 换任务后 state-σ 价值需重评（本任务 state-σ 净收益未证明）
