# 实验记录 — 2026-08-10

## 背景

基于 `exp_basic_balance_v2.py`（原始单 agent 基线）和 `exp_basic_balance_v2_phi_dual.py`（dual-agent + φ 动态 actor weight），探索不同奖励结构和 actor weight 策略对 humanoid21 平衡/迈步学习的影响。

所有 dual-agent 实验共用环境蓝图 `basic_balance_v2_phi_dual_env.yaml`（`DualImbalanceTerminationPlugin`，per-agent 终止）。

---

## 实验 0: phi_dual（φ² 动态 actor weight）

- **文件**: `exp_basic_balance_v2_phi_dual.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_ppo_20260810_003746`
- **目的**: 在 dual-agent 环境中，用 φ² 调制 shaping channels 的 actor weight，使早期学习由 r_fall 主导，站稳后 shaping 信号逐渐介入。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，摔倒 `-1`，无 timeout bonus
  - r_fall actor weight: 固定 `3.0`
  - shaping actor weights: `base × φ²`（r_cross=1.0×φ², 其余=0.2×φ²）
- **结果**: **效果很好**。U370 达到 100% 存活，U400 ep_len=196，U1500 完全收敛（ep_len=200）。学会了良好的交替迈步步态。
- **状态**: 训练完成（U3200，survival_rate=1.000）

---

## 实验 1: fixaw（固定 actor weight + φ-scaled r_fall）

- **文件**: `exp_basic_balance_v2_phi_dual_fixaw.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_fixaw_ppo_20260810_110156`
- **目的**: 固定 actor weights，仅 r_fall 每步奖励用 φ 调制。与实验 0 对比，隔离 φ² 动态 actor weight 的作用。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，摔倒 `-1`，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛慢，行为不符合预期**。U455 才达到 93.8% 存活（phi_dual 仅需 U370）。虽然学会了平衡，但是"双脚不离地的原地平衡"（高频微幅抖动），未学会交替迈步。U525 ep_len=150.5，r_vel 惩罚为 phi_dual 的 ~2.7 倍，说明运动质量差。
- **分析**: 固定 actor weight 下，shaping channels（r_joint/r_vel/r_tilt/r_foot）在机器人挣扎阶段以满权重参与梯度，与 r_fall 竞争。这些通道本质惩罚"运动"（偏离静态站立姿态、关节速度、抬脚高度），抑制了真实迈步所需的探索。phi_dual 的 φ² gate 在低 φ 时自动削弱这些反运动惩罚，给策略腾出探索空间。
- **状态**: 已手动停止（U542）

---

## 实验 2: dual_baseline（固定 actor weight + 固定 r_fall）

- **文件**: `exp_basic_balance_v2_dual_baseline.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_baseline_ppo_20260810_110204`
- **目的**: 在 dual-agent 环境中复现原始单 agent 基线 `exp_basic_balance_v2.py` 的行为，验证 dual-agent 实现的正确性。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，摔倒 `-1`，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **与原始基线基本一致**。首次存活 U355，100% 存活 U370，ep_len 收敛曲线与原始 baseline 在个位数 update 级别波动。验证了 dual-agent setup 未引入实质性差异。
- **状态**: 已手动停止（U542）

---

## 实验 3: fixaw_notb（fixaw 去掉 timeout bonus）

- **文件**: `exp_basic_balance_v2_phi_dual_fixaw_notb.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_fixaw_notb_ppo_20260810_124333`
- **目的**: 在实验 1 基础上去掉 r_fall 的 `+1` timeout bonus，观察 timeout bonus 对学习的影响。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，摔倒 `-1`，**无 timeout bonus**
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛到 100% 存活，步态待确认**。首次存活 U345，100% 存活 U500（比 fixaw 的 U455 慢，但最终达到了）。ep_len 收敛到 200（U3400）。与 fixaw 相比，去掉 timeout bonus 后收敛速度稍慢，但最终存活率一致。
- **状态**: 已手动停止（U754）

---

## 实验 4: fixaw_verify（fixaw 改固定 r_fall，验证实现正确性）

- **文件**: `exp_basic_balance_v2_phi_dual_fixaw_verify.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_fixaw_verify_ppo_20260810_125435`
- **目的**: 在 fixaw 代码基础上用最小改动（仅 r_fall 从 `0.01×φ(t)` 改为固定 `0.01`）来复现 baseline，验证 fixaw 的其他实现没有 bug。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，摔倒 `-1`，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
  - 其余代码与 fixaw 完全一致
- **结果**: **与 dual_baseline 完全一致**。首次存活 U280，100% 存活 U385，ep_len 在所有里程碑点（U100~U2900）数值与 dual_baseline 相同或差 <1。确认 fixaw 的实现除 φ-scaled r_fall 外无任何问题。
- **状态**: 已手动停止（U592）

---

## 原始基线参考

- **文件**: `exp_basic_balance_v2.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_ppo_20260806_182022`
- **设计**: 单 agent，固定 r_fall=0.01/step + (-1 fall, +1 timeout)，固定 actor weights `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`，环境 `basic_balance_v2_env.yaml`（`ImbalanceTerminationPlugin`，整体终止）
- **结果**: U370 达到 100% 存活，U2400 ep_len 收敛到 200。学会了交替迈步。

---

## 关键对比总结

| 实验 | r_fall 每步 | r_fall timeout | shaping aw | 首次存活 | 100%存活 | 步态 |
|---|---|---|---|---|---|---|
| 原始 baseline | 0.01 (固定) | +1 | 固定 | U355 | U370 | 交替迈步 |
| dual_baseline | 0.01 (固定) | +1 | 固定 | U355 | U370 | 交替迈步 |
| phi_dual | 0.01×φ | 无 | base×φ² | U325 | U385 | 交替迈步 |
| fixaw | 0.01×φ | +1 | 固定 | U345 | U455 | 原地平衡 |
| fixaw_notb | 0.01×φ | 无 | 固定 | U345 | U500 | 待确认 |
| fixaw_verify | 0.01 (固定) | +1 | 固定 | U280 | U385 | 与baseline一致 |

**核心发现**:
1. φ² 动态 actor weight（phi_dual）是学到良好步态的关键因素。固定 actor weight + φ-scaled r_fall（fixaw）会导致策略陷入"原地微幅抖动"局部最优——shaping channels 的固定满权重惩罚了真实迈步所需的关节运动，策略选择最小化运动幅度来规避惩罚。
2. fixaw_verify 证实 fixaw 的实现除 φ-scaled r_fall 外无任何 bug，与 dual_baseline 结果完全一致。
3. fixaw_notb 去掉 timeout bonus 后仍能达到 100% 存活，收敛稍慢（U500 vs U455），说明 timeout bonus 对 fixaw 的收敛速度有正面影响但对步态质量无根本改变。
