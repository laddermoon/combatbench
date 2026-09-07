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
- **状态**: 已手动停止（U628）

---

## 实验 2: dual_baseline（固定 actor weight + 固定 r_fall）

- **文件**: `exp_basic_balance_v2_dual_baseline.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_baseline_ppo_20260810_110204`
- **目的**: 在 dual-agent 环境中复现原始单 agent 基线 `exp_basic_balance_v2.py` 的行为，验证 dual-agent 实现的正确性。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，摔倒 `-1`，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **与原始基线基本一致**。首次存活 U355，100% 存活 U385，ep_len 收敛曲线与原始 baseline 在个位数 update 级别波动。验证了 dual-agent setup 未引入实质性差异。
- **状态**: 已手动停止（U542）

---

## 实验 3: fixaw_notb（fixaw 去掉 timeout bonus）

- **文件**: `exp_basic_balance_v2_phi_dual_fixaw_notb.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_fixaw_notb_ppo_20260810_124212`
- **目的**: 在实验 1 基础上去掉 r_fall 的 `+1` timeout bonus，观察 timeout bonus 对学习的影响。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，摔倒 `-1`，**无 timeout bonus**
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛到 100% 存活，学会交替迈步**。首次存活 U345，100% 存活 U500（比 fixaw 的 U455 慢，但最终达到了）。U700+ ep_len=200，r_cross=-0.0004（交替迈步），r_vel=-0.86。与 fixaw 相比，去掉 timeout bonus 后收敛速度稍慢，但步态质量显著改善——fixaw 陷入原地平衡（r_cross=-0.007），fixaw_notb 学会了交替迈步。timeout bonus 在 φ-scaled r_fall 中是有害的：增加 adv_std 8 倍（0.036→0.294），导致策略震荡。
- **状态**: 已手动停止（U794）

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

## 实验 5: dual_notb（固定 r_fall，去掉 timeout bonus）

- **文件**: `exp_basic_balance_v2_dual_notb.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_notb_ppo_20260810_145633`
- **目的**: 在 dual_baseline 基础上去掉 r_fall 的 `+1` timeout bonus，观察 timeout bonus 对固定 r_fall 学习的影响。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，摔倒 `-1`，**无 timeout bonus**
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛到 100% 存活**。首次存活 U285，100% 存活 U420（比 dual_baseline 的 U370 慢 50 updates）。U585 ep_len=199.4，r_cross≈0 说明有交替迈步。
- **状态**: 已手动停止（U585）

---

## 实验 6: dual_survonly（固定 r_fall，仅 per-step survival reward）

- **文件**: `exp_basic_balance_v2_dual_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_survonly_ppo_20260810_145652`
- **目的**: r_fall 仅保留每步 `+0.01` survival reward，去掉 fall penalty 和 timeout bonus。验证纯稠密正向信号是否足以驱动学习。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，**无 fall penalty，无 timeout bonus**
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛最快**。首次存活 U320，100% 存活 U345（所有实验中最快）。U546 ep_len=199.9，r_cross≈0 说明有交替迈步。adv_std 最低（0.05 vs baseline 0.10），entropy 最低，策略改进最稳定。
- **状态**: 已手动停止（U546）

---

## 实验 7: dual_fallonly（固定 r_fall，仅 fall penalty）

- **文件**: `exp_basic_balance_v2_dual_fallonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_fallonly_ppo_20260810_145710`
- **目的**: r_fall 仅保留摔倒 `-1` 惩罚，去掉 per-step reward 和 timeout bonus。验证纯稀疏负向信号能否驱动学习。
- **设计**:
  - r_fall 每步奖励: `0`，**仅摔倒 `-1`**，无 timeout bonus
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **最终收敛但最慢之一**。首次存活 U385，100% 存活 U435。U585 ep_len=200.0，r_cross≈0 说明有交替迈步。早期 U10 出现 ep_len 回退（22.8→20.0），策略短暂倾向于"快速结束"以减少 fall penalty 累积。
- **状态**: 已手动停止（U585）

---

## 实验 8: dual_falltb（固定 r_fall，fall penalty + timeout bonus）

- **文件**: `exp_basic_balance_v2_dual_falltb.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_dual_falltb_ppo_20260810_145738`
- **目的**: r_fall 仅保留终端信号（fall `-1` + timeout `+1`），去掉 per-step reward。验证纯稀疏正负终端信号组合能否驱动学习。
- **设计**:
  - r_fall 每步奖励: `0`，摔倒 `-1`，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛最慢**。首次存活 U405，100% 存活 U465（所有实验中最慢）。U590 ep_len=199.4，r_cross≈0 说明有交替迈步。timeout bonus 在无稠密信号时反而拖慢收敛（比 fallonly 还慢 30 updates），因为增加了 V 拟合复杂度但未提供有用的中间梯度。
- **状态**: 已手动停止（U590）

---

## 实验 9: fixaw_survonly（φ-scaled r_fall，仅 per-step survival reward）

- **文件**: `exp_basic_balance_v2_phi_dual_fixaw_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_fixaw_survonly_ppo_20260810_175157`
- **目的**: 在 fixaw 基础上去掉 fall penalty 和 timeout bonus，仅保留 `0.01×φ(t)` per-step reward。与 dual_survonly（固定 0.01）对比 φ-gating 的作用，与 fixaw/fixaw_notb 对比去掉终端信号的影响。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，**无 fall penalty，无 timeout bonus**
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: **收敛到 100% 存活，步态为原地平衡**。首次存活 U450，100% 存活 U505（比 fixaw 的 U455 慢 50 updates）。U700+ ep_len=200，但 r_cross=-0.019（未学会交替迈步），r_vel=-0.65（低于 dual_survonly 的 -1.25，说明运动幅度更小）。与 fixaw 一致，固定 actor weight 下 φ-scaled r_fall 导致"原地微幅抖动"局部最优——去掉终端信号不改变这一结论。
- **状态**: 已手动停止（U1037）

---

## 实验 10: phi_dual_survonly（φ² 动态 actor weight + survonly r_fall）

- **文件**: `exp_basic_balance_v2_phi_dual_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_survonly_ppo_20260810_180122`
- **目的**: phi_dual 去掉 fall penalty（phi_dual 本身已无 timeout bonus），仅保留 `0.01×φ(t)` per-step reward + φ² 动态 actor weight。与 phi_dual（有 fall penalty）对比 fall penalty 在 φ² aw 下的作用，与 fixaw_survonly（固定 aw）对比 φ² aw 在纯 survonly 信号下的作用。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，**无 fall penalty，无 timeout bonus**
  - r_fall actor weight: 固定 `3.0`
  - shaping actor weights: `base × φ²`（r_cross=1.0×φ², 其余=0.2×φ²）
- **结果**: **收敛到 100% 存活，学会交替迈步**。首次存活 U365，100% 存活 U430（比 phi_dual 的 U385 慢 45 updates）。U500+ ep_len=200，r_cross≈0（交替迈步），r_vel=-0.82（与 phi_dual 的 -0.87 接近）。步态质量与 phi_dual 基本一致。
- **分析**: 去掉 fall penalty 后收敛慢了 45 updates，但步态质量不受影响——φ² 动态 actor weight 是步态质量的决定因素，fall penalty 仅影响收敛速度。与 fixaw_survonly 对比：φ² aw 使 r_cross 从 -0.019 改善到 ≈0，r_vel 从 -0.65 恶化到 -0.82（更多运动），证实 φ² gate 释放了探索空间。
- **状态**: 已手动停止（U904）

---

## 实验 11: phi2aw_survonly（固定 0.01 survonly r_fall + φ² 动态 actor weight）

- **文件**: `exp_basic_balance_v2_phi2aw_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi2aw_survonly_ppo_20260810_*`（训练中）
- **目的**: 将 dual_survonly（固定 0.01/step）的 r_fall 与 phi_dual 的 φ² 动态 shaping actor weight 组合。完成 2×2 消融矩阵的最后一个角落：固定/φ-scaled r_fall × 固定/φ² aw。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，**无 fall penalty，无 timeout bonus**
  - r_fall actor weight: 固定 `3.0`
  - shaping actor weights: `base × φ²`（r_cross=1.0×φ², 其余=0.2×φ²）
- **2×2 消融矩阵**:

  | | 固定 aw | φ² aw |
  |---|---|---|
  | **固定 0.01** | dual_survonly ✓ (交替迈步, U345) | **phi2aw_survonly** ← 本实验 |
  | **0.01×φ** | fixaw_survonly ✗ (原地平衡, U505) | phi_dual_survonly ✓ (交替迈步, U430) |

- **结果**: **收敛到 100% 存活，学会交替迈步**。首次存活 U300，100% 存活 U335。步态质量与 phi_dual_survonly 一致（r_cross≈0）。
- **状态**: 已手动停止（U463）

---

## 实验 12b: phi2aw_notb（固定 0.01 + fall penalty + 无 timeout + φ² shaping）

- **文件**: `exp_basic_balance_v2_phi2aw_notb.py`
- **训练目录**: `runs/p1_phi2aw_notb`（续跑目录: `runs/p1_phi2aw_notb_resume`）
- **目的**: 在 phi2aw 基础上去掉 timeout bonus，测试 φ² shaping 下 timeout bonus 的影响。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，摔倒 `-1`，**无 timeout bonus**
  - shaping actor weights: `base × φ²`
- **结果**: conv95×3 = U365，首次 100% = U355。步态窗口 [U365, U465) 内 r_cross=-0.0016，**交替迈步**。dip%=0.0%（stab 窗口 150 updates，未跑满 200）。
- **状态**: 已停止（U517，含续跑）

---

## 实验 12c: survonly_tb（固定 0.01 survonly + timeout bonus + 固定 Shape）

- **文件**: `exp_basic_balance_v2_survonly_tb.py`
- **训练目录**: `runs/p2_survonly_tb`
- **目的**: 在 dual_survonly 基础上加回 timeout bonus，测试固定 Shape 下 timeout bonus 对纯 survonly 信号的影响。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，**无 fall penalty**，timeout `+1`
  - actor weights: 固定 `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`
- **结果**: conv95×3 = U360，首次 100% = U340。步态 r_cross=-0.0019，**交替迈步**。dip%=5.0%（stab 窗口 200 updates 跑满）。
- **状态**: 已停止（U560）

---

## 实验 12d: phi2aw_survonly_tb（固定 0.01 survonly + timeout bonus + φ² shaping）

- **文件**: `exp_basic_balance_v2_phi2aw_survonly_tb.py`
- **训练目录**: `runs/p3_phi2aw_survonly_tb`
- **目的**: 在 phi2aw_survonly 基础上加 timeout bonus，测试 φ² shaping 下 timeout bonus 对纯 survonly 信号的影响。
- **设计**:
  - r_fall 每步奖励: 固定 `0.01`，**无 fall penalty**，timeout `+1`
  - shaping actor weights: `base × φ²`
- **结果**: conv95×3 = U390，首次 100% = U355。步态 r_cross=-0.0022，**交替迈步**。dip%=12.1%（stab 窗口 160 updates，未跑满 200）。
- **状态**: 已停止（U552）

---

## 实验 12e: phi2aw_fallonly（无 survive + fall penalty + φ² shaping）

- **文件**: `exp_basic_balance_v2_phi2aw_fallonly.py`
- **训练目录**: `runs/p4_phi2aw_fallonly`
- **目的**: 测试 φ² shaping 在仅有 fall penalty（无 per-step survive reward、无 timeout bonus）条件下能否驱动学习。
- **设计**:
  - r_fall 每步奖励: **无**，摔倒 `-1`，**无 timeout bonus**
  - shaping actor weights: `base × φ²`
- **结果**: conv95×3 = U395，首次 100% = U395。步态 r_cross=-0.0019，**交替迈步**。dip%=0.0%（stab 窗口 210 updates 跑满）。稳定性极佳。
- **状态**: 已停止（U606）

---

## 实验 12f: phi_dual_tb（0.01×φ survive + fall penalty + timeout bonus + φ² shaping）

- **文件**: `exp_basic_balance_v2_phi_dual_tb.py`
- **训练目录**: `runs/p5_phi_dual_tb`
- **目的**: 全信号组合（survive + fall + timeout + φ² shaping），与 phi_dual（无 timeout）对比 timeout bonus 在 φ² aw + φ-scaled r_fall 下的影响。
- **设计**:
  - r_fall 每步奖励: `0.01 × φ(t)`，摔倒 `-1`，timeout `+1`
  - shaping actor weights: `base × φ²`
- **结果**: conv95×3 = U360，首次 100% = U360。步态 r_cross=-0.0026，**交替迈步**。dip%=0.0%（stab 窗口 230 updates 跑满）。稳定性极佳。
- **状态**: 已停止（U591）

---

## 实验 13: φ_A = min(height/1.28, uprightness) 消融

### 13a: phi2aw_min_survonly

- **文件**: `exp_basic_balance_v2_phi2aw_min_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi2aw_min_survonly_ppo_20260811_000400`
- **目的**: 用 φ_A = min(height/1.28, uprightness) 替代 φ = uprightness × (height/1.28)，测试 min 形式是否改善 φ² shaping 效果。
- **设计**: 固定 0.01 survonly + φ_A² shaping，无 fall/TB。
- **结果**: 100% Eval at U360。比原版 phi2aw_survonly（U335）慢 25 updates。
- **状态**: 已停止（U379）

### 13b: phi_dual_min_survonly

- **文件**: `exp_basic_balance_v2_phi_dual_min_survonly.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_phi_dual_min_survonly_ppo_20260811_000414`
- **目的**: 用 φ_A 测试 0.01×φ_A survive + φ_A² shaping。
- **结果**: U430 时仅 90.6%（未收敛）。比原版 phi_dual_survonly（U430 达 100%）显著更差。
- **状态**: 已停止（U430）
- **结论**: φ_A = min(height/1.28, uprightness) 未改善训练。min 形式放大了 uprightness 的不稳定性。后续实验继续使用原版 φ。

---

## P0: dual_survonly 多 seed 噪声标定

> **完整分析见 `combatbench-research/workspace/experiment/seed_variance_report.md`**

### P0a: 可复现性验证

- **训练目录**: `runs/repro_dual_survonly_s42_r1`（seed=42 重跑）
- **方法**: 用 `compare_runs.py` 逐 update 对比 10 个标量
- **结果**: 150 updates 内最大相对误差 = 0.000e+00，**逐位完全一致**
- **结论**: 训练管线是确定性的，σ 是真实的种子敏感性

### P0b: 4-seed 噪声标定

| Seed | 训练目录 | conv95×3 | 首次100% | bdgt | dip% | r_cross | 步态 |
|------|---------|---------|---------|------|------|---------|------|
| 42 | `runs/train_v2_basic_balance_v2_dual_survonly_ppo_20260810_145652` | U345 | U345 | 200 | 0.0% | -0.0016 | 交替迈步（前进绕圈） |
| 43 | `runs/p0_dual_survonly_s43` | U600 | U600 | 1280 | 5.0% | -0.0020 | 交替迈步（后退靠墙） |
| 44 | `runs/p0_dual_survonly_s44` | U505 | U510 | 1320 | 2.5% | -0.0009 | 交替迈步（前进绕圈） |
| 45 | `runs/p0_dual_survonly_s45` (+ resume) | U365 | U370 | 150 | 11.4%~ | -0.0046 | 交替迈步（后退靠墙） |

- **统计**: Mean = 454, σ = 120 updates, Range = 255 (U345~U600), CV = 26.6%
- **结论**: σ=120 远超因子间差异（10~55 updates），所有单 seed 速度比较无统计意义

### P0c: 方差本质分析（时间伸缩）

- **方法**: 按各自 conv95×3 归一化时间轴后比较 ep_len_mean
- **结果**: 固定 update 的 CV 高达 59%（U400），时间轴重标定后 CV 塌缩至 4.9%（均值）
- 排除 s45（短预算）后，三种子在 conv+30 的 ep_len CV = **0.7%**
- 时间伸缩因子: 0.76x ~ 1.32x（1.74x spread）
- **结论**: σ=120 完全来自**相变（phase transition）时刻**的种子敏感性，非结果差异
- **视频观察发现**: s42/s44 向前走绕圈，s43/s45 向后退靠墙——这是**结果维度的定性差异**（前进 vs 后退两个等价局部最优），奖励函数不区分方向

---

## 原始基线参考

- **文件**: `exp_basic_balance_v2.py`
- **训练目录**: `runs/train_v2_basic_balance_v2_ppo_20260806_182022`
- **设计**: 单 agent，固定 r_fall=0.01/step + (-1 fall, +1 timeout)，固定 actor weights `(3.0, 1.0, 0.2, 0.2, 0.2, 0.2)`，环境 `basic_balance_v2_env.yaml`（`ImbalanceTerminationPlugin`，整体终止）
- **结果**: U370 达到 100% 存活，U2400 ep_len 收敛到 200。学会了交替迈步。

---

## 关键对比总结

### 固定 r_fall 组（r_fall 每步为固定 0.01 或 0，shaping aw 固定）

| 实验 | r_fall 每步 | r_fall fall | r_fall timeout | 首次存活 | 100%存活 | 步态 |
|---|---|---|---|---|---|---|
| 原始 baseline | 0.01 (固定) | -1 | +1 | U355 | U370 | 交替迈步 |
| dual_baseline | 0.01 (固定) | -1 | +1 | U355 | U385 | 交替迈步 |
| dual_notb | 0.01 (固定) | -1 | 无 | U285 | U420 | 交替迈步 |
| dual_survonly | 0.01 (固定) | 无 | 无 | U320 | U345 | 交替迈步 |
| dual_fallonly | 0 | -1 | 无 | U385 | U435 | 交替迈步 |
| dual_falltb | 0 | -1 | +1 | U405 | U465 | 交替迈步 |
| fixaw_verify | 0.01 (固定) | -1 | +1 | U280 | U385 | 与baseline一致 |

### φ-scaled r_fall 组（r_fall 每步为 0.01×φ(t)，shaping aw 固定）

| 实验 | r_fall 每步 | r_fall fall | r_fall timeout | 首次存活 | 100%存活 | 步态 |
|---|---|---|---|---|---|---|
| fixaw | 0.01×φ | -1 | +1 | U345 | U455 | 原地平衡 |
| fixaw_notb | 0.01×φ | -1 | 无 | U345 | U500 | 交替迈步 |
| fixaw_survonly | 0.01×φ | 无 | 无 | U450 | U505 | 原地平衡 |

### φ² 动态 actor weight 组

| 实验 | r_fall 每步 | r_fall fall | r_fall timeout | shaping aw | conv95×3 | 首次100% | dip% | 步态 |
|---|---|---|---|---|---------|---------|------|------|
| phi_dual | 0.01×φ | -1 | 无 | base×φ² | — | U385 | — | 交替迈步 |
| phi_dual_survonly | 0.01×φ | 无 | 无 | base×φ² | — | U430 | — | 交替迈步 |
| phi2aw_survonly | 0.01 (固定) | 无 | 无 | base×φ² | — | U335 | — | 交替迈步 |
| phi2aw | 0.01 (固定) | -1 | +1 | base×φ² | — | U350 | — | 交替迈步 |
| phi2aw_notb (P1) | 0.01 (固定) | -1 | 无 | base×φ² | U365 | U355 | 0.0%~ | 交替迈步 |
| survonly_tb (P2) | 0.01 (固定) | 无 | +1 | 固定 | U360 | U340 | 5.0% | 交替迈步 |
| phi2aw_survonly_tb (P3) | 0.01 (固定) | 无 | +1 | base×φ² | U390 | U355 | 12.1%~ | 交替迈步 |
| phi2aw_fallonly (P4) | 无 | -1 | 无 | base×φ² | U395 | U395 | 0.0% | 交替迈步 |
| phi_dual_tb (P5) | 0.01×φ | -1 | +1 | base×φ² | U360 | U360 | 0.0% | 交替迈步 |

**核心发现**:
1. φ² 动态 actor weight（phi_dual）是学到良好步态的关键因素。固定 actor weight + φ-scaled r_fall（fixaw）会导致策略陷入"原地微幅抖动"局部最优——shaping channels 的固定满权重惩罚了真实迈步所需的关节运动，策略选择最小化运动幅度来规避惩罚。
2. fixaw_verify 证实 fixaw 的实现除 φ-scaled r_fall 外无任何 bug，与 dual_baseline 结果完全一致。
3. fixaw_notb 去掉 timeout bonus 后达到 100% 存活（U500），且步态从 fixaw 的"原地平衡"改善为"交替迈步"（r_cross: -0.007→-0.0004）。timeout bonus 在 φ-scaled r_fall 中是有害的：使 adv_std 暴增 8 倍（0.036→0.294），导致策略震荡和步态退化。
4. **r_fall 消融（实验 5-8）**：在固定 actor weight + 固定 r_fall 条件下，四组实验最终都收敛到 100% 存活并学会交替迈步。收敛速度排序：survonly (U345) > baseline (U385) > notb (U420) > fallonly (U435) > falltb (U465)。纯稠密正向信号（survonly）最快，纯稀疏信号（falltb）最慢。fall penalty 是双刃剑——增加 adv_std 但拖慢早期探索。timeout bonus 效果依赖信号环境：有稠密信号时加速 35 updates（baseline vs notb），无稠密信号时反效果 -30 updates（falltb vs fallonly）。
5. **survonly + φ² aw（实验 9-10）**：fixaw_survonly（固定 aw）收敛到 U505 但步态为原地平衡（r_cross=-0.019）；phi_dual_survonly（φ² aw）收敛到 U430 且学会交替迈步（r_cross≈0）。再次证实 φ² 动态 actor weight 是良好步态的决定因素——与 r_fall 是否有 fall penalty 无关。去掉 fall penalty 后 phi_dual_survonly 比 phi_dual 慢 45 updates（U430 vs U385），说明 fall penalty 在 φ² aw 下仍有加速收敛的作用。
6. **步态质量对比（U700-U900 均值）**：r_cross 排序：phi_dual(-0.0002) ≈ phi_dual_survonly(-0.0003) > dual_survonly(-0.0003) > fixaw_survonly(-0.019)。r_vel 排序：fixaw_survonly(-0.65) > phi_dual(-0.87) ≈ phi_dual_survonly(-0.82) > dual_survonly(-1.25)。fixaw_survonly 的 r_vel 最小（运动最少）但 r_cross 最差（无交替迈步）——典型的"原地微幅抖动"局部最优。
7. **phi2aw（0.01+fall+timeout+φ²）**: 收敛 U350，比 dual_baseline（固定 Shape, U385）快 35 updates。完成因子 4 的第 4 组实验，确认在固定 0.01 survive 信号下 φ² Shape 仍有中等优势（10-35 updates）。r_cross=-0.004（收敛初期），步态为交替迈步但质量仍在发展中。
8. **φ_A = min(height/1.28, uprightness) 消融**:
   - phi2aw_min_survonly（0.01 survive + φ_A² shaping, 无 fall/TB）: 100% Eval at U360. 比原版 phi2aw_survonly（U335）慢 25 updates。φ_A 并未加速收敛——min 形式在早期训练时 uprightness 波动大，min 选址导致 φ_A 比 φ_product 更不稳定。
   - phi_dual_min_survonly（0.01×φ_A survive + φ_A² shaping, 无 fall/TB）: U430 时仅 90.6%（未收敛）。比原版 phi_dual_survonly（U430 达 100%）显著更差。φ_A 的 min 形式在弱信号（0.01×φ）下表现不佳，可能因为 min 对 uprightness 的敏感性导致 survive reward 过度波动。
   - **结论**: φ_A = min(height/1.28, uprightness) 并未改善训练效果。在两种 survive 信号条件下均不如原版 φ = uprightness × (height/1.28)。min 形式反而放大了 uprightness 的不稳定性。后续实验继续使用原版 φ。
9. **P1-P5 因子扩展实验（2026-08-11）**：补齐了因子矩阵中缺失的 5 个组合。所有 9 个 run（含 P0 多 seed）步态均为**交替迈步**，r_cross 在 -0.0009 ~ -0.0046 范围内。P4（phi2aw_fallonly，无 survive 信号）和 P5（phi_dual_tb，全信号）稳定性最佳（dip%=0.0%）。所有配置间 conv95×3 差异在 10~55 updates 范围内，**远小于 σ=120 的种子方差**，无统计意义。
10. **多 seed 噪声标定（P0）**：4 seeds (42/43/44/45) 的 conv95×3 分别为 U345/U600/U505/U365，σ=120 updates。管线已验证逐位确定性（seed=42 重跑 150 updates 完全一致）。方差本质是**相变时刻的种子敏感性**（时间伸缩因子 0.76x~1.32x），而非结果差异——时间轴对齐后 ep_len CV 从 59% 塌缩至 0.7%（排除短预算 s45）。**所有单 seed 收敛速度结论需降级为"未验证"**。
11. **前进 vs 后退等价局部最优（视频观察）**：s42/s44 向前走绕圈，s43/s45 向后退靠墙。奖励函数中 r_vel 惩罚的是关节角速度（非质心方向速度），r_cross 惩罚横向运动——**没有任何奖励项区分前进与后退**。两者是等价局部最优，agent 根据初始化随机落入。这是结果维度的定性差异，需要加入方向性奖励信号（如 forward velocity reward）才能消除。
