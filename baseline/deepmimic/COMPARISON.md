# Actuation Profile 对比实验：simchar vs legacy

日期：2026-11-26。同一 clip、同一配方（2000 updates, 256 eps/update）、同一
评估协议（确定性 rsi=first 整段回放），唯一变量 = `actuation_profile`。

## 结论先行

**legacy 全面优于 simchar**：精度相当或更好，抖动（关节速度）低 4.5-6.5 倍。
simchar 在本管线没有带来任何收益，建议默认改用 legacy。

机理：simchar 的 PD 目标范围扩展（alpha∈[1.2,2.04]）+ 更高 KP 刚度，
把 action 空间相同的抖动放大成 2.5 倍以上的物理关节抖动；奖励中没有
平滑项，抖动零成本，策略收敛到"抖动换精度"解——但最终精度反而更差。

## Clip 1: accad_m2ma_punches_e1_jab_left（3.3s, T=66）

确定性整段回放（u2000 final export）：

| 指标 | simchar | legacy | 优胜 |
|---|---|---|---|
| completed | ✅ | ✅ | — |
| reward_mean | 0.955 | **0.990** | legacy |
| kp_err_m_mean | 0.071 | **0.063** | legacy |
| q_rmse_deg_mean | 4.66 | **3.78** | legacy |
| root_dyaw_deg_mean | 2.41 | **2.11** | legacy |
| action_flip_rate | 28.9% | **8.8%** | legacy |
| **jvel_abs_mean** | **3.21** | **0.50** | **legacy 6.4x** |
| jvel_flip_rate | 85.2% | 73.2% | legacy |

run 目录：
- `runs/dm_accad_m2ma_punches_e1_jab_left_simchar_cont/`（u700 被杀后续训）
- `runs/dm_accad_m2ma_punches_e1_jab_left_legacy_1126_1507/`
- 视频：各自 `eval_final.mp4`；指标：`eval_final.json`

## Clip 2: cmu_111_19（8.6s, T=171, strike_punch）

| 指标 | simchar | legacy | 优胜 |
|---|---|---|---|
| completed | ✅ | ✅ | — |
| reward_mean | 0.784 | **0.800** | legacy |
| kp_err_m_mean | **0.217** | 0.264 | simchar |
| q_rmse_deg_mean | 13.35 | **13.10** | ≈平 |
| root_dxy_m_mean | 0.169 | **0.139** | legacy |
| root_dyaw_deg_mean | 10.28 | **8.93** | legacy |
| action_flip_rate | 30.5% | **13.0%** | legacy |
| **jvel_abs_mean** | **3.79** | **0.84** | **legacy 4.5x** |
| jvel_flip_rate | 78.2% | **55.7%** | legacy |

run 目录：
- `runs/dm_cmu_111_19_1126_1557/`（simchar）
- `runs/dm_cmu_111_19_legacy_1126_1557/`（legacy）

长 clip 训练动态差异：legacy 在 u800 即达成 complete=1.0 且 fall=0 全程
稳定；simchar 到 u1300 才 complete=1.0，中期 fall 率反复 6-19%。
两版 kp_err/q_rmse 绝对值都明显差于短 clip（组合动作长片段本来就更难），
kp_err max 达 0.52-0.78m，说明高强度段落仍有明显漂移——这是后续改进点
（与 actuation 无关）。

## 抖动指标的解读

- `action_flip_rate`：相邻控制步 action 符号变化比例（策略输出层）
- `jvel_abs_mean`：实际关节速度（归一化）绝对值均值（物理层）
- `jvel_flip_rate`：关节速度符号翻转率。**注意**：小幅振荡也会计入翻转，
  因此该指标需结合 jvel_abs 一起看——legacy jab 的 73% 翻转发生在
  0.5 量级的速度上，视觉上几乎不可感知；simchar 同等翻转率发生在
  3.2 量级，即肉眼可见的剧烈抖动。

## 对 DESIGN.md 的修订

D11 原将 simchar 标为"对齐项"。实验证据表明：在本框架 20Hz 控制率 +
无平滑奖励的组合下，simchar 的高增益执行器是**净负面因素**。
论文 DeepMimic 使用 30-60Hz 控制且训练量远大于本次实验，simchar 的
对齐价值无法在本管线复现。默认 actuation_profile 改为 legacy；
simchar 保留为对照开关。
