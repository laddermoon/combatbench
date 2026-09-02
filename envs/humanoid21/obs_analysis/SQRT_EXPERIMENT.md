# sqrt 观测变换实验对比报告

> 日期: 2026-09-01
> 分支: `obs-sqrt-transform`
> 基线 commit: `86b4f14` (sqrt 变换前)
> 变换 commit: `d595081` (sqrt 变换后)

## 1. 变换内容

对 96 维 flat observation 中的三个速度段做 `sign(v)·√|v|·c` 非线性压缩：

| 维度段 | 变换 | 说明 |
|---|---|---|
| `joint_vel [21:42]` (21维) | `sign(v)·√\|v\| / 2` | v 为线性归一化值 |
| `ang_vel [49:52]` (3维) | `sign(v)·√\|v/2\|` | v 为原始 rad/s |
| `opp_kp_vel [84:96]` (12维) | `sign(v)·√\|v\| / 2` | 手脚部分, 不含头部 [81:84] |

字典字段 (`root_state`, `opponent_keypoint_vel`, `core_state`) 保持原始物理值, 仅供 observer/reward plugin 使用。

**代价**: 所有现有 checkpoint 失效, 需要重新训练。

## 2. 实验设计

三组对照实验, 每组一个旧 run (无 sqrt) vs 一个新 run (有 sqrt), 使用相同 experiment 配置:

| 任务 | 旧 run (无 sqrt) | 新 run (有 sqrt) | 旧 updates | 新 updates |
|---|---|---|---|---|
| basic_balance_step | `train_basic_balance_step_ppo_20260901_003045` | `train_basic_balance_step_ppo_20260901_152102` | 2339 | 710 |
| standup | `train_standup_ppo_20260901_092840` | `train_standup_ppo_20260901_134103` | 2143 | 714 |
| basic_balance | `train_basic_balance_ppo_20260831_235604` | `train_basic_balance_ppo_20260901_150925` | 158 | 168 |

所有 run 使用相同的 PPO 超参数和 experiment 配置, 唯一差异是观测是否做了 sqrt 变换。

## 3. 结果

### 3.1 学习速度 (达到 ep_len 阈值所需 update 数)

| 阈值 | basic_balance_step | basic_balance |
|---|---|---|
| ep_len ≥ 30 | +30.0% (u20→u14) | +30.0% (u20→u14) |
| ep_len ≥ 40 | +22.5% (u80→u62) | +12.3% (u65→u57) |
| ep_len ≥ 50 | +10.4% (u134→u120) | +7.3% (u110→u102) |
| ep_len ≥ 75 | +8.3% (u180→u165) | +4.7% (u148→u141) |
| ep_len ≥ 100 | +7.9% (u189→u174) | +3.2% (u156→u151) |
| ep_len ≥ 190 | +6.4% (u203→u190) | — |

standup 任务 ep_len 始终为 200 (全 timeout), 无法用此指标对比。

### 3.2 Critic 冷启动 (update 1 的 EV)

| 任务 | 旧 (无 sqrt) | 新 (有 sqrt) | 改善 |
|---|---|---|---|
| basic_balance_step | -6.760 | -1.751 | 显著 |
| basic_balance | -6.760 | -1.751 | 显著 |
| standup | -1.593 | +0.506 | 显著 (从负转正) |

三个任务全部一致: sqrt 变换让 critic 在第一个 update 就有更好的初始 EV。

### 3.3 Entropy 行为

#### standup (关键差异)

| update | 旧 entropy | 新 entropy |
|---|---|---|
| 1 | 8.798 | 8.798 |
| 10 | 10.112 | 8.999 |
| 25 | 10.235 | 9.009 |
| 50 | 10.335 | 9.149 |
| 100 | 8.494 | 8.123 |
| 300 | -0.678 | 1.148 |
| 500 | -2.513 | -2.953 |
| 700 | -3.615 | -3.242 |

旧 run 前 50 步 entropy 从 8.8 **涨到 10.3** (策略发散), 然后才开始下降。新 run 从 8.8 **直接稳定在 9.0**, 没有发散阶段。

#### basic_balance_step

| update | 旧 entropy | 新 entropy |
|---|---|---|
| 1 | 8.798 | 8.798 |
| 100 | 6.908 | 7.131 |
| 300 | -1.530 | -0.695 |
| 500 | -6.880 | -5.915 |
| 700 | -7.411 | -6.932 |

新 run entropy 始终略高, 下降略慢, 但差异不大。

### 3.4 Tanh 饱和 (后期)

| 任务 @u700 | 旧 tanh_sat | 新 tanh_sat |
|---|---|---|
| basic_balance_step | 0.0010 | 0.0072 (**新更差**, 呈上升趋势) |
| standup | 0.0047 | 0.0010 (新更好) |

basic_balance_step 中新 run 的 tanh 饱和更高且随时间增长 (u400: 0.0003 → u700: 0.0072)。
standup 中新 run tanh 饱和更低。

### 3.5 EV (后期)

| 任务 @u700 | 旧 EV | 新 EV |
|---|---|---|
| basic_balance_step | 0.860 | 0.873 |
| standup | 0.957 | 0.969 |

后期 EV 水平接近, 新 run 略好但差异在噪声范围内。

### 3.6 Critic Value (return) — standup

| update | 旧 ret_mean | 新 ret_mean |
|---|---|---|
| 50 | 0.105 | 0.140 |
| 100 | 0.216 | 0.205 |
| 200 | 0.492 | 0.365 |
| 400 | 0.811 | 0.806 |
| 600 | 0.925 | 0.938 |

return 爬升曲线几乎重合, u400 后基本同步。

## 4. 结论

### 明确的正面效果

1. **早期学习加速 20-30%** — 两个 balance 任务在 ep_len 30-40 区间快 22-30%。最大的一致收益。
2. **Critic 冷启动显著改善** — update 1 的 EV 从 -6.76 提升到 -1.75 (balance), 从 -1.59 到 +0.51 (standup)。三个任务全部一致。
3. **standup 消除了早期 entropy 发散** — 旧 run entropy 涨到 10.3 后才下降, 新 run 稳定在 9.0。

### 收益衰减

加速效果随训练进行快速衰减: 30% → 22% → 10% → 8% → 6%。到后期两条曲线基本重合。sqrt 变换主要帮助**早期探索阶段** — 当策略还很随机、观测中噪声占主导时, 压缩极端值有帮助。一旦策略学会基本控制, 噪声减少, 变换的作用消失。

### 负面信号

basic_balance_step 中新 run 的 tanh 饱和更高 (0.0072 vs 0.0010 @u700) 且呈上升趋势。可能因为 sqrt 压缩后观测方差变小, 网络需要更大权重放大信号, 导致第一层更容易饱和。虽然 0.7% 还很低, 但趋势值得关注。

### 整体判定

**收益有限, 代价 (checkpoint 失效) 较高。**

- 早期加速 20-30% 在长训练中节省的时间有价值, 但不是决定性的
- 后期收敛速度基本一致, 不改变任务的可学习性
- checkpoint 失效是硬代价: 所有现有训练成果需要重来
- tanh 饱和的负面趋势需要长期观察

**决策: 暂不合入主分支。** 保留在 `obs-sqrt-transform` 分支, 后续如果:
- 需要从头重新训练所有策略, 或
- 发现 tanh 饱和成为瓶颈, 或
- 有更难的任务需要更好的早期探索

再考虑合入。

## 5. 文件位置

- 变换实现: `envs/humanoid21/simulator.py` (commit `d595081`)
- 文档更新: `envs/humanoid21/DATASPEC.md`, `envs/humanoid21/OBSERVATION_zh.md`
- 测试更新: `envs/humanoid21/tests/test_data_interfaces.py`
- 观测分布分析: `envs/humanoid21/obs_analysis/REPORT.md` (在主分支, 不受 sqrt 变换影响)
