# 观测空间分布分析报告

> 生成日期: 2026-09-01
> 数据来源: 两个训练完成的策略, 各 5 episodes x 600 steps x 2 robots = 6010 frames

## 1. 实验设计

### 策略来源

| 标签 | Run 目录 | 策略行为 |
|---|---|---|
| `attack_prep` | `baseline/runs/train_attack_prep_ppo_20260831_103818` | 攻击准备姿态, 站立为主 |
| `standup_step_v3` | `baseline/runs/train_standup_step_v3_ppo_20260826_112808` | 起身迈步, 含摔倒和站立 |

选择两个行为差异大的策略, 是为了区分**通用的尺度问题**和**策略依赖的分布特征**。

### 数据采集

- EnvBlueprint: `envs/humanoid21/blueprint.yaml`
- 采集方式: `RoundRunner` + `BaseFrameRecorder` (save_observation=True)
- 每帧记录 pre-action observation (96 维), 两个策略各 5 个 episode
- Seeds: attack_prep 10000-10004, standup_step_v3 20000-20004
- 所有 episode 均跑满 600 步 (timeout)

### 文件清单

| 文件 | 说明 |
|---|---|
| `obs_attack_prep.npy` | attack_prep 原始 obs 数组 (6010, 96) |
| `obs_standup_step_v3.npy` | standup_step_v3 原始 obs 数组 (6010, 96) |
| `attack_prep_all_dims.png` | 96 维直方图 (matplotlib 子图) |
| `standup_step_v3_all_dims.png` | 96 维直方图 (matplotlib 子图) |
| `hist_compact.txt` | ASCII 紧凑直方图 (含 mean/std/min/max/p1/p99) |

## 2. 观测空间结构 (96 维)

| 维度段 | 范围 | 维数 | 物理量 | 单位 | 当前归一化 |
|---|---|---|---|---|---|
| joint_pos | [0:21] | 21 | 关节位置 | rad | (q - ref) / scale, scale=range/2 |
| joint_vel | [21:42] | 21 | 关节速度 | rad/s | vel / scale, scale=range/2 (位置尺度) |
| proj_gravity | [42:45] | 3 | 重力投影 (机体系) | 无量纲 | 无 |
| height | [45:46] | 1 | root body 高度 | m | 无 |
| lin_vel | [46:49] | 3 | 质心线速度 (机体系) | m/s | 无 |
| ang_vel | [49:52] | 3 | 质心角速度 (机体系) | rad/s | 无 |
| feet_forces | [52:54] | 2 | 足底接触力 | BW (体重倍数) | 已除以 m*g |
| arena_center | [54:57] | 3 | 场心位置 (机体系) | m | 无 |
| opp_rel_pos | [57:60] | 3 | 对手相对位置 (机体系) | m | 无 |
| opp_rel_vel | [60:63] | 3 | 对手相对速度 (机体系) | m/s | 无 |
| opp_face | [63:66] | 3 | 对手朝向向量 (机体系) | 无量纲 | 无 |
| opp_kp_pos | [66:81] | 15 | 对手关键点位置 (机体系) | m | 无 |
| opp_kp_vel | [81:96] | 15 | 对手关键点线速度 (机体系) | m/s | 无 |

## 3. 分段统计

### attack_prep (站立为主)

| 维度段 | std_min | std_max | abs_max | p1 | p99 |
|---|---|---|---|---|---|
| joint_pos | 0.097 | 0.612 | 1.151 | -0.976 | 1.006 |
| **joint_vel** | **1.029** | **5.460** | **28.517** | **-10.202** | **8.859** |
| proj_gravity | 0.020 | 0.122 | 1.000 | -0.999 | 0.181 |
| height | 0.027 | 0.027 | 1.327 | 1.178 | 1.312 |
| lin_vel | 0.238 | 0.450 | 1.552 | -0.893 | 1.037 |
| ang_vel | 0.913 | 2.032 | 6.259 | -4.029 | 3.634 |
| feet_forces | 0.737 | 0.886 | 9.581 | 0.000 | 3.635 |
| arena_center | 0.155 | 0.945 | 2.561 | -1.945 | 2.354 |
| opp_rel_pos | 0.114 | 0.562 | 2.000 | -1.087 | 1.387 |
| opp_rel_vel | 0.243 | 0.502 | 1.896 | -1.256 | 0.919 |
| opp_face | 0.161 | 0.607 | 1.000 | -1.000 | 0.984 |
| opp_kp_pos | 0.112 | 0.663 | 2.122 | -1.392 | 1.536 |
| **opp_kp_vel** | **0.243** | **2.720** | **14.096** | **-5.652** | **5.478** |

全局 std ratio: **278x** (dim 38 std=5.46 vs dim 44 std=0.020)

### standup_step_v3 (起身迈步)

| 维度段 | std_min | std_max | abs_max | p1 | p99 |
|---|---|---|---|---|---|
| joint_pos | 0.076 | 0.525 | 1.243 | -0.986 | 1.007 |
| **joint_vel** | **1.022** | **6.469** | **15.581** | **-7.748** | **8.075** |
| proj_gravity | 0.092 | 0.164 | 1.000 | -0.994 | 0.682 |
| height | 0.135 | 0.135 | 1.300 | 0.407 | 1.298 |
| lin_vel | 0.185 | 0.397 | 4.084 | -0.749 | 1.010 |
| ang_vel | 0.877 | 1.479 | 10.132 | -2.676 | 3.211 |
| feet_forces | 0.537 | 0.582 | 5.527 | 0.000 | 2.233 |
| arena_center | 0.251 | 0.945 | 2.112 | -1.672 | 2.030 |
| opp_rel_pos | 0.784 | 1.889 | 4.072 | -2.785 | 3.802 |
| opp_rel_vel | 0.270 | 0.555 | 3.098 | -1.195 | 1.225 |
| opp_face | 0.151 | 0.294 | 1.000 | -0.978 | 0.653 |
| opp_kp_pos | 0.558 | 1.905 | 4.440 | -2.877 | 3.880 |
| opp_kp_vel | 0.270 | 1.734 | 10.194 | -2.427 | 2.909 |

全局 std ratio: **85x** (dim 41 std=6.47 vs dim 13 std=0.076)

## 4. 关键发现

### 4.1 joint_vel — 唯一的通用尺度问题

两个行为截然不同的策略中, joint_vel 都是 std 最大的维度段:

- attack_prep: std 1.03~5.46, abs_max 28.5
- standup_step_v3: std 1.02~6.47, abs_max 15.6

Top 10 高 std 维度全部来自 joint_vel。

**根因**: 当前归一化 `vel / (range/2)` 量纲不正确 — rad/s 除以 rad 得到 1/s, 不是无量纲量。且不同关节的 range 差 4 倍 (0.349 vs 1.484), 导致同样的角速度在小 range 关节上被放大 4 倍。

极端值 (28.5 rad/s) 主要来自关节抖动, 不是有意义的运动。

### 4.2 height — bias 浪费

两个策略 height mean 都在 1.264~1.265, std 仅 0.027~0.135。网络第一层需要一个大 bias (~1.265) 来抵消均值。减去站立参考高度 (~1.28) 可让 mean 归零, 但实际性能收益可忽略。

### 4.3 对手相关维度 — 高度策略依赖

| 维度段 | attack_prep std | standup_step_v3 std | 差异 |
|---|---|---|---|
| opp_rel_pos | 0.11~0.56 | 0.78~1.89 | 3x |
| opp_kp_pos | 0.11~0.66 | 0.56~1.91 | 3x |

standup_step_v3 中机器人在地上挣扎, 对手位置变化大。attack_prep 中双方站立对峙, 距离稳定。**无法在仿真端给通用配置。**

### 4.4 opp_kp_vel — 杠杆效应

opp_kp_vel 是关键点 (头/手/脚) 的线速度, 不是质心速度。手脚在挥动时末端线速度远大于质心 (杠杆效应), abs_max 到 10~14 m/s 是正常的。头部速度通常较小 (~1 m/s), 手脚速度大 (~10 m/s)。

### 4.5 proj_gravity — 策略依赖的方差

dim 44 (重力投影 z 分量) 在 attack_prep 中 std=0.020 (站直时恒为 -1), 在 standup_step_v3 中 std=0.164 (有时倒下)。这是预期行为, 不是问题。

### 4.6 feet_forces — 零膨胀右偏

大量零值 (腾空) + 少数大值 (着地冲击)。attack_prep dim 53 abs_max=9.58 BW (重落地)。分布形状是预期的, 归一化后尺度合理。

## 5. 量纲优化候选

| 维度段 | 问题 | 优先级 | 修复方案 | 代价 |
|---|---|---|---|---|
| **joint_vel** | 量纲错误, 抖动极端值 | **高** | `sign(v)*sqrt(|v|)/2` | 破坏 checkpoint |
| height | 未减参考值 | 低 | `height -= 1.28` | 破坏 checkpoint |
| ang_vel | 尺度略大 | 低 | `sign(v)*sqrt(|v/2|)` | 破坏 checkpoint |
| opp_kp_vel (手脚) | 杠杆效应大值 | 低 | `sign(v)*sqrt(|v|)/2` | 破坏 checkpoint |

注: 所有仿真端变换都会破坏现有 checkpoint 兼容性, 因为网络第一层权重是在原始观测尺度下训练的。非线性变换 (sqrt) 无法用权重迁移精确复现, 现有 checkpoint 需要重新训练。

## 6. 结论

1. **joint_vel 是唯一值得在量纲层面优化的维度** — 两个策略中都突出, 量纲不正确, 极端值来自抖动噪声
2. **其他维度的尺度差异要么是 O(1) 级别 (网络能处理), 要么高度策略依赖 (无法给通用配置)**
3. **foot-force 归一化 (已完成) 已捕获了最主要的饱和问题** — 修复后 99.8% 网络单元有活跃梯度
4. **可学习 Normalization 层不提供新能力** — 网络第一层已能学习任意 per-dimension 线性缩放
5. **固定非线性变换 (sqrt) 有实际价值但需重新训练** — 压缩抖动噪声, 放大微调信号
