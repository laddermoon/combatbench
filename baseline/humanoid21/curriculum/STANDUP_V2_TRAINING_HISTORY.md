# StandupV2 训练历史与关键决策记录

## 概述

StandupV2 实验的目标是训练 21-DoF 人形机器人从随机摔倒姿态恢复站立。使用 PBRS (Potential-Based Reward Shaping) 作为核心奖励机制，配合高度课程 (Curriculum) 逐步增加难度。

**最终成功模型**: `standup_v2_r14/checkpoints/checkpoint_u04615.pt` (4615 updates)

**复现问题**: 从零开始训练 (2026-07-02) 在 1109 updates 后仍无法突破 Stage 3→4 过渡，success=0.0。

---

## 训练时间线

### 第一阶段：初始实验 (r2, 2026-06-27 22:47 ~ 23:12)

**Commit**: `93bab5f` — feat: add standup_v2 experiment

**初始配置**:
| 参数 | 值 |
|------|-----|
| learning_rate | 3e-4 |
| critic_learning_rate | 3e-4 |
| entropy_coef | 1e-3 |
| log_std_min | -2.5 |
| potential_reward_scale | 1.0 |
| height_reward_scale | 0.0 |
| terminal_success_bonus | 0.0 |
| reward_keys | `r_potential` |

**Potential 函数 (V1, `StandupPotentialRewarder`)**:
| Stage | 描述 | Potential 范围 |
|-------|------|---------------|
| 5 | 完美站立 | 0.75 ~ 1.00 |
| 4 | 双脚站立 | 0.60 ~ 0.75 |
| 3 | 单脚支撑 | 0.45 ~ 0.60 |
| 2 | 手+脚支撑 | 0.30 ~ 0.45 |
| 1 | 手撑地 | 0.20 ~ 0.30 |
| 0 | 地面 | 0.00 ~ 0.20 |

**终止条件**: success_height=0.75, success_uprightness=0.85, success_hold_steps=10

**训练量**: r2 从 u1 训练到 u190 (190 updates)

**问题**: potential scale=1.0 太弱，reward 信号几乎被 PPO 噪声淹没。entropy_coef=1e-3 导致 std 无法降到 0.373 以下。

---

### 第二阶段：放大奖励信号 (r3, 2026-06-27 23:12 ~ 23:42)

**Commit**: `1594b36` — fix: amplify standup_v2 reward signal, lower entropy for precise control

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| reward_keys | `r_potential` | `r_standup` | 合并为单一 reward key |
| log_std_min | -2.5 | -4.0 | 允许更精确的平衡动作 |
| entropy_coef | 1e-3 | 5e-4 | 减少探索噪声 |
| potential_reward_scale | 1.0 | 5.0 | 5x 更强的梯度信号 |
| height_reward_scale | 0.0 | 2.0 | 添加连续向上梯度 |
| terminal_success_bonus | 0.0 | 10.0 | 添加稀疏目标信号 |

**新增**: `HeightObserver` 插件提供 per-step 高度数据

**训练量**: r3 从 u175 训练到 u315 (140 updates，从 r2 的 checkpoint 恢复)

**问题**: entropy bonus (5e-4 * 9 = 0.0045) 仍然是 policy gradient (0.0004~0.001) 的 4-10 倍，阻止 std 下降。

---

### 第三阶段：关闭熵奖励 (r3→r4 过渡, 2026-06-27 23:42)

**Commit**: `b31e2dd` — fix: disable entropy_coef to allow policy std reduction

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| entropy_coef | 5e-4 | 0.0 | entropy bonus 远大于 policy gradient，阻止 std 下降 |

**训练量**: r4 从 u315 训练到 u435 (120 updates)

---

### 第四阶段：提高学习率和奖励 (r4→r5 过渡, 2026-06-28 00:15)

**Commit**: `37c1ecd` — tune: increase LR 5e-4, height_reward 5.0, terminal_bonus 50

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| learning_rate | 3e-4 | 5e-4 | 加速学习 |
| critic_learning_rate | 3e-4 | 5e-4 | 加速 critic |
| height_reward_scale | 2.0 | 5.0 | 更强的向上梯度 |
| terminal_success_bonus | 10.0 | 50.0 | 更强的目标信号 |

**训练量**: r5 从 u435 训练到 u530 (95 updates)

**问题**: height_reward_scale 与 PBRS potential 在下蹲过渡时冲突（高度下降但 potential 可能上升）。

---

### 第五阶段：移除高度奖励，添加时间惩罚 (r5→r6 过渡, 2026-06-28 00:40)

**Commit**: `b045eaa` — fix: remove height delta reward, add time penalty

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| height_reward_scale | 5.0 | 0.0 | 与 PBRS 在下蹲过渡时冲突 |
| time_penalty | 0.0 | -0.01 | 添加紧迫感，鼓励快速站起 |

**训练量**: r6 从 u1 重新开始训练到 u320 (**320 updates，从零开始！**)

**关键发现**: r6 是一次**从零开始的完整重训**，与 r5 配置完全相同。这说明之前尝试从零复现，但 r6 只达到 u320 就停了（对比 r14 最终达到 u4615）。

---

### 第六阶段：新 V2 Potential 函数 (r7, 2026-06-28 00:59 ~ 01:32)

**Commit**: `f598dfa` — feat: new V2 potential function with transition gaps

**核心变更**: 替换 `StandupPotentialRewarder` → `StandupPotentialRewarderV2`

**V2 Potential 函数关键改进**:
| Stage | 描述 | V1 范围 | V2 范围 | 变化 |
|-------|------|---------|---------|------|
| 5 | 完美站立 | [0.75, 1.00] | [0.85, 1.00] | 提高 |
| 4 | 双脚站立 | [0.60, 0.75] | [0.65, 0.85] | 提高 |
| 3 | 单脚支撑 | [0.45, 0.60] | [0.40, 0.55] | 降低 |
| 2 | 手+脚支撑 | [0.30, 0.45] | [0.25, 0.40] | 降低 |
| 1 | 手撑地 | [0.20, 0.30] | [0.15, 0.25] | 降低 |
| 0 | 地面 | [0.00, 0.20] | [0.00, 0.15] | 降低 |

**关键**: Stage 3→4 之间创造了 **0.10 的 potential gap** (0.55→0.65)，为 risky 的 "放下第二只脚" 过渡提供明确奖励。

**新增 Stage 3.5**: 双脚+膝触地（过渡容忍），potential [0.55, 0.65]

**训练量**: r7 从 u435 恢复（**回到 r4 的 checkpoint，放弃 r6**）训练到 u535 (100 updates)

**决策**: r6 从零训练只到 u320 就停了，说明从零训练在当时也无法快速收敛。回到 r4 的 u435 checkpoint 继续训练。

---

### 第七阶段：降低 std，提高 potential scale (r7→r8 过渡, 2026-06-28 02:21)

**Commit**: `2e06734` — tune: reduce std to 0.22, increase potential scale to 10

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| potential_reward_scale | 5.0 | 10.0 | 更强的 PBRS 梯度 |
| time_penalty | -0.01 | -0.005 | 降低时间惩罚 |

**训练量**: r8 从 u535 训练到 u2670 (**2135 updates，最长的一次训练！**)

**这是突破性阶段**: 机器人从 Stage 3 突破到 Stage 4-5，开始有 success 记录。

---

### 第八阶段：降低成功阈值 (r8→r9 过渡, 2026-06-28 08:46)

**Commit**: `137e7df` — fix: create Stage 4→5 gap, lower success thresholds

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| success_height | 0.75 | 0.60 | 机器人到达 Stage 4 但无法维持严格原始标准 |
| success_uprightness | 0.85 | 0.70 | 同上 |
| success_hold_steps | 10 | 5 | 降低维持时间要求 |
| Stage 5 potential | [0.85, 1.00] | [0.90, 1.00] | 创建 Stage 4→5 gap (0.85→0.90) |
| Stage 5 h_score | (h-0.75)/0.15 | (h-0.60)/0.20 | 适配新阈值 |
| Stage 5 u_score | (u-0.85)/0.15 | (u-0.70)/0.20 | 适配新阈值 |

**训练量**: r9 从 u2670 训练到 u3360 (690 updates)

---

### 第九阶段：防止 jump-up exploit (r9→r10 过渡, 2026-06-28 09:58)

**Commit**: `415b0cc` — fix: require sustainable standing, prevent jump-up exploit

**问题**: success_hold_steps=5 (0.1s) 太短，机器人学会了瞬间向上冲触发 success，不维持平衡。

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| success_hold_steps | 5 | 50 | 要求 1 秒维持 (50Hz) |
| terminal_success_bonus | 50.0 | 100.0 | 更强的目标信号 |
| time_penalty | -0.005 | 0.0 | 移除时间惩罚（鼓励 rushing/exploiting） |
| Stage 5 速度门控 | 无 | mean_abs_joint_vel < 2.0 | 防止高速跳跃触发 Stage 5 |
| Stage 4.5b | 无 | 高速站立 potential [0.80, 0.85] | 过渡态降低 potential |
| v_score in Stage 5 | 一次方 | 三次方 | 更强的稳定性梯度 |

**训练量**: r10 从 u3360 训练到 u4035 (675 updates)

---

### 第十阶段：墙壁检测与惩罚 (r10→r12 过渡, 2026-06-28 11:40 ~ 12:00)

**Commit `2714635`**: fix: detect wall contacts, forbid wall-leaning in Stage 5
**Commit `e6b3aef`**: fix: strengthen anti-wall incentives
**Commit `ce10577`**: fix: only apply wall penalty at standing height (>0.45m)

**问题**: 机器人学会靠墙站立来通过 success 判定。

**变更**:
- Stage 5 新增 `not has_wall` 条件 — 靠墙不能进入 Stage 5
- Stage 4.5: 墙辅助站立 potential 降到 [0.40, 0.48]（低于 Stage 3）
- `wall_penalty = -0.05`：站立高度 (>0.45m) 时靠墙惩罚
- 墙在爬起过渡阶段不惩罚（height < 0.45m 时 free）

**训练量**: r11 从 u4035 到 u4140 (105 updates)，r12 从 u4035 到 u4430 (395 updates)

---

### 第十一阶段：Stage 5 持续奖励 (r13→r14, 2026-06-28 13:47)

**Commit**: `800c54f` — feat: add per-step Stage 5 bonus for sustained balance

**问题**: PBRS 只奖励状态转换，维持 Stage 5 时 reward 为零。

**变更**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| stage5_per_step_bonus | 0.0 | 0.1 | 每步 Stage 5 给 0.1 dense reward |

**训练量**: r14 从 u4485 训练到 u4615 (130 updates)

**最终模型**: `checkpoint_u04615.pt` — 被用于 hybrid 实验的 standup_net 初始化。

---

## 完整训练链路图

```
r2 (u0001→u0190)  从零开始，pot=1.0, ent=1e-3, lr=3e-4
  ↓ resume from u0175
r3 (u0175→u0315)  pot=5.0, ent=5e-4, h_reward=2.0, term=10
  ↓ ent=0
  ↓ resume from u0315
r4 (u0315→u0435)  lr=5e-4, h_reward=5.0, term=50
  ↓ h_reward=0, time_penalty=-0.01
  ↓ resume from u0435
r5 (u0435→u0530)  移除高度奖励
  ↓ 【从零重训，放弃 r5】
r6 (u0001→u0320)  从零开始，与 r5 配置相同 — 未继续
  ↓ 【放弃 r6，回到 r4 的 u0435】
  ↓ 新 V2 potential 函数 + transition gaps
r7 (u0435→u0535)  V2 potential, Stage 3→4 gap=0.10
  ↓ pot=10, time_penalty=-0.005
  ↓ resume from u0535
r8 (u0535→u2670)  ★ 突破阶段，2135 updates，Stage 3→4→5
  ↓ 降低成功阈值
  ↓ resume from u2670
r9 (u2670→u3360)  success: 0.75→0.60, uprightness: 0.85→0.70, hold: 10→5
  ↓ 防 jump-up exploit
  ↓ resume from u3360
r10 (u3360→u4035) hold=50, vel gate, term=100, time_penalty=0
  ↓ 墙壁检测与惩罚
r11 (u4035→u4140) Stage 5 禁止靠墙
r12 (u4035→u4430) wall_penalty=-0.05, 站立高度才惩罚
  ↓ resume from u4430
r13 (u4430→u4485) 微调
  ↓ stage5_per_step_bonus=0.1
  ↓ resume from u4485
r14 (u4485→u4615) ★ 最终模型，130 updates
```

**总训练量**: ~4615 updates (其中 r2-r5 贡献 530, r6 贡献 320 但被放弃, r7-r14 贡献 4080)

---

## 关键决策总结

| # | 决策 | 时间 | 影响 | 训练量 |
|---|------|------|------|--------|
| 1 | 放大 reward 信号 (pot 1→5, term 0→10) | r2→r3 | 让 reward 不被噪声淹没 | +140u |
| 2 | 关闭 entropy_coef | r3→r4 | 允许 std 下降到精确控制范围 | +120u |
| 3 | 提高 LR 到 5e-4 | r4→r5 | 加速学习 | +95u |
| 4 | 移除 height_reward (与 PBRS 冲突) | r5→r6 | 消除 reward 矛盾 | r6: 320u (从零) |
| 5 | **新 V2 potential 函数 + transition gaps** | r6→r7 | **核心改进**：Stage 3→4 gap | +100u |
| 6 | **pot scale 5→10** | r7→r8 | **突破性**：足够强的梯度 | +2135u |
| 7 | 降低成功阈值 | r8→r9 | 让 success 可达 | +690u |
| 8 | 防 jump-up (hold=50, vel gate) | r9→r10 | 防止 reward hacking | +675u |
| 9 | 墙壁惩罚 | r10→r12 | 防止靠墙作弊 | +500u |
| 10 | Stage 5 持续奖励 | r13→r14 | dense reward 维持站立 | +130u |

---

## 复现失败分析

### 从零训练 vs 原始训练的关键差异

**原始成功路径** (r2→r14):
1. r2-r5: 从零训练 530 updates，逐步调参
2. r6: 从零训练 320 updates — **被放弃**
3. r7-r14: 从 r4 的 u435 checkpoint 恢复，配合 V2 potential 函数训练到 u4615

**关键发现**: 原始训练**不是一次从零到成功的连续训练**。它是：
- 前 530 updates 从零训练建立基础（r2-r5）
- 然后回到 u435 checkpoint（r4 的最后状态）
- 配合**全新的 V2 potential 函数**重新开始训练
- V2 potential 函数是突破的关键

### 从零复现失败的可能原因

1. **V2 Potential 函数的 transition gaps 是后来才加的**
   - 原始 r2-r5 使用 V1 potential（无 gap），训练了 530 updates 建立基础
   - r7 开始使用 V2 potential（有 gap），从 u435 恢复
   - 从零使用 V2 potential 可能无法建立早期基础（gap 太大，早期 reward 稀疏）

2. **r6 的从零训练也被放弃了**
   - r6 使用 V1 potential + r5 的配置从零训练 320 updates
   - 但没有继续，而是回到 r4 的 u435 + V2 potential
   - 这暗示 r6 的从零训练效果也不好

3. **当前复现使用的配置是最终配置 (r14)**
   - pot=10, term=100, wall_penalty=-0.05, stage5_bonus=0.1
   - 这些参数是在已有 u4485 基础上微调的，不是从零训练的最佳配置
   - 从零训练可能需要更小的 pot scale 和更简单的 reward 结构

4. **1109 updates 仍卡在 Stage 3.88**
   - max_stage ≈ 3.88，接近 Stage 4 但无法稳定进入
   - max_potential ≈ 0.626，刚好在 Stage 3→4 gap 的下边界 (0.55) 之上
   - 机器人学会了接近 Stage 4 但无法完成 "放下第二只脚" 的过渡

### 建议的复现策略

1. **分阶段训练**：先用 V1 potential (无 gap, pot=5) 从零训练 ~500 updates 建立基础，然后切换到 V2 potential (有 gap, pot=10) 继续训练
2. **或者降低 V2 gap**：将 Stage 3→4 gap 从 0.10 减小到 0.05，让早期训练更容易获得过渡奖励
3. **降低 pot scale**：从零开始时用 pot=5，等机器人到达 Stage 3 后再提高到 pot=10
4. **参考 r6 的教训**：r6 从零训练 320 updates 后被放弃，说明即使 V1 potential 从零训练也不容易

---

## 最终配置 (r14, 用于 hybrid 实验)

```python
# PPO 参数
learning_rate = 5e-4
critic_learning_rate = 5e-4
entropy_coef = 0.0
log_std_min = -4.0
grad_clip_norm = 1.0
target_kl = 0.05
update_epochs = 4
minibatch_size = 4096

# Reward 配置
potential_reward_scale = 10.0
height_reward_scale = 0.0
terminal_success_bonus = 100.0
time_penalty = 0.0
wall_penalty = -0.05
stage5_per_step_bonus = 0.1

# 课程
height_thresholds = [0.5, 0.3, 0.15]
phase_transition_success_rate = 0.5

# 终止条件
success_height = 0.60
success_uprightness = 0.70
success_hold_steps = 50
stagnation_height = 0.25
stagnation_steps = 150

# V2 Potential 函数
# Stage 5: [0.90, 1.00] — 双脚+无墙+低速度+高+直立
# Stage 4.5: [0.80, 0.85] — 高速站立 (过渡态)
# Stage 4.5w: [0.40, 0.48] — 墙辅助站立 (capped low)
# Stage 4: [0.65, 0.85] — 双脚站立
# Stage 3.5: [0.55, 0.65] — 双脚+膝触地 (过渡容忍)
# Stage 3: [0.40, 0.55] — 单脚支撑
# Stage 2: [0.25, 0.40] — 手+脚支撑
# Stage 1: [0.15, 0.25] — 手撑地
# Stage 0: [0.00, 0.15] — 地面
```

---

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `experiments/exp_standup_v2.py` | 9 次修改 (93bab5f → 800c54f) |
| `rewards/standup_v2.py` | 5 次修改 (f598dfa → e6b3aef) |
| `plugins/standup_termination.py` | 3 次修改 (93bab5f → 415b0cc) |
| `blueprints/standup_v2_env.yaml` | 4 次修改 (93bab5f → 415b0cc) |
