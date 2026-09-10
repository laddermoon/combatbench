# Debug Viewer — 场景四：Update 时间线训练动态视图

## 定位与边界

**场景四聚焦于"整个 update 的训练过程时间线"**——即 PPO 多 epoch × 多 minibatch 的训练循环中，全局指标如何随时间演化。

场景三是**空间视角**（一条 trajectory × epoch 的 per-frame 矩阵），场景四是**时间视角**（minibatch 序列 × 全局指标的时间线）。

**与场景三的区别**：
- 场景三回答："这条 trajectory 的帧，在训练中怎么被对待？"
- 场景四回答："这个 update 的训练过程，整体健康度如何？KL 什么时候爆？early stop 在哪触发？"

## 目的

回答一个问题：**"这个 update 的 PPO 训练循环，每个 minibatch 发生了什么？KL 怎么演化？什么时候 early stop？actor 和 critic 的 loss/grad 怎么走？"**

用户看到一条 minibatch 时间线（epoch 0 mb 0 → epoch 0 mb 1 → ... → epoch 3 mb 49），拖动时间游标，同步看到：
1. 当前 minibatch 的 KL / clip_frac / ratio_mean / ratio_max
2. 当前 minibatch 的 policy_loss / actor_grad_norm
3. 当前 minibatch 的 per-channel critic_loss / critic_grad_norm
4. 当前 minibatch 的 actor 状态（active / stopped）
5. KL 随 minibatch 的演化曲线（trust region 接近极限的过程）
6. early stop 触发点（时间线上的标记）
7. epoch 边界（时间线上的分隔线）

## 核心概念

### PPO 训练循环的时间结构

```
update = 4 epochs × 50 minibatches = 200 minibatch steps

时间线:
  epoch 0: mb 0  mb 1  mb 2  ... mb 49  | epoch 1: mb 0  mb 1 ... mb 49 | ...
  
  每个 mb step:
    1. shuffle 后取一个 minibatch（~4096 帧）
    2. critic 更新: new_V = critic(s) → MSE(new_V, return) → backward → step
    3. actor 更新（如果未 early stop）:
       ratio = exp(new_lp - old_lp)
       surr = min(ratio × adv, clip(ratio) × adv)
       loss = -mean(surr) → backward → step
       KL = (ratio - 1) - log(ratio)
    4. 检查 running_mean_kl > target_kl → early stop

  关键时间点:
    epoch 0 mb 0:  actor = θ_old, ratio ≈ 1.0, KL ≈ 0
    epoch 0 mb 10: actor 移动, ratio 偏离, KL 上升
    epoch 1 mb 0:  新 epoch, 重新 shuffle, 但 actor ≠ θ_old
    epoch 2 mb 15: running_mean_kl > target_kl → actor_stopped = True
    epoch 2 mb 16~49: actor 冻结, critic 继续
    epoch 3: actor 全程冻结, critic 继续
```

### 关键关系

- `KL_mb = mean((ratio - 1) - log(ratio))`，per-minibatch 的 k3 估计
- `running_mean_kl = mean(KL_mb for mb in current epoch)`，用于 early stop 判断
- `early_stop`: `running_mean_kl > target_kl` → actor 冻结，critic 继续
- `clip_frac_mb = mean(|ratio - 1| > clip_eps)`，per-mb 被 clip 的帧比例
- `policy_loss_mb = -mean(min(surr1, surr2))`，per-mb actor loss
- `critic_loss_mb = MSE(new_V, return) / n_active`，per-mb per-channel critic loss
- epoch 边界：每 epoch 重新 shuffle，但 actor/critic 参数跨 epoch 累积变化

### 时间线的两种粒度

**per-minibatch（默认）**：200 个数据点，每个 mb step 一个采样点。最细粒度。

**per-epoch（可选聚合）**：4 个数据点，每个 epoch 一个聚合点。用于快速概览。

## 数据来源

### 新增 dump artifact: `timeline.npz`

per-minibatch 的全局统计，在 trainer.py 的 mb 循环中收集。

| 字段 | 类型 | 说明 |
|------|------|------|
| `n_epochs` | int | 实际运行的 epoch 数 |
| `n_batches` | int | 每 epoch 的 minibatch 数 |
| `n_steps` | int | 总 minibatch 步数 = n_epochs × n_batches |
| `epoch_idx` | (n_steps,) int | 每个 step 的 epoch 编号 |
| `mb_idx` | (n_steps,) int | 每个 step 的 mb 编号 |
| `actor_active` | (n_steps,) bool | 该 step actor 是否运行 |
| `kl` | (n_steps,) float32 | per-mb KL（k3 估计） |
| `clip_frac` | (n_steps,) float32 | per-mb 被 clip 的帧比例 |
| `ratio_mean` | (n_steps,) float32 | per-mb ratio 均值 |
| `ratio_max` | (n_steps,) float32 | per-mb ratio 最大值 |
| `policy_loss` | (n_steps,) float32 | per-mb actor loss |
| `actor_grad_norm` | (n_steps,) float32 | per-mb actor 梯度范数 |
| `critic_loss.{channel}` | (n_steps,) float32 | per-mb per-channel critic loss |
| `critic_grad_norm.{channel}` | (n_steps,) float32 | per-mb per-channel critic 梯度 |
| `early_stop_step` | int | early stop 触发的 step（-1 = 未触发） |
| `target_kl` | float | PPO target_kl 配置 |
| `clip_eps` | float | PPO clip_eps 配置 |

### 数据量估算

```
n_steps = 4 × 50 = 200
channels = 4

per-step scalars:  200 × 8 × 4 bytes = 6.4 KB
per-channel:       200 × 4 × 2 × 4 bytes = 6.4 KB
metadata:          ~100 bytes
─────────────────────────────────────────────
总计: ~13 KB / dump
```

微不足道。

### 数据收集方式

这些统计在 trainer.py 的 minibatch 循环中**已经在计算**（`all_clip_fracs`、`all_ratio_means`、`all_actor_kls` 等列表），只是最后聚合成了全局平均。改为同时保存 per-minibatch 的原始序列即可。

```python
# trainer.py 现有代码（已计算但只存了聚合值）:
all_clip_fracs.append(clip_frac)        # per-mb
all_ratio_means.append(float(ratio.mean().item()))  # per-mb
all_ratio_maxs.append(float(ratio.max().item()))     # per-mb
all_actor_kls.append(approx_kl)         # per-mb
all_grad_norms_actor.append(float(grad_norm_a))     # per-mb
val_losses[key].append(float(val_loss))  # per-mb per-channel
all_grad_norms_critic[key].append(float(grad_norm_c))  # per-mb per-channel
epoch_pol_losses.append(float(policy_loss))  # per-mb
```

只需在 dump_callback 活跃时，将这些列表连同 epoch_idx/mb_idx/actor_active 一起序列化。

## 界面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│  Update Timeline  4 epochs × 50 batches = 200 steps                  │
│  [◄━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►] step 35/200│
│  epoch 0 | epoch 1 | epoch 2 | epoch 3                               │
│  early stop at step 115 (epoch 2, mb 15)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ KL 演化曲线                                                      │ │
│  │                                                                    │ │
│  │  target_kl ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │ │
│  │                  ╱─────╲                                          │ │
│  │           ╱─────╱       ╲─────╲                                  │ │
│  │      ╱────╱                   ╲────× early stop                  │ │
│  │  ───╱                              (actor frozen)                │ │
│  │  0    25    50    75   100   125   150   175   200  (step)       │ │
│  │  |epoch 0 |epoch 1 |epoch 2 |epoch 3 |                            │ │
│  │                ↑ 当前 step                                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ 多指标叠加（可切换显示哪些线）                                    │ │
│  │                                                                    │ │
│  │ [线开关: ☑KL ☑clip_frac ☑ratio_max ☐policy_loss ☐actor_grad]    │ │
│  │                                                                    │ │
│  │  KL          ╱─────╲                                              │ │
│  │  clip_frac   ────╱───╲─── (右轴)                                 │ │
│  │  ratio_max   ─────╱─────╲── (右轴)                               │ │
│  │                ↑ 当前 step                                        │ │
│  │  |epoch 0 |epoch 1 |epoch 2 |epoch 3 |                            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  Current Step: 35  |  Epoch 0, Minibatch 35  |  Actor: active          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Actor Stats (step 35)                                            │ │
│  │                                                                    │ │
│  │ KL:           0.0089    target_kl: 0.015  → below target ✓       │ │
│  │ clip_frac:    12.3%     clip_eps: 0.20                          │ │
│  │ ratio_mean:   1.0021    ratio_max: 1.84                         │ │
│  │ policy_loss:  -0.0072                                              │ │
│  │ actor_grad:   2.81       (pre-clip)                               │ │
│  │                                                                    │ │
│  │ running_mean_kl (this epoch): 0.0072  (target: 0.015)            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Critic Stats (step 35, per-channel)                              │ │
│  │                                                                    │ │
│  │ channel        critic_loss   critic_grad    value_loss (cumul)   │ │
│  │ ──────────── ────────────── ────────────── ───────────────────── │ │
│  │ r_potential   5.5e-5        0.0083         0.00019                │ │
│  │ r_fall        4.4e-5        0.0093         0.00019                │ │
│  │ r_left_foot   3.2e-4        0.0081         0.00032                │ │
│  │ r_right_foot  3.5e-4        0.0089         0.00035                │ │
│  │                                                                    │ │
│  │ (critic continues even after actor early stop)                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Early Stop Analysis                                              │ │
│  │                                                                    │ │
│  │ status: not triggered (step 35 < 115)                            │ │
│  │ if triggered:                                                     │ │
│  │   step 115, epoch 2, mb 15                                        │ │
│  │   running_mean_kl = 0.0152 > target_kl = 0.0150                   │ │
│  │   actor frozen from step 115 onward                               │ │
│  │   critic continues for remaining 85 steps                         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 布局说明

**顶部**：时间线进度条 + epoch 边界标记
- 进度条范围 0 ~ n_steps-1
- epoch 边界用竖线分隔
- early stop 点用 × 标记
- 当前 step 显示 `epoch / mb / actor_status`

**KL 演化曲线**：核心视图
- KL 随 minibatch step 的变化
- target_kl 水平虚线
- early stop 点标记
- epoch 边界竖线
- 当前 step 竖线

**多指标叠加图**：可切换的指标线
- KL / clip_frac / ratio_max / policy_loss / actor_grad
- 左右双轴（KL 和 clip_frac 量级不同）
- epoch 边界标记

**当前 step 数据区**：三个表格

1. **Actor Stats**：KL / clip_frac / ratio / policy_loss / grad + running_mean_kl
   - running_mean_kl 是 early stop 的判断依据
   - 与 target_kl 对比显示是否接近极限

2. **Critic Stats**：per-channel critic_loss / critic_grad
   - critic 在 actor early stop 后继续运行
   - 显示 critic 是否在稳定收敛

3. **Early Stop Analysis**：early stop 状态
   - 未触发：显示 "not triggered" + 距离 target_kl 的余量
   - 已触发：显示触发的 step/epoch/mb + running_mean_kl vs target_kl

## 交互

| 操作 | 行为 |
|------|------|
| 拖动时间线进度条 | 所有曲线竖线、三个表格同步更新 |
| 点击 KL 曲线的某个点 | 跳转到对应 step |
| 切换指标线 | 多指标叠加图更新 |
| 点击 early stop 标记 | 跳转到触发 step |

## 关键设计决策

### 1. KL 曲线是核心

PPO 的 trust region 机制是训练健康度的核心指标。KL 曲线让用户直观看到：
- KL 是否在上升（policy 在移动）
- KL 是否接近 target_kl（即将 early stop）
- early stop 后 KL 是否冻结（actor 不再更新）

### 2. epoch 边界可见

每 epoch 重新 shuffle，但 actor/critic 参数跨 epoch 累积。epoch 边界让用户区分：
- epoch 内的 KL 上升（同一 epoch 内 actor 持续移动）
- epoch 边界的 KL 跳变（新 shuffle 引入不同帧分布）

### 3. running_mean_kl 可见

early stop 的判断依据是 `running_mean_kl = mean(KL for mb in current epoch)`，不是单个 mb 的 KL。表格中显示 running_mean_kl 与 target_kl 的对比，让用户看到 early stop 的触发过程。

### 4. actor vs critic 分离

actor early stop 后冻结，但 critic 继续。表格明确标注 actor 状态（active/stopped），critic stats 始终显示（critic 全程运行）。

### 5. per-minibatch 粒度

200 个数据点足够画出平滑的演化曲线，也不会太密。per-epoch 聚合（4 个点）丢失了 epoch 内的动态。

## 技术方案

| 层 | 选择 | 理由 |
|----|------|------|
| 后端 | Python HTTP server（复用 Scene 1/2/3） | 同一 server，新增 API |
| 前端 | 单页 HTML + vanilla JS + SVG | 无框架 |
| 图表 | SVG path（手写） | 200 个数据点的折线图 |
| 启动 | `debug.py viewer <dump_dir>` | 与 Scene 1/2/3 同一入口 |

### 后端 API（新增）

```
GET /api/timeline/overview
  → {
      n_epochs, n_batches, n_steps,
      early_stop_step, target_kl, clip_eps,
      channels: [...],
      // 所有 step 的数据（用于画曲线）
      steps: [
        {
          step: 0, epoch: 0, mb: 0, actor_active: true,
          kl: 0.0001, clip_frac: 0.0, ratio_mean: 1.0, ratio_max: 1.0,
          policy_loss: -0.008, actor_grad: 2.5,
          critic_loss: { ch: float, ... },
          critic_grad: { ch: float, ... },
          running_mean_kl: 0.0001,
        },
        ...
      ]
    }

GET /api/timeline/step/<step>
  → {
      step, epoch, mb, actor_active,
      kl, clip_frac, ratio_mean, ratio_max,
      policy_loss, actor_grad,
      critic_loss: { ch: float, ... },
      critic_grad: { ch: float, ... },
      running_mean_kl, target_kl,
      early_stop_step, is_early_stopped,
    }
```

## 前置工作

### 1. trainer.py 收集 per-minibatch 统计

这些统计**已经在计算**，只需在 dump_callback 活跃时保存原始序列：

```python
# trainer.py，在 mb 循环中已有:
all_clip_fracs.append(clip_frac)
all_ratio_means.append(float(ratio.mean().item()))
all_ratio_maxs.append(float(ratio.max().item()))
all_actor_kls.append(approx_kl)
all_grad_norms_actor.append(float(grad_norm_a))
val_losses[key].append(float(val_loss))
all_grad_norms_critic[key].append(float(grad_norm_c))
epoch_pol_losses.append(float(policy_loss))

# 新增：dump_callback 活跃时，收集 per-step 元数据
if dump_callback is not None:
    timeline_step = {
        "epoch": epoch,
        "mb": mb_idx,
        "actor_active": not actor_stopped,
        "kl": approx_kl,
        "clip_frac": clip_frac,
        "ratio_mean": float(ratio.mean().item()),
        "ratio_max": float(ratio.max().item()),
        "policy_loss": float(policy_loss),
        "actor_grad": float(grad_norm_a),
        "critic_loss": {k: float(v) for k, v in ...},
        "critic_grad": {k: float(v) for k, v in ...},
        "running_mean_kl": float(np.mean(epoch_kls)) if epoch_kls else 0.0,
    }
    dump_callback("timeline_step", timeline_step)
```

或者更简单：在 `ppo_update` 返回后，将已有的 `all_*` 列表通过 dump_callback 一次性传出：

```python
# ppo_update 末尾，dump_callback 活跃时
if dump_callback is not None:
    dump_callback("timeline", {
        "epoch_idx": epoch_indices,       # (n_steps,)
        "mb_idx": mb_indices,              # (n_steps,)
        "actor_active": actor_active_flags, # (n_steps,)
        "kl": np.array(all_actor_kls),     # (n_steps,)
        "clip_frac": np.array(all_clip_fracs),
        "ratio_mean": np.array(all_ratio_means),
        "ratio_max": np.array(all_ratio_maxs),
        "policy_loss": np.array(pol_losses),
        "actor_grad": np.array(all_grad_norms_actor),
        "critic_loss": {k: np.array(v) for k, v in val_losses.items()},
        "critic_grad": {k: np.array(v) for k, v in all_grad_norms_critic.items()},
        "early_stop_step": early_stop_step,  # -1 if not triggered
        "target_kl": target_kl,
        "clip_eps": clip_eps,
    })
```

第二种方式更简洁，不需要修改 mb 循环内部，只在 `ppo_update` 末尾一次性传出。

### 2. dump_capture.py 增加 timeline.npz 序列化

```python
# dump_capture.py
if "timeline" in dump_collector:
    np.savez_compressed(dump_dir / "timeline.npz", **dump_collector["timeline"])
```

### 3. 生产路径零开销

所有数据收集都在 `if dump_callback is not None:` 守卫内。正常训练（无 dump）不执行任何额外操作。实际上 `all_*` 列表在正常训练中也会构建（用于聚合统计），但不会序列化。

## 不做的事

- 不做 per-frame 数据（那是场景三的空间视角）
- 不做跨 update 对比（那是更高级的场景）
- 不做梯度可视化（per-mb 的 per-param gradient 数据量太大）
- 不做 critic 网络结构可视化
- 不做 reward shaping / curriculum 可视化
