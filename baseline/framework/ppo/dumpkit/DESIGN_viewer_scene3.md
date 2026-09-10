# Debug Viewer — 场景三：Trajectory × Epoch 训练动态视图

## 定位与边界

**场景三聚焦于"一条 trajectory 在训练过程中的动态"**——即 PPO 多 epoch 训练循环中，这条 trajectory 的每一帧被 actor 和 critic 如何对待。

场景二展示的是 epoch-invariant 的训练目标（value/adv/return/combined_adv），场景三展示的是 epoch-variant 的训练动态（ratio/clip/new_value 随 epoch 变化）。

**与场景四的区别**：
- 场景三是**空间视角**：选定一条 trajectory，看它的帧 × epoch 矩阵
- 场景四是**时间视角**：看 minibatch 序列 × 指标的训练过程

## 目的

回答一个问题：**"这条 trajectory 的每一帧，在 PPO 的每个 epoch 中，policy 对它的评估怎么变了？被 clip 了吗？critic 的值怎么移动了？"**

用户选定一条 trajectory，选择一个 epoch，拖动进度条，同步看到：
1. 当前帧在当前 epoch 的 ratio（new/old log_prob）
2. 当前帧是否被 PPO clip
3. 当前帧在当前 epoch 的 new_log_prob
4. 当前帧在当前 epoch 的 new_value（critic 当前估计，随 epoch 变化）
5. 当前帧的 policy_loss 贡献（surr1 vs surr2）
6. 当前帧的 critic_loss 贡献
7. 整条 trajectory 在当前 epoch 的 ratio/clip/new_value 趋势曲线
8. epoch 间对比：ratio 如何从 epoch 0 的 ≈1.0 逐渐偏离

## 核心概念

### PPO 训练循环中的 per-frame 动态

```
epoch 0:
  actor = θ_old (rollout 时的参数)
  ratio = exp(log_prob(θ_old) - log_prob(θ_old)) = 1.0  ← 几乎不变
  clip_mask = False
  new_value = V(s) (critic 初始估计，= gae.npz 的 values_all)

epoch 0 mb 0:
  actor.step() → θ_0_1 (actor 移动了一点)
  critic.step() → critic_0_1 (critic 移动了一点)

epoch 0 mb 1:
  ratio = exp(log_prob(θ_0_1) - log_prob(θ_old))  ← 开始偏离 1.0
  new_value = critic_0_1(s)  ← critic 也变了
  ...

epoch 1:
  actor = θ_1 (经过 epoch 0 的全部 mb 更新)
  ratio = exp(log_prob(θ_1) - log_prob(θ_old))  ← 明显偏离 1.0
  clip_mask = |ratio - 1| > clip_eps  ← 部分帧开始被 clip
  ...

epoch 2:
  KL 可能超过 target_kl → actor early stop
  如果 actor stopped: ratio 不再变化（actor 不再更新）
  critic 继续: new_value 继续变化
```

### 关键关系

- `ratio = exp(new_log_prob - old_log_prob)`，epoch 0 ≈ 1.0，后续偏离
- `clip_mask = |ratio - 1| > clip_eps`，被 clip 的帧梯度被截断
- `surr1 = ratio × adv`，`surr2 = clip(ratio) × adv`，`policy_loss = -min(surr1, surr2)`
- `new_value = critic_current(s)`，随 critic 更新而变化
- `critic_loss = (new_value - return)²`，critic 试图追上 return
- early stop 后 actor 冻结，ratio 不再变化，但 critic 继续，new_value 继续变化

### epoch 采样策略

**关键设计决策：每个 epoch 结束后做一次 full-batch forward pass 采样。**

PPO 每 epoch shuffle 所有帧并分 minibatch 更新。trajectory 的 200 帧分散到 50 个 minibatch 中。要获得"这条 trajectory 在 epoch e 结束时的完整状态"，需要在 epoch e 的所有 mb 更新完成后，对这条 trajectory 的所有帧做一次 full-batch forward pass：

```python
# epoch e 结束后（dump_callback 活跃时）
with torch.no_grad():
    new_lp_e = actor.evaluate_actions(obs_t, act_t, explore_factor=ei_t).log_prob
    ratio_e = torch.exp(new_lp_e - old_lp_t)
    new_val_e = {ch: critics[ch](obs_t).reshape(-1) for ch in reward_keys}
    # 保存 per-frame 数据，标记 epoch=e
```

**为什么不在 minibatch 内采样**：minibatch 内只能看到该 mb 的帧，不是 trajectory 的完整视图。full-batch 采样保证每条 trajectory 在每个 epoch 有完整的 per-frame 快照。

**生产路径零开销**：`dump_callback is None` 时不执行 forward pass。

## 数据来源

### 新增 dump artifact: `epoch_frames.npz`

每个 epoch 结束后采样，保存 per-frame × per-epoch 数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `n_epochs` | int | 实际运行的 epoch 数 |
| `n_frames` | int | 总帧数（= sum(ep_lengths)） |
| `ep_lengths` | (n_trajs,) | 每条 trajectory 帧数（用于切片） |
| `ratio.{epoch}` | (N,) float32 | 该 epoch 结束时的 per-frame ratio |
| `clip_mask.{epoch}` | (N,) bool | 该 epoch 结束时是否被 clip |
| `new_log_prob.{epoch}` | (N,) float32 | 该 epoch 结束时的 new_log_prob |
| `new_value.{epoch}.{channel}` | (N,) float32 | 该 epoch 结束时 critic 的 V(s) 估计 |
| `actor_stopped_epoch` | int | actor 在哪个 epoch early stop（-1 = 未 stop） |

### 数据量估算

```
N = 204800 帧
n_epochs = 4
channels = 4

ratio:        204800 × 4 × 4 bytes = 3.3 MB
clip_mask:    204800 × 4 × 1 byte  = 0.8 MB
new_log_prob: 204800 × 4 × 4 bytes = 3.3 MB
new_value:    204800 × 4 × 4 × 4 bytes = 13.1 MB
─────────────────────────────────────────────
总计: ~20 MB / dump
```

可接受。如果需要减小，可以只保存 ratio + clip_mask + new_value（去掉 new_log_prob，因为它可以从 ratio 反推）。

### 复用已有数据

- `gae.npz: values_all` = epoch 0 之前的 critic 估计（= epoch -1 的 new_value）
- `combine.npz: combined_adv` = actor 使用的 advantage（epoch-invariant）
- `buffer.npz: log_probs` = old_log_prob（θ_old 下的，epoch-invariant）
- `buffer.npz: ep_lengths` = trajectory 切片信息

## 界面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│  Trajectory [42 ▼]  ep0000:robot_a  len=200  frame 50/200            │
│  Epoch [0 ▼]  (actor: active / stopped at epoch 2)                   │
│  [◄━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━►]  [▶ play]  speed: [1x ▼] │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ 趋势图（整条 trajectory，当前 epoch，当前帧竖线标记）              │ │
│  │                                                                    │ │
│  │ [线开关: ☑ratio ☑clip_mask ☑new_value ☐old_value]                │ │
│  │ [channel: r_potential ▼]  (new_value 是 per-channel 的)          │ │
│  │                                                                    │ │
│  │  ratio    1.0─────────────────────── (epoch 0: ≈1.0)             │ │
│  │          / \                                                       │ │
│  │         /   \  ← epoch 1: 偏离 1.0                                 │ │
│  │  1±eps ───────────────────────────── clip band                   │ │
│  │         \   /                                                      │ │
│  │          \_/  ← epoch 2: 更大偏离，部分被 clip                     │ │
│  │                                                                    │ │
│  │  new_value ─────────────────────── (critic 估计，随 epoch 变化)   │ │
│  │  old_value ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (epoch 0 前的初始估计)          │ │
│  │  return    ─────────────────────── (回归目标，不变)              │ │
│  │                                                                    │ │
│  │          0     50    100    150    200  (frame)                   │ │
│  │                ↑ 当前帧                                            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Epoch 对比（ratio 偏离过程）                                      │ │
│  │                                                                    │ │
│  │  epoch 0  ───────────────────────  (≈1.0, 几乎不变)              │ │
│  │  epoch 1  ────╲___╱──────────────  (轻微偏离)                    │ │
│  │  epoch 2  ──╲_____╱──────────────  (明显偏离，部分 clip)         │ │
│  │  epoch 3  ╱─────────╲────────────  (actor stopped, 冻结)         │ │
│  │          0     50    100    150    200  (frame)                   │ │
│  │                ↑ 当前帧                                            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  Current Frame: 50  |  Epoch: 1                                        │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Actor Dynamics (frame 50, epoch 1)                                │ │
│  │                                                                    │ │
│  │ old_log_prob:  -2.3412  (θ_old, epoch-invariant)                 │ │
│  │ new_log_prob:  -2.2987  (θ after epoch 1)                        │ │
│  │ ratio:         1.0438   (exp(new - old))                          │ │
│  │ clip_eps:      0.20     → clip band [0.80, 1.20]                 │ │
│  │ clip_mask:     False    (ratio in band, not clipped)             │ │
│  │                                                                    │ │
│  │ surr1 = ratio × adv = 1.0438 × (-0.0126) = -0.0132               │ │
│  │ surr2 = clip(ratio) × adv = 1.0438 × (-0.0126) = -0.0132         │ │
│  │ policy_loss_contrib = -min(surr1, surr2) = +0.0132               │ │
│  │ (if clipped: surr2 ≠ surr1, gradient truncated)                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Critic Dynamics (frame 50, epoch 1, per-channel)                 │ │
│  │                                                                    │ │
│  │ channel        old_value   new_value   return    critic_loss     │ │
│  │ ──────────── ────────── ────────── ────────── ────────────────── │ │
│  │ r_potential   +0.0183    +0.0191    +0.0053   (0.0138)²=1.9e-4  │ │
│  │ r_fall        -0.0783    -0.0791    -0.0823   (0.0032)²=1.0e-5  │ │
│  │ r_left_foot   +0.5412    +0.5398    +0.5002   (0.0396)²=1.6e-3  │ │
│  │ r_right_foot  +0.5238    +0.5211    +0.4784   (0.0427)²=1.8e-3  │ │
│  │                                                                    │ │
│  │ old_value = epoch 0 之前的 critic 估计 (gae.npz: values_all)     │ │
│  │ new_value = epoch 1 结束后的 critic 估计 (epoch_frames.npz)      │ │
│  │ return = critic 回归目标 (epoch-invariant, gae.npz: rets_all)   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 布局说明

**顶部**：Trajectory 选择器 + **Epoch 选择器** + 进度条
- Epoch 下拉框显示 `epoch 0/1/2/3` + actor 状态（active/stopped）
- 如果 actor 在 epoch 2 early stop，epoch 3 的 ratio 与 epoch 2 相同（actor 冻结）

**趋势图区**：当前 epoch 的整条 trajectory 曲线
- ratio 线 + 1±eps clip band（水平带）
- new_value vs old_value vs return（critic 移动过程）
- clip_mask 用底色标记被 clip 的帧区域
- 当前帧竖线

**Epoch 对比图**：多条 ratio 线叠加，展示偏离过程
- 每个 epoch 一条线，颜色深浅区分
- 直观看到 ratio 从 ≈1.0 逐渐偏离
- early stop 后的 epoch 线与前一 epoch 重合（actor 冻结）

**当前帧数据区**：两个表格

1. **Actor Dynamics**：ratio / clip / surr / policy_loss_contrib
   - 展示 PPO clipped surrogate 的计算细节
   - clip 时 surr1 ≠ surr2，梯度被截断

2. **Critic Dynamics**：per-channel old_value / new_value / return / critic_loss
   - 展示 critic 如何随 epoch 移动，是否在追上 return
   - old_value 是 epoch 0 前的估计，new_value 是当前 epoch 的估计

## 交互

| 操作 | 行为 |
|------|------|
| 拖动进度条 | 趋势图竖线、两个表格同步更新 |
| 切换 Epoch | 趋势图切换到该 epoch 的数据，表格更新 |
| 切换 Trajectory | 加载新 trajectory，进度条归零，epoch 归零 |
| 趋势图线开关 | 切换 ratio/clip_mask/new_value/old_value |
| channel 选择器 | 切换 new_value 显示哪个 channel |
| 点击趋势图 | 跳转到对应帧 |

## 关键设计决策

### 1. Epoch 是第一类维度

场景二没有 epoch 维度（数据 epoch-invariant）。场景三的核心就是 epoch 维度——用户需要看到"epoch 0 → epoch 1 → epoch 2"的演化过程。

Epoch 选择器与 Trajectory 选择器并列，两者都是第一类控件。

### 2. clip band 可视化

PPO 的 clip 机制是核心调试点。趋势图中用水平带标记 `[1-eps, 1+eps]` 区间，ratio 线超出带的部分就是被 clip 的帧。底色标记被 clip 的区域。

### 3. Epoch 对比图

单看一个 epoch 的 ratio 不够，需要对比多个 epoch。Epoch 对比图将所有 epoch 的 ratio 线叠加，直观展示偏离过程。这是场景三独有的视图。

### 4. Critic 追踪

critic 的 job 是追上 return。old_value（epoch 0 前）vs new_value（当前 epoch）vs return（不变）三条线，让用户看到 critic 是否在收敛。

### 5. early stop 可见性

如果 actor 在 epoch 2 early stop：
- epoch 2 和 epoch 3 的 ratio 线完全重合（actor 冻结）
- epoch 选择器中 epoch 3 标注 "actor stopped"
- new_value 仍在变化（critic 继续）

## 技术方案

| 层 | 选择 | 理由 |
|----|------|------|
| 后端 | Python HTTP server（复用 Scene 1/2） | 同一 server，新增 API |
| 前端 | 单页 HTML + vanilla JS + SVG | 无框架 |
| 图表 | SVG path（手写） | 多条线叠加，数据量小 |
| 启动 | `debug.py viewer <dump_dir>` | 与 Scene 1/2 同一入口 |

### 后端 API（新增）

```
GET /api/trajectory/<traj_idx>/epoch/<epoch>/overview
  → {
      traj_idx, epoch, ep_lengths[traj_idx],
      actor_stopped_epoch,
      channels: [...],
      clip_eps,
      // 整条 trajectory 在该 epoch 的逐帧数据
      frames: [
        {
          local_frame: 0,
          ratio: float,
          clip_mask: bool,
          new_log_prob: float,
          new_value: { channel: float, ... },
          // epoch-invariant 参考
          old_log_prob: float,
          old_value: { channel: float, ... },
          return: { channel: float, ... },
          combined_adv: float,
        },
        ...
      ]
    }

GET /api/trajectory/<traj_idx>/epoch/<epoch>/frame/<local_frame>
  → {
      local_frame, epoch,
      ratio, clip_mask, new_log_prob, old_log_prob,
      clip_eps,
      combined_adv,
      surr1, surr2, policy_loss_contrib,
      new_value: { channel: float, ... },
      old_value: { channel: float, ... },
      return: { channel: float, ... },
      critic_loss: { channel: float, ... },
    }

GET /api/trajectory/<traj_idx>/epoch_compare
  → {
      traj_idx, ep_lengths[traj_idx],
      n_epochs, actor_stopped_epoch, clip_eps,
      // 所有 epoch 的 ratio，用于叠加对比
      ratios_by_epoch: [
        { epoch: 0, ratios: [float, ...] },
        { epoch: 1, ratios: [float, ...] },
        ...
      ]
    }
```

## 前置工作

### 1. trainer.py 增加 per-epoch full-batch 采样

在 epoch 循环的末尾（所有 mb 更新完成后），当 `dump_callback` 活跃时，做一次 full-batch forward pass：

```python
# epoch 循环末尾，dump_callback 活跃时
if dump_callback is not None:
    with torch.no_grad():
        epoch_eval = actor.evaluate_actions(obs_t, act_t, explore_factor=ei_t)
        epoch_new_lp = epoch_eval.log_prob
        epoch_ratio = torch.exp(
            torch.clamp(epoch_new_lp - old_lp_t, -20.0, 20.0)
        )
        epoch_clip_mask = (epoch_ratio - 1.0).abs() > clip_eps
        epoch_new_val = {
            ch: critics[ch](obs_t).reshape(-1) for ch in reward_keys
        }
    dump_callback("epoch_frames", {
        "epoch": epoch,
        "ratio": epoch_ratio.cpu().numpy(),
        "clip_mask": epoch_clip_mask.cpu().numpy(),
        "new_log_prob": epoch_new_lp.cpu().numpy(),
        "new_value": {ch: v.cpu().numpy() for ch, v in epoch_new_val.items()},
        "actor_stopped": actor_stopped,
    })
```

### 2. dump_capture.py 增加 epoch_frames.npz 序列化

收集所有 epoch 的采样数据，合并保存：

```python
# dump_capture.py
epoch_frames_data = dump_collector.get("epoch_frames", [])
if epoch_frames_data:
    n_epochs = len(epoch_frames_data)
    n_frames = sum(ep_lengths)
    data = {
        "n_epochs": np.array(n_epochs),
        "n_frames": np.array(n_frames),
        "ep_lengths": np.array(ep_lengths),
        "actor_stopped_epoch": np.array(
            next((e["epoch"] for e in epoch_frames_data if e["actor_stopped"]), -1)
        ),
    }
    for e_data in epoch_frames_data:
        e = e_data["epoch"]
        data[f"ratio.{e}"] = e_data["ratio"]
        data[f"clip_mask.{e}"] = e_data["clip_mask"]
        data[f"new_log_prob.{e}"] = e_data["new_log_prob"]
        for ch, v in e_data["new_value"].items():
            data[f"new_value.{e}.{ch}"] = v
    np.savez_compressed(dump_dir / "epoch_frames.npz", **data)
```

### 3. 生产路径零开销

所有新增的 forward pass 都在 `if dump_callback is not None:` 守卫内。正常训练（无 dump）不执行任何额外计算。

## 不做的事

- 不做 per-minibatch 的 per-frame 数据（数据量太大，且 minibatch 内帧是随机子集，不是完整 trajectory 视图）——这是场景四的时间视角
- 不做梯度可视化（per-frame gradient 需要逐帧 backward，开销巨大）
- 不做跨 trajectory 对比
- 不做多 update 对比
- 不做 critic 网络结构可视化
