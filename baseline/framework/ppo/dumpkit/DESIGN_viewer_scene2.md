# Debug Viewer — 场景二：Trajectory → Value / Advantage / Return 视图

## 定位与边界

**场景二聚焦于"训练输入数据的截面"**——即 PPO update 中 epoch 循环**之前**计算的所有数据。

这些数据是 epoch-invariant 的：value / advantage / return / combined_adv 在整个多 epoch 训练循环中不变，是 actor 和 critic 的训练目标。

**不在场景二范围内**：
- per-epoch 演化（KL 随 epoch 变化、ratio 偏离、early stop 触发）
- per-minibatch 动态（per-frame ratio / clip mask / gradient）
- policy 参数变化（θ_old → θ_new 的移动）

这些属于"训练过程动态"，是后续场景的职责。当前 dump 也没有保存 per-epoch / per-minibatch 数据。

## 目的

回答一个问题：**"这条 trajectory 的每一帧，critic 估了什么值，GAE 算了什么 advantage，combine 后 actor 实际用了什么信号？"**

用户选定一条 trajectory，拖动进度条，同步看到：
1. 当前帧的 reward（per-channel）
2. 当前帧的 value（per-channel，critic 估计）
3. 当前帧的 advantage（per-channel，GAE 输出）
4. 当前帧的 return（per-channel，critic 回归目标）
5. 当前帧的 combined advantage（actor 实际使用的信号）
6. 当前帧的 actor_weight / confidence / EV（combine 的权重链路）
7. 整条 trajectory 的趋势曲线（value / advantage / return / reward）

## 核心概念

### 数据流水线（trainer.py `ppo_update`，epoch 循环之前）

```
reward (per-channel, per-frame)
  ↓ critic(s_t) → V_c(s_t)           [gae.npz: values_all]
  ↓ GAE(reward, V, γ, λ) → A_c, R_c  [gae.npz: advs_all, rets_all]
  ↓ z-score normalize → norm_adv_c   [combine.npz 内部计算]
  × confidence_c = sqrt(clip(EV_c))   [combine.npz: confidences]
  × actor_weight_c (L1-normalized)    [combine.npz: key_actor_weight_frame]
  ↓ Σ_c → combined_adv                [combine.npz: combined_adv]
```

这些数据是 **epoch-invariant** 的：在整个多 epoch 训练循环中不变，是 actor 和 critic 的训练目标。

epoch 循环（参数更新、KL 演化、early stop）不属于场景二，是后续场景的职责。

### 关键关系

- `return = advantage + value`（GAE 定义）
- `combined_adv = Σ_c aw_c_normed × conf_c × norm_adv_c`（actor 看到的单一信号）
- `confidence_c = sqrt(clip(EV_c, 0, 1))`（EV 低 → 置信度低 → 贡献小）
- `aw_c_normed = aw_c / Σ|aw|`（L1 归一化，解耦 channel 平衡与有效学习率）
- `norm_adv_c = z-score(adv_c)`（per-channel 归一化，scale-invariant）

### 通道状态

每个 channel 在每条 trajectory 上有独立的状态：
- `key_seg_active[channel][traj_idx]`：channel 是否参与此 trajectory
- `key_seg_terminated[channel][traj_idx]`：trajectory 是否真正终止（vs 截断需 bootstrap）
- `key_frame_mask[channel]`：per-frame 展开的 active 标志
- inactive channel 的 value/adv/ret 全为 0

## 数据来源

所有数据都是 per-frame flat 数组，按 trajectory 顺序拼接。trajectory `j` 的帧范围：
```
[seg_offset[j], seg_offset[j] + ep_lengths[j])
seg_offset[j] = sum(ep_lengths[:j])
```

### gae.npz

| 字段 | 类型 | 说明 |
|------|------|------|
| `values_all.{channel}` | (N,) float32 | critic 对每帧的 V(s) 估计 |
| `advs_all.{channel}` | (N,) float32 | GAE advantage A_c(s) |
| `rets_all.{channel}` | (N,) float32 | return target R_c(s) = A_c + V_c |
| `key_frame_mask.{channel}` | (N,) bool | channel 在此帧是否 active |
| `bootstrap_values.{channel}` | (n_trajs,) float32 | 截断 trajectory 的 V(s_next) bootstrap 值 |
| `key_seg_active.{channel}` | (n_trajs,) bool | channel 是否参与此 trajectory |
| `key_seg_terminated.{channel}` | (n_trajs,) bool | trajectory 是否真正终止 |

### combine.npz

| 字段 | 类型 | 说明 |
|------|------|------|
| `combined_adv` | (N,) float32 | actor 实际使用的 advantage |
| `key_actor_weight_frame.{channel}` | (N,) float32 | per-frame actor_weight（已展开） |
| `confidences.{channel}` | scalar float | per-channel confidence = sqrt(clip(EV)) |
| `aw_l1_sum` | (N,) float32 | per-frame Σ|aw|（L1 归一化分母） |
| `explained_variances.ev_{channel}` | scalar float | per-channel EV = 1 - Var(R-V)/Var(R) |

### update.npz（全局统计，非 per-frame）

| 字段 | 说明 |
|------|------|
| `approx_kl` | 平均 KL 散度（k3 估计） |
| `max_kl` | 最大 KL |
| `clip_frac` | 被 PPO clip 的帧比例 |
| `ratio_mean` / `ratio_max` | new/old log_prob 比率 |
| `policy_loss` | actor loss |
| `critic_loss.{channel}` | per-channel critic loss |
| `grad_norm_actor` | actor 梯度范数 |
| `critic_grad_norm.{channel}` | per-channel critic 梯度范数 |
| `param_grad_norms` | per-parameter 梯度范数（epoch 0, mb 0） |

### trajectories.npz（per-frame，与 gae/combine 对齐）

| 字段 | 说明 |
|------|------|
| `ep_lengths` | (n_trajs,) 每条 trajectory 帧数 |
| `reward.{channel}` | (N,) per-channel 逐帧 reward |
| `actor_weight.{channel}` | (N,) per-channel 逐帧 actor_weight |
| `frame_id` | (N,) `ep{pos:04d}:{agent}:{t}` |

## 界面布局

```
┌──────────────────────────────────────────────────────────────────────┐
│  Trajectory [42 ▼]  ep0000:robot_a  t_start=0  len=200  frame 50/200 │
│  [◄━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━►]  [▶ play]  speed: [1x ▼] │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  趋势图（整条 trajectory，当前帧用竖线标记）                      │ │
│  │                                                                    │ │
│  │  [channel selector: ☑r_potential ☑r_fall ☐r_left_foot ☐r_right_foot]│ │
│  │                                                                    │ │
│  │  Reward     ┌─────╮     ┌──╮                                      │ │
│  │             │     │     │  │                                      │ │
│  │  Value      ───────╱────────── (V(s), critic 估计)                │ │
│  │             │     │     │  │                                      │ │
│  │  Advantage  ────╲___╱── (A(s), GAE 输出)                         │ │
│  │             │     │     │  │                                      │ │
│  │  Return     ───────╱────────── (R(s) = A + V, critic 回归目标)    │ │
│  │             │     │     │  │                                      │ │
│  │             0     50    100  150  200  (frame)                    │ │
│  │                    ↑ 当前帧                                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  Current Frame: 50                                                     │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Per-Channel Data (frame 50)                                      │ │
│  │                                                                    │ │
│  │ channel        reward      value       advantage    return       │ │
│  │ ──────────── ────────── ────────── ────────── ────────────────── │ │
│  │ r_potential   +0.000001   +0.0183    -0.0130     +0.0053  ✓active │ │
│  │ r_fall        +0.000000   -0.0783    -0.0040     -0.0823  ✓active │ │
│  │ r_left_foot   +0.050000   +0.5412    -0.0410     +0.5002  ✓active │ │
│  │ r_right_foot  +0.050000   +0.5238    -0.0454     +0.4784  ✓active │ │
│  │                                                                    │ │
│  │ inactive channels (if any): shown greyed out with "inactive" tag │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Combine Chain (frame 50)                                          │ │
│  │                                                                    │ │
│  │ channel        aw      aw_normed  conf    EV       contribution   │ │
│  │ ──────────── ──────── ────────── ──────── ──────── ─────────────── │ │
│  │ r_potential   1.000   1.000      0.971    0.943    +1.000×0.971   │ │
│  │ r_fall        0.000   0.000      0.970    0.941      0.000       │ │
│  │ r_left_foot   0.000   0.000      0.910    0.827      0.000       │ │
│  │ r_right_foot  0.000   0.000      0.814    0.663      0.000       │ │
│  │ ─────────────────────────────────────────────────────────────── │ │
│  │ combined_adv = Σ contribution × norm_adv = -0.0126              │ │
│  │ aw_l1_sum = 1.000                                                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Bootstrap (this trajectory)                                      │ │
│  │                                                                    │ │
│  │ status: truncated → V(s_next) used for GAE bootstrap             │ │
│  │   r_potential: +0.0071  r_fall: -0.1032  r_left_foot: +0.5071     │ │
│  │   r_right_foot: +0.5050                                            │ │
│  │ (terminated → "no bootstrap (terminated)")                       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 布局说明

**顶部**：Trajectory 选择器 + 进度条 + 播放控制
- Trajectory 下拉框显示 `traj_idx` + `frame_id` 首帧（`ep0000:robot_a`）
- 进度条范围 0 ~ len-1
- 显示 `t_start`、`length`、当前 `local_frame`

**趋势图区**：整条 trajectory 的逐帧曲线
- Y 轴：reward / value / advantage / return（可切换显示哪些线）
- X 轴：trajectory-local frame（0 ~ len-1）
- 当前帧用竖线标记
- 多 channel 可叠加或分面（默认叠加，颜色区分）
- 点击曲线可跳转到对应帧

**当前帧数据区**：三个表格

1. **Per-Channel Data**：reward / value / advantage / return + active 状态
   - inactive channel 灰显，标注 "inactive"
   - 4 个数值列对齐

2. **Combine Chain**：展示 combined_adv 的计算链路
   - `aw` → `aw_normed`（÷ aw_l1_sum）→ × `conf` → × `norm_adv` → contribution
   - 底部显示 `combined_adv` 最终值
   - 让用户看到每个 channel 如何贡献到 actor 信号

3. **Bootstrap**：当前 trajectory 的 bootstrap 信息
   - 如果 truncated（非 terminated），显示 per-channel bootstrap V(s_next)
   - 如果 terminated，显示 "no bootstrap (terminated)"

## 交互

| 操作 | 行为 |
|------|------|
| 拖动进度条 | 趋势图竖线、三个表格同步更新 |
| 点击 ▶ | 自动播放 |
| 选择 trajectory | 加载新 trajectory 数据，进度条归零 |
| channel 复选框 | 切换趋势图显示哪些 channel |
| 趋势图线开关 | 切换 reward/value/advantage/return 哪些线 |
| 点击趋势图 | 跳转到对应帧 |

## 关键设计决策

### 1. 趋势图是核心

Scene 1 是数值表格（单帧），Scene 2 的核心价值是**趋势**——让用户看到 value 如何沿 trajectory 演化，advantage 在哪里翻转，return 是否被 critic 追上。

趋势图用 SVG 折线图（无第三方库），支持：
- 多 channel 叠加（颜色区分）
- 4 种线（reward / value / advantage / return）可独立开关
- 当前帧竖线 + tooltip

### 2. Combine Chain 透明化

combined_adv 的计算涉及 5 步变换，用户需要看到每一步：
```
aw (实验设定)
  → aw_normed (L1 归一化)
  → × conf (EV 置信度)
  → × norm_adv (z-score 归一化 advantage)
  → Σ → combined_adv
```

表格逐行展示每个 channel 的链路，底部显示最终 combined_adv。

### 3. Bootstrap 可见性

GAE 的 bootstrap 值是容易出错的点：
- terminated → last_value = 0（无 bootstrap）
- truncated → last_value = V_c(s_next)（critic 估计）

表格明确显示当前 trajectory 是哪种情况，以及 bootstrap 值是多少。

### 4. inactive channel 处理

某些 channel 在某些 trajectory 上可能 inactive（`key_seg_active = False`），此时 value/adv/ret 全为 0。表格灰显并标注 "inactive"，不隐藏——让用户知道这个 channel 存在但不参与。

## 技术方案

| 层 | 选择 | 理由 |
|----|------|------|
| 后端 | Python HTTP server（复用 Scene 1） | 同一 server，新增 API |
| 前端 | 单页 HTML + vanilla JS + SVG | 无框架，SVG 画折线图 |
| 图表 | SVG path（手写） | 4 条线 × N channel，数据量小 |
| 启动 | `debug.py viewer <dump_dir>` | 与 Scene 1 同一入口 |

### 后端 API（新增）

```
GET /api/trajectory/<traj_idx>/overview
  → {
      traj_idx, ep_lengths[traj_idx], frame_id_first,
      channels: ["r_potential", "r_fall", ...],
      // 整条 trajectory 的逐帧数据（用于趋势图）
      frames: [
        {
          local_frame: 0,
          reward: { r_potential: 0.0001, ... },
          value:   { r_potential: 0.0183, ... },
          adv:     { r_potential: -0.013, ... },
          ret:     { r_potential: 0.0053, ... },
          combined_adv: -0.0126,
          active:  { r_potential: true, ... },
        },
        ...
      ]
    }

GET /api/trajectory/<traj_idx>/frame/<local_frame>
  → {
      local_frame,
      reward: { channel: value, ... },
      value: { channel: value, ... },
      adv: { channel: value, ... },
      ret: { channel: value, ... },
      active: { channel: bool, ... },
      combined_adv: float,
      // combine chain
      aw: { channel: float, ... },
      aw_normed: { channel: float, ... },
      conf: { channel: float, ... },
      ev: { channel: float, ... },
      aw_l1_sum: float,
      // bootstrap
      is_terminated: { channel: bool, ... },
      bootstrap_value: { channel: float or null, ... },
    }
```

### 前端交互流

```
用户选择 trajectory 42
  → GET /api/trajectory/42/overview → 画趋势图，进度条归零
  → GET /api/trajectory/42/frame/0 → 填充三个表格

用户拖动进度条到 frame 50
  → GET /api/trajectory/42/frame/50 → 更新表格，竖线移动

用户点击趋势图的 value 线在 frame 73
  → 跳转到 frame 73，更新所有数据
```

## 数据索引

trajectory `j` 的 flat 帧范围：
```python
seg_offsets = np.concatenate([[0], np.cumsum(ep_lengths)])
traj_start = seg_offsets[j]
traj_end = seg_offsets[j + 1]
```

per-channel per-frame 数据切片：
```python
def get_traj_data(gae, combine, traj_idx, ep_lengths):
    """获取 trajectory 的所有 per-frame 数据。"""
    seg_offsets = np.concatenate([[0], np.cumsum(ep_lengths)])
    s, e = seg_offsets[traj_idx], seg_offsets[traj_idx + 1]
    channels = list(gae["values_all"].item().keys())
    frames = []
    for t in range(e - s):
        frame = {"local_frame": t}
        for ch in channels:
            frame["reward"] = {ch: float(traj_reward[ch][s + t]) for ch in channels}
            frame["value"] = {ch: float(gae["values_all"].item()[ch][s + t])}
            frame["adv"] = {ch: float(gae["advs_all"].item()[ch][s + t])}
            frame["ret"] = {ch: float(gae["rets_all"].item()[ch][s + t])}
            frame["active"] = {ch: bool(gae["key_frame_mask"].item()[ch][s + t])}
        frame["combined_adv"] = float(combine["combined_adv"][s + t])
        frames.append(frame)
    return frames
```

bootstrap 值：
```python
def get_bootstrap(gae, traj_idx):
    """获取 trajectory 的 per-channel bootstrap 值。"""
    channels = list(gae["bootstrap_values"].item().keys())
    terminated = {ch: bool(gae["key_seg_terminated"].item()[ch][traj_idx])
                  for ch in channels}
    active = {ch: bool(gae["key_seg_active"].item()[ch][traj_idx])
              for ch in channels}
    bootstrap = {}
    for ch in channels:
        if not active[ch]:
            bootstrap[ch] = None  # channel inactive
        elif terminated[ch]:
            bootstrap[ch] = None  # terminated, no bootstrap
        else:
            # truncated: V(s_next) from bootstrap_values
            # Note: bootstrap_values only has entries for trajectories
            # that needed bootstrap. Need to map traj_idx → bootstrap index.
            bootstrap[ch] = float(gae["bootstrap_values"].item()[ch][?])
    return bootstrap
```

注意：`bootstrap_values` 只包含需要 bootstrap 的 trajectory，需要通过 `bootstrap_indices` 映射回 `traj_idx`。当前 dump 没有保存 `bootstrap_indices`，需要补充。

## 前置工作

### 1. dump capture 保存 bootstrap_indices

当前 `gae.npz` 保存了 `bootstrap_values` 但没有保存 `bootstrap_indices`（哪些 trajectory 需要 bootstrap）。需要在 `dump_capture.py` 的 GAE payload 中增加：

```python
# trainer.py GAE dump hook 增加:
"bootstrap_indices": np.array(bootstrap_indices, dtype=np.int64),
```

这样 viewer 可以判断 trajectory `j` 是否需要 bootstrap，以及它在 `bootstrap_values` 中的位置。

### 2. traj_map.json 复用

Scene 1 的 `traj_map.json` 已经建立了 trajectory → episode 映射。Scene 2 可以复用，在 trajectory 选择器中显示关联的 episode 信息（`ep0000:robot_a`）。

### 3. 趋势图性能

单条 trajectory 最多 200 帧 × 4 channel × 4 线 = 3200 个数据点，SVG 完全可以处理。不需要 WebGL 或 Canvas。

## 不做的事

- 不做 per-epoch 演化（KL 随 epoch 变化、ratio 偏离、early stop 触发）——属于训练过程动态，是后续场景的职责
- 不做 per-minibatch 动态（per-frame ratio / clip mask / gradient）——同上
- 不做 policy 参数变化可视化（θ_old → θ_new）——同上
- 不做跨 trajectory 对比
- 不做梯度可视化（数据量太大，且 update.npz 只保存了 epoch 0 mb 0 的梯度）
- 不做 critic 网络结构可视化
- 不做多 update 对比
- 不做 reward shaping / curriculum 可视化
