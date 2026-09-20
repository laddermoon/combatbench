# PPO 训练数据流（逻辑视图）

## 概述

从环境交互到梯度更新，PPO 训练的数据流分为五个阶段。每个阶段产出特定的数据结构，供下游阶段消费或供 debug viewer 展示。

## 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Episode（环境交互）                                               │
│                                                                      │
│  per-agent, per-frame:                                               │
│    observation, action, terminated                                    │
│  per-episode:                                                        │
│    observer_outputs (reward 信号、metrics、自定义数据)                │
│                                                                      │
│  视角：两个 agent 各自看到什么、做了什么、环境判定是否结束            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ experiment.build_trajectories()
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Trajectory（奖励通道化）                                           │
│                                                                      │
│  per-frame:                                                          │
│    obs, action, log_prob (θ_old), explore_factor, floor_weight       │
│  per-channel:                                                        │
│    reward, actor_weight, is_active, is_terminated                    │
│  per-trajectory:                                                     │
│    last_obs (for bootstrap), importance                              │
│                                                                      │
│  视角：每个 agent 的轨迹被拆解为多个奖励通道的训练数据                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ critic forward + GAE + combine
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. 训练目标（epoch-invariant，循环前算一次）                          │
│                                                                      │
│  per-channel, per-frame:                                             │
│    value      = critic(s_t)           ← critic 对状态的估计         │
│    advantage  = GAE(reward, value)    ← 每帧的相对优势               │
│    return     = advantage + value      ← critic 的回归目标           │
│  per-frame:                                                          │
│    combined_adv = Σ_c aw_normed × conf_c × norm_adv_c                │
│                   ← actor 实际使用的单一信号                         │
│  per-channel (标量):                                                  │
│    confidence = sqrt(clip(EV))         ← critic 可信度               │
│    EV         = 1 - Var(R-V)/Var(R)    ← explained variance          │
│                                                                      │
│  视角：critic 估了什么，GAE 算了什么，combine 后 actor 用什么         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ 进入训练循环（见下方 3.5 gradsig 截面）
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Epoch（per-epoch 快照）                                            │
│                                                                      │
│  每个 epoch 结束后，actor/critic 参数已更新：                         │
│                                                                      │
│  per-frame (该 epoch 结束时的状态):                                   │
│    ratio      = exp(new_log_prob - old_log_prob)  ← 偏离 θ_old 程度  │
│    clip_mask  = |ratio - 1| > clip_eps            ← 是否被 PPO 截断  │
│    new_value  = critic_current(s_t)               ← critic 当前估计  │
│                                                                      │
│  per-epoch (标量):                                                   │
│    kl_mean, kl_max, kl_std, actor_active                             │
│                                                                      │
│  视角：每个 epoch 结束时，policy 对每帧的评估怎么变了                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ 每个 epoch 内部拆成 minibatches
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Minibatch（per-step 训练动态）                                     │
│                                                                      │
│  每个 mb step（actor + critic 更新一次）:                             │
│                                                                      │
│  actor:                                                              │
│    ratio, clip_frac, policy_loss, actor_grad, KL                     │
│    window_mean_kl (滑动窗口) → 判断是否 early stop                              │
│  critic (per-channel):                                              │
│    critic_loss, critic_grad                                          │
│  状态:                                                               │
│    epoch_idx, mb_idx, actor_active                                   │
│                                                                      │
│  early stop: window_mean_kl > target_kl → actor 冻结，critic 继续   │
│                                                                      │
│  视角：训练过程的每一步发生了什么，KL 怎么走，何时 early stop         │
└─────────────────────────────────────────────────────────────────────┘
```

## 数据与 Viewer 场景的对应

| 阶段 | 数据 | Viewer 场景 |
|------|------|-------------|
| 1. Episode | obs, action, terminated, observer | Scene 1：Episode → Trajectory |
| 2. Trajectory | reward, actor_weight, per-channel | Scene 1：Episode → Trajectory |
| 3. 训练目标 | value, advantage, return, combined_adv | Scene 2：Trajectory → 训练目标 |
| 3.5 gradsig | grad_sig_* 标量 + cos×norm 二维分布 | run 首页图表 + Update Detail |
| 4. Epoch | ratio, clip_mask, new_value | Scene 3：Trajectory × Epoch |
| 5. Minibatch | KL, loss, grad, clip_frac | Scene 4：Update 时间线 |

## 阶段 3.5：ADV 梯度信号诊断（θ_old 截面）

在 combined_adv 确定之后、epoch 循环开始之前——此时 actor 恰为
θ_old——框架对每帧定义**真实训练损失**的改善方向梯度：

```
scalar_i = w_i·A_i·log_prob_i − coef·relu(floor − U_i)²·floor_weight_i
g_i      = ∇_θ scalar_i          (θ = θ_old，含 surrogate + floor 两项)
```

先对**整个 buffer** 分块反向累积得到聚合梯度 `G = mean_i g_i`
（等价于完整 mean-loss 的一次梯度），再从 buffer 随机抽样
`grad_sig_sample_size` 帧（专用 RNG，`seed = f(run_seed, update)`，
不消耗训练随机流）逐帧求梯度并投影到聚合方向 `Ĝ = G/‖G‖`：

```
p_i = g_i·Ĝ          (带符号投影：>0 = 本 update 方向使该帧获益，<0 = 被牺牲)
c_i = cos(g_i, G)    (纯方向)
```

产出：

- **标量**（进 `__RAW_STATS__` → `stats.*`）：`grad_sig_gnorm`
  （‖G‖，净拉力强度）、`grad_sig_coherence`（‖G‖/mean‖g_i‖，总
  拉力聚合后的存活率）、`grad_sig_proj_mean`（mean p_i，‖G‖ 的
  采样估计——恒等式 mean(p)=‖G‖ 在完整 buffer 上严格成立，偏差
  大说明采样不具代表性）、`grad_sig_proj_std`（逐帧获益离散
  度）、`grad_sig_frac_neg`（P(p<0)，被本 update 方向牺牲的帧
  占比）、`grad_sig_dir_cos`（cos(G_u, G_{u−1})，合力方向跨
  update 持续性；G 由 loop 在内存中传递不落盘）、
  `grad_sig_frames`（有效帧数）、`grad_sig_norm_mean`（mean‖g_i‖，
  coherence 的分母）、`grad_sig_time_s`。
- **二维分布 + 逐帧数组**（`run_dir/gradsig/uNNNNN.npz`）：
  `hist[norm_bin, cos_bin]` 帧计数 + 分箱边界 + 覆盖计数 +
  范数分位数 + 逐帧 `grad_norm`/`cos`/`proj`/`valid` 数组（前端
  自行分箱投影直方图，无需冻结轴）。norm 轴为**等质量分位数
  分箱**（`grad_sig_norm_bins` 个 bin，派生 update 上每行恰装
  ~1/norm_bins 帧），边界冻结进 `gradsig/meta.json`（带
  `norm_axis: "per_frame_norm"` 标记，旧 pairwise 格式的 meta
  不会被复用）；超出首 update 范围的帧收进
  `hist_under`/`hist_over` 兜底行。
- **dump 细节**（仅 dump 时，`dumps/uNNNNN/gradsig.npz`）：采样帧
  的扁平 buffer 索引、逐帧梯度范数、w·A 与 floor 标量系数、
  逐帧 p_i/c_i。
  有 dump 请求时会强制运行诊断，即使该 update 被 interval 跳过。

解读边界：‖G‖ 是聚合后的**净**强度，同时受个体拉力与方向一致性
影响——配 coherence 拆开看；coherence 低不必然是有害冲突（方向
散开也稀释合力），是否真有帧被牺牲看 frac_neg；p_i 是相对"已选
定方向"的一阶获益，正交方向的可能性不在其中；G 是 θ_old 处的原
始意图，不等于 Adam/clip/minibatch 后的实际位移；零范数帧（无
方向）单独计数不进分布；‖G‖≈0 时方向无定义，proj/cos 统计不产
出。语义详见 metric_catalog.py 的 hint。

## 关键边界

- **阶段 1-2**：从环境到训练数据。实验负责（`build_trajectories`）。
- **阶段 3**：训练目标。epoch 循环之前算一次，不变。
- **阶段 3.5**：θ_old 截面诊断。只读不改，专用 RNG，训练无感知。
- **阶段 4-5**：训练过程。参数在变，数据在变。

Scene 2 展示阶段 3（训练输入），Scene 3-4 展示阶段 4-5（训练动态）。
