# PPO 训练调试指南（Debug Guide）

**目标**：让你在**任何情况下**都「心里有底，手里有牌」——知道训练内部正在发生什么，且知道下一步该看哪个数据。

**反面目标**：盲调参数。如果你正在改一个超参而说不出「我预期哪个指标会怎么变」，就停下来先读 §4。

**关联文档**：`GUIDE.md`（框架用法）、`DESIGN_unified_exploration_control.md`（探索语义）、`FIXPLAN.md`（历史缺陷修复记录）

---

## 0. 状态图例

本文档描述的是**调试能力完备后**的框架。当前实现进度用标记区分：

| 标记 | 含义 |
|---|---|
| ✅ | **已实现**，现在就能用 |
| 🚧 | **待建**，本文档同时作为设计规格 |
| ⚠️ | **设计缺口**，见 §11，实现前需先决策 |

> 写这份文档的过程本身是一次倒逼设计。§11 记录了写作时暴露出的框架缺陷——**那一节是本文档最重要的产出**。

---

## 1. 三层可观测性模型

不要一上来就 dump 全量数据。调试成本应该和问题难度匹配。

| 层 | 名称 | 成本 | 覆盖 | 何时用 |
|---|---|---|---|---|
| **L1** | **常态聚合观测** | 零（始终开启） | 「有没有病」 | **总是先看这里** |
| **L2** | **快照 + 离线重算** | 一次 update 的落盘 | 「病在哪个环节、哪一帧」 | L1 报警但看不出根因 |
| **L3** | **进程内深挖** | 侵入训练进程 | 优化器状态、逐 minibatch 梯度 | L2 无法复现的问题 |

### 1.1 为什么 L2 是「离线重算」而不是「在线 dump」

**训练管线里绝大部分环节是 `(episodes, checkpoint)` 的纯函数**：

| 环节 | 纯函数？ | 离线可重算 |
|---|---|---|
| 奖励 / 权重计算（`build_trajectories`） | `f(episodes)` | ✅ |
| GAE / returns | `f(trajectories, critic参数)` | ✅ |
| advantage 归一化 + confidence + L1 合成 | `f(同上)` | ✅ |
| PPO minibatch 更新 / 梯度 | `f(同上, RNG种子)` | ✅（需固定种子） |
| Adam 动量状态 | 有状态 | ❌ → L3 |

所以 L2 只需要在进程内做**一件事**：把 `episodes` + `θ_old` + 种子快照落盘。其余全部由离线工具**调用同一套框架代码**重算。

这样做的四个好处：

1. 进程内改动极小，不容易把正在跑的训练搞坏
2. **调试路径不会和生产路径漂移**——重算调用的就是真实的 `build_trajectories`
3. 分析可以反复迭代，改分析脚本不用重跑训练
4. 快照天然变成**回归测试 fixture**

---

## 2. 七个环节与责任边界

```
[1] 环境/observer  →  [2] 奖励与权重  →  [3] θ_old 一致性  →  [4] GAE
                                                                  ↓
                              [7] Eval  ←  [6] PPO 更新  ←  [5] advantage 合成
```

| # | 环节 | 代码位置 | 谁负责 | 主要覆盖层 |
|---|---|---|---|---|
| 1 | 环境 / observer 原始输出 | `rollout/`, observer plugins | 框架 + 环境 | L2 |
| 2 | 奖励与权重计算 | `experiment.build_trajectories` | **实验** | L2（需实验钩子） |
| 3 | θ_old 一致性 / 数据对齐 | `PPOBuffer` | 框架 | L1 不变量 |
| 4 | GAE / returns | `ppo_update` | 框架 | L1 聚合 + L2 逐帧 |
| 5 | advantage 合成 | `ppo_update` | 框架 | L1 聚合 + L2 逐帧 |
| 6 | PPO 更新 / 梯度 | `ppo_update` epoch×mb | 框架 | L1 + L3 |
| 7 | Eval | `experiment.on_eval` | **实验** | L2 |

**环节 1 和 3 最容易被忽略，却藏着最阴的 bug。** 环节 1 是奖励的**源头**——问「奖励算得对不对」，第一步是看输入对不对，而不是看输出的 reward 数值（全零的 reward 看起来"很正常"）。

---

## 3. L1：常态聚合观测（先看这里）

### 3.1 主入口：`analyze_training.py` ✅

```bash
# 全量报告：趋势表 + 健康诊断
PYTHONPATH=. python3 baseline/framework/analyze_training.py <run_dir>/train.log

# 只看诊断结论
... --diagnostics-only

# 实时跟随（类似 tail -f，但带诊断）
... --watch --interval 2

# 全历史 sparkline，可按指标名过滤
... --history ev
... --history survived

# 列出所有可用指标名
... --list-metrics
```

它已内置 **9 条诊断规则**（含严重级别），窗口默认 10 个 update：

| 规则 | 级别 | 触发含义 |
|---|---|---|
| Exploration Collapse | CRITICAL | `std_min` 贴在下界 → 策略坍缩 |
| Episode Death Spiral | CRITICAL | 平均 episode 长度过短 |
| PPO Early Stop | WARNING | 每轮都 KL 超标 → 步长过大 |
| Critic Blind | WARNING | 某通道 `EV <= 0` → critic 没学到东西 |
| Vanishing Advantage | WARNING | `adv_std ≈ 0` → 该通道不产生梯度 |
| High Clip Fraction | WARNING | 大比例样本被 clip |
| Large / Vanishing Actor Gradient | WARNING | 梯度过大或过小 |
| Policy Ratio Divergence | WARNING | `ratio_max` 异常 |
| Rollout Bottleneck | INFO | rollout 占墙钟比例过高 |

### 3.2 内联诊断行 ✅

`ppo_update` 每轮直接打印到日志，`grep` 即可：

| 前缀 | 含义 | 关注点 |
|---|---|---|
| `[warn] first-minibatch \|ratio-1\|` | **θ_old 重算不一致** | 出现即为严重问题，见 §5.3 |
| `[warn] all active channels have confidence=0` | 所有 critic EV≤0，actor 无梯度 | warmup 期正常，持续则异常 |
| `[warn] channel 'X' has N active frame(s) but zero-variance advantages` | 该通道本轮不贡献梯度 | |
| `[warn] epoch=N mean_kl too small` | 策略卡住或 LR 过低 | |
| `[warn] epoch=N KL monotonically increasing` | 过冲风险 | |
| `[warn] epoch=N KL jump at step i` | 单步 KL 翻倍 | |
| `[early_stop] epoch=N mb=M` | KL 早停触发（critic 继续） | |
| `[kl_stats] epoch=N mean/std/cv` | 每 epoch 的 KL 分布 | `cv` 大 = minibatch 间不均匀 |
| `[GradDiag] log_std grad` | floor loss 与 policy loss 对 `log_std` 的梯度对比 | `ratio` ≪1 说明 floor 无力 |
| `[skip] build_trajectories returned no usable frames` | 本轮空 buffer | |

### 3.3 `__RAW_STATS__` JSON ✅

每轮一行机器可读 JSON，`analyze_training.py` 的数据源。三个区块：

- `episode_stats`：长度分布、终止原因计数
- `buffer_stats.per_channel`：每通道 `reward_*`、`actor_weight_*`（**归一化前**）、`active_ratio`、轨迹长度
- `stats`：PPO 核心量 + 每通道 `ev/confidence/adv_*/ret_*/vloss/grad_norm` + 策略贡献量（`uncertainty`、`std_*`、`eff_std_mean`、`mean_abs`）

### 3.4 L1 待补的四个量 🚧

以下四项**现在完全看不到**，且都是聚合量（不需要 dump 基础设施）。按价值排序：

| 待补指标 | 定义 | 为什么关键 |
|---|---|---|
| `aw_normed_*` | L1 归一化**后**的 actor_weight：`aw / Σ_c\|aw_c\|` | 现在只报归一化前的值。**真正乘到 advantage 上的是归一化后的**——你现在看不到真实的通道混合比例 |
| `influence_share_*` | `Σ_frames \|aw_normed × conf × normed_adv\|`，按通道归一 | 「**到底哪个奖励在驱动策略**」的唯一直接答案 |
| `dead_frame_ratio` | `Σ_c\|aw_c\| == 0` 的帧占比 | 这些帧对 actor **完全无梯度**，白采样 |
| `grad_norm_per_action_dim` | actor 输出层按动作维度的梯度范数 `(action_dim,)` | 某个关节到底有没有拿到梯度？若为 0，**再怎么调 reward 都没用** |

> **实践建议**：遇到「策略学不出某个行为」时，先加这四个量。经验上它们直接给出答案的概率远高于翻逐帧数据。

---

## 4. 症状 → 排查路径（核心速查表）

这是「手里有牌」的部分。**从症状出发**，不要从超参出发。

### 4.1 策略完全不学（`policy_loss ≈ 0`，`approx_kl ≈ 0`）

| 步骤 | 看什么 | 若异常说明 |
|---|---|---|
| 1 | `dead_frame_ratio` 🚧 | 接近 1 → 所有帧 `actor_weight` 全零 |
| 2 | `[warn] all active channels have confidence=0` | 所有 critic EV≤0，梯度被 confidence 归零 |
| 3 | 各通道 `adv_std` | 全部 ≈0 → advantage 无方差（reward 恒定？） |
| 4 | `grad_norm_actor` | ≈0 但上面都正常 → 检查 optimizer / LR / `requires_grad` |
| 5 | L2 重算，看 `combined_adv` 逐帧分布 | 定位到具体通道和帧 |

### 4.2 学了，但学不出想要的行为（最常见、最难）

**这不是「不学」，是「学错方向」或「信号太弱」。**

| 步骤 | 看什么 | 判据 |
|---|---|---|
| 1 | 目标通道的 `influence_share` 🚧 | < 5% → 该奖励基本没参与决策 |
| 2 | 目标通道 `aw_normed` 的**均值符号** 🚧 | 负值 → 该通道在**抑制**当前行为 |
| 3 | 目标通道 `adv_std` vs `reward_std` | `adv_std` ≪ 期望 → critic 已把该信号解释掉了 |
| 4 | `grad_norm_per_action_dim` 🚧 相关关节维度 | ≈0 → 物理上不可能学出来 |
| 5 | L2 + 实验钩子：奖励是否在**该出现的帧**出现 | 相位掩码 / 状态机是否按预期触发 |
| 6 | L2：reward 的**分布**而非均值 | 均值 0.01 可能是「1% 帧为 1.0」也可能是「全帧 0.01」，含义完全不同 |

### 4.3 reward 数值看起来正常，但训练无效

优先怀疑**环节 1**（源头）而不是环节 2：

| 步骤 | 看什么 |
|---|---|
| 1 | 不变量 `obs[45] == observer.h_torso` 🚧（§6） |
| 2 | L2：observer 原始字段逐帧值。**是不是全零 / 全常数？** |
| 3 | L2：observer 数组长度是否 == `T`（长度不符现在会 raise，但截断会掩盖） |
| 4 | 实验钩子：中间掩码（如 `balance_mask`）是否全 False |

> **真实案例**：`h_torso` 缺失时曾静默回退全零 → 相位掩码全 False → foot 权重全零 → 训练照跑、不报错、永远学不出 stepping。已在 commit `60017b3` 改为 raise。这类 bug 从 reward 数值上完全看不出来。

### 4.4 critic 学不动（`EV <= 0`）

| 步骤 | 看什么 | 判据 |
|---|---|---|
| 1 | `is_terminated` 设置 | **最常见的 bug**。设错会让 bootstrap 完全错 |
| 2 | 该通道 `gamma` 与 reward 尺度 | reward 恒正且 γ=0.99 → return ≈ 常数，EV 天然低 |
| 3 | `ret_std` | ≈0 → 目标无方差，EV 无意义（不是 bug） |
| 4 | L2：逐帧 `V(s)` vs `return` 🚧 | 在哪类状态上错 |
| 5 | `vloss` 趋势 | 不降 → LR / 容量问题 |

### 4.5 σ 坍缩 或 σ 爆炸

| 症状 | 先看 | 再看 |
|---|---|---|
| 坍缩 | `std_min`、`uncertainty` | `[GradDiag]` 的 `ratio` — floor loss 是否被 policy 梯度压制 |
| 爆炸 | `uncertainty_coef`、`uncertainty_floor` | `eff_std_mean` vs `std_mean` — 是策略本身大还是 `explore_factor` 放大 |

> **已知陷阱**：`uncertainty_floor` 用的是二次 hinge `coef·relu(floor-U)²`，**接近目标时梯度趋零**，所以 U 通常停在 floor 之下。若用 floor 值本身作为阶段切换阈值，会永远不触发。必须把「floor 目标」和「触发阈值」解耦（见 commit `57df509`）。

### 4.6 KL 爆炸 / 频繁早停

`[kl_stats]` 的 `cv` → `[warn] KL jump` → `ratio_max` → `clip_frac` → 降 LR 或降 `update_epochs`。

### 4.7 eval 指标与训练指标背离

| 可能原因 | 如何确认 |
|---|---|
| eval 是确定性的（`stochastic=False`），训练是随机的 | 对比 `std_mean`；σ 大时两者天然不同 |
| `explore_factor` 在 eval 被绕过 | 确认 `_wrap_policy` 的 `stochastic` 分支 |
| eval 指标本身定义有问题 | L2：把 eval episode 落盘，**手工重算指标** |

> **真实案例**：`_count_steps` 数的是支撑脚切换次数，**不要求最小支撑/摆动时长**，所以接触抖动会被计成"步数"。指标涨了但物理上没迈步。指标定义必须能被独立重算验证。

### 4.8 resume 后行为突变

`--reset-update` 是否重置了不该重置的状态 → `experiment.load_state` 是否恢复完整 → 观测归一化常量是否变过（如 `feet_forces` 从牛顿改为体重倍数会使旧 checkpoint 失效）。

---

## 5. L2：快照 + 离线重算 🚧

### 5.1 触发方式：哨兵文件

在 `run_dir/` 放一个请求文件，训练循环在**每轮 update 顶部**检查一次：

```bash
cat > baseline/runs/<run>/debug_request.json <<'EOF'
{
  "hypothesis": "foot 通道 influence_share 过低，怀疑 startup_bias 从未触发",
  "max_episodes": 8,
  "include_observer_outputs": true,
  "include_grads": false
}
EOF
```

**语义约定**：

- **一次性消费**：读到后立刻 `os.rename` 到输出目录成为 `request.json`（原子操作）。防止手滑连续 dump 塞满磁盘，同时自带「当初请求了什么」的记录。
- `hypothesis` 字段**必填**。它会跟着输出目录留下来——这是防止重新陷入盲调的机制性约束。
- 抓取粒度 = 一个 update，与哨兵文件的检查粒度天然匹配。

**为什么不用信号（SIGUSR1）**：无法携带参数；Python 信号处理有重入问题；rollout 是多进程，投递目标需额外处理；且无法留下请求记录。唯一优势「即时」对本场景无价值。

### 5.2 落盘布局

```
<run_dir>/debug/u00123/
├── request.json              # 原始请求（含 hypothesis）
├── manifest.json             # 抓取时的 update / 时间 / 配置 / 代码 commit
├── episodes/                 # EpisodeCollection.save() ✅ 组件已存在
│   ├── collection.json
│   ├── blueprint.yaml
│   └── episodes/episode_000NN.npz + .json
├── theta_old/                # 本轮 rollout 用的策略（= θ_old）
├── rng_state.pt              # torch / numpy RNG 状态，供精确 replay
└── (离线重算产物写到 replay/ 子目录)
```

> `Episode.save/load`（format v3）与 `EpisodeCollection.save/load`（含 blueprint hash 校验）✅ **已经实现**，只是训练循环还没调用。L2 的进程内部分主要是接线，不是新建。

### 5.3 离线重算工具

```bash
PYTHONPATH=. python3 baseline/framework/ppo/replay_update.py \
    baseline/runs/<run>/debug/u00123 --stages all
```

它**调用同一套框架代码**依次重跑并落盘每层中间量：

| stage | 重算内容 | 产出的逐帧数组 |
|---|---|---|
| `trajectories` | `experiment.build_trajectories(episodes)` | 各通道 `reward`、`actor_weight`（归一化前）、`is_terminated` |
| `experiment` | `experiment.debug_arrays(...)`（§8） | 实验特有中间量 |
| `buffer` | `PPOBuffer` 装填 + θ_old 整批前向 | `old_log_prob`、`first-minibatch ratio` |
| `gae` | 各通道 GAE | `V(s)`、`δ`、`advantage`、`return`、bootstrap 值 |
| `combine` | advantage 合成 | `aw_normed`、`conf`、`normed_adv`、`combined_adv`、`norm_mask`、逐通道 `contribution` |
| `update` | 单轮 `ppo_update`（固定种子） | 逐 minibatch `ratio`/`clip_frac`/loss/梯度范数 |

**核心不变量**：重算得到的 `UpdateStats` 应与训练日志中该 update 的 `__RAW_STATS__` **逐字段一致**。不一致 = 快照不完整或存在非确定性，本身就是一个必须先解决的 bug。这条自校验让 L2 可信。

### 5.4 数据量必须有预算

一个典型 update = 512 episodes × 200 帧 × 2 agents = **204,800 帧**。仅 obs 就 `96×4B×204800 ≈ 78 MB`。

| 项 | 默认 | 理由 |
|---|---|---|
| `max_episodes` | 8 | 逐帧诊断不需要全量；8 条足以看清模式 |
| `include_grads` | false | 全参数逐 minibatch 梯度是天文数字 |
| 梯度默认粒度 | 逐 minibatch **范数** + 单个 minibatch 全量 | |
| 输出目录总量上限 | 需显式配置 | `baseline/runs/` 已经很大且 gitignored |

---

## 6. 不变量清单（自动报警）🚧

**这是投入产出比最高的一类调试设施**：它把「什么应该成立」编码进代码，无需人盯着看。已有的 `first-minibatch |ratio-1|` 检查 ✅ 就是这个模式的成功案例。

| # | 不变量 | 违反说明 | 建议级别 |
|---|---|---|---|
| 1 | `obs[45] == observer.h_torso`（逐帧，容差内） | observer 错位 / agent 串号 / 时间偏移 | raise |
| 2 | observer 字段长度 `== T` | ✅ 已由 `coerce_per_step` 保证 | raise |
| 3 | 所有 reward / weight 有限（无 NaN/Inf） | 数值爆炸 | raise |
| 4 | `len(obs) == len(actions) == len(reward) == len(actor_weight)` | 实验侧数组错位 | raise |
| 5 | `last_obs` 是第 `T+1` 帧而非第 `T` 帧 | bootstrap 用错状态 | raise |
| 6 | 首 minibatch `\|ratio-1\| < 1e-4` | ✅ 已有 | warn |
| 7 | `explore_factor` 回放与 rollout 记录一致 | importance ratio 错误 | raise |
| 8 | `dead_frame_ratio < 1.0` | actor 完全无梯度 | warn |
| 9 | 重算 `UpdateStats` == 日志 `__RAW_STATS__` | 非确定性 / 快照不全 | raise（L2 内） |

> **原则**（见 `CLAUDE.md` "Fail Loud"）：**缺数据必须 raise，不能静默回退。** 静默零回退会让训练继续跑但学错东西，比崩溃危险得多。

---

## 7. 指标字典

只列**容易误读**的。完整列表用 `analyze_training.py --list-metrics` ✅。

| 指标 | 含义 | 健康范围 | 误读风险 |
|---|---|---|---|
| `uncertainty` | 策略**自身**不确定度 ∈[0,1]，**不含** explore 缩放 | 视任务 | 和 `std_mean` 不是一回事 |
| `std_mean` | 策略原始 σ 均值 | — | |
| `eff_std_mean` | 有效 σ（**含** explore 缩放） | — | `eff/std` 比值 = 实际 explore 放大倍数。`explore_factor=0` 时两者应严格相等 |
| `actor_weight_mean` | **归一化前** | — | **不是**真正乘到 advantage 上的值 |
| `confidence` | 由 EV 推出的通道可信度 | >0.5 | 为 0 时该通道梯度被完全屏蔽 |
| `adv_std` | 归一化**前** advantage 标准差 | >0 | ≈0 表示该通道本轮无贡献 |
| `ev` | explained variance | >0.5 | `ret_std≈0` 时 EV 天然低，**不是 bug** |
| `clip_frac` | 被 clip 样本比例 | 0.1–0.3 | 过低可能是步长太小 |
| `epochs_done` | 恒等于 `update_epochs` | — | **不携带早停信息**；要看 `actor_epochs_done` |
| `ratio_max` | 单帧最大 importance ratio | <5 | 长尾很正常，看 `ratio_mean` 更稳 |

---

## 8. 实验侧调试钩子 🚧

### 8.1 为什么必须有

**环节 2 完全是实验特有的。** 框架不知道 `r_left_foot` 是什么、不知道 `balance_mask`、不知道状态机分支。**只有实验能解释「为什么第 137 帧的 actor_weight 是 -1.0」。**

### 8.2 接口

```python
def debug_arrays(self, episodes, trajectories) -> Dict[str, np.ndarray]:
    """可选。返回逐帧命名数组，供 L2 dump 保存。

    默认返回 {}。数组第 0 维必须与 trajectories 的帧拼接顺序对齐
    （见 §11.1 关于 provenance 的未决问题）。
    """
    return {}
```

**设计要点**：

1. **加法式、默认 no-op**，不破坏现有实验。
2. **不做控制反转**——不把 sink 穿进 `build_trajectories`，热路径零改动、零风险。
3. **实现时必须调用同一批私有 helper**（如 `_compute_phase_mask`、`_compute_foot_weights_masked`），而**不是重写逻辑** → 逻辑单一来源，不会和生产路径漂移。
4. 这里的**静默默认是合法的**：「本实验没有额外调试数据」是一个**完整的答案**，与「observer 数据缺失」（信息缺失，必须 raise）性质不同。不要过度套用 fail-loud。

### 8.3 建议导出的内容（以相位型实验为例）

| 数组 | 用途 |
|---|---|
| `balance_mask` / `standup_mask` | 相位判定是否符合预期 |
| `h_torso`, `phi`, `potential` | 奖励源头原始值 |
| `state_machine_branch`（逐帧整数码） | **哪个分支产生了权重**——否则无法确认 `startup_bias` 是否触发过 |
| `foot_height_l/r`, `contact_l/r` | 接触信号是否二值、抬脚幅度分布 |

---

## 9. L3：进程内深挖 🚧

只在 L2 无法复现时使用。

| 抓什么 | 为什么 L2 抓不到 |
|---|---|
| Adam `m` / `v` 范数 | 有状态，无法从 `(episodes, θ)` 重建 |
| 逐 minibatch 全量梯度轨迹 | 数据量过大，只在明确假设下抓 |
| policy loss vs floor loss 的梯度分解（扩展到 mean） | ✅ `log_std` 部分已有 `[GradDiag]` |

**必须避开的陷阱：RNG 漂移。** 如果调试路径消耗了 RNG（多一次 `randn`、多一次 shuffle），**这轮之后的训练全部偏离**原轨迹——你会得到一个「被观测行为改变了的」训练。解法：调试用独立 `Generator`，或在 dump 前后 `get_rng_state()` / `set_rng_state()` 包夹。**必须显式处理。**

---

## 10. 标准调试流程

```
症状出现
   ↓
[L1] analyze_training.py --diagnostics-only        ← 30 秒
   ↓ 有结论？ → 按 §4 对应条目处理 → 结束
   ↓ 无结论
[L1] 按 §4 查缺失的四个聚合量是否已加 🚧          ← 大概率在这里结束
   ↓ 仍无结论
写下 hypothesis → 投放 debug_request.json          ← 强制明确假设
   ↓
[L2] replay_update.py --stages all                 ← 离线，可反复
   ↓ 定位到具体环节/帧 → 修复 → 补一条不变量(§6) + 一个回归用例
   ↓ 仍无结论
[L3] 进程内抓优化器状态 / 梯度轨迹
```

**每次修复后必须做两件事**：

1. **补一条不变量**（§6）——让同类 bug 下次自动暴露
2. **把快照转成回归用例**——`debug/uNNNNN/episodes/` 可直接作为 `build_trajectories` 的测试 fixture

否则下次还会踩同一个坑。

---

## 11. ⚠️ 设计缺口（写本文档时暴露的问题）

**这一节是本文档最重要的产出。** 以下问题必须在实现 L2 之前决策。

### 11.1 【阻塞级】缺少帧溯源（frame provenance）

**问题**：`Trajectory` **完全没有身份信息**。`PPOBuffer` 把所有 trajectory 拼成平坦数组（`n = Σ T`）后，**无法把第 `i` 个平坦索引映射回 `(episode_index, agent_id, t)`**。

**后果**：整个 L2 的价值大打折扣——你可以 dump 出所有数组，但**无法把框架侧数组（advantage/梯度）、实验侧数组（`balance_mask`）、Episode 侧数组（observer 原始值）join 起来**。而「第 137 帧为什么权重是 -1」这类问题恰恰需要三边 join。

**为什么不能由框架自动推断**：`build_trajectories(episodes) -> List[Trajectory]` 是**多对多**映射，且完全由实验决定（1 个 episode 可产生 0/1/N 条 trajectory；相位实验按 agent 产生 2 条，也可能按相位切成多段）。框架**原理上无法**知道来源。

**待决策**：
- 方案 A：`Trajectory` 加可选 `provenance: Optional[TrajectoryProvenance]`（`episode_index`, `agent_id`, `t_start`），由实验填写。调试时必填，生产可空。
- 方案 B：`build_trajectories` 改为返回 `List[Tuple[Trajectory, Provenance]]`。破坏性大。
- 方案 C：只在调试模式下要求实验通过 `debug_arrays` 返回一个 `frame_id` 数组。侵入最小但契约最弱。

倾向 **A**。这是 L2 的前置依赖，**必须先做**。

### 11.2 实验侧中间量与轨迹的时间基准可能不同

实验常在完整 episode（`T_full`）上计算中间量，随后按终止步**截断**成 `T`（如 `exp_basic_balance` 在失衡步截断）。于是 `debug_arrays` 返回的数组可能是 `T_full` 长度，而框架侧是 `T`。

**待决策**：契约规定「必须已截断、与 trajectory 对齐」，还是允许两种基准并显式标注？倾向前者（更简单），但需在 docstring 写死。

### 11.3 `is_terminated` 丢失了「为什么终止」

诊断 §4.4 时想知道每条 trajectory 的终止原因。`Episode.agent_termination_proposal_records` 有这个信息，但到 `Trajectory` 只剩一个 `bool`。建议随 provenance 一起带上 `termination_reason: str`。

### 11.4 逐帧 `V(s)` 从不出框架

只有聚合 `ev`。逐帧 `V(s)`、`δ`、`advantage` 目前是 `ppo_update` 内的局部变量。L2 重算能拿到，但需要 `ppo_update` 支持一个「dump sink」参数，否则要在离线工具里复制 GAE 逻辑——**那就违反了「不复制生产逻辑」的原则**。

**待决策**：给 `ppo_update` 加可选 `debug_sink` 参数（默认 `None`，零开销）。

### 11.5 θ_old 与 RNG 状态的可获得性未验证

L2 精确重算需要**本轮 rollout 实际使用的策略**（θ_old）和 update 起始的 RNG 状态。

- `policy_exports/uNNNNN/` 每轮都在导出 ✅，但**需要验证它确实是 θ_old**（更新前）而非更新后
- RNG 状态**目前没有任何地方记录** → minibatch shuffle 顺序无法复现 → §5.3 的自校验不变量无法成立

**待决策**：快照时显式保存 `torch.get_rng_state()` + `np.random.get_state()`。

### 11.6 `UpdateStats` 不是调试数据的载体

`UpdateStats` 是 frozen 的聚合量，且 `on_update` 是实验的调度钩子。**不要**把大数组塞进去——会污染 `__RAW_STATS__`、拖慢日志、破坏 `analyze_training.py` 的解析。调试数据必须走独立的 sink 路径。

同时注意：`analyze_training.py` 按路径解析 `stats.*`，§3.4 新增指标时要确认它能自动发现（`--list-metrics` 验证）。

---

## 12. 快速参考

```bash
# L1 健康检查（最常用）
PYTHONPATH=. python3 baseline/framework/analyze_training.py <run>/train.log --diagnostics-only

# L1 实时跟随
... --watch

# 抓所有告警
grep -E '\[warn\]|\[early_stop\]|\[skip\]' <run>/train.log

# 看 floor loss 是否还在生效
grep '\[GradDiag\]' <run>/train.log | tail -5

# 确认 explore_factor 是否真的在缩放 σ（两者相等 = 未缩放）
grep -o 'eff_std_mean=[0-9.]*\|std_mean=[0-9.]*' <run>/train.log | tail -10

# 冒烟测试（2 轮 update，验证代码能跑通）
PYTHONPATH=. python3 baseline/framework/train.py --experiment <name> --algo ppo --smoke

# L2 触发（🚧）
cat > <run>/debug_request.json <<< '{"hypothesis":"...","max_episodes":8}'
```
