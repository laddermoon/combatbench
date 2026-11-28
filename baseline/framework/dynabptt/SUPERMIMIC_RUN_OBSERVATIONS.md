# SuperMimic 官方训练过程观察记录

目的：通过跟进官方实现在本机的完整训练过程，积累"健康/病态指标长什么样"的体感，
反哺我们自己的 dynabptt 实现。持续追加。

Run: `/tmp/supermimic_mjlab/outputs/run512_15k`
配置: 512 envs / wm-batch 1024 / policy-batch 512 / 15000 iters（官方为 4096 envs / wm 2048 / policy 4096，我们缩 8×）
监控: `python /tmp/supermimic_mjlab/track.py outputs/run512_15k`

## 指标字典（与我们实现的对应）

| 官方指标 | 含义 | 我们的对应 |
|---|---|---|
| `wm_loss` / `wm_state` / `wm_termination` | WM 状态预测 + termination head 损失 | wm 训练 loss |
| `imagined_reward` | 32 步想象链上**每步** tracking reward | eval_surv 链条内的逐步 φ |
| `survival_at_horizon` | 想象链末端软存活 `alive(H)=cumprod(1-P_term)` | 无（我们没有软存活） |
| `real_reward_per_step` / `real_*_error` | 真实 rollout 采集时的 tracking 质量 | 部署口径指标 |
| `physical_failures` | 采集中的物理失败事件比例 | fall 事件 |
| `critic_raw_value_max` / `critic_value_above_cap_fraction` | critic 数值健康度 | 无对应（我们无 critic） |
| `actor_gradient_norm` / `wm_gradient_norm` | 梯度健康度 | grad clip 前的范数 |

## iter ~760/15000（5%）判读 — 早期相，WM 先行，策略未起

### 健康的部分

- **WM 拟合良好且在继续**：wm_loss 5.6→0.45 单调降；诊断显示 horizon-8 归一化 MSE
  **0.018 vs 恒速基线 0.185（10× 优）**——官方资格门槛（打败恒速基线）轻松通过；
  termination head Brier=0.0012 基本完美
- **无 model exploitation**：imagined_reward≈0.020 vs real_reward_per_step≈0.029，
  现实略高于想象——策略**没有**钻 WM 的空子。与我们 dynabptt 的核心病态（想象-现实
  gap 先行恶化）形成对照；原因猜测：tracking reward 被 mocap 参考锚定，"想象里的
  奇怪状态"拿不到奖励
- **梯度健康**：actor/wm grad norm 都在 2-6，无爆炸；critic_loss 在早期 ~800 尖峰
  （想象目标漂移期）后稳定在 ~1.2
- **失败率方向对**：physical_failures 94%→~45%（前 760 轮减半）

### 还没到位的部分（正常，不急）

- eval 仍 0/40，mean_completion 0.36%（最好的 clip 在 ~37 步就 height_failure）
- real_reward_per_step 平在 ~0.03，pos_err 0.24→0.38 甚至略升
- surv_H 在 0.4-0.6 振荡——想象里也只有一半概率"活"到 32 步

### 值得注意的现象

- **wm_state 在 iter ~550 后从 0.17 回升到 0.22**：策略开始探索新状态区，WM 被拖去
  拟合移动的目标分布——on-policy 非平稳性的正常表现，若持续上升才是坏事
- **critic 早期巨峰（~12000）**：imagined return 快速爬坡导致 TD target 漂移，正常；
  `critic_value_above_cap_fraction` 可关注是否触顶
- eval 协议：确定性策略、无噪声、从 clip 首帧全段播、height/tilt 失败即终止——
  比我们 eval 严格得多（我们 StartSet 种子 eval 本就是优势区间）

## 全程结果（15000 轮跑完，~2h）

**1/8 规模（512 envs）达到 30/40 success、86.9% completion、norm_return 0.498。**
官方 4096 envs 报 40/40——缩小配置有效，README 的"未验证"已验证通过。

### 曲线三段结构（这是最重要的体感）

```
iter 0-1500 (10%):   死亡平台期 —— eval 0/40，real_r 平在 0.03，
                     pos_err 反而涨到 2.0m（策略还在乱挥），
                     所有"下游"指标零信号。此时唯一能看的是 wm_loss 在降。
iter 1300-2000:      相变期 —— wm_term 0.21→0.03 雪崩，
                     surv_H 0.4→0.985，fail% 51→3，tilt_err 砍半，
                     eval 一次评估间从 0/40 跳到 15/40。
iter 2000-15000:     精修期 —— 缓慢研磨到 30/40，
                     4000 轮后 completion 就 ~80%+，之后全是长尾。
```

**核心判读：termination head 的学习是"离散的锁"**——它一旦学会（wm_term 从 0.2 跌到
0.03 发生在 ~500 轮内），软存活权重 `alive` 才有意义，想象的奖励分配瞬间变得可用，
策略立刻起飞。在此之前整条链都在"无 survival 信号"状态空转。

### imag↔real 全程贴合——最硬的一条体感

imagined_reward 与 real_reward_per_step **全程几乎重合**（终点 0.069 vs 0.0665，gap~2%），
从未分叉。这证实之前的推断：mocap 锚定的 tracking reward 天然抗 model exploitation；
我们 dynabptt 的 imag-real gap 病态是**内在奖励语义**的问题，不是 WM 训练的问题。

### 剩余 10 条失败 clip 的画像

- 5/8 个 fallAndGetUp 全挂（5%-93% completion 不等）——倒地-爬起是接触最复杂的相
- jumps、快节奏 dance——空中相 + 冲击接触
- 失败原因全是 height/tilt_failure——模型和策略都倒在了高动态相
- 合理推测：这些 clip 要么需要更多 envs/batch（探索覆盖），要么需要 wm-only 预训练期

### 数值健康基线（"正常应该长这样"）

| 指标 | 健康轨迹 |
|---|---|
| wm_loss | 5.6 → 0.015，单调降，相变后趋稳 |
| wm_term | 平台 0.2 很久 → 相变时雪崩到 ~0.001 |
| surv_H | 死亡期 0.2-0.5 振荡 → 相变跳到 0.98+ |
| real_r vs imag_r | 全程贴合，gap<10% 即健康 |
| physical_failures | 94% → ~1%，相变期降幅最陡 |
| actor/wm grad norm | 2-6 起步，收敛期降到 0.3-1.7，无尖峰 |
| critic_loss | 早期巨峰（想象目标漂移）后稳在 ~1 |
| pos_err（采集口径） | 含失败 rollout 漂移，收敛后仍有 ~0.7m——会被失败样本拉偏，别当 tracking 精度读 |

### 对我们实现的反哺清单

1. **死亡平台 10% 是正常的**——我们之前看到 eval 不动就改方向可能过早；
   正确的等待判据是"底层指标（WM loss）是否还在动"
2. **termination/存活信号是离散的锁**——我们的实验里没有等价的"alive"结构，
   值得考虑给想象链加权存活软终止
3. **imag↔real gap 是唯一需要盯的剥削信号**，且只在内在奖励下才危险
4. **长尾是结构性的**——最难 clip（高动态/倒地爬起）天然需要更多覆盖，
   30/40→40/40 靠的是规模不是等更久
