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

## 待观察的转折点

- [ ] real_reward_per_step 何时脱离 0.03 平台开始爬（策略真正起形的信号）
- [ ] 第一个 eval success clip 出现的迭代（官方 4096 envs 下预期远早于 15000）
- [ ] imag_r 与 real_r 是否出现分叉（model exploitation 的早期信号）
- [ ] 缩小配置（512 envs）能否收敛到有意义的成功率——README 明说未验证
