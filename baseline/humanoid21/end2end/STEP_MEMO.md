# Step 任务 Memo — 从 Standup 到迈步

> 本文件是"standup → stepping"任务的持续工作日志。新尝试、新发现按时间
> 追加在文末，旧内容不删改（修正用新条目注明）。

## 任务定义

- **目标**：在已收敛的 standup 模型上续训，让机器人学会交替迈步（stepping），
  为后续 follow-target 走路（end2end 第三步）做铺垫。
- **成功标尺**：**3 个不同 seed** 都能学会迈步（eval `step` 指标达标 +
  视频确认交替步态）。
- **方法论**：试 → 用 Debug 系统查 → 再试 → 再查。策略与 debug 工具双重迭代。
- **路线**：`experiments_ppo/exp_step.py` — φ² 软门控 + 逐脚高度奖励 +
  步态状态机（reward 承载物理事实，actor_weight 承载意图）。
  不采用 v3 的 h_torso 平台期硬相位切换路线。

## 关键资产

| 资产 | 位置 |
|---|---|
| 实验代码 | `baseline/experiments_ppo/exp_step.py`（`--experiment step`） |
| 环境蓝图 | `baseline/humanoid21/end2end/step_env.yaml` |
| 步态状态机 + 迈步检测器 | `baseline/humanoid21/end2end/stepping_state_machine.py` |
| warm-start ckpt | `baseline/runs/verify_resume_A/checkpoints/checkpoint_u01500.pt` |
| — standup_floor04 跑满 u1500，success≈1.0，max_h≈1.28，单 critic `r_potential` | |
| 检测器单测 | `baseline/experiments_ppo/test_step_detect.py` |

## 奖励结构（exp_step 当前配置）

```
r_potential  = 0.01 × φ_4stage    γ=0.99  aw = 3.0 固定（全程）
r_left_foot  = clip(h_L, 0, 0.05) γ=0.90  aw = 状态机权重 × φ²
r_right_foot = clip(h_R, 0, 0.05) γ=0.90  aw = 状态机权重 × φ²
```

- PPO：lr=3e-5, critic_lr=3e-4, target_kl=0.03（warm-start 保守档）
- 探索：`uncertainty_floor=0.4, coef=1.0`（对齐 standup_floor04；
  原 floor=0.3/coef=1e-3 是惰性的，U≈0.46 在铰链之上、系数弱 3 个数量级）
- 无失衡终止；max_steps=400；双 agent 同训

## Eval 指标（detect_step_cycles，2026-09-22 起）

一个**有效迈步周期** = 脚 F 在去抖接触序列上：
站立帧(φ≥0.9)抬脚 → 单支撑连续 airborne ≥3 帧 → 峰值 h≥0.05 → 站立帧落地。
跳跃（双脚离地）、抖动、摔倒落地均被构造性排除。

| 指标 | 含义 |
|---|---|
| `swings` | 摆动尝试次数/episode（最早动的信号） |
| `hmax` | 摆动峰值高度均值（连续进步信号） |
| `cycles` | 有效迈步数/episode |
| `alt` | 左右交替率（需 ≥2 cycles） |
| `step` | 双脚各≥1 有效迈步的 agent 比例（验收判据） |
| `success` | max φ ≥ 0.9（站立保护，必须维持 ~1.0） |

## 启动命令

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<N> python3 -B baseline/framework/train.py \
  --experiment step --algo ppo \
  --resume-from baseline/runs/verify_resume_A/checkpoints/checkpoint_u01500.pt \
  --reset-update --background
# 可选: --seed 43/44 多 seed; --dump-at N 预约 dump; --param KEY=VAL@U 热干预
```

## Debug 工具速查

```bash
debug.py metrics <run> --keys eval.                    # 迈步指标曲线
debug.py summary <run> --keys pc.,stats.               # 通道/critic/σ 健康度
debug.py dump <run> --hypothesis "..."                 # 下个 update 边界抓全截面
debug.py inspect/samples/trace/timeline <run>:uNNNNN   # dump 分析
debug.py render <dump_dir> --episode N                 # 逐帧渲染看步态
```

---

## 进展日志

### 2026-09-22 — 任务启动

**改动（commit a80d8fb, fd1f090, 4afc7f0）**
1. `exp_step.py` 从 `todo/` 移入 registry（`from .base` 直接可用，无冲突）。
2. 脚高 clip `±0.05` → `[0, 0.05]`（负奖励会惩罚深蹲；压低由 -W aw 表达）。
3. `on_eval` 换成周期检测五指标（原"单帧离地+过阈值"太粗）。
4. 探索参数对齐 floor04：`floor=0.4, coef=1.0`。

**首轮 dump 分析（u21, run 170015）**
- 站立帧占比 93%；脚通道 aw 在站立段 84% 帧非零（+W 42%/-W 58%）——信号链全通。
- 新 critic 收敛快：ev 0.94/0.91，conf 0.95+。
- **核心瓶颈**：每 episode ~7 次抬脚尝试（swings≈7），但平均峰值仅 0.9cm ——
  "尝试但抬不起来"，不是"不尝试"。旧二值指标的 ~3% "step" 是噪声。
- `combined_adv` 有 ±47 尖峰（r_potential adv_std=0.005 被 zscore 放大）——
  观察项，暂不动。

**当前 run**：`train_step_ppo_20260922_174438`（seed 42, floor04 探索档）
- u5-u15：`success=1.0`，`swings` 3.6→2.3，`hmax` 6→3mm（探索噪声上来后
  尝试略降），U 0.46→0.34 已被 floor 咬住（floor_loss>0）。
- 预约 dump：u50 / u150 / u300。
- 下判据：u50 看 `hmax` 是否触底回升；若停滞，候选干预依次是
  ① φ 切段状态机（摔倒段不污染 last_swing）② FOOT_WEIGHT 调大
  ③ adv winsorize 削尖峰 ④ lr/ef 加强。

**历史包袱说明**：更早的尝试（v1 cross_support、v2/v3 硬相位切换、以及
exp_step 的老 run）背景不完整，不作为决策依据；一切以当前 run 的
metric + dump 证据为准。

### 2026-09-22（晚）— 第一次"查"出死锁 + 双重干预

**run 174438 跑到 u335 的终局数据**：success=1.0 全程保持，但 swings
从 3.6 衰减到 ~0.8/episode，hmax 停在 3-12mm，cycles≈0.01 —— **335
个 update 的停滞确认不是"需要时间"，是策略在收敛到"不抬脚"**。

**u50 dump 死锁证据链**：
- +W 抬脚意图帧 128k/141k，仅 0.3%/0.8% 在 15 帧内真抬起 >3cm
- 真抬起的帧 adv=+0.07（87% 正，信用回传正常），但 65% 落在 -W
  窗口（Phase C 催落地 / DOUBLE-settle）→ aw×adv≈0，成功不被强化
- 没抬的 +W 帧 adv≈0（critic 正确预测"不会发生"）→ 无 surprise 无梯度
- `explore_factor` 全零 —— per-frame ef 设施存在但 exp_step 未驱动

**干预（commit 9ef3bec）**：
1. **Phase C 门控**：`w_swing = -W if h≥0.05 else +W` —— 没抬够的脚
   永远收到 +W，迟到的抬脚不再被惩罚（4 个新单测，共 11 个全过）。
2. **站立帧 ef=+0.631（σ×2），低位帧 ef=-0.631（σ×0.5）** —— obs[45]
   为 h_torso，与 v3 同参数；保护 standup 技能同时集中放大站立期探索。

**新 run**：`train_step_ppo_20260922_222700`（pid 410677, GPU2）
- u1 验证：ef-verify OK（ef∈{-0.631,+0.631}），eff_std=0.471 vs
  std=0.311（比值 1.52 符合相位占比），脚通道 adv_std=0.14
  （旧 run 同期 ~0.09→0.01 衰减），foot reward_mean ~0.02（5×提升）
- 预约 dump：u50/u150/u300
- 判据：`swings` 应回升 >3，`hmax` 应爬向 0.02+；`success` 必须
  维持 ~1.0（σ×2 有拖垮站立的风险，盯紧）
