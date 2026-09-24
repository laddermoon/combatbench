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

**run 222700 u50/u70 结果（第二次"查"）**：
- 干预生效：`swings` 2.5→13.5/ep（5×），+W→抬脚转化率 0.5%→8-11%
  （15×），站立帧 h p99 20→31mm，success=1.0 守住。
- **新落差**：rollout（ef 噪声）p99=31mm vs eval（确定性 policy）
  hmax=8mm —— 能力在分布尾部，均值还没吸收。
- **lr 过保守证据**：kl=0.006 vs target 0.03（只用 20% trust region）；
  warm-start ckpt 本来就是 lr=3e-4 训的，3e-5 是 v1 继承的过度谨慎。

**干预 #2（无代码改动，`--param` 启动补丁）**：lr 3e-5→**1e-4**。
新 run `train_step_ppo_20260922_230600`（pid 2047479, GPU2），
u1-u2 确认 `actor_lr=1e-4`、`param_overrides` 已记录、kl=0.011-0.014
（仍 <0.03，健康区间）、ef/foot adv_std 正常。

**判据（u50 dump 复查）**：
- `hmax`（eval 确定性）应从 8mm 明显爬升 → 均值在吸收尾部行为
- rollout 转化率应 >10% 继续走高；`success` 维持 ≥0.99
- 若 hmax 仍平：下一个嫌疑是 FOOT_WEIGHT 幅值或脚通道在
  combined_adv 中的占比被 r_potential 尖峰稀释

### 2026-09-23 — 根因确认：r_potential 否决了抬脚（第三次"查"）

**run 230600（lr=1e-4）u50/u55 数据**：rollout 统计与上一版几乎一致
（lr 不影响采样），但暴露了两个趋势：
- `foot reward_mean` 逐 update **下降**（L 0.0225→0.0095）——尾部在变瘦
- eval hmax 停在 7mm —— 均值没有吸收尾部

**根因（u50 dump 逐帧分解 combined_adv）**：

| lifted 帧 | potcontrib(aw=3.0) | footcontrib | combined |
|---|---|---|---|
| h=3-5cm | **-0.82** | +0.025 | **-0.18** |
| h>5cm | **-2.62** | -0.06 | **-0.73** |

抬脚 → φ 瞬时下跌（critic 无法预料随机抬脚何时发生）→ 负 adv ×
aw=3.0 → **站立通道以 ~30 倍力量否决每次真抬脚**。策略学的是
"抬脚对站立有害" —— 这解释了所有历史路线（v3 的 r_fall aw=1.0 是
同构否决，只是弱 3 倍）为何全部停滞。

**干预 #3（commit e55460c）— 摆动豁免门**：
```
aw_pot = 3.0 × (1 − φ_trail² × ss_mask)
```
- `ss_mask` = 去抖单支撑（新 helper `single_support_mask`，FLIGHT
  不继承——跳跃不是被命令的摆动，不豁免）
- `φ_trail` = φ 的 15 帧滑动 max：摆动的瞬时 φ dip 保持豁免；真摔倒
  （φ 持续低）0.75s 内保护恢复
- 倒地 φ=0 → aw_pot=3.0 全保护；站立+DOUBLE → 3.0；站立+单支撑 → ≈0

**新 run**：`train_step_ppo_20260923_000000`（pid 3394695, GPU2,
lr=1e-4）。u1-u2 确认 `aw_pot mean=2.08`（站立单支撑帧豁免生效）、
min=0、ef-verify OK、kl=0.011-0.013 健康。

**判据**：lifted 帧的 combined_adv 应由负转正；`foot reward_mean`
应止跌回升；eval `hmax` 应脱离 8mm。若 combined 仍负 →
查 ss_mask/φ_trail 实现细节；若转正但 hmax 不涨 →
探索瓶颈仍在，考虑继续放大站立 σ 或 FOOT_WEIGHT。

---

## [2026-09-23] u50 dump 验证（run 000000）— 豁免门生效，残留否决在 >5cm 帧

**判读工具升级**：`dumpkit/frame_access.py`（DumpDataset 懒加载访问层）
已落地并迁移全部消费方（server/dump_analysis/dump_delta/dump_render/
debug.py，无 DumpData shim）。跨通道归因现在 ~10 行。

**修正**：此前分析误用 `observer.standing_balance_a.*`（agent A 数据套到
所有帧）。正确用法是不带后缀的 `observer.standing_balance.*`——访问层
按 traj_map 逐轨迹路由到对应 agent。

**豁免门验证（u50）**：

| 帧集 | n | aw_pot | phi |
|---|---|---|---|
| stand + 去抖单支撑 | 79071 | **0.040** | 0.988 |
| stand + 非单支撑 | 298984 | 3.000 | 0.988 |
| 非站立 | 31545 | 2.577 | 0.328 |

门按设计工作。**lifted 帧梯度**：

| h | n | combined | contrib.pot | contrib.feet |
|---|---|---|---|---|
| 3-5cm | 9044 | **+0.072** | -0.026 | +0.049×2 |
| >5cm | 780 | **-0.209** | -0.204 | L+0.04/R-0.05 |

>5cm 帧分解：334 帧为 !ss（flight/去抖滞后 → 豁免不适用，aw=3 →
contrib -0.26）；446 帧为 ss（aw_pot=0.075 已豁免，但 r_potential
critic 的 normed_adv 本身为负 → contrib -0.16，且 Phase C 对超阈
摆动脚 -W → 顶点的正 adv 被负权重打回）。

**支付排序已正确**（per-traj）：pk>5cm 轨迹 foot_ret=8.92 >
2-5cm 的 8.36，pot_ret 几乎不掉（3.73 vs 3.76）——没有微步陷阱，
只是 >5cm 帧的瞬时梯度仍为负。

**rollout 分布**：站立帧 h p50=11.5/p90=22/p99=36/p99.9=60mm，
h>3cm 占 2.6%、>5cm 占 0.25%；+W→抬脚(>3cm) 转化率 ~12%。

**eval 平台期**：hmax 7→9mm（90 eval 几乎不动），swings 9.4/ep
稳定，step≈0。事件在 rollout 尾部存在且梯度转正，但均值吸收极慢
—— >5cm 事件仅 0.24% 帧 × 残负梯度；同时 u70+ 起 KL early-stop
频发（actor_steps 47-104/400，kl_max 冲到 0.137），更新被截断。

**候选下一干预**（若 u150 仍平台）：
1. **事件完成奖励**：detect_step_cycles 命中的摆动窗内给 bonus
   （如 +0.5 摊到 airborne 帧）——真实迈步 vs 微步的支付比从
   ~1.07× 拉到 ~50×，直接对齐 eval 指标
2. **Phase C 顶点对齐**：`w_swing = -W` 仅在脚下降时（dh≤0），
   上升中的高脚不再被罚
3. 继续加大站立 ef 或 FOOT_WEIGHT

---

## [2026-09-23] run 000000 判决：平台期，u101 终止；干预 #4 上线

**判决**：u90 eval —— hmax 7→9mm 90 eval 几乎不动，swings ~9.4/ep
平台，step≈0；u70+ KL early-stop 频发（10/15 update actor_steps
<400）。信号符号已修对但 >5cm 事件（仅 0.24% 帧）仍净负梯度，
均值吸收速度≈0。判定平台期，u101 终止。

**干预 #4（commit 358ca11）— 三处联动，目标：>5cm 残余否决 + 支付比**：

1. **事件完成奖励** `step_cycle_bonus=0.5`：`detect_step_cycles`
   （与 eval 同一判定）命中的摆动窗内逐帧加 bonus —— 真实迈步 vs
   微步支付比 ~1.07× → ~50×
2. **ss_mask ±hold 膨胀**：去抖滞后导致摆动边界帧落进 !ss 吃满
   aw=3 否决（u50: 334/780 帧）——膨胀后边界帧同豁免
3. **Phase C/A/B 顶点对齐**：超阈但仍在上升的摆动脚保持 +W，
   -W 只在过顶点后 —— 上升顶点不再被惩罚

**新 run**：`train_step_ppo_20260923_014732`（pid 2602274, GPU2,
ckpt u01500, lr=1e-4, dump u50/150/300）。

u1 即验证奖励流改变：foot reward_max 0.05→**0.22**（bonus 命中），
reward_mean ~2×，adv_std 4×（0.02→0.08），aw_pot mean 2.08→0.83，
foot critic ev 0.53/0.48。

**判据**：eval `cycles`/`hmax` 应开始爬升；u50 dump 看 >5cm 帧
combined_adv 是否转正、bonus 是否落在正确的帧。

**修正**：本 run resume 计数器从 1501 起（旧 run 从 1 起），
`--dump-at 50/150/300` 全部 <start_update 永不触发（loop 有 warning）。
已改用 sentinel 机制（`dump_request.json` 一次性触发下一 update）+
后台 watcher 在绝对 u1550/1650/1800 武装 —— 等价于 run 内第
50/150/300 个 update。教训：resume 后 dump-at 必须用绝对编号。

### run 014732 — u01551 dump 分析（run 内第 51 update）

**梯度符号全面转正**（对比 run 000000 u50 同口径）：

| lifted 帧 | run000000 | run014732 |
|---|---|---|
| h=3-5cm combined | +0.035 | **+0.126** |
| h>5cm combined | -0.070 | **+1.329** |
| >5cm contrib.pot | -0.097 | **+0.008** |
| >5cm contrib.feet | — | **+0.66/+0.66** |
| lift&ss aw_pot | 0.075 | 0.49（膨胀后更宽豁免）|

事件奖励生效：reward>0.1 帧 0.07-0.09%（~330 帧，对应完成的周期），
foot reward_max=0.217。h>5cm 站立帧占比 0.38%（前值 0.25%）。

**frame_access 修复**（commit 8f13888）：`observer.X_a.field` 此前对
所有 traj 都 gather agent-a 数组（一半 traj 错配）——现 `_[a-z]`
后缀=agent 过滤（不匹配 traj 置 NaN），canonical 名
`observer.X.field` 按 traj agent 解析。此前 u50 veto 结论方向不变
（两 agent 同 episode 分布近似），但绝对值略有偏差。

**eval 早期**（u1505-1545）：success=1.0 守住，swings 3.9→12.2/ep，
hmax 4→10mm 缓升，cycles 偶现（0.01-0.03）。下一判据 u1650 dump。

### run 014732 判决（u1651，~151 update）：吸收失败，终止

**梯度符号已修对但事件在灭绝**：lifted 站立帧 10570→2233（-79%），
h>5cm 占站立帧 0.38%→0.17%，foot reward_mean 0.010→0.0076，
eval swings 17.4→13/ep、hmax 12→7mm —— 全部指标在 u1570 见顶后
单调回落。每帧梯度反而更强（rare → normed_adv 放大），但数量在塌。

**根因（本次最重要的发现）**：状态机的 +W "意图"不在观测里。
u1651 dump：stand 帧 30% 收到 +W 指令，但指令期间真抬脚率 0.5% ——
策略看不见指令，抬脚只能靠探索噪声碰巧发生。PPO 的优势加权只能
把 mean 推向"碰巧好的噪声方向"，但没有可学的状态特征区分"该抬"
vs"该压"，mean 无法把抬脚固化成决策 → 事件永远稀有 → 被 99.5%
站立帧稀释 + KL 刹车截断 → 吸收≈0。三轮干预修的是"信号正确性"，
真正的瓶颈是"意图不可观测"。

**下一干预（#5）设计**：把迈步意图放进 obs —— 给观测加相位/指令
通道（如摆动时钟 sin/cos 或状态机相位 one-hot）。warm-start 兼容：
首层权重按新维度补零列，新输入初始惰性，policy 与 ckpt 完全等价，
再学习"cmd_lift_L=1 → 抬左脚"。这是让抬脚成为*决策*而非*巧合*
的必要条件 —— 与文献中 phase-conditioned locomotion 一致。

**干预 #5 已上线（commit 2a89bab + optimizer pad）**：
`GaitClockSimulator` 在 obs[96:99] 注入 cmd_L/cmd_R/窗口进度（动作步
索引的确定性函数，与轨迹帧号对齐）；reward 侧改用
`clock_foot_weights`（指令脚 +W 抬升/−W 落地，支撑脚恒 −W）。
load_checkpoint 支持输入维零填充扩展（权重+Adam 状态），暖启动
policy 初始与 ckpt 完全等价。

新 run：`train_step_ppo_20260923_053633`（pid 1196165, GPU2,
lr=1e-4, dump u1551/1651/1801 —— 本次为绝对编号）。

**判据**：eval swings/hmax 应突破前值平台（17/12mm）；u1551 dump
看 cmd_L=1 窗口内 h_left 响应率（目标 ≫0.5%）、combined_adv 符号。

### run 064853（u1550 ckpt 续跑）— u1551 dump：指令通道打开

**时钟条件化生效**。stand 帧 87%：

| 窗口 | 指令脚 h | 支撑脚 h | >3cm |
|---|---|---|---|
| cmd_L | 14.3mm | 7.3mm | 6.5% |
| cmd_R | 13.6mm | 8.2mm | 5.1% |

指令脚 ≈2× 支撑脚；窗口内相位结构匹配设计（9→15.3→16.8→16.1mm，
末段 -W 落地已见响应）。指令→抬脚转化率 ~12× 提升（0.5%→6.5%）。

eval：cycles 0→0.25、hmax 49mm、alt 首次非 None；final_pot 0.86
（真摆动施压站立，正常转换期）。中途崩过一次：prev_gvec 旧维度
（96042 vs 96810）——已修（commit 9af767f）。

### run 064853 终局（u1705 自动停止）：迈步已学会，但停在饱和指标上

eval 末段（u1630-1700）：success=1.0、**step=0.77-0.86**（约八成
agent 双脚各完成≥1有效周期）、cycles 4.0-4.8/ep、alt 0.50-0.58、
hmax 10-12cm、final_pot 0.85-0.89 —— 单 seed(42) 迈步已达标。
swings ~10/ep 与周期 40 步时钟吻合。

**停止原因 = 框架 bug**：early_stop 追踪 `mean_max_pot`（站立势），
暖启动后恒 1.000 → 严格大于永不成立 → u1705 触发
"no improvement for 200"。真实目标（step rate）仍在缓升。
已修（见下 commit）：best 追踪拆成 `_best_potential` +
`_best_step`（success≥0.9 时记 step rate），u01500 系 ckpt 继续
训也可恢复。

**后续**：以此 ckpt(u01550) 为起点跑 seed 43/44 验证三 seed 标尺。

### 视频验证推翻指标结论（u01700.mp4）—— reward hacking 实锤

人工查看 eval 视频：**不是迈步**。机器人双脚交替踮脚尖/侧沿
（~0.3s 一换，正好是时钟周期），脚无明显离地，20 秒内摔 2 次。
step=0.85 的 eval 指标是被骗出来的：

- `r_foot = clip(h,0,.05)` 用 foot body midpoint z —— 踮脚倾转把
  中点撬起 >5cm，无需离地
- `detect_step_cycles` 要求 contact=0 ≥3 帧 —— 换支点的瞬间整脚
  瞬时空载，bounce 恰好满足
- 周期 0.3s ≈ 时钟半窗 —— 策略踩着时钟节拍晃脚

### 干预 #6（commit d861c4b）：sole clearance 物理量替换 midpoint

- `FootStateObserver` 新增 `sole_clear_{left,right}`：四个脚胶囊
  端点世界 z 最小值 − 半径。踮脚 pivot 时 ≈0（下端点仍贴地），
  只有整脚腾空才 >0 —— 数学验证：45° heel-pivot → 0，平抬 5cm → 5cm
- 奖励值：`r_foot = clip(sole_clear, 0, .05)`；`clock_foot_weights`
  的 lift/rising 判定、`detect_step_cycles` 有效性（要求摆动段
  sole_peak ≥ 3cm）全部切到 sole_clear；eval 新增 `solepk` 指标
- 附带修复：early-stop 指标饱和 bug（6686cdb）
- 20/20 测试过（新增 rocking 回归用例）

旧 run s43/s44（hackable 奖励）已终止；三 seed 重新从 u01550 开跑：
s42_v2(GPU0, pid 3830278) / s43_v2(GPU1, pid 3838856) /
s44_v2(GPU5, pid 3844282)。u1551 基线 foot reward_mean≈0.003
（hacking 版 ~0.01+）—— 踮脚收益归零，需重新学真腾空。

**判据**：eval 的 `solepk`/`step` 指标。若 solepk 长期 ≈0 → 探索
产生不了真腾空，需回到 ef（探索增强）或动作先验。三 seed 全达标
才算过验收标尺。

### s42_v2 固化验证（u1651-1795）：sole_clear 奖励下真迈步收敛

u1651 dump：指令窗口内 sole>1cm 帧占 8.6%(L)/5.8%(R)，
aw=+0.35/+0.51，combined_adv=+0.11/+0.13 —— 真腾空帧净梯度为正，
固化路径成立。

eval 轨迹（u1555→1795）：solepk −0.027→+0.048、step 0→0.859、
cycles 0→9.4/ep、alt →0.34、success 全程 ≈1.0。

u1750 视频人工复核：真短暂离地 + 大量踮脚残留；模式为"一脚窗口内
尝试 ~3 次再换脚"（=时钟半窗 0.4s）；躯干晃动调节不摔 ——
真实但粗糙的步态，与低 alt 指标吻合。踮脚残留因"收益 0 但无惩罚"
而自然存在，随真腾空收益固化预期被挤出。

**与 hack 版对比**：同样的 step≈0.85 水平，hacking 版 ~150 update
到达（踮脚零成本），真版 ~250 update（真腾空更难）—— 但这次的
每个周期都过 sole≥3cm 判据。

### 三 seed 验收进展（sole_clear 版）

| seed | run | 轨迹 | 终态 |
|---|---|---|---|
| 42 | s42_v2 | solepk −0.03→+0.10, step u1665 非零→u1815 饱和 | step≈1.0, cycles~10.5, alt~0.60, success 1.0 |
| 43 | s43_v3 | 与 s42 同型曲线，略快 | step≈0.97-0.99, cycles~9.3, alt~0.54 |
| 44 | s44_v3 | 运行中 | — |

两 seed 收敛到同一平台（solepk≈0.10, step≥0.97, success=1.0），
配方跨 seed 复现确认。final_pot 降至 ~0.8（迈步耗平衡余量，正常）。

### 三 seed 验收完成（u1755 更新）

| seed | run | step 峰值 | solepk | alt | success | 收敛点 |
|---|---|---|---|---|---|---|
| 42 | s42_v2 | 0.99-1.00 | 0.10 | 0.60 | 1.0 | u2015 自动停 |
| 43 | s43_v3 | 0.97-0.99 | 0.10 | 0.54 | 1.0 | u1995 自动停 |
| 44 | s44_v3 | 0.93（还在涨） | 0.048→ | 0.41→ | 1.0 | 收敛中 |

三个 seed 走出几乎相同的曲线：~100 update 踮脚消退期
（solepk 负→零）→ ~50 update 固化爬升（step 0→0.9+）→
平台期 step≥0.9。**"站起来→真腾空交替迈步"配方跨 seed 成立**。

### 三 seed 验收 — 通过（u1785 更新）

| seed | run | step | solepk | alt | cycles/ep | success |
|---|---|---|---|---|---|---|
| 42 | s42_v2 | **0.99-1.00** | 0.10 | 0.60 | ~11 | 1.0 |
| 43 | s43_v3 | **0.97-0.99** | 0.10 | 0.54 | ~9.3 | 1.0 |
| 44 | s44_v3 | **0.97-1.00** | 0.06 | 0.45 | ~12 | 1.0 |

"用三个不同的 Seed 都可以成功完成迈步" —— **达标**。
且为真腾空（sole≥3cm 判据 + s42 视频人工复核），非踮脚刷分。

**共同收敛路径**（~200 update from u1550 ckpt）：
1. ~100 update 消退期：旧踮脚策略收益归零 → 渐灭（solepk 负→零）
2. ~80 update 固化期：稀有真腾空帧拿正梯度 → mean 漂移 →
   step 0→0.9+（s44 更快，u1690 即拐点）
3. 平台期：step≥0.97，cycles ~10/ep，alt 0.4-0.6，success 1.0

**产物**：各 run `policy/`（best-of-run）与 checkpoints/ —
可作为后续目标跟随/行走/战斗任务的暖启动底座。
