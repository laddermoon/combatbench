# PPO 训练调试指南

**这份文档的目标**：让你在训练出问题时，**心里有底，手里有牌**。

- **心里有底** = 你有一个正确的心智模型，知道问题只可能出在哪几个地方
- **手里有牌** = 每个地方都有一个明确的工具能看清它

**这份文档不解决的问题**：调参技巧。本文档的立场是——**在你能说出「我预期哪个量会怎么变」之前，不要动任何超参**。做不到这一点时，说明你缺的是观测，不是参数。

> 本文档描述调试体系的**目标形态**，同时充当其功能规格。文末 §8 列出它对实现提出的要求。

---

## 1. 心智模型：两个轴

调试 PPO 训练时，所有问题都落在两个轴上。**先判断问题在哪个轴，再选工具。**

### 1.1 纵轴：信号链（因果）

策略之所以改变，**唯一原因**是每一帧上的 `combined_adv`。上游所有东西——observer、reward、critic、权重、归一化——存在的唯一目的就是产生这一个数。

于是从「物理世界发生的事」到「策略行为改变」是一条**九环因果链**。任何「学不出来」的问题，本质都是**链条在某一环断了**。

```
 ①物理  →  ②观测  →  ③奖励  →  ④预测  →  ⑤优势
                                              ↓
 ⑨行为  ←  ⑧梯度  ←  ⑦合成  ←  ⑥门控  ←───┘
```

| 环 | 含义 | 健康判据 | 断裂时的现象 |
|---|---|---|---|
| **①物理** | 目标行为在 rollout 中**是否曾经发生**（哪怕偶然、笨拙） | 出现率 > 0 | reward 恒为常数或恒为 0 |
| **②观测** | observer 是否如实捕捉到它 | 与独立来源交叉一致 | 数值全零 / 全常数 / 与 obs 不符 |
| **③奖励** | reward 是否在**正确的帧**、以**合适的量级**出现 | 非零帧位置与物理事件重合 | 均值正常但分布错（见 §7.1） |
| **④预测** | critic 能否解释它 | `EV > 0.5` | `EV ≤ 0`，advantage 全是噪声 |
| **⑤优势** | 「好于预期」的部分是否有区分度 | 信噪比 `\|adv_mean\|/adv_std` 不接近 0 | `adv_std ≈ 0` 或信噪比 ≈ 0 |
| **⑥门控** | `actor_weight` 是否在该帧打开、**符号是否正确** | 目标帧上非零且符号符合意图 | 权重为 0、或均值为负（净抑制） |
| **⑦合成** | 归一化 + confidence 后是否还剩份额 | 影响份额 > 5% | 份额 < 5%，被其他通道淹没 |
| **⑧梯度** | 是否转化为**相关参数**的梯度 | 相关动作维度梯度占比合理 | 目标关节梯度 ≈ 0 |
| **⑨行为** | 参数变化是否转化为**可观测的行为改变** | 行为探针判定为真 | 指标涨了但行为没变（见 §7.2） |

**这张表最重要的用途不是诊断，是决定你该做哪一类改动。** 断在不同环，正确的修法完全不同：

| 断在 | 该做的事 | **不该做的事** |
|---|---|---|
| ①物理 | 加探索、加课程、给示教、放宽初始条件 | ❌ **调 reward 完全无用**——行为从未发生，没有东西可以被奖励 |
| ②观测 | 修 observer / 修数据通路 | ❌ 调 reward 权重（在修一个不存在的问题） |
| ③奖励 | 改 reward 形状、量级、触发帧 | ❌ 改探索（信号有，只是形状不对） |
| ④⑤预测/优势 | 改 critic 容量/LR、改 γ、改 reward 尺度 | ❌ 改 actor_weight |
| ⑥⑦门控/合成 | 改 actor_weight、改通道数、改 confidence | ❌ 改 reward |
| ⑧梯度 | 改 LR、梯度裁剪、网络容量、检查冻结 | ❌ 改 reward 或权重 |
| ⑨行为 | 修指标定义 | ❌ 改任何训练配置——**问题在你的尺子上** |

> **最常见的错误是类别错误**：链条断在①（探索不足），却去反复改 reward 权重。这会烧掉很多天而毫无进展，且每次都"看起来有一点变化"（因为噪声）。**§3.2 的信号链剖面工具存在的全部意义，就是让你在动手前先知道断在第几环。**

### 1.2 横轴：可信度（认识论）

即使因果链完好，你看到的结论也可能是假的。三种典型失真：

| 失真 | 含义 | 工具 |
|---|---|---|
| **测量不可信** | 指标测的不是你以为的东西 | §3.6 指标验证 / §3.7 行为探针 |
| **变化不显著** | 变化量小于 run-to-run 噪声 | §3.9 噪声基线 |
| **干预未生效** | 你改的配置根本没进入数据通路 | §3.8 干预验证 |

**纪律**：在把任何数字变化当成结论之前，先过一遍这三条。

---

## 2. 你会遇到的问题 → 用什么工具

按真实发生频率排序。每一行都是一条完整的牌路。

| 你的问题 | 第一手牌 | 若不够 | 再不够 |
|---|---|---|---|
| 训练健康吗？ | `health` | `timeline` | — |
| **想要的行为没出现** | `chain --channel X` | `frame` 抽检 | `probe` 确认①环 |
| 它到底在学什么？ | `attribute` | `chain` 逐通道 | `frame` |
| 这个指标是真的吗？ | `metric --verify` | `probe` | 人眼看视频 |
| **我该调什么？** | `whatif`（离线试算） | `chain` 定位环 | — |
| 我改的东西生效了吗？ | `intervene-check` | `frame` 查逐帧值 | — |
| 两个实验哪个好？ | `compare` | `noise` 定噪声带 | — |
| 突然崩了 | `timeline` | `health` 回溯窗口 | `snapshot` 崩溃前后 |
| resume 后不对 | `intervene-check` | `compare` 与原 run | — |
| 某一帧为什么是这个值 | `frame` | — | — |
| reward 算得对吗 | `chain` 的②③环 | `frame` | 回归用例 |

---

## 3. 工具箱

统一入口：

```bash
PYTHONPATH=. python3 baseline/framework/ppo/debug.py <子命令> [...]
```

### 3.1 `health` — 训练体检

**回答**：现在有没有病？

```bash
debug.py health <run_dir>              # 一次性报告
debug.py health <run_dir> --watch      # 实时跟随
```

输出一个**结论优先**的报告：先给判定，再给证据，最后给建议动作。

```
训练体检 @ update 250   [ 需要注意 ]

CRITICAL  探索坍缩风险
          std_min 0.055 已贴近下界，最近 10 轮持续下降
          → 检查 uncertainty_floor 是否生效：debug.py health --knob uncertainty

WARNING   r_left_foot 通道影响份额 2.1%
          该通道基本未参与策略更新
          → debug.py chain <run> --channel r_left_foot

OK        critic 健康（所有通道 EV > 0.6）
OK        KL 稳定（mean 0.021，无早停）
OK        无数据通路异常
```

**用法纪律**：这是唯一一个「不知道该干什么时先跑」的命令。它不会告诉你根因，但会告诉你**该往哪看**。

### 3.2 `chain` — 信号链剖面 ⭐

**回答**：这个奖励的信号，在九环里的**哪一环死掉了**？

这是本体系最重要的工具。它把 §1.1 的心智模型变成一张可读的表。

```bash
debug.py chain <run_dir> --channel r_left_foot
debug.py chain <run_dir> --channel r_left_foot --behavior "swing_foot_clear"
```

```
r_left_foot 信号链剖面 @ update 250

环             量                              判定
①物理    抬脚事件出现率 0.4%              ⚠  极低（阈值 h>2cm 持续≥3帧）
②观测    h_left_foot 与足端几何一致        ✓
         非零帧 68%，量级 [0, 0.31]
③奖励    非零帧 68%，量级 [0, 0.05]        ✓  与②一致
         但 P50=0.002（半数帧信号≈0）      ⚠  实际有效帧远少于 68%
④预测    EV 0.71                           ✓  critic 学到了
⑤优势    adv_std 0.060  |mean| 0.0002      ✗  信噪比 0.003
⑥门控    aw 非零帧 68%  mean −0.04         ⚠  均值为负 = 净抑制当前行为
⑦合成    影响份额 2.1%                     ✗  ← 信号在此死亡
⑧梯度    腿部关节梯度占比 4%               ✗
⑨行为    swing_foot_clear 判定 False        ✗

诊断：主断点在 ①，次断点在 ⑦。
      目标行为几乎从未发生（0.4%），因此 ③–⑦ 的信号本质上是噪声。
      → 这是探索问题，不是奖励问题。修 reward 权重不会有效果。
      → 建议：debug.py whatif 试算探索类改动；或先用 probe 确认行为可达性。
```

**关键设计**：它必须**自动指出主断点**，并且**明确排除掉不该做的改动类别**。用户不应该需要自己解读九行数字。

### 3.3 `attribute` — 归因

**回答**：这一轮更新，策略**为什么**变成了这样？谁在驱动它？

```bash
debug.py attribute <run_dir>                    # 最近一轮
debug.py attribute <run_dir> --window 20        # 近 20 轮平均
debug.py attribute <run_dir> --by action-dim    # 按动作维度分解
```

```
更新归因 @ update 250

通道影响份额（Σ|aw_normed × conf × normed_adv|，归一化）
  r_potential   ████████████████████░░░░  62.3%
  r_fall        ████████░░░░░░░░░░░░░░░░  33.5%
  r_left_foot   ░░░░░░░░░░░░░░░░░░░░░░░░   2.1%
  r_right_foot  ░░░░░░░░░░░░░░░░░░░░░░░░   2.1%

无梯度帧占比  12.4%   （Σ|aw| = 0，这些帧对 actor 完全无贡献）

按动作维度的梯度分布（top / bottom 3）
  最大  torso_yaw 0.083 | hip_r_pitch 0.071 | shoulder_l 0.066
  最小  ankle_l_roll 0.0004 | knee_l 0.0007 | ankle_r_roll 0.0009
        ⚠ 踝/膝关节梯度接近零——腿部行为在物理上无法通过当前梯度改变
```

**为什么这个工具不可或缺**：`actor_weight` 的配置值（比如 3.0 / 1.0）**不是**它的实际影响力。经过逐帧 L1 归一化、confidence 加权、advantage 归一化之后，实际份额可能与配置比例相差一个数量级。**不看归因就调权重，等于闭眼调。**

### 3.4 `frame` — 帧级检查器

**回答**：第 137 帧到底发生了什么？为什么它的权重是 −1？

```bash
debug.py frame <snapshot> --id ep0003:robot_a:137
debug.py frame <snapshot> --id ep0003:robot_a:137 --render   # 附带该帧渲染图
debug.py frame <snapshot> --where "aw.r_left_foot < 0" --limit 20   # 按条件筛帧
```

```
帧 ep0003:robot_a:137        （episode 3，robot_a，第 137 步）

观测        h_torso(obs[45]) 1.243    与 observer 一致 ✓
            joint_vel 范数 2.31

observer    standing_balance: potential 0.998  h_torso 1.243
            height_phi:       phi 1.000
            foot_state:       h_left 0.004  h_right 0.001
                              contact_l True  contact_r True

实验中间量  balance_mask      True
            相位              BALANCE（第 118 帧进入）
            状态机分支        DOUBLE_grace_expired
            startup_bias      未触发（已过前 40 帧窗口）

通道        reward      aw(原始)  aw(归一化)  V(s)     adv     贡献
r_potential  0.0100      1.00      0.333     0.891   −0.004   −0.001
r_fall       0.0100      1.00      0.333     0.462   +0.051   +0.014
r_left_foot  0.0004     −1.00     −0.167     0.152   −0.031   +0.004
r_right_foot 0.0001     −1.00     −0.167     0.148   −0.028   +0.004

combined_adv  +0.021
```

`--where` 是这个工具的杀手用法：**按条件批量抽检**，比如「所有权重为负的帧」「所有 reward 异常大的帧」「所有相位切换帧」，用来验证你对逻辑的理解是否与实际一致。

### 3.5 `whatif` — 离线反事实试算 ⭐

**回答**：如果我改这个参数，这一轮更新会有什么不同？

**这是打破盲调循环的核心工具。** 它在一个快照上重跑更新，应用你的假设改动，对比梯度方向和量级——**不需要训练**。

```bash
debug.py whatif <snapshot> --set foot_actor_weight=3.0
debug.py whatif <snapshot> --set foot_height_clip=0.30
debug.py whatif <snapshot> --set 'channel.r_potential.actor_weight=0'
debug.py whatif <snapshot> --sweep foot_actor_weight=1,3,5,10
```

```
反事实试算   基线 vs foot_actor_weight: 1.0 → 3.0

影响份额     r_left_foot    2.1%  →   5.8%
             r_potential   62.3%  →  54.1%
combined_adv 与基线余弦相似度  0.981
梯度方向     与基线余弦相似度  0.994
腿部关节梯度占比               4.0%  →   6.1%

判定：梯度方向变化 0.6%，远小于 seed 间噪声（实测 4.2%，见 noise 基线）。
      → 这个改动的效果无法与噪声区分。不建议花训练时间验证。
      → 根因在①环（行为出现率 0.4%），加大权重无法放大不存在的信号。
```

**这个工具改变工作方式**：过去你要花 4 小时训练来验证一个猜想；现在几十秒就能排除掉大部分无效猜想。**只有通过了 `whatif` 检验的改动才值得投入训练。**

局限（必须知道）：它只能看**单轮更新的边际效应**，看不到多轮累积的动力学变化。所以它能可靠地**排除**无效改动（方向变化 < 噪声 ⇒ 一定无效），但不能**保证**有效改动一定成功。**用它做否证，不做证实。**

### 3.6 `metric --verify` — 指标可信度验证

**回答**：这个指标测的是我以为的东西吗？

```bash
debug.py metric <run_dir> --verify steps
debug.py metric <run_dir> --verify steps --at u00250
```

它用**更严格的独立定义**重算同一个指标并对比：

```
指标验证   steps @ update 250

当前定义   支撑脚切换次数（无时长约束）              21.4
严格定义   摆动脚离地 ≥3cm 且持续 ≥5 帧，
           另一脚保持支撑                              0.8
接触抖动   单帧接触翻转次数                           19.2

判定：✗ 当前指标 96% 来自接触抖动，不反映物理迈步。
      → 该指标不可用于判断训练进展。修正定义后重新评估历史曲线。
```

**为什么必须有这个工具**：一个失真的指标比没有指标更糟——它会让你在错误方向上持续投入并"看到进展"。**任何你打算用来做决策的指标，都必须先通过一次 verify。**

### 3.7 `probe` — 行为探针

**回答**：策略**到底会不会**做这件事？

指标是间接的，探针是直接的。探针从**固定初始状态**跑确定性 episode，判定**行为谓词**是否成立。

```bash
debug.py probe <run_dir> --at u00250
debug.py probe <run_dir> --suite locomotion --at u00250 --render
debug.py probe <run_dir> --suite locomotion --sweep-updates 100:300:20
```

```
行为探针   locomotion @ update 250   （16 个固定初始状态，确定性 rollout）

谓词                              通过率    最佳样本
stand_up_from_fallen              16/16     ✓
maintain_balance_300f             15/16     ✓
swing_foot_clear_3cm_5f            0/16     ✗
alternating_support_2cycles        0/16     ✗
forward_displacement_0.2m          0/16     ✗

对比 update 100:  无变化（0/16 → 0/16）
判定：150 轮训练未产生任何迈步行为。①环持续断裂。
```

**探针相对指标的三个优势**：确定性（无采样噪声）、固定初值（可跨 update 比较）、谓词化（不会被抖动欺骗）。

**推荐做法**：任何一个训练目标，在开训之前先写好它的探针谓词。**如果一个目标无法写成谓词，说明你还没定义清楚要什么。**

### 3.8 `intervene-check` — 干预验证

**回答**：我改的配置真的进入数据通路了吗？

配置改了但没生效，是最浪费时间的一类问题——因为一切看起来都正常。

```bash
debug.py intervene-check <run_dir>                 # 检查所有旋钮
debug.py intervene-check <run_dir> --knob explore_factor
```

```
干预验证 @ update 250

旋钮                配置值    数据通路实测              判定
explore_factor      相位相关   σ 比值 1.56（期望 1.5–1.7）   ✓ 已生效
uncertainty_floor   0.35      floor loss 梯度 0.000          ✓ 已按预期在 u110 关闭
foot_actor_weight   1.0       逐帧 |aw| 唯一值 {0, 1, 2}     ✓ 已生效
foot_height_clip    0.05      reward 上界实测 0.0500         ✓ 已生效
resume checkpoint   u01500    参数指纹匹配                   ✓
observer 一致性     —         obs[45] vs h_torso 最大偏差 0   ✓
```

**每一个可配置的旋钮都必须有一条对应的通路实测。** 「配置里写了」不等于「生效了」——这条断言应该是自动的，而不是靠人想起来去验。

### 3.9 `noise` 与 `compare` — 显著性与对照

**回答**：这个差异是真的，还是噪声？两个实验哪个好？

```bash
debug.py noise --experiment standup_step_v3 --seeds 4 --updates 50
debug.py compare <runA> <runB>
debug.py compare <runA> <runB> --metric max_pot --with-noise-band
```

```
跨 run 对照   A: 固定 ef=0      B: 相位相关 ef
              （噪声基线：4 seeds × 50 updates）

指标            A          B          差异     噪声带    判定
max_pot        0.999      0.999      +0.000   ±0.003   无差异
bal_frac       0.851      0.874      +0.023   ±0.041   噪声内
steps(严格)    0.8        1.1        +0.3     ±0.9     噪声内
uncertainty    0.183      0.241      +0.058   ±0.012   ✓ 显著
腿部梯度占比    4.0%       6.8%      +2.8%    ±1.1%    ✓ 显著

判定：B 的探索确实更强（uncertainty、腿部梯度显著），
      但尚未转化为行为差异（steps 在噪声内）。
      → 结论：机制生效，但不足以突破①环。继续观察或加大力度。
```

**纪律**：**没有噪声带的对照结论一律不采信。** 这是防止把随机波动当成进展的唯一手段。跑一次噪声基线的成本，远低于因误信噪声而走错方向的成本。

### 3.10 `timeline` — 事件时间线

**回答**：曲线在这里转折，是因为我做了什么？

```bash
debug.py timeline <run_dir>
debug.py timeline <run_dir> --overlay uncertainty,max_pot
```

```
事件时间线   train_standup_step_v3_ppo_20260908_165855

u0000  ├─ resume from train_standup_truncnorm/u01500（--reset-update）
u0001  ├─ floor 阶段 1 启动（floor=0.35 coef=5.0）
u0110  ├─ floor 阶段 2 切换（uncertainty 连续 5 轮 ≥ 0.30）
u0142  ├─ best checkpoint（max_pot 0.999）
u0250  └─ 当前

uncertainty  ▁▂▃▅▆▇▇▇▆▅▄▃▃▂▂▂▂  峰值 u110 后单调下降
max_pot      ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  平坦
```

把**你的干预**和**指标变化**放在同一时间轴上，是建立因果判断的最低要求。

### 3.11 `snapshot` — 抓取现场

**回答**：我想仔细看看现在这一轮到底发生了什么。

对**正在运行**的训练下指令，抓取一轮完整现场：

```bash
debug.py snapshot <run_dir> \
    --hypothesis "foot 影响份额过低，怀疑 startup_bias 从未触发" \
    --episodes 8
```

抓取在下一轮 update 的边界发生，不打断训练，产物落在 `<run_dir>/debug/uNNNNN/`。

**`--hypothesis` 是必填的。** 它会随快照保存。这不是形式主义——**说不出假设就说明你还不该抓数据，而应该先跑 `health` 或 `chain`。** 这条约束的存在就是为了阻断「先抓一堆数据再说」的无效循环。

快照是 `frame` / `whatif` / `metric --verify` 的输入，也是回归用例的来源（见 §5）。

---

## 4. 场景剧本

### 剧本 A：想要的行为没出现（最常见、最容易走错）

```
1. debug.py chain <run> --channel <目标通道>
   → 读「主断点」。这一步决定后面的一切。

2. 若主断点 = ①物理
   ├ debug.py probe --sweep-updates   确认历史上是否真的从未发生
   ├ debug.py whatif --set <探索类改动>   离线试算
   └ ⛔ 不要改 reward / 权重

3. 若主断点 = ③奖励
   ├ debug.py frame --where "<目标事件条件>"   看 reward 是否出现在正确的帧
   └ 检查分布而非均值（§7.1）

4. 若主断点 = ⑥⑦门控/合成
   ├ debug.py attribute   看实际影响份额与无梯度帧占比
   └ debug.py whatif --sweep <权重>   找出能把份额推到 >10% 的取值

5. 若主断点 = ⑧梯度
   └ debug.py attribute --by action-dim   确认目标关节是否有梯度

6. 若主断点 = ⑨行为
   └ debug.py metric --verify   你的尺子坏了，不是训练坏了
```

### 剧本 B：我不知道该调什么（防盲调）

```
1. ⛔ 先停手。不要改任何参数。

2. debug.py chain   定位断点环 → 查 §1.1 表得到「该做的事」的类别

3. 在该类别内列出 2–5 个候选改动，写下每个的预期指标变化

4. debug.py snapshot --hypothesis "<你的假设>"

5. debug.py whatif --sweep   对每个候选试算
   ├ 方向变化 < 噪声带  → 直接淘汰，不训练
   └ 方向变化显著       → 进入候选池

6. 候选池中最多选 1 个投入训练；用 debug.py compare --with-noise-band 判定

7. 训练结束后：无论成败，把结论写进实验文件的 docstring
```

**这套流程的核心是第 5 步**：它把「4 小时训练一次的试错」变成「几十秒一次的筛选」。历史上一天内 13 次盲调提交的代价，本可以用几分钟的 `whatif` 避免。

### 剧本 C：指标涨了，但我不信

```
1. debug.py metric --verify <指标>        用严格定义重算
2. debug.py probe                         行为谓词是否也变了？
3. debug.py compare --with-noise-band     变化是否超过噪声？
4. 三者皆通过 → 可以相信。任一不通过 → 先修尺子。
```

### 剧本 D：突然崩了

```
1. debug.py timeline        崩溃点前后我做了什么？
2. debug.py health --at <崩溃前若干轮>    崩前是否已有预警？
3. debug.py snapshot（若仍在运行）
4. debug.py compare <崩溃run> <上一个正常run>
```

### 剧本 E：resume 后行为不对

```
1. debug.py intervene-check --knob resume    参数指纹是否匹配
2. 确认 experiment.load_state 恢复完整
3. 确认观测归一化常量未变（改过会使旧 checkpoint 失效）
4. debug.py compare <新run> <原run> 前若干轮
```

---

## 5. 安全网：不变量与回归用例

工具需要你主动去用；**安全网不需要**。这是投入产出比最高的部分。

### 5.1 不变量守卫（始终开启）

违反即报错，不允许静默通过。

| 类别 | 不变量举例 |
|---|---|
| **数据一致性** | `obs[45] == observer.h_torso`（逐帧）；observer 长度 == T；obs/action/reward/weight 长度一致 |
| **数值健全** | 无 NaN / Inf；reward 在通道声明范围内 |
| **时序正确** | `last_obs` 是第 T+1 帧；`obs[t]` 与 `action[t]` 对齐 |
| **通路一致** | rollout 采样分布与训练重算分布一致（首 minibatch `\|ratio−1\| < 1e-4`） |
| **梯度存在** | 无梯度帧占比 < 100% |
| **旁路检测** | 每个已配置旋钮都能在数据通路中被观测到（§3.8） |

核心原则（见 `CLAUDE.md` "Fail Loud"）：**缺数据必须报错，不能静默回退。** 静默回退会让训练继续跑但学错东西——比崩溃危险得多，因为它消耗的是你的时间而不是你的注意力。

### 5.2 每次修复后必做两件事

1. **补一条不变量** —— 让同类 bug 下次自动暴露，而不是靠你再debug一遍
2. **把快照转成回归用例** —— `debug/uNNNNN/` 可直接作为 `build_trajectories` 的测试 fixture

不做这两件事，同一个坑一定会再踩。**调试的产出不是「问题解决了」，而是「这类问题以后会自己暴露」。**

---

## 6. 纪律清单

工具再好，用错顺序也没用。这几条是硬约束：

1. **不知道该干什么时，跑 `health`。** 不要凭感觉改参数。
2. **改任何超参之前，先能说出预期哪个量怎么变。** 说不出 ⇒ 缺观测，去跑 `chain`。
3. **调 reward 之前，先确认断点不在①环。** 这是最高频的类别错误。
4. **任何用于决策的指标，先过一次 `metric --verify`。**
5. **任何对照结论，必须带噪声带。**
6. **抓快照必须写 hypothesis。** 写不出来就先别抓。
7. **候选改动先过 `whatif` 否证，再投入训练。**
8. **每次修复补一条不变量 + 一个回归用例。**
9. **实验结论写进实验文件的 docstring**，无论成败。下一个人（包括三个月后的你）需要它。

---

## 7. 常见误读

### 7.1 均值骗人，要看分布

`reward_mean = 0.017` 可以是两种完全不同的情况：

- 全部帧都是 0.017（信号平坦，无区分度）
- 3% 的帧是 0.5，其余为 0（稀疏强信号）

**训练动力学完全不同，但均值一样。** 所以 `chain` 报告分位数而不只是均值，`frame --where` 用来抽检分布的两端。

### 7.2 配置值 ≠ 实际影响力

`actor_weight = 3.0` 只是原始配置。经过**逐帧 L1 归一化**（`aw / Σ|aw|`）、**confidence 加权**、**advantage 归一化**之后，实际影响份额可能与配置比例差一个数量级。

**永远看 `attribute` 的影响份额，不要看配置值。**

### 7.3 其他易误读指标

| 指标 | 误读风险 |
|---|---|
| `uncertainty` vs `std_mean` | 前者是归一化不确定度，后者是 σ 均值，**不是一回事** |
| `eff_std_mean` vs `std_mean` | 比值 = 实际探索放大倍数。相等表示未放大 |
| `EV ≤ 0` | 当 `ret_std ≈ 0` 时 EV 天然低，**不是 bug**（目标本身无方差） |
| `epochs_done` | 恒等于配置值，**不携带早停信息**；要看 `actor_epochs_done` |
| `ratio_max` | 长尾正常，看 `ratio_mean` 更稳健 |
| 二次 hinge floor | `coef·relu(floor−U)²` 在接近目标时梯度趋零，U 通常**停在 floor 之下**。用 floor 值本身做阶段切换阈值会永不触发——必须解耦目标与阈值 |

---

## 8. 本文档对实现提出的要求

本文档描述目标形态。为了让上述每一条都能真正成立，实现必须满足：

### 8.1 数据模型

1. **帧溯源（provenance）**——阻塞级。`frame`、`chain` 的分位数、`--where` 筛选、以及框架/实验/episode 三侧数组的 join，都要求能把平坦 buffer 索引映射回 `(episode, agent, t)`。`build_trajectories` 是实验拥有的多对多映射，框架**原理上无法推断**，因此溯源信息必须由实验提供。
2. **终止原因**要随轨迹保留，而不是只留一个 `bool`。
3. **实验中间量**需要一个导出通道（相位掩码、状态机分支等），且与轨迹时间基准明确对齐。

### 8.2 可重算性

4. **逐帧 `V(s)`、advantage、归一化后权重、逐通道贡献**必须能被导出，且导出路径要**复用生产代码**而非复制 GAE 逻辑。
5. **θ_old 与 RNG 状态**必须随快照保存，否则 `whatif` 的基线无法与训练日志对齐，工具不可信。
6. **自校验**：从快照重算出的聚合量必须与训练日志逐字段一致。不一致本身是必须先修的 bug。

### 8.3 新增可观测量

7. 逐帧 **L1 归一化后**的 actor_weight（当前只有归一化前）
8. **逐通道影响份额**
9. **无梯度帧占比**
10. **按动作维度的梯度范数**

### 8.4 声明式扩展点

11. **行为谓词**需要一套声明方式（探针 suite），且能从固定初始状态确定性回放。
12. **指标的严格定义**需要能与生产定义并存，供 `metric --verify` 对比。
13. **每个旋钮的通路实测**需要一个注册机制，让 `intervene-check` 能自动覆盖新旋钮而不是硬编码。

### 8.5 流程约束

14. 快照请求**必须携带 hypothesis** 才被接受。
15. 调试路径**不得改变训练动力学**——特别是不能消耗 RNG（否则观测行为会改变被观测对象）。
