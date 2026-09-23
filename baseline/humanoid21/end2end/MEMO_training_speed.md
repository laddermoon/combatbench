# 训练速度优化 Memo

> 主线任务：在 `standup_floor04` 基线上持续优化策略训练速度。
> 工作方式：**分析 → 提议 → 实现 → 训练验证 → 再分析** 的双重迭代——
> 既提升对训练动力学的理解，也迭代 Debug 系统本身。
> 本文档持续追加：每次尝试、证据、结论都记录在此。

## 任务定义

- **基线实验**: `baseline/experiments_ppo/exp_standup_floor04.py`
  （4 阶段 dense potential 站立；uncertainty_floor=0.4, coef=1.0）
- **参照 Run**: `verify_resume_A`（seed=42，u380 逃逸，1500 updates 跑满）
- **策略锁定**: `TruncatedNormalPolicy`（tanh_gaussian_mlp），不可更换。
  策略架构类实验（statesig/low-rank/MoG/RealNVP）不属于本主线。
- **Debug 系统**: `baseline/framework/ppo/dumpkit/` + viewer（localhost:8873 /
  180.76.152.227:8873）。所有分析走 dump/debug 数据流，不另建管线。

## 速度的度量（操作化指标）

| 指标 | 定义 | verify_resume_A 锚点 |
|---|---|---|
| 逃逸点 | eval.success ≥0.5 首次出现 | ~u380 |
| 收敛点 | eval.success ≥0.9 稳定 | ~u500 |
| 收敛后效率 | 早停率 / 有效 minibatch 占比 | 66–82% 早停（浪费） |

wall-clock 暂不作为主指标：rollout 占每 update 26.4s 中的 24.5s（93%），
update 数学仅 1.3s；除非改动影响 rollout 成本，优化空间在 sample
efficiency。episode 数改动除外——它直接压缩 rollout 壁钟。

## 杠杆池（已确认边界）

| 杠杆 | 参数 | 现状 | 备注 |
|---|---|---|---|
| 探索机制 | `uncertainty_floor`, `uncertainty_coef`, `explore_factor` | floor=0.4/coef=1.0；hinge 从 init U≈0.46 起咬合 | 本任务历史核心变量 |
| Episode 数 | `episodes_per_update` | 512→204K帧/update | 直接速度杠杆；与 epochs 耦合为"数据预算" |
| ADV 变换 | `adv_norm`, `adv_winsorize_sigma`, `dual_clip`, `actor_weight` | zscore 基线 | 帧级梯度方向结构 |
| KL/LR 预算 | `lr`, `critic_lr`, `target_kl`, `clip_eps`, early-stop 窗口 | lr=3e-4, tkl=0.05 | 收敛后早停浪费的主要回收口 |
| 数据复用 | `update_epochs`, `minibatch_size` | 4 epoch × 50 mb | 与 episode 数是同一数据预算的两维 |

**排除**: 策略架构替换。

## 已确立的事实（证据库）

1. **逃逸是离散事件**：平台期 max_pot ~0.66 停滞 → 某个 update 附近
   success 阶跃。非渐进爬升。
2. **Seed 方差压过臂间差异（逃逸序）**：
   - s42/s1: base > winz4 > grank
   - s2: grank > winz4 > base（完全倒转）
   → "裁 adv 尾部放慢逃逸"不是稳健效应；单 seed A/B 不可信。
3. **收敛后早停差异跨 seed 一致**：
   - gaussrank: s42=23%, s1=0%，且梯度范数维持 base 的 2–3 倍
   - base: s42=82%, s1=66%
   → ADV 秩变换让 update 持续"满负荷"，但学的是不是有用信号待定。
4. **平台期不撞 KL**：三组对比 run 逃逸前早停率均 ~1%。
   早停纯粹是收敛后现象。
5. **梯度相干度极低**：u452 截面 coherence=0.008——204K 帧逐帧
   梯度互相抵消，聚合 ‖G‖=0.544 是 mean‖g‖≈71 的残差。
   加数据买的可能主要是噪声。
6. **三组 run 代码完整性已核实**：s2 组快照树一致、配置正确、
   启动后无运行时代码污染。

## 验证协议（吸取 s2 倒转教训）

- **初筛**: 候选臂跑 seed 42，对比 verify_resume_A 曲线。
  u500 仍未逃逸 → 杀（基线 u380，给 ~30% 余量）。
- **确认**: 初筛通过的臂补 seed {1,2}；3 seed 中 ≥2 一致改进才算数。
- 8 GPU 并行，初筛可同时铺 2–3 臂。
- 新 run 必须带 `--dump-at` 排程——逃逸是瞬态，稳态截面抓不到。

## Debug 系统已知缺口

- 逃逸瞬态未被抓到（现有 dump 全是稳态截面）
- 跨 update 演化视角缺失（dump 是单 update 截面；
  verify_resume_A 的 gradsig/u*.npz 旧格式有全量 per-update 数据可先用）
- 逃逸归因：trajectory 级 ADV 贡献视图（trace_frame 是帧级）

### 已补（本任务线内）

- Run 页信息架构修正：Run Info = 全量静态初始参数
  （common_params/ppo_params/reward_channels，被 override 的字段打 ▲）；
  Update Detail = 本 update 生效参数（调度值 stats.uncertainty_floor/
  coef/actor_lr/critic_lr + param_overrides），stats/pc/ep/time 数字表
  删除（与曲线重复）。初始+覆盖=有效参数的读法闭环。

## 分析 #1：逃逸点画像与假设清单（2026-09-22）

数据源：verify_resume_A（1500u 全量 gradsig + train.log）、三 seed × 三臂
对比组、base_from0_s2（同臂未逃逸对照）。

### 核心发现：逃逸不是相变，是阈值穿越

verify_resume_A 逃逸窗口 u350–u400 内，coherence/frac_neg/σ/KL/clip
全部连续无跳变。真实结构是：

- **fpm（final_potential_mean）从 u60 起以 ~2.5e-3/update 线性爬坡**，
  u340 达 ~0.85，u365–u400 穿越 0.9 阈值 → eval.success 阶跃。
- "逃逸快慢" = 爬坡斜率。速度优化 = 提高 d(fpm)/du。
- 基线爬坡期（u60–u365，~300 updates）占逃逸前时间的 ~95%。
  最后过顶只占 ~40 updates。

### 发现：KL 预算半闲置（最强信号）

- approach 期 kl_mean ≈ 0.025，target_kl = 0.05 → **利用率 ~50%**
- 早停率仅 2.8%（4 epoch × 50mb 全部跑完），epoch 间 KL 均匀分布
- 跨 run corr(kl_mean, fpm_slope) = **0.53**；同臂内也成立：
  base_s2 kl_util 48%/slope 1.84 vs verify kl 56%/slope 2.51
- 弱旁证：standup_tune 系 lr=2e-4+tkl=0.03 → esc u720（更慢的爬坡）；
  lr=1e-3+tkl=0.03+ue=6 → u28 崩（过猛）

### 发现：收敛后早停浪费

post-escape 早停率 86%（s42）、66–82%（s1）。逃逸后才撞 KL cap。
对"逃逸时间"指标无影响，对"收敛时间"是主要浪费源。

### 发现：ADV 信号先天稀薄

raw adv_std ≈ 0.008（dense potential 单帧差异小），z-score 放大到
±19 → 近均值帧符号翻转 → coherence ≈ 0.008（帧级梯度大面积互抵）。
这是所有"信号结构"类杠杆的理论基础。

### 发现：探索量单调缓降

std 0.368→0.27、U 0.456→0.31（floor=0.4 下 hinge 全程咬合，
floor_loss>0）。grank 臂 std 更低（0.243），逃逸 u295 时 std=0.222。

### 假设清单（按证据强度排序）

**A. KL 预算类 —— 每 update 位移量不足**
| 臂 | 参数 | 机制 | 成本 |
|---|---|---|---|
| A1 epochs↑ | update_epochs 4→8 | 同数据多榨更新，早停自限在 cap | ppo 1.3s→~2.6s，几乎免费 |
| A2 lr↑ | lr 3e-4→5~6e-4 | 直接放大步长 | 稳定性风险 |
| A3 adaptive lr | exp_standup_floor04_lr 已有 | KL 欠填→涨，早停→降 | 已写好 |
| A4 clip_eps↑ | 0.2→0.3 | clip_frac 32% 偏高，放行大 ratio 帧 | 免费 |

**B. 数据预算类**
| 臂 | 参数 | 检验什么 |
|---|---|---|
| B1 episodes↓ | 512→256（+epochs 4→8 保优化步数） | 逃逸是"发现驱动"还是"精度驱动"：若 slope 掉 <50% 则净赚 ~1.7× update 率 |
| B2 episodes↑ | 512→1024 | 若 cancellation 需更多数据对冲（贵，2× rollout） |

**C. 探索机制类**
| 臂 | 参数 | 检验什么 |
|---|---|---|
| C1 floor schedule | floor 0.4→0.25 @u250（--param 免改码） | 前期保探索后期要精度 |
| C2 explore_factor | 0→0.1 | 纯噪声探索是否助发现 |

**D. ADV 结构类**
| 臂 | 参数 | 检验什么 |
|---|---|---|
| D1 advstd | adv_norm=std（exp 已存在） | 去中心化保符号，减 cancellation |
| D2 γ/λ | γ0.99→0.98 / λ0.95→0.9 | 加厚局部 adv 信号 |

**E. 收敛后效率类**（不加速逃逸，加速精修）
| 臂 | 参数 |
|---|---|
| E1 | target_kl/lr 收敛后下调（A3 覆盖） |

### 待定首发臂建议

A1（epochs 8）或 A2（lr 5e-4）+ B1（eps256/ue8）+ C1（floor 排程）。
3 GPU 并行初筛，u500 未逃逸即杀。

## 进展日志

### 2026-09-22 初筛第一轮开跑

三臂 seed=42（与 verify_resume_A 同 seed 对照），GPU 0/1/4，
`--dump-at 200,300,400` 抓逃逸瞬态：

| run | GPU | 臂 | override（日志已确认生效） |
|---|---|---|---|
| `ue8_s42` | 0 | A1 | `update_epochs=8` |
| `eps256_s42` | 1 | B1 | `episodes_per_update=256, update_epochs=8`（数据减半、单帧消费 8 次，优化步数与基线持平） |
| `floorsched_s42` | 4 | C1 | ~~`uncertainty_floor=0.25@250`~~ **配置错误已修正** |

**事故记录**：`uncertainty_floor` 不是 CommonParams/PPOParams 字段——
它走 `experiment.exploration(update) → ExplorationSpec` 通道，`--param`
白名单里没有它。run 在 u250 补丁生效时抛 `ValueError` 崩溃。

**修正**：新建 `exp_standup_floor04_floorsched.py`，用 `exploration()`
钩子做排程（u<250: floor=0.4，u≥250: floor=0.25），从
`floorsched_s42/checkpoint_u00245.pt` 续跑为 `floorsched_s42_r245`。
u1–249 段本就是 floor=0.4（与基线逐位一致），续跑无损失。
教训：排程类改动先查字段归属——`--param` 只管 cp/pp 数据类字段。

早期观测：ue8 的 asteps=400（8×50）正常；eps256 每 update 102K 帧
（asteps=200=8×25mb）。u1 均有初始 KL 尖峰早停（基线同样现象，
非异常）。杀掉线：u500 未逃逸即停。预期逃逸点若有效 ~u200-300。

### 2026-09-22 第一轮结果：两臂已判死并停止

| 臂 | 逃逸点 | 壁钟到逃逸* | 判定 |
|---|---|---|---|
| `ue8_s42` (A1) | u345 vs 基线 u365 | ~167min vs ~161min（同期 ~29s vs ~26s/u） | **无增益**：update 数仅省 5%，单 update 成本 +10%；approach 早停率 3%→22%——多 epoch 只是更快撞满 KL cap，总位移不变 |
| `eps256_s42` (B1) | u735 vs 基线 u365 | ~189min vs ~161min | **负**：爬坡斜率掉到 1/3（0.87e-3），update 数翻倍不止。逃逸是信号/精度驱动，不是发现驱动——数据不能砍 |

\* 壁钟跨时期不可比（verify_resume_A 历史日志显示 10.5s/u，
当时机器负载/rollout worker 不同）；以 update 数为主指标。
| `floorsched_s42_r245` (C1) | u370 vs 基线 u365 | — | **无效**：floor 如期在 u250 降（floor_loss→0），但 post-drop slope 1.06e-3 ≈ 基线同窗口 1.08e-3，逃逸点在噪声内。σ 下压力不是爬坡约束 |

**修正"KL 预算"假设**：kl_util 低不是因为优化步数不够，而是
KL cap 本身就限制每 update 位移。ue8 只是把同一份数据消费得
更狠（早停 22%），并没有让策略走得更远。真正要试的是
**提高 KL cap 本身**（target_kl↑）或 **lr↑**（在 cap 内走更大步）——
下一轮候选。

**新增事实**：数据量减半 → slope ×0.35（非线性惩罚），
说明 204K 帧的梯度估计质量是爬坡的瓶颈之一，不是冗余。

### 2026-09-22 第二轮开跑（推断：梯度信号质量是瓶颈）

第一轮三个负结果排除了"位移量"和"探索强度"两类机制，
且数据量高度敏感 → 爬坡速率由"每 update 估计 φ 上升方向的精度"
决定。据此起两臂 + 一验证臂，全部 seed=42、`--dump-at 200,300,400`：

| run | GPU | 假设 |
|---|---|---|
| `eps768_s42` | 0 | **B2（正向推断）**：eps 512→768，若信号质量是瓶颈则 slope 应上升 |
| `tkl10_s42` | 0→（前次启动） | A2：`target_kl=0.10`——cap 本身是约束则 slope 上升 |
| `lr5e4_s42` | 1 | A3：`lr=5e-4`——若方向质量是瓶颈，大步长应无效或更差（对照验证） |

预期判读：
- eps768 slope >2.5e-3 且逃逸 <u365 → 信号质量假设坐实，方向是堆数据/堆精度
- lr5e4 逃逸 ≈或晚于基线 → 佐证"方向质量受限"而非步长受限
- tkl10 逃逸提前 → cap 约束假设仍有价值（与 ue8 结果张力，需细看）

杀掉线不变：u500 未逃逸即停。

### 2026-09-22 第二轮中段：tkl10 判死，advstd/clip03 开跑

中段对比（同 update 点 fpm vs 基线）：

| run | 读数 | 判定 |
|---|---|---|
| `eps768_s42` | u121 fpm=0.373 vs 基线 0.304；u61–121 斜率 ~3.1e-3 vs 基线 ~2.1e-3 | **正向兑现中**，预计 ~u290 逃逸 |
| `tkl10_s42` | u160 前微领先，之后持续掉队：u201 −0.06、u261 −0.11；斜率退化到 1.8e-3 | **已杀**——cap 翻倍反而更慢 |
| `lr5e4_s42` | 全程贴身，u281 落后 0.04；斜率 ~2.1e-3 | 预计 ~u370 逃逸，跑完取 lr 响应曲线第二点 |

**tkl10 解读（A2 判负）**：KL cap 翻倍后策略位移不受约束，但斜率反而退化——
大胆猜测：更大的单步位移增加了目标移动（moving target）噪声，
rollout 数据更快过时，等效于降低了有效信号质量。与 ue8 结果合并
后结论收敛：**KL cap 不是爬坡瓶颈，放开它反而有害**。A 族（KL 预算）
杠杆基本关闭——剩余只有收敛后效率问题（属次要指标）。

**新起两臂**（GPU3/4，seed=42，`--dump-at 200,300,400`）：

| run | 假设 |
|---|---|
| `advstd_s42` | D1：`adv_norm=std`——zscore 的批次均值中心化会把 ~45% 近零帧翻成负号，std 保留原始符号→正向推力更连贯 |
| `clip03_s42` | A4 残留：`clip_eps=0.3`——clip_frac 32% 偏高，放松让更多梯度信息通过（u2 已验证：kl 0.031、clip_frac 0.20） |

### 2026-09-22 中段监控 #2：lr5e4 判死，clip03 浮现正信号

| run | 读数 | 判定 |
|---|---|---|
| `lr5e4_s42` | u386 未逃逸（基线 u365），全程 Δfpm≈−0.04 | **已杀**——A3 判负：lr↑ 不加速反而微慢，佐证方向受限 |
| `eps768_s42` | Δ 从 u121 的 +0.084 缩到 u195 的 +0.015 | 早期优势蒸发中，跟踪到 ~u250 再判 |
| `clip03_s42` | Δ 单调爬升：u29 +0.04 → u87 **+0.100** | **首个正信号**，clip 放开让每 update 在 KL cap 内通过更多梯度信息 |
| `advstd_s42` | Δ≈0（u87 −0.012） | 中性，跟踪 |

**lr 响应曲线已有点**：lr 3e-4→u365，5e-4→~u400+（慢），1e-3→u28 崩。
lr 杠杆关闭——它动的是"步长"，而瓶颈在"方向质量"。

**新起 `clip04_s42`**（GPU1，`clip_eps=0.4`）：clip03 的剂量响应点——
若 0.4 比 0.3 更快则响应曲线单调，clip 杠杆可继续推进；若 0.4 更慢
则最优点在 0.3 附近。

### 2026-09-22 中段监控 #3：eps768 判死，机制合成浮现

| run | 读数 | 判定 |
|---|---|---|
| `eps768_s42` | Δ 从 u159 +0.095 衰减到 u210 **−0.008** | **已杀**——早期优势蒸发，且 1.5× 帧成本追平即亏 |
| `clip03_s42` | u108 Δ**+0.141** 仍加速 | 正信号最强，预计 ~u300-320 逃逸 |
| `advstd_s42` | u109 Δ−0.043 | 漂移向下，~u150 无起色即杀 |
| `clip04_s42` | u21 Δ+0.010 | 早期，等读数 |

**机制合成（重要）**：eps256（数据减半→斜率×0.35）与 eps768（数据
+50%→无效）构成**不对称响应**——瓶颈不是样本量而是**有效梯度吞吐**。
基线 clip_frac≈0.34（eps768 u211 实测），即 ~1/3 样本的梯度被 clip
置零。clip03 把 clip_frac 降到 ~0.20 后 Δ 单调爬升——放开截断等价于
每 update 多 ~15% 有效梯度，且不增加 rollout 成本。

**修正后的瓶颈模型**：爬坡速率 ∝ 每 update 有效（未被截断的）梯度
信息量。数据量和 clip_eps 是同一底层量的两个表面杠杆——这解释了
为什么 ue8/tkl10（只动位移预算）全部失败。

### 2026-09-22 中段监控 #4：advstd 判死，clip01 证伪臂开跑

| run | 读数 | 判定 |
|---|---|---|
| `advstd_s42` | Δ 单调恶化至 u121 −0.053 | **已杀**——D1 判负：保号归一化无增益，批次均值中心化不是损耗源 |
| `clip03_s42` | u120 Δ**+0.170** | 若斜率保持预计 ~u220-250 逃逸（vs 基线 u365） |
| `clip04_s42` | u34 +0.017 | 早期 |
| `clip01_s42` | GPU0 新起 | **证伪臂**：clip_eps=0.1（clip_frac 0.37、kl 0.007），吞吐模型预测它应**显著慢于**基线 |

clip 剂量曲线在成形：0.1（预测最慢）→ 0.2=基线 → 0.3（+0.17）→ 0.4（待读）。

### 2026-09-22 中段监控 #5：clip 剂量曲线完成，clip01/clipinf 判死

| clip_eps | 峰值 Δ | u | asteps | 判定 |
|---|---|---|---|---|
| 0.1 | −0.067 | u99 | 200（KL 用不满 0.006） | **已杀**：太紧→每步位移不足 |
| 0.2（基线） | 0 | — | ~200 | — |
| 0.3 | +0.170 → +0.033 | u221 | 27-57 | 领跑后 Δ 衰减中 |
| 0.4 | +0.138 → +0.084 | u138 | 17-20 | **当前最优**，衰减较慢 |
| 10（实质无截断） | −0.036 | u57 | 5-7 | **已杀**：步子过大，5 步穿 cap 浪费预算 |

**机制终版**：固定 KL 预算下，clip_eps 决定每 minibatch 的截断损耗。
甜点 0.3-0.4——太紧（0.1）KL 花不出去，太松（10）几步穿 cap。
clipinf 的 asteps=5 vs clip04 的 17：同样 ~0.05 KL，后者把预算摊在
更多步里→每步更接近局部线性区→位移质量更高。

**系统性观察**：所有臂的 Δfpm 在 u100-150 见顶后向基线收敛——
基线中段（u150-250）斜率本身最陡，早期增益被压缩。评估臂的真实
收益要看逃逸点，不看 Δ 峰值。

**确认 seed 已起**：`clip04_s1`（GPU0，seed=1）、`clip04_s2`（GPU3，
seed=2），均 `--seed N --param clip_eps=0.4`。注意 `--param seed=`
无效——`rollout_seed` 读 base `cp.seed`，必须走 `--seed` CLI。

**已杀汇总**：ue8、eps256、floorsched、tkl10、lr5e4、eps768、advstd、
clip01、clipinf（9 臂）。存活：clip03_s42、clip04_s42 + 确认 s1/s2。

### 2026-09-22 逃逸裁决：clip03 温和胜，clip04 在 s42 转负

| run | 结果 | 判定 |
|---|---|---|
| `clip03_s42` | **逃逸 u347**（基线 u365，−18u） | s42 上温和正收益，待 s1/s2 确认 |
| `clip04_s42` | u264 Δ−0.17 且加深 | **已杀**：0.4 在 s42 上明确差于 0.3，s42 剂量曲线非单调 |
| `clip04_s1` | u122 Δ+0.19 | s1 上强正——**跨 seed 效应不稳** |
| `clip04_s2` | u127 Δ+0.01 | s2 上近零 |

**Δ-衰减的教训**：所有臂 Δfpm 在 u100-150 见顶后衰减，中段 Δ 峰值
不能预测逃逸点（clip03 峰值 +0.17 → 逃逸只快 18u；clip04_s42 峰值
+0.14 → 最终转负）。裁决只看逃逸 update。

**clip04 的 seed 分化**（s42 −0.17 / s1 +0.19 / s2 +0.02）与之前三
seed 倒转同型——clip_eps 的收益大概率是 seed 路径依赖而非普适。
clip03 的 s1/s2 确认臂已起（GPU1/2），若 clip03 在三个 seed 上都
≥基线，则 +0.3 是稳健小改进；若也分化，则 clip 杠杆整体是噪声。

### 2026-09-22 分析 #2：Δ-衰减的机制 = KL 预算的相位饱和

基线 verify_resume_A 的时序数据：

| 相位 | u | clip_frac | kl_mean | 含义 |
|---|---|---|---|---|
| 早期 | 30-120 | 0.22→0.26 | 0.014→0.018 | KL 预算只用 ~30-40%，**有闲置预算可挖** |
| 中期 | 150-300 | 0.26→0.40 | 0.018→0.034 | 预算利用率升到 ~70%，收益空间收窄 |
| 逃逸后 | 360+ | 0.37-0.42 | 0.035-0.042 | 预算饱和 + 大量早停（asteps 20-200 抖动） |

**Δ-衰减的解释**：clip 放开只在"预算闲置"相位有增益——中期基线
自己就把 KL 吃到接近 cap，所有臂被同一预算封顶 → Δ 归零。这不是
臂退化，是**约束面合拢**。

**推论**：在 target_kl 固定的体制下，逃逸时间存在由任务几何决定
的下限——单参数调节最多压缩早期闲置段（~20-40u 量级）。要实质
提速需要改变"每单位 KL 换来的方向质量"，即探索/rollout 质量
（explore_factor、探索排程）或 reward 整形——这两类尚未验证
（floorsched 只测了 floor 常数下移，explore_factor 未测）。

**下一步方向候选**：
- explore_factor>0（需 experiment 钩子，不在 --param 白名单）
- clip_eps=0.3 若三 seed 都 ≥基线 → 收下 ~20u 的稳健改进作为新默认
- γ/λ 调整（reward channel 配置，需实验子类）

### 2026-09-22 中段监控 #6：同 seed 基线修正 + clip03 跨 seed 稳健

**方法修正**：此前 Δfpm 全部对 s42 基线（verify_resume_A）比较——
跨 seed 对比有系统性偏差（各 seed 的基线斜率不同）。启用 per-seed
基线：s42→verify_resume_A，s1→base_from0_s1（esc u343），
s2→base_from0_s2（u342 未逃逸，fpm 仅 0.66——"慢 seed"）。

同 seed 重算后画面变化很大：

| run | 同 seed Δ | 结论 |
|---|---|---|
| clip03_s42 | esc u347（−18u） | ✅ s42 正 |
| clip03_s1 | +0.118 @u103 | 进行中，正 |
| clip03_s2 | +0.175 @u106 | 进行中，正——**若逃逸则翻盘最慢 seed** |
| clip04_s42 | −0.17 已杀 | 负 |
| clip04_s1 | −0.045 @u229 | 衰减到负 |
| clip04_s2 | −0.010 @u237 | 中性 |
| ef05_s42 | +0.048 @u73 | 早期正，跟踪 |

**clip04 同 seed 判定：≈中性/负**（跨 seed 的 +0.19 是 s1 基线慢的
假象）。**clip03（clip_eps=0.3）在三 seed 同 seed Δ 全正——目前是
唯一稳健改进候选**，幅度约 −20~−40u 逃逸。

clip03_s42 逃逸后数据已够（succ 0.93 @u439），已停收 GPU4。
clip04_s1/s2 同 seed Δ 回落到 −0.045/−0.01——三 seed 全非正，
**clip_eps=0.4 判死**（剂量峰值在 0.3 附近），两臂已停。

**新起 `mb2048_s42`**（GPU0，`minibatch_size=2048`）：同一 KL 预算
摊到 400 asteps（vs 基线 200）——检验"步数粒度×每步精度"权衡，
吞吐模型预测中性或微负（更小的批→更噪的单步梯度）。

**已起 `ef05_s42`**（GPU5，`standup_floor04_ef`，e=0.5 → rollout
σ×1.73）：探索宽度杠杆的首次测试。机制：explore_factor 只放大
采样 σ（训练 σ 不变，重要性比率已含 ei 修正）。已验证生效：
u2 eff_std=0.637 vs std=0.368。若方向质量是瓶颈，更宽的探索应
换来更准的梯度方向；若探索稀释数据则变慢。

### 2026-09-22 里程碑：clip03_s2 在"死 seed"上逃逸

| seed | clip03 逃逸 | 同 seed 基线逃逸 | Δ |
|---|---|---|---|
| s42 | u347 | u365（verify_resume_A） | −18u |
| s1 | **u381**（2026-09-23 终局） | u343（base_from0_s1） | **+38u，更慢** |
| s2 | **u268** | **u342 未逃逸**（fpm 峰值 0.66） | **≥−74u，翻盘** |

**clip_eps=0.3 终局判定（修正）**：非均匀改进——s1 上反而慢 38u。
真实画像是**方差压缩器**：救坏 seed（s2 翻盘）但轻微伤好 seed
（s1 +38u、s42 −18u）。mean(Δ)≈−19u，但收益分布高度 seed 依赖。
作为"稳健性"改进有价值；作为"提速"改进不成立。
mb2048 中性已停（Δ≈0 全程——步数粒度不是杠杆）。

### 2026-09-23 ef05 逃逸终局：第二个稳健杠杆成立

| seed | ef05 逃逸 | 同 seed 基线逃逸 | Δ |
|---|---|---|---|
| s42 | **u335** | u365 | **−30u** |
| s1 | **u277** | u343 | **−66u** |
| s2 | 进行中 @u290 fpm 0.840 Δ+0.257 | u342 未逃逸 | 预计翻盘 |

**explore_factor=0.5 是第二个稳健改进，且比 clip03 更强**：
s42/s1 双双大幅提前，s2 正在改写另一个死 seed。机制：rollout
σ×1.73 提升梯度方向质量（方向质量假说再次兑现）。
clip03_s1（esc u381）与 ef05_s42（esc u335）均已收敛停掉。

**组合臂 `clip03ef05_s42` @u271 Δ+0.106**，轨迹指向 ~u290-310
逃逸——若显著早于 ef05 单臂的 u335 则加性成立。

### 2026-09-23 逃逸终局：ef05 三连确认 + 组合臂无加性

| 臂 | s42 | s1 | s2 | 判定 |
|---|---|---|---|---|
| 基线逃逸 | u365* | u343 | u342 停时 fpm=0.663（远未逃逸） | — |
| clip03 | u347 | u381（+38u） | u268 | 方差压缩器 |
| **ef05** | **u335（−30u）** | **u277（−66u）** | **u351**（基线 342 停时 fpm 仅 0.663→翻盘） | **三 seed 全胜，最强杠杆** |
| clip03+ef05 | **u345** | — | — | ≈clip03 单臂，**无加性** |

\* verify_resume_A 按 online_success≥0.5 判 u387；全表同口径。

**ef05 判定：第一个均匀稳健改进**。机制再次指向方向质量——
rollout σ×1.73 让同样 KL 预算买到更准的上升方向。

**组合臂教训**：clip03+ef05=u345 ≈ max(单臂)——两杠杆消耗同一
"每 update KL 位移预算"，不叠加。**后续单杠杆推进，不测组合。**

**ef05_s2 逃逸点细节**：u351 数字上晚于基线被停的 u342，但基线
停时 fpm 仅 0.663（同点 ef05 已 ~0.9），基线实际逃逸需 400+，
故仍为翻盘。

### 2026-09-23 ef 剂量扫描已起

`ef03_s42`（GPU3，σ×1.39）、`ef08_s42`（GPU4，σ×2.41）——
bracket ef=0.5 的最优点。新增实验文件 exp_standup_floor04_ef03/
ef08.py（EF 类属性子类化）。生效确认 eff_std=0.511/0.886。
ef05_s1/s2 已收敛停掉（esc u277/u351），GPU 5-7 + 2 空闲。

### 2026-09-23 第三方向：`gr_s42` adv_norm=gauss_rank（GPU5）

- 机制假设：advantage 分布尾部离群 episode 拉扯梯度方向；
  gauss_rank 把 adv 映射到分位数（有界、抗离群），若方向质量是
  瓶颈则应有效。与 ef（探索宽度）、clip（截断损耗）正交。
- 生效确认 `param_overrides={adv_norm: gauss_rank}`，adv_std 量级
  正常。对照同 seed：verify_resume_A。
- 判读预案：u100-150 Δss 持续正 → 方向质量假说再添证据；
  ≈0/负 → adv 分布形状不是瓶颈，关闭该分支。
- **已停**：发现 `gaussrank_from0` 就是同一臂（同 exp/seed42/同
  adv_norm=gauss_rank override；额外 overrides 仅 grad_sig 采样
  关闭+max_updates 日志项，无训练语义差异）。它已跑满 1500u：
  **逃逸 u532 vs 基线 u387 —— gauss_rank 显著更差**。winsorize4
  同批 u488 也差。adv 整形方向整体关闭（std 负、gauss_rank 负、
  winsorize 负）——zscore 已是最优。

### 2026-09-23 第三方向改为：`gc4_s42` grad_clip_norm=4.0（GPU5）

- 线索：基线 grad_norm_actor_mean 从 u1 的 1.4 一路涨到 approach
  期 3-4+（max 6.0），而 grad_clip_norm=1.0 —— **整个爬坡期每个
  minibatch 都被裁到 1/3-1/4**。Adam 尺度不变性下主要效应是 mb
  间权重压平：高梯度 mb 被压到与低梯度 mb 同权。
- 检验：若 clip 在压制高信号 mb 的有效权重 → gc4 加速；若 clip
  的方向平滑在承重 → gc4 变慢。这是与 ef/clip/adv 均正交的第四
  个机制族（梯度处理）。
- 生效确认 overrides={grad_clip_norm: 4.0}，u4 pre-clip norm 1.414
  （低于 4.0，早期步不再被裁）。

### 2026-09-23 逃逸终局：ef 剂量曲线非单调 + gc4 中性

| 臂 | 逃逸（s42） | vs 基线 u365/u387 | 判定 |
|---|---|---|---|
| ef03（σ×1.39） | u401 | 更慢 | 轻度放宽反而负（或噪声） |
| **ef05（σ×1.73）** | **u335** | **−30u~-52u，最优** | 剂量甜点 |
| ef08（σ×2.41） | u349 | −16u~-38u | 正但弱于 0.5 |
| gc4（grad_clip_norm=4.0） | u386 | ≈0 | **中性**——clip 压平不承重 |

**ef 剂量曲线（s42）**：0→365，0.3→401，0.5→335，0.8→349。
峰值在 ~0.5；ef03 为负是反直觉点（单 seed 噪声带 ±30u 内，
但方向性值得记下）。全部三臂已收敛停掉。

### 2026-09-23 两个排程臂已起

- `efdecay_s42`（GPU3，`standup_floor04_efdecay`）：ef 0.5→0 线性
  衰减至 u300。检验 ef 收益的时间定位——逃逸 <u335 → 后期宽探索
  是纯浪费；>u335 → ef 全程有贡献。生效：u9 eff_std=0.628。
- `tklearly_s42`（GPU4，`standup_floor04_tklearly`）：target_kl
  =0.10 仅 u≤150，之后回 0.05。直接攻击观测到的"早期 KL 预算
  闲置"（kl 0.016 vs cap 0.05）——tkl10 恒定失败点恰在自然饱和
  期（u160+），早窗增压是修正版。生效：overrides 已应用。

**`mb8192_s42` 补臂**（GPU5，`minibatch_size=8192`）：mb2048 判中
性（更多更噪的步不赢），本臂测反向——每 mb 用 2× 数据（更准的
方向估计）但每 update 最多 100 步 vs 基线 200。生效确认：
n_batches=25、asteps=100。判读：正 → 步质量>步数量；≈0 → 该
维度整体非杠杆；负 → 步数下限有硬约束。

### 2026-09-23 判死：tklearly / mb8192；efdecay 收尾中

| 臂 | @u | Δss | 判定 |
|---|---|---|---|
| tklearly_s42 | 315 | **−0.142** | **已杀**——早窗 KL 增压同样失败（与 tkl10 恒定版一致）："早期 KL 闲置"不是可被直接兑现的预算，加压即劣化 |
| mb8192_s42 | 308 | **−0.151** | **已杀**——步质量换步数量净亏：minibatch 维度两端测完（2048 中性、8192 负），4096 已是最优，**该维度关闭** |
| efdecay_s42 | 324 | +0.020 | 收尾中：u300 后 ef=0，轨迹回到基线附近——初判 ef 收益不在纯早期，衰减版丢失大部分增益（ef05 恒定 u335） |

**"KL 预算闲置"理论修正**：kl 0.016 vs cap 0.05 的闲置不是浪费
而是**优化状态的自然结果**——早期 policy 位移天然小，强塞更多
KL（tklearly）与恒定放大（tkl10）一样劣化。KL cap 维度整体关闭：
0.03/0.05/0.10 三个点 + 早窗排程全部非正。

### 2026-09-23 新一轮双臂：探索孪生 + lr 下行括线

| run | GPU | 设定 | 假设 |
|---|---|---|---|
| `ufloor05_s42` | 4 | `uncertainty_floor=0.5`（子类，类属性） | ef 的孪生杠杆：ef 只宽 rollout σ，floor 顶训练侧熵。若也正 → 探索机制两侧都活；若无效/负 → 收益专属采样侧 |
| `lr15e4_s42` | 5 | `learning_rate=1.5e-4` | lr 只测过上行（负）；下行括响应曲线。也负 → 3e-4 已最优（维度关闭）；正 → 小步更准成立 |

生效确认：ufloor05 stats.floor=0.5；lr15e4 stats.actor_lr=1.5e-4。
后备池：dual_clip_c=3.0、--no-confidence 消融、GAE λ（需通道子
类）、early_stop_kl_window（次要）。

### Debug 系统迭代：grad_clip_frac 指标落地

发现 grad clip 观测缺口（只有 pre-clip norm，无触发率）→ 新增
`stats.grad_clip_frac`（actor）+ `grad_clip_frac_<ch>`（critic），
viewer Actor Gradient Norm 图 + Critic 健康组登记，GUIDE 字段清单
同步。注意：老 run（含 gc4/ef03/ef08）不回溯，新 run 才发。

ef05_s42 同 seed Δ +0.15-0.22 @u230 区间，轨迹指向 ~u300-330 逃逸，
若成立则探索宽度是第二个稳健杠杆（且与 clip 机制正交——一个提
方向质量，一个降截断损耗，理论上可叠加）。

### 2026-09-22 ef05 确认 seed 已起 + clip03_s1 转弱

- `ef05_s1`（GPU3，seed=1）、`ef05_s2`（GPU4，seed=2）已启动，
  eff_std=0.637 生效确认。判据同 clip03：同 seed 逃逸点。
- `clip03_s1` @u285 Δss=−0.053（fpm 0.757）——在 s1 上已落后同
  seed 基线，预计逃逸 ≈或晚于 u343，确认 clip03 收益集中在坏
  seed（s2 翻盘、s42 −18u、s1 ≈中性/微负）——**即 clip03 的真实
  价值是压缩最坏情形，不是均匀加速**。
- `ef05_s42` @u244 Δss=+0.135、fpm 0.809，指向 ~u300-320 逃逸
  （vs 基线 u365）。
- clip03_s2 逃逸后 succ 0.674→0.82 已收敛，已停收 GPU2。
- **新起 `clip03ef05_s42`**（GPU2）：`standup_floor04_ef` +
  `clip_eps=0.3`，两杠杆叠加（探索宽度 × 低截断损耗，机制正交）。
  生效确认：eff_std=0.636 + overrides clip_eps=0.3。判读：组合逃逸
  若 ≪ min(clip03 u347, ef05 ~u310) → 加性成立；≈clip03 → 相互
  干扰。ef05 未单独确认完毕，s42 上两者均正故组合可归因。

## 进展日志

- 基线/边界/指标/协议确立（见上）。
- 资源：s2 对比组三条 run 已停（结论已收），GPU 0/1/2 空出。
  GPU2/3 上有两个非主线 run 在跑（step resume、statesig）。
- **下一步**: gradsig 时间序列取证——verify_resume_A 全量 per-update
  gradsig 扫 u380 逃逸点前后的 coherence/‖G‖/proj 变化；
  对照 base_from0_s2（同臂未逃逸）隔离 seed 路径差异。
  产出逃逸点梯度画像 → 定首发臂。
