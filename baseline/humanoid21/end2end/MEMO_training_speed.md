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

## 进展日志

- 基线/边界/指标/协议确立（见上）。
- 资源：s2 对比组三条 run 已停（结论已收），GPU 0/1/2 空出。
  GPU2/3 上有两个非主线 run 在跑（step resume、statesig）。
- **下一步**: gradsig 时间序列取证——verify_resume_A 全量 per-update
  gradsig 扫 u380 逃逸点前后的 coherence/‖G‖/proj 变化；
  对照 base_from0_s2（同臂未逃逸）隔离 seed 路径差异。
  产出逃逸点梯度画像 → 定首发臂。
