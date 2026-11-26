# DynaBPTT 通用训练框架 — Balance 实施方案

> 状态：已定稿（2026-11-26 用户拍板），本文档是唯一权威设计。
> 框架形式化见 `bootstrip.md`；本文件是 Balance 任务的实例化决策集。

## 框架骨架（S0-S6）

```
概念：Env / WM / StartSet / SubStartSet / ReplayBuffer / NStep
每 Update：
  S0 更新 StartSet（实验自维护规则）
  S1 采样 SubStartSet（1024 个）
  S2 Rollout：Env + SubStartSet + Policy(确定性+高斯噪声) → D 入 ReplayBuffer
  S3 更新 WM（ReplayBuffer）+ 门控判定
  S4 Imagine：WM + Policy + SubStartSet（与 Rollout 同集）→ I
  S5 Loss(I)（含 critic 拟合）
  S6 更新 Policy（BPTT 直传）
```

## Balance 实例化决策（全部已定）

### A. 奖励与终止
- 想象奖励 `r_t = φ_t`（解析：uprightness×z/standing_height，可微，恒正）
- **不用** PPO 通道奖励、不用 r_cross、无终点惩罚——φ≥0 恒正保证
  存活单调占优，排序反转从根上消失
- 软终止塑形：analytic `d̂ = sigmoid((0.45−φ)/0.05)`，存活权重 c_t 连乘
  （可选机制，默认开）

### B. StartSet 机制（框架核心）
- **严格波前推进**：init={standup}（φ=1）；每 update 从 SubStartSet
  每种子 rollout 8 步，**只把第 1 步后继状态 s₁ 入集**；后 7 步只作
  WM 拟合数据 + 配对校验数据 + 种子可行性判定
- 死轨迹照样产生 s₁（φ 低→权重低→自然边缘化，无准入过滤）
- **采样权重**：`w_i ∝ φ_i^6`，P = 0.85·加权 + 0.15·均匀地板
  （κ=6：φ0.9 对 0.7 是 4.5×、对 0.5 是 34×、对 0.3 是 735×；
  均匀地板保证边缘态偶尔被采——恢复学习数据的唯一来源）
- 入集去重：rep 空间最近邻距离 <ε 跳过（ε 初值 0.1，可调）
- 容量 cap=4096，超出按 φ 升序淘汰（死区先出）
- **SubStartSet=1024**，Imagine 与 Rollout 共用同一子集

### C. 步数与噪声
- NStep=8（rollout/imagine 同长，配对校验天然可得）
- 策略语义：确定性 π(s) + 外加高斯噪声 σ=0.15（固定，不学、不进
  BPTT 图）；**想象用确定性动作**（官方语义）；obs 加官方式均匀抖动；
  feet 通道想象时置零

### D. 门控（硬性）
| 信号 | 阈值 | 不达时动作 |
|---|---|---|
| 配对 φ-MAE（同种子 8 步真/幻轨迹） | <0.08 | 只采集不更新策略 |
| (s,a) 覆盖率（策略动作落在种子动作云 ±2σ 内比例） | >60% | 同上 + 告警归因 |
| 留出集状态 NRMSE h8 | <0.15 | 同上 |
- 不迁就现状：达不到就持续采集拟合直到达标，全程记录

### E. critic
- RepCritic(rep+aux)，在**真实分支**上拟合 GAE（同种子真实 rollout
  的 φ 奖励 + done）——接地 bootstrap，配对分支白送
- bootstrap：J 末端 γ⁸c₈V(ŝ₈)

### F. 诊断（生产级）
每 update 输出 JSONL 健康报告，关键信号：
- wm_*：拟合损失分通道、留出 NRMSE
- paired_*：配对 φ-MAE、adv_corr
- coverage：动作云覆盖率、离群种子 dump（归因指针）
- startset：集大小、φ 分布、前沿外扩速率、淘汰数
- honest：想象 J vs 真实回报相关（滚动窗）
- gate：降级状态机日志

## 模块划分（dynabptt/core/）

```
startset.py   StartSetManager：入集(仅s₁)/去重/淘汰/φ加权采样；覆盖统计
collect.py    分支 rollout：set_core_state 注入（原生支持，
              validation.py:_paired_branch_adv 已验证此路径）
model.py      复用 hybrid_ppo.SelfWorldModel（GenCoord Δv+解析积分，
              1024³）+ fit_model
imagine.py    H=8 想象：φ 奖励+obs 抖动+feet 置零+analytic done+确定性动作
critic.py     RepCritic + real-branch GAE 拟合
gate.py       三信号硬门控 + 降级状态机
diag.py       JSONL 健康报告 + 归因标签
train.py      S0-S6 编排器
```

## 与原版 SuperMimic 的关系

框架是泛化：原版 = 本框架的 tracking 实例（StartSet=clip 帧回报自适应
采样器；NStep 解耦 rollout 2.5/imagine 32）。本实施与其差异：φ 奖励、
无 goal、短 H、硬门控——均记录在 `SUPERMIMIC_BOUNDARY.md` T9-T11
的边界分析中。

## 运行参数（首发）
```
workers=96, episodes 以分支计（1024 种子×8 步=8192 env步/update）
model: hidden 1024³, fit 40批×256窗 k=8 / update
imagine: 1024 种子 × H=8，actor 4 梯度步/update
critic: real-branch GAE 4 批/update
容量：replay ~4000 轨迹段；startset cap 4096
```

---

## 实验教训附录（2025-01 core 系列实证）

### L1 视界-难度联合陷阱（Horizon-Difficulty Trap）

若 NStep 太短，失败动力学在时间尺度上进不了窗口，则连真实 rollout
都表现不出任何信号（v2@H8 实测：surv8=1.00，actor 梯度归零 0.0000，
策略完全冻结但系统自洽地"健康"）。平衡类任务失稳时间尺度 ~20-50
控制步，H 必须覆盖它。实测闭环 phi-MAE：h16=0.021 h24=0.028
h32=0.067——在分布内 WM 完全撑得住 32 步，"WM 撑不住长视界"须以
实测为准而非先验。

### L2 评估回声室（Evaluation Echo Chamber）——比模型不诚实更隐蔽

StartSet 的准入-采样循环是自选择回路：只有当前策略能存活的状态才
能产出后继入集，分支 eval 又在这批状态上测量。训测同源，越训练
eval 越虚高。实证：v4 分支 eval 显示 0.90 存活，自然初始部署实测
仅 59 步（1.2s）。**唯二的诚实基准**：
1. 自然初始状态的完整 episode 部署测试（不参与训练的状态）
2. 配对闭环 fidelity（同种子同策略，真/幻 phi 轨迹对比）

所有 run 必须带部署 eval，分支内 eval 仅作参考。

### L3 critic 的真角色与 inflation 陷阱

Dyna-BPTT 的 critic 不是压方差（那是 PPO），是**视界补丁**：
γ^H·V(s_H) 把窗口外未来压缩进边界。但短分支数据若不含真实死亡，
GAE 目标 ≈ γ^H·V(s_H) 自我引用 → V 无约束上漂（v1 实测 J 飙至 45
而真实存活反降）。接地要求：终止分支的纯 MC return-to-go 目标 +
终止样本超采样。H=32 后窗口内已有真实信号，critic 必要性下降，
v4/v5 消融裁决中。

### L4 StartSet 前沿坍缩

严格首步准入 + phi^6 加权 + 低 phi 淘汰三力合一，使 StartSet 向
phi≈1 简单态坍缩（实测 phi10 0.91→0.99），前沿不扩反缩。对策实验
（v6/v7）：稀疏中间帧准入（phi∈[0.45,0.98] 且分支续存 4 步的
"已验证可救"硬状态）+ 随机淘汰保分布 + cap 16384。首测 phi10
降至 0.51，覆盖真实困难态。

### L5 噪声-敏感度地板（Noise-Sensitivity Floor）与开环 brace 陷阱

平衡任务的核心技能是"对小倾斜信号的精细反馈修正"。实测 strict 组
收敛出的策略是输入不敏感的开环抱死姿态：corr(dφ,da)≈0.04，obs 扰动
+0.1 仅改变动作 0.002，多关节饱和。机制推断：想象中逐步注入的 obs
噪声使"输入敏感"本身被惩罚（敏感→噪声放大成动作抖动→抖动破坏平衡），
噪声量级设定敏感度地板。SuperMimic 官方用同种噪声无碍，因为 tracking
参考信号幅度大；平衡类精细控制任务会被淹死。对策方向：噪声消融实验
（v10）+ 硬状态覆盖（粗反馈先行建立，细反馈随后跟进）+ 部署锚点。

### L6 部署锚点（Deployment Anchor）

自然初始状态是零速度测度为零的角落，噪声扩散的 StartSet 永远覆盖不到
它——但它恰恰是部署状态本身。实测想象从该状态出发能预测摔倒全程，
但它在 16k 状态中被采样概率≈0。修复：每轮 SubStartSet 强制保留名额
（anchor-slots，默认 32/1024）给自然初始态；其分支中间帧经前沿准入
把真实部署轨迹持续注入 StartSet。v8/v9 实测部署存活 84/93 步 vs
strict 组停滞在 55-65（u200 仍无进展）。
