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
