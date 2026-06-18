# CombatBench 论文框架

> 本文件是论文的结构化骨架，用于指导后续写作。每个章节标注了**核心论点**、**要覆盖的内容**、以及对应的项目素材路径。正文撰写时再填充学术化语言。

---

## 拟定标题

**主标题候选：**
- CombatBench: An Adversarial Humanoid Benchmark with AI-in-the-Loop Training
- CombatBench: Benchmark and Closed-Loop AI Training Methodology for Humanoid Combat

**论文定位：** 双贡献论文
1. **Benchmark 贡献**：双人 humanoid 对抗的仿真平台 + 框架 + baseline
2. **方法论贡献**：AI-in-the-loop 强化学习训练方法论（RL 训练的 TDD）

---

## Abstract

**核心三段式：**

1. **问题**：双人对抗的 humanoid 控制兼具平衡、反应、策略博弈，研究价值高；但现有 benchmark 多为单 agent / manipulation / 导航，缺乏连续物理对抗场景。
2. **我们做了什么**：
   - 提出 CombatBench：HP-based 双人 21-DOF humanoid 对抗 benchmark
   - 提供可扩展的 plugin-based 框架（World Plugin / Observer plugin 分离，Blueprint 序列化）
   - 给出 staged curriculum + safety gating 的 baseline
   - 提出 AI-in-the-loop 训练方法论：把"什么是健康训练"编码为可观测指标，让 AI 监控-诊断-修复闭环
3. **结果**：[TBD - baseline 训练曲线、生存率、对战胜率、GPU 算力成本]

---

## 1. Introduction

### 1.1 Motivation

**论点链：**
- 双人 humanoid 对抗是 RL 的硬任务：高自由度（21-DOF）+ 长 horizon + 对抗的非平稳性 + 稀疏奖励（HP 触发）
- 现有 benchmark 的空白：
  - MuJoCo / DM Control Suite / IsaacGym → 单 agent locomotion
  - RoboSuite → 单 agent manipulation
  - AlphaGo / AlphaStar / OpenAI Five → 对抗但非物理连续控制
- 仿真是研究该问题的最佳载体：可尝试摔倒、可注入扰动、可绝对公平
- 这类任务**不依赖海量算力**，策略精巧度 > 资源规模，降低研究门槛

**素材：** `docs/RULE.md`、`docs/ENVIRONMENT.md`、`whitepaper.md` 的"动机"段

### 1.2 Challenges（论文要解决的难题）

- C1. 高难度 humanoid 任务的训练稳定性（直接训练 fight 会崩）
- C2. Benchmark 需要兼顾公平性、可扩展性、可复现性
- C3. RL 训练高度依赖专家经验，非专家难以调参——这是阻碍研究多样化的隐性门槛

### 1.3 Contributions（明确编号列出）

- **C1. CombatBench benchmark**：HP-based 双人 humanoid 对抗任务 + 规则集 + 评测协议
- **C2. Humanoid21 仿真环境**：96-dim 观测、21-dim 动作、500Hz 物理、严格对称
- **C3. 可扩展框架**：分层架构 + World/Observer plugin 分离 + Blueprint 序列化 + 多后端抽象
- **C4. Staged curriculum baseline with safety gating**：4 阶段课程 + 自监督安全门 + 8 reward 模块
- **C5. AI-in-the-loop 训练方法论**：RL 训练的 TDD——指标先行 + 机器可读日志 + AI 闭环监控-诊断-修复
- **C6. 公开榜单与开源实现**

### 1.4 Paper Organization

一段话交代后续结构。

---

## 2. Related Work

> **不能留空**。Related Work 定义 novelty 的坐标系。

### 2.1 RL Benchmarks

- MuJoCo continuous control (Duan et al. 2016, Rajeswaran et al. 2017)
- DM Control Suite (Tassa et al. 2018)
- IsaacGym / IsaacLab (Makoviychuk 2021, Mittal 2023)
- RoboSuite (Zhu et al. 2020)
- ProcGen, Atari/ALE
- **对照点**：均为单 agent；缺乏 competitive continuous-control 对抗场景

### 2.2 Competitive & Adversarial RL

- AlphaGo / AlphaZero (Silver et al.)
- OpenAI Five, AlphaStar
- Self-play in continuous control（Bansal et al. emergent complexity）
- MAS multi-agent benchmarks（MPE, StarCraft Multi-Agent Challenge）
- **对照点**：多为棋盘 / 离散游戏；我们的连续物理对抗 + 高自由度 humanoid 是空白

### 2.3 Humanoid Control

- DeepMimic (Peng et al. 2018)
- AMP / ASE (Peng et al. 2021/2022)
- 全身运动 / 平衡控制（Yin et al., Liu et al.）
- **对照点**：聚焦 locomotion / 模仿 / 平衡，非对抗

### 2.4 ML Observability & Automated Training

- AutoML、HPO（Hyperband, BOHB）
- ML 可观测性工具（MLflow, W&B, TensorBoard）
- LLM-based agents for code/debugging（SWE-agent, Devin 等）
- **对照点**：现有工具把人留在循环里；我们把 AI 升格为**主动诊断者与修复者**，并把"训练健康标准"显式编码为可观测契约

---

## 3. CombatBench Task Definition

> **规则在 framework 之前**：先告诉读者"打什么"，再讲"怎么实现"。

### 3.1 Combat Rules

**素材：** `docs/RULE.md`

- HP-based：每方 100 HP，归零即 KO
- 6 局 × 30 秒，每局重置
- 有效打击：8 个攻击部位（手/前臂/肘/上臂/足/小腿/膝/大腿）→ 2 个目标部位（头 -3 HP，躯干 -1 HP）
- 物理条件：相对速度阈值 + 非连续接触
- 决策频率 20Hz，物理 500Hz

### 3.2 Design Philosophy: HP-only

**论点（把 RULE.md 的设计哲学学术化）：**
- 为什么 HP 而非判分：**减少主观性**，HP 是唯一可量化、可复现的胜负判据
- 为什么无犯规限制：**最大化策略空间**，让 AI 自由演化最优策略，不强行模仿人类武术
- 公平性 formaly 定义：物理参数严格对称、固定步长、确定性 reset

### 3.3 Evaluation Protocol

- Elo rating（用于公开榜单）
- 6 局 head-to-head 胜率
- 评测是确定性的（给定 seed 可复现）

---

## 4. Environment Design

### 4.1 Humanoid21 Robot

**素材：** `envs/humanoid21/simulator.py`、`docs/ROBOT.md`、`envs/humanoid21/CONTROLSPEC.md`

- 21 DOF：3 腹部 + 每腿 6 + 每臂 3
- PD 控制，固定 KP（150-1000）/ KD（10-100）
- 归一化位置控制 [-1, 1]

### 4.2 Observation Space (96-dim)

**素材：** `envs/humanoid21/OBSERVATION_zh.md`

| 模块 | 维度 | 内容 |
|------|------|------|
| Proprioception | 42 | joint_pos_norm(21) + joint_vel_norm(21) |
| Root state | 13 | height(1) + local_orientation(6) + local_lin_vel(3) + local_ang_vel(3) |
| Tactile | 2 | 双足力传感器 |
| Opponent | 39 | root(7) + keypoints(18) + keypoint_vel(14) |

**设计论证：**
- 为什么 96 维：屏蔽全局坐标，强化本地坐标系，避免策略过拟合到绝对位置
- 对手信息：keypoint 表示而非全自由度，平衡信息量与泛化

### 4.3 Action Space

- 归一化关节位置目标（绝对位置，非增量）
- 20Hz 决策，25 个物理步 / 决策
- 选用绝对位置的论证：便于初始化、便于跨策略迁移

### 4.4 Arena & Physics

**素材：** `docs/ENVIRONMENT.md`、`assets/battle_v1.xml`

- 6.10m 标准拳击台
- 500Hz 物理，严格对称
- 9 机位相机

### 4.5 Disturbance Interface（可挖掘点）

**素材：** `envs/humanoid21/disturbance_plugins.py`

- ContinuousWind / InstantPush / InitialStatePerturbation
- 论证：平台支撑**鲁棒性研究**（safe RL、sim-to-real）

---

## 5. Framework Architecture

> 这一章把工程抽象包装为**设计贡献**，不是代码说明。

### 5.1 Design Goals

- 可扩展（extensible）：新机器人 / 新规则 / 新奖励
- 可复现（reproducible）：环境完整序列化
- 后端无关（backend-agnostic）：MuJoCo / PyBullet / IsaacGym

### 5.2 Layered Architecture

**素材：** `envs/framework/DESIGN.md`

```
Algorithm/Adapter Layer
Policy Runtime Layer (EnvRuntime + ObserverPlugins)
Physical Sandbox Layer (_RuntimeCore + WorldPlugins)
Backend Layer (BaseSimulator)
```

### 5.3 Capability-based Accessor/Mutator

**论点：** 能力分离 = 安全性 + 可组合性
- 读权限 always available
- 写权限 selectively granted（require_mutator flag）
- 对照：比直接暴露 simulator handle 的开源环境更安全

### 5.4 World Plugin vs Observer Plugin

**核心论点：这是对 RL 研究最重要的设计**
- World plugin（写）：pre/post hooks 改物理
- Observer plugin（只读）：算 observation / reward
- **分离的好处**：研究人员可以**换 reward 不动物理**，对 reward shaping 研究极友好

**素材：** `envs/framework/plugin.py`、`envs/framework/runtime_plugin.py`

### 5.5 Blueprint: Serializable Environment

**素材：** `envs/framework/blueprint.py`、`envs/humanoid21/blueprint.yaml`

**论点：**
- 完整环境 = 一个 YAML
- 可复现实验的基础设施
- 对照 ML 社区对可复现性危机的关注

### 5.6 Extensibility Evidence

**素材：** `envs/t800/`

- T800 已有完整 mesh + URDF + XML + simulator + plugins
- 用作 existence proof：框架支持多个不同形态机器人
- 这是**benchmark 扩展性**的最强论证

---

## 6. Baseline: Staged Curriculum with Safety Gating

> 把 baseline 从"参考实现"升格为**方法论贡献**。

### 6.1 Why Curriculum?

- 直接训练 fight 任务不稳定（长 horizon + 稀疏 HP 奖励）
- 分解为可学习的子技能金字塔

### 6.2 Four-Stage Curriculum

**素材：** `baseline/humanoid21/curriculum/experiments/`

| Stage | 目标 | 关键挑战 |
|-------|------|----------|
| 1. Basic balance | 站立 | 关节协调 |
| 2. Balance recovery | 抗扰动恢复 | 初始状态扰动 |
| 2+. Recovery plus | 强扰动 | progressive difficulty |
| 3. Follow | 跟随对手 | 平衡 + 移动 |
| 4. Fight | 完整对战 | 攻防权衡 |

### 6.3 Safety Gating Network（独立亮点）

**素材：** `baseline/humanoid21/curriculum/train_gating_network.py`、`mixed_policy.py`

- **动机**：高难度阶段探索黑洞，rollout 全是失败轨迹
- **方法**：自监督训练 gating classifier（recoverable vs not）
- **混合策略**：gate 置信度高时用主策略，否则回退到安全策略
- **数据收集**：weakened policy 生成覆盖难例的轨迹
- 论证：这是 stable high-difficulty humanoid training 的通用方法

### 6.4 Reward Engineering

**素材：** `baseline/humanoid21/rewards/`（8 个模块）

- cross_support、balance、damage、follow_opponent、opponent_relation、action_limit、posture、standing
- 课程相关的 reward 切换

### 6.5 Implementation

- 基于 `baseline/common/`（PPO / GAE / normalize / rollout / eval）
- PPO clipped surrogate，value clipping，entropy bonus

---

## 7. AI-in-the-Loop Training Methodology（新贡献）

> 这是论文的第二支柱。论点：把"什么是健康训练"显式编码为可观测契约，让 AI 成为训练的主动参与者。

### 7.1 Motivation

**痛点：**
- RL 训练高度依赖专家调参，"黑盒盲目调参"是常态
- 非专家无法判断训练是否朝着目标前进
- 这阻碍了研究多样化（只有调参老手能玩）

**机会：**
- 大模型具备通识 RL 知识（PPO 应该监控什么、什么是合理值）
- 训练日志可以结构化为机器可读
- AI 可以承担监控 + 诊断 + 修复的闭环

### 7.2 The TDD Analogy（核心 framing）

| 传统 TDD | 我们的 RL 训练 TDD |
|----------|---------------------|
| 先写测试（定义成功） | 先定义训练健康指标与合理区间 |
| 跑代码 | 跑训练，输出机器可读日志 |
| 测试 runner 报告失败 | 日志分析程序提取异常 |
| 开发者看 stack trace 修复 | AI 读原始日志 + 代码定位问题 |
| 重跑测试 | 调参 / 改 reward / 改课程，重训 |

**一句话总结：** 训练领域的 TDD = 先把"正常的、朝着目标前进的训练"长什么样定义下来，让 AI 不断监控这些指标并修复偏离。

### 7.3 The Closed-Loop Methodology

**6 步闭环：**

**Step 1 — Metric Definition（先验知识编码）**
- AI 先总结通用 PPO 训练方法论：应该监控什么、合理值是多少
- 输出：训练健康检查表（指标 + 物理含义 + 健康区间 + 红色警报）
- **素材：** `baseline/humanoid21/curriculum/OBSERVABILITY.md` 已经实现了这一步——
  - Rollout 子系统：episode 长度、生存率
  - Policy 子系统：entropy、std（含"标准差锁死"警报）
  - PPO Opt 子系统：epochs、kl_mean、kl_max（含"LR 过大早停"警报）
  - Critics 子系统：explained_var（含"价值网络破产"警报）

**Step 2 — Observable Training（机器可读日志）**
- 训练输出结构化 dashboard，**为程序消费而非人眼**
- 每 update 一组四行格式化指标
- **素材：** OBSERVABILITY.md 中的 dashboard 示例

**Step 3 — Automated Monitoring（程序提取）**
- 日志分析程序定期提取关键指标、识别趋势、触发告警
- **素材：** `analyze_logs.py`、`analyze_follow_logs.py`、`analyze_fight_logs.py`

**Step 4 — AI Diagnosis（AI 诊断）**
- AI 定期调用分析程序查看训练状态
- 异常时回溯原始日志，关联代码定位根因
- 复用 OBSERVABILITY.md 中的"三步排查法"作为诊断协议

**Step 5 — AI Remediation（AI 修复）**
- 学习率过大 → 降 lr
- KL 早停过快 → 降 lr / 调 target_kl
- Critic EV 为负 → 非对称 lr / 降 γ / 混合课程回退
- 探索黑洞 → 降课程难度 / mixed batch

**Step 6 — Retrain & Loop**
- 改完代码 / 配置后重训，回到 Step 3

### 7.4 Implementation in CombatBench

- OBSERVABILITY.md = Step 1 的产物
- dashboard 输出格式 = Step 2
- analyze_*.py = Step 3
- Claude Code / 类似 agent = Step 4-5
- 闭环运行 = Step 6

### 7.5 Discussion

- **优势**：降低 RL 训练门槛，非专家也能稳定训练；方法论不限于 CombatBench，可推广到任何 PPO 任务
- **局限**：诊断质量依赖 Step 1 的先验编码质量；当前 remediation 仍需人在回路审核代码改动
- **未来**：把 Step 1 也自动化——AI 从文献和过往 run 自动总结健康标准

---

## 8. Experiments

> **不能省略**。Benchmark 论文的全部说服力在此。

### 8.1 Baseline Training Results

**素材：** `baseline/humanoid21/runs/`（115+ runs）

- 4 阶段训练曲线（reward / survival / EV / entropy）
- 关键里程碑：
  - Balance recovery level 4: 82-89% survival
  - Refined recovery: 80.9% overall survival across levels 0-6

### 8.2 Ablations

- With/without safety gating（论证 gating 必要性）
- Reward component ablation
- Curriculum vs end-to-end training（论证课程必要）

### 8.3 Combat Performance

- Win rate vs Random / Standing policy
- Elo rating
- Sample efficiency（达到稳定所需帧数）

### 8.4 Compute Cost Analysis

**这是"不拼算力"论点的硬数据：**
- 训练所需 GPU 小时
- 参数量
- 与 IsaacGym-scale benchmark（需数千并行环境）对比
- 论证：消费级 GPU 可训

### 8.5 AI-in-the-Loop Effectiveness（如果可行）

- 对比有 / 无 AI 监控的训练效率
- Case study：AI 诊断出的具体问题与修复

---

## 9. Discussion

- **Limitations**：
  - 当前仅 humanoid21 一个完整 baseline（T800 在做）
  - HP 规则相对简化（未来可扩展判分制）
  - AI-in-the-loop 的 remediation 仍需人审
- **Broader Impact**：
  - 对抗 RL 的安全/伦理（双机器人物理对抗，无现实伤害）
  - 降低研究门槛的正面影响

---

## 10. Conclusion & Future Work

- CombatBench 已开源 + 公开榜单
- 未来工作：
  - T800 / G1 多机器人集成
  - 纯视觉感知方向（去对手 keypoint，用主观视角图像）
  - AI-in-the-loop 方法论推广到更多 RL 任务族

---

## Appendix

- **A. Leaderboard & Submission Protocol**（网站 + combat-submit 工具）
- **B. Hyperparameters**（PPO / 课程 / gating）
- **C. Full Observation Specification**
- **D. Reward Functions**（8 个模块的数学定义）
- **E. OBSERVABILITY Dashboard Spec**（AI-in-the-loop 的契约文档）

---

## 写作优先级建议

1. **先做 Related Work 调研**（第 2 章）——没有它就无法定义 novelty
2. **从 runs/ 提取实验数据**（第 8 章）——这是论文硬通货，先有数据再写文字
3. **写 Task Definition + Environment Design**（第 3-4 章）——素材最齐全
4. **写 Framework**（第 5 章）——已有 DESIGN.md，学术化包装即可
5. **写 Baseline + AI-in-the-loop**（第 6-7 章）——核心方法论贡献
6. **最后写 Intro / Abstract**（第 1 章 + Abstract）——总结全文

---

## 关键素材速查表

| 章节 | 项目素材路径 |
|------|--------------|
| 规则 | `docs/RULE.md` |
| 环境 | `docs/ENVIRONMENT.md`、`envs/humanoid21/OBSERVATION_zh.md`、`CONTROLSPEC.md` |
| 框架 | `envs/framework/DESIGN.md`、`README.md`、`blueprint.py` |
| 多机器人 | `envs/t800/` |
| Baseline | `baseline/humanoid21/curriculum/`、`experiments/`、`rewards/`、`plugins/` |
| Safety gating | `train_gating_network.py`、`mixed_policy.py`、`weakened_policy.py` |
| AI-in-the-loop | `OBSERVABILITY.md`、`analyze_logs.py`、`analyze_follow_logs.py`、`analyze_fight_logs.py` |
| 实验数据 | `baseline/humanoid21/runs/`（115+ runs） |
| 提交工具 | `tools/binaries/combat-submit-*` |
