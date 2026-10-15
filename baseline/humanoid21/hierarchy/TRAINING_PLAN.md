# 第一版底层执行器（Low-Level Executor v1）深化训练方案

> 类型：记录 ｜ 状态：规划中 ｜ 日期：2026-10-15

本文档定义 `humanoid21` 分层控制路线中 **第一版底层模型（v1）** 的输入输出契约、自训练环境机制、奖励通道设计与分阶段训练节奏。

---

## 0. 核心设计准则

1. **单人自训练（Opponent-Free Self-Training）**：
   底层把一切任务抽象为**机体系下的末端点追踪与姿态维持**，训练过程不依赖对手策略，仅通过虚拟目标点采样与物理外力扰动（躯干推力 + 命中反震冲量）完成闭环。
2. **不倒优先铁律（Balance-First Iron Law）**：
   **复用既有 Standup 成果，先完成并锁死 Standup 与抗推平衡，再逐步解锁上肢打点与下肢移动；在做任何动作的全过程中，平衡与不倒拥有最高优先级。**
3. **物理语义组合编码（Factorized Semantic Command）**：
   拒绝不可扩展的 One-Hot 与易发生线性串扰的二进制码，采用 `部位 + 左右 + 模式 + 相位 + 三维坐标` 的 8 维正交语义码，一次定型，后续扩充踢腿（Kick）、格挡（Guard）、闪避（Duck）无需修改网络结构。

---

## 1. 观测与指令契约（65 维输入）

### 1.1 剥离对手观测，保留 57 维纯本体
标准 `Humanoid21Simulator` 观测为 96 维（详见 `envs/humanoid21/OBSERVATION_zh.md`），其中：
- `obs[0:57]`（57 维）：本体关节角度（21）+ 关节角速度（21）+ 重力投影/高度/速度/场地中心（13）+ 足底触地力（2）。
- `obs[57:96]`（39 维）：对手观测。

底层模型**截断剔除后 39 维对手观测**，原因：
- 杜绝底层将动作执行耦合到特定对手姿态分布上（规避 `TRAINING_HISTORY.md` §3 中环境分布变更导致策略崩塌的隐患）；
- 前 57 维与既有 `standup_mbs_ef05_s42` 完全对齐，第一层权重 $W_1[:, :57]$ 可 **100% 无损继承**。

### 1.2 尾部追加 8 维物理语义指令（`obs[57:65]`）
所有维度控制在 $[-1.5, 1.5]$ 的 $O(1)$ 无量纲区间内：

| 观测索引 | 指令内索引 | 字段名 | 取值范围 | 物理语义 |
|---|---|---|---|---|
| `57` | `cmd[0]` | `eff_type` | `{-1, 0, +1}` | 受控部位层级：`+1`=上肢(手)，`0`=躯干/全身，`-1`=下肢(脚，v2启用) |
| `58` | `cmd[1]` | `eff_side` | `{-1, 0, +1}` | 受控部位左右：`+1`=左侧(L)，`0`=中轴/双侧，`-1`=右侧(R) |
| `59` | `cmd[2]` | `act_mode` | `{-1, 0, +1}` | 动作意图模式：`+1`=爆发打击(Strike)，`0`=待机恢复(Stand)，`-1`=定位/移动(Move/Guard) |
| `60` | `cmd[3]` | `phase` | `[0.0, 1.0]` | 当前指令时间窗归一化进度 $t / T_{\text{cmd}}$（`STAND` 常驻时为 `0.0`） |
| `61` | `cmd[4]` | `target_x` | `[-1.5, 1.5]` | 目标点在**当前帧机体系**下的前向坐标 $x^*$ (m) |
| `62` | `cmd[5]` | `target_y` | `[-1.5, 1.5]` | 目标点在**当前帧机体系**下的横向坐标 $y^*$ (m，左正右负) |
| `63` | `cmd[6]` | `target_z` | `[-1.5, 1.5]` | 目标点在**当前帧机体系**下的垂向坐标 $z^*$ (m，相对躯干根部高度) |
| `64` | `cmd[7]` | `target_aux`| `[-1.0, 1.0]` | 辅助参数：`MOVE` 时为期望朝向角误差 $\Delta\psi / \pi$；`PUNCH` 时为期望力度系数 |

### 1.3 原语编码对照表（v1 实装 vs v2 预留）

| 原语名称 | 版本 | `(eff_type, eff_side, act_mode)` | `phase` | `target_xyz` | `target_aux` |
|---|---|---|---|---|---|
| `STAND`（起身/抗推/收手护架） | **v1** | `( 0,  0,  0)` | `0.0` | `(0, 0, 0)` | `0.0` |
| `PUNCH_L`（左拳三维打点） | **v1** | `(+1, +1, +1)` | $t/T$ | 机体系击打点 $(x^*, y^*, z^*)$ | `1.0` |
| `PUNCH_R`（右拳三维打点） | **v1** | `(+1, -1, +1)` | $t/T$ | 机体系击打点 $(x^*, y^*, z^*)$ | `1.0` |
| `MOVE`（全向移动与转向） | **v1** | `( 0,  0, -1)` | `0.0` | 机体系航点 $(x^*, y^*, 0)$ | $\Delta\psi / \pi$ |
| `GUARD_L/R`（指定点架盾格挡）| v2 预留 | `(+1, ±1, -1)` | $t/T$ | 护架位置 $(x^*, y^*, z^*)$ | `0.0` |
| `KICK_L/R`（左右脚三维踢点） | v2 预留 | `(-1, ±1, +1)` | $t/T$ | 踢击目标点 $(x^*, y^*, z^*)$ | `1.0` |

---

## 2. “不倒优先”的四重保障机制

为了确保**机器人在做任何动作时，先保证绝不摔倒**，方案从权重初始化、梯度门控、运行时调度、课程晋级四个层面建立硬约束：

### 2.1 保障一：65 维 Standup 无损迁移与底座先行
- 从 `baseline/runs/standup_mbs_ef05_s42/checkpoints/checkpoint_u00350.pt` 提取 MBS Actor（`StateMixtureBoundedStdTruncatedNormalPolicy`）与 `r_potential` Critic 权重：
  - 输入层 $W_1 \in \mathbb{R}^{H \times 96} \rightarrow \big[W_1[:, :57],\ \mathbf{0}_{H \times 8}\big] \in \mathbb{R}^{H \times 65}$。
- **底座先行原则**：在开启任何打点或移动训练之前，必须先在 65 维架构下完成 **Stage 0（Standup 巩固与抗推平衡）** 训练，确保机器人在倒地随机初始化下起身成功率 $\ge 0.99$，且在持续随机推力与指令扰动下稳如泰山。

### 2.2 保障二：平衡通道全时最高优先级 + 动作通道安全门控
- **平衡通道永不关断**：`r_potential`（`aw = 3.0`）与 `r_fall`（`aw = 2.0`）在每一帧、每一种指令下**始终保持全额权重**，从不因正在执行出拳或移动而降权。
- **动作通道带死区平方门控（Safety-Gated Actor Weight）**：
  定义当前帧直立度势能 $\varphi_t \in [0, 1]$，所有非平衡通道（`r_strike`、`r_move`、`r_guard`）的 `actor_weight` 统一乘以安全门控因子：
  $$g_{\text{safe}}(\varphi_t) = \mathbb{I}(\varphi_t \ge \varphi_{\text{safe}}) \cdot \varphi_t^2, \quad \varphi_{\text{safe}} = 0.75$$
  - 当机器人站得极稳（$\varphi_t \approx 0.95$）时，$g_{\text{safe}} \approx 0.90$，正常学习出拳与走位；
  - 一旦出拳过猛或受外力冲击导致身体倾斜（$\varphi_t < 0.75$），$g_{\text{safe}}$ **瞬时归零**，动作任务梯度彻底切断，网络 100% 梯度容量回归 `r_potential` 和 `r_fall` 进行重心自救。

### 2.3 保障三：指令调度器的“前置准入”与“中途失衡熔断”
在 `CommandSimulator` 的指令状态机中内置两条铁律：
1. **发招前置准入（Launch Pre-condition）**：
   仅当机器人连续 $K=5$ 步满足 $\varphi_t \ge 0.85$ 且双脚至少一脚有效触地时，才允许从 `STAND` 切入 `PUNCH_L/R` 或 `MOVE`；若刚起身或刚受推力尚未站稳，调度器强制维持 `STAND` 指令。
2. **中途失衡紧急熔断（Mid-Action Abort）**：
   在执行 `PUNCH_L/R` 或 `MOVE` 过程中，若任意一帧出现 $\varphi_t < 0.70$（明显倾覆风险），调度器**立即中断当前动作窗口，下一帧强制切回 `STAND (0,0,0)`**，让策略在训练中形成“重心一旦失控就立刻放弃出拳、收手扎马步”的肌肉本能。

### 2.4 保障四：课程晋级与 Best 评选的“不倒一票否决制”
- 在所有阶段的 `on_eval()` 中，同时统计任务指标（`hit_rate`、`hold_ratio`）与生存指标（`stand_ratio`、`no_fall_rate`）：
  - **硬门槛**：仅当 `no_fall_rate >= 0.95` 且 `stand_ratio >= 0.92` 时，才允许记录 `is_new_best` 或累加课程晋级计数；
  - **动态退火**：若连续 3 次 eval 的 `no_fall_rate < 0.90`（说明打点距离过远或推力过大开始破坏平衡），课程自动退回上一级，优先修复不倒能力。

---

## 3. 训练节奏（四阶段稳健递进）

调整后的训练流水线严格遵循 **“先固化 Standup 与抗推 $\rightarrow$ 再练近身稳态打点 $\rightarrow$ 再扩全空间发力与抗反震 $\rightarrow$ 最后合入全向移动”**：

```text
standup_mbs_ef05_s42 (96d, u350)
       │  (权重切片 W1[:,:57] + 0_{H×8})
       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 0: Standup 迁移与抗推巩固 (exp_hier_s0_stand)      │
│ • 指令：100% STAND (0,0,0) + 护架约束 r_guard            │
│ • 环境：50% 倒地随机初始化(练起身) + 50% 站立12级推力课程│
│ • 验收：起身成功率 ≥ 0.99，100N 推力 recovery ≥ 0.92，  │
│         站立后双拳自然收回胸前护架位                     │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: 平衡包络内定点打靶 (exp_hier_s1_punch_safe)     │
│ • 指令：[STAND 20~30步] ↔ [PUNCH_L/R 12步] 交替         │
│ • 目标点：仅在近身安全包络采样 x*∈[0.45,0.65], y*∈[±0.25]│
│ • 核心：每打一拳必须切回 STAND 站稳收手；验证打拳 0 倒地 │
│ • 验收：no_fall_rate ≥ 0.97，hit_rate ≥ 0.85，收手率≥0.9│
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2: 全空间爆发打点 + 命中反震 (exp_hier_s2_punch)   │
│ • 指令：同上，叠加 40N~80N 背景躯干扰动                  │
│ • 课程：x* 扩至 [0.45, 0.85]m（逼出转腰送肩与小垫步），  │
│         判定球 r_hit 0.18m → 0.10m，要求拳速 ≥ 2.0 m/s，│
│         命中瞬间在拳头施加 50~150N 反向冲击力            │
│ • 验收：no_fall_rate ≥ 0.95，远距重拳命中后反震不倒      │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3: 合入 MOVE 全指令协同 (exp_hier_s3_full_v1)      │
│ • 指令：STAND (30%) + MOVE (40%) + PUNCH_L/R (30%) 交替  │
│ • 核心：训练 MOVE → 急停扎稳 → PUNCH → STAND 收手 → MOVE │
│ • 验收：全指令切换下 no_fall_rate ≥ 0.95，即可交付上层   │
│         Code-as-Policy 直接投入完整对战                  │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 多 Critic 奖励通道设计（6 通道）

延续 `baseline/framework/ppo/` 的多 Critic + EV-confidence 架构，6 个通道职责正交：

| 通道 | $\gamma$ | `actor_weight` 调度 | 信号定义 |
|---|---|---|---|
| `r_potential` | 0.99 | **全程恒定 `3.0`** | `0.01 * φ(t)`（四阶段起身与站立势能，继承自 `exp_standup`） |
| `r_fall` | 0.99 | **全程恒定 `2.0`** | `-0.2 * fall_onset - wall_lean_penalty + push_clean_bonus`（继承自 `exp_balance_pure`） |
| `r_cross` | 0.99 | `1.0 * g_safe(φ)` | `CrossSupportBalanceRewarder`（跨步/垫步时的左右脚交替约束，防单脚连跳） |
| `r_guard` | 0.99 | `1.5 * g_safe(φ)` | **非攻击手护架奖励**：对处于非出拳状态的手（`STAND`/`MOVE` 时的双手、`PUNCH_L` 时的右手、`PUNCH_R` 时的左手），当屈肘角 `elbow_norm >= 0.6` 且手部高度在胸颌区间时给分。**作用：既防止闲置手乱甩破坏重心，又在 `PUNCH -> STAND` 切换时天然驱动出拳手收回胸前** |
| `r_strike` | 0.99 | **仅 `PUNCH_L/R` 帧** `4.0 * g_safe(φ)`（其余帧 `0`） | 针对指定手 $h \in \{L, R\}$ 与三维目标点 $\mathbf{p}^*$：<br>1. **肩手相对伸展**：$\text{clip}(\|\mathbf{p}_{\text{shoulder}}-\mathbf{p}^*\| - \|\mathbf{p}_{\text{hand}}-\mathbf{p}^*\|, -0.2, 0.6)$（防只靠身体前冲刷分）<br>2. **正向出拳速度**：$\mathbb{I}(\dot{\theta}_{\text{elbow}} < 0) \cdot \max(0, \mathbf{v}_{\text{hand}} \cdot \hat{\mathbf{d}}_{\text{hand}\to\mathbf{p}^*})$<br>3. **命中脉冲**：当 $\|\mathbf{p}_{\text{hand}} - \mathbf{p}^*\| < r_{\text{hit}}$ 时奖励 $1.0 + 0.5 v_{\text{hit}}$，并立即完成该次打点切回 `STAND` |
| `r_move` | 0.99 | **仅 `MOVE` 帧** `3.0 * g_safe(φ)`（其余帧 `0`） | 朝向机体系目标航点 $(x^*, y^*)$ 的径向速度奖励 + 切向漂移惩罚 + 期望朝向对齐 $\cos(\Delta\psi)^+$（复用 `follow_opponent.py`） |

> **关键解耦机制**：在 `PUNCH` 窗口内 `r_move` 的 `actor_weight = 0`；在 `MOVE` 窗口内 `r_strike` 的 `actor_weight = 0`；在 `STAND` 窗口内两者皆为 `0`。通过指令时间窗掩码实现梯度正交，彻底消除稠密走位奖励淹没打点/收手奖励的问题。

---

## 5. 代码模块落地方案

所有分层专属环境组件统一放在本目录 `baseline/humanoid21/hierarchy/` 下，实验入口注册在 `baseline/experiments_ppo/`：

```text
baseline/humanoid21/hierarchy/
├── README.md                      # 目录指南
├── TRAINING_PLAN.md               # 本深化方案
├── init_hier_ckpt.py              # 权重迁移脚本：96d standup_mbs_ef05_s42 → 65d 初始 ckpt
├── command_simulator.py           # CommandSimulator：截取前57维 + 8维指令调度 + 失衡熔断 + 命中反震力
├── command_observer.py            # CommandTaskObserver：计算手/脚到目标点距离、拳速、护架度、命中事件
└── blueprints/
    ├── hier_s0_stand_env.yaml     # Stage 0 环境蓝图（起身 + 12级推力 + 护架）
    ├── hier_s1_punch_env.yaml     # Stage 1/2 环境蓝图（打点 + 命中反震 + 推力）
    └── hier_s3_full_env.yaml      # Stage 3 环境蓝图（全指令协同）

baseline/experiments_ppo/
├── exp_hier_s0_stand_mbs.py       # Stage 0 实验定义
├── exp_hier_s1_punch_safe_mbs.py  # Stage 1 实验定义
├── exp_hier_s2_punch_mbs.py       # Stage 2 实验定义
└── exp_hier_s3_full_mbs.py        # Stage 3 实验定义

policy/humanoid21/hier_code/
└── policy.py                      # 部署适配器：接收平台 96 维观测 → 上层 Python 规则产出 8 维 cmd → 调 65 维底层
```

---

## 6. 第一步行动项（Stage 0 启动清单）

1. 编写 `baseline/humanoid21/hierarchy/init_hier_ckpt.py`，将 `standup_mbs_ef05_s42` (`u350`) 转换为 65 维初始 checkpoint，并跑通单元测试验证前 57 维在零指令（`cmd = 0`）下的动作输出与原 96 维模型（对手观测置零时）完全等价。
2. 实现 `CommandSimulator` 与 `CommandTaskObserver` 的 Stage 0 子集（`STAND` 指令 + 护架奖励 `r_guard`）。
3. 注册 `exp_hier_s0_stand_mbs.py`，先跑 `--smoke` 冒烟验证，再正式启动 Stage 0 训练，拿到**“倒地能起、推不倒、站立自动收拳护胸”**的 65 维铁底座。
