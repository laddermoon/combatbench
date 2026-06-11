# Humanoid 21 扰动平衡站立：问题建模与调参指南

本项目旨在让具有 21 个自由度（DOF）的双足人形机器人学会**在外部强随机扰动下快速恢复站立平衡**。为了避免“盲目试错”，本文件对该控制问题进行数学建模，梳理系统架构，并提供精简高效的调参抓手。

---

## 一、 系统数学建模

### 1. 状态与动作空间
*   **状态空间 $\mathcal{S}$（96维）**：
    *   **本体感觉 (42维)**：21个关节的归一化位置和速度。
    *   **根节点状态 (13维)**：躯干高度、根节点局部朝向、局部线速度与角速度。
    *   **触觉感知 (2维)**：双脚受到的地面支持力。
    *   **对手状态 (39维)**：在平衡任务中作为填充占位或静态特征。
*   **动作空间 $\mathcal{A}$（21维）**：
    *   输出 21 个关节的 PD 控制器目标角度（取值 $\in [-1, 1]$）。

### 2. 课程学习（Curriculum Learning）设计
机器人面临的初始化扰动是**渐进式加剧**的：
*   **扰动规模（Perturbation Scale）**：由一个标量 $s \in [0, 1]$ 控制。在 Episode 初始化时，关节位置/速度、躯干倾角、线/角速度的最大随机偏差乘以 $s$。
*   **难度晋升**：由一组离散的难度滑块定义（`LEVEL_SCALES = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)`）。
*   **自动晋升机制**：当机器人在当前 Level 的评估存活率 $\ge 90\%$（`PROMOTE_SURVIVAL = 0.9`）并保持一定代数（`PROMOTE_PATIENCE = 1`）时，自动晋升到下一 Level。

### 3. 多 Critic 优势融合（Multi-Critic Advantage Fusion）
系统包含两个并行的 Critic 网络，分别评估不同的奖励目标：
1.  **`r_fall`**：生存奖励（存活每步给 $+0.01$，摔倒判定扣除 $-penalty$，撑过给 $+penalty$）。
2.  **`r_cross`**：双脚交叉支撑力平衡惩罚（辅助动作柔顺、减少抽搐）。

**Advantage 融合公式**：
为了消除不同奖励项尺度量级差异、保证梯度平稳，系统在每轮 PPO 更新前，对各 Critic 的 Advantage 信号**先独立归一化，再加权融合**：
$$A_{norm}^{(i)} = \frac{A^{(i)} - \mu^{(i)}}{\sigma^{(i)} + \epsilon}$$
$$A_{combined} = \sum_{i} w^{(i)} \cdot A_{norm}^{(i)}$$
*其中 $w^{(i)}$ 为当前难度阶段各奖励成分的融合权重（例如 `stage_weights = (3.0, 1.0)`）*。最终将 $A_{combined}$ 作为 Actor 策略更新的唯一 Advantage 信号。

---

## 二、 代码结构与入口

```text
curriculum/
├── train.py                          # 1. 训练启动主入口 (命令行参数、设备配置)
├── analyze_logs.py                   # 2. 智能进度监控与异常自动诊断脚本
├── OBSERVABILITY.md                  # 3. 诊断面板各指标物理含义指南
└── framework/
    ├── training_loop.py              # 4. 核心训练大循环 (数据收集、模型评估、保存)
    ├── ppo_trainer.py                # 5. PPO 优化计算 (Advantage 融合、Trust Region 更新)
    └── config.py                     # 6. 基础实验配置基类
└── experiments/
    └── exp_balance_recover.py        # 7. 本任务的专属参数/奖励定义、课程控制器
```

---

## 三、 核心调参抓手与物理指南

当训练进展停滞或发生退化时，可调节的底层参数（ Knob ）及其物理影响如下：

| 参数名称 | 所在位置 | 物理含义与调参导向 |
| :--- | :--- | :--- |
| **`learning_rate`** | `exp_balance_recover.py` | **Actor 学习率**。若 PPO 频繁在 Epoch 1 早停，说明 LR 太高，步子迈得太大，导致 KL 瞬爆。**强烈建议降至 `3e-5`**（当前设置），动作更细腻、能跑满 Epochs。 |
| **`update_epochs`** | `exp_balance_recover.py` | **每一批数据的训练 Epoch 数量**。通常设为 `4`。在 LR 适中时，跑满 Epoch 能够深度压榨每批数据的价值，加快收敛。 |
| **`n_batches`** | `ppo_trainer.py` | **数据分割份数（Minibatch 分割数）**。系统将一轮总样本数 $N$ 均分为 `n_batches = 24` 份。**Minibatch Size = $N / 24$**。如果想要增加梯度估计的精确度，可调小 `n_batches`（即增大单次梯度更新的 batch size）。 |
| **`episodes_per_update`** | `exp_balance_recover.py` | **单代数据收集量**（目前为 `1024`）。代表每次更新前机器人尝试的总 Episode 数量。如果由于扰动太大导致生存极短、总 Step 数暴跌时，应调大该值以补充足够的样本密度。 |
| **`target_kl`** | `exp_balance_recover.py` | **信任域 KL 散度硬阈值**（当前为 `0.05`）。它是策略安全更新的“刹车线”。不建议调得太高，否则会导致关节在极限动作下抽搐。 |
| **`log_std_min`** | `exp_balance_recover.py` | **探索方差下限**（当前设为 `-2.0`）。它锁定了关节探索的最小方差。如果机器人在高难度下陷入硬性肌肉记忆不肯动弹，可调大该值（如调至 `-1.5`）强制增加乱动探索。 |
| **`gammas`** | `exp_balance_recover.py` | **折现因子**（当前为 `0.99`）。对于“不摔倒”这种即时、短长远的任务，可将对应的 `r_fall` 折现因子降到 `0.95`，能极大降低 Critic 的预测难度，提高 Critic $EV$。 |
| **融合权重 `weights`** | `exp_balance_recover.py` | **`r_fall` 与 `r_cross` 的权重比例**（当前为 `(3.0, 1.0)`）。通过调节初始权重 `initial_weights()` 与动态权重 `next_weights()`，控制站立生存梯度与柔顺平衡梯度的博弈。 |
