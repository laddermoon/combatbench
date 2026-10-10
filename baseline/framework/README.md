# baseline/framework — 统一训练框架

> 类型：指南

自研的 on-policy/off-policy 训练框架（**不用** Stable-Baselines3）。
核心设计：multi-critic（每个奖励通道一个 value/Q critic）、
课程由实验层在 `build_trajectories()` 里调 `actor_weight`、
实验注册表自动发现 `exp_*.py`。

## 🏆 这项成果的价值

本子项目交付**两个东西**，价值分开论证：

### A. PPO 训练框架

**探索干预被设计成一等框架公民**——不是埋在策略代码里的超参，而是三条独立机制通道：

| 通道 | 旋钮 | 干预侧 |
| :--- | :--- | :--- |
| 采样强度 | `explore_factor` ∈[-1,1]（标量/callable/声明式程序三形态） | rollout 采样分布 |
| 不确定性地板 | `uncertainty_floor`+`uncertainty_coef`（`relu(floor−U)` 损失） | 训练侧防分布崩塌 |
| 参考漂移 σ 地板 | `delta_factor`+`reference_horizon`+`delta_mode`（`σ_eff=max(σ,c·|Δ|)`，只增宽不收缩） | 相对历史策略集的漂移驱动探索 |

配套机制使这些干预**可声明、可记录、可回放**：`SamplingContext` 逐帧记入 `action_extras` 并经 `Episode→Trajectory→Buffer` 原样回放（importance ratio 正确性的前提）；ef 的声明式程序形态使同一 spec 在 CPU 与 device collector 上语义一致（device 侧不能逐帧回调 Python callable）。

其余结构优势：

- **multi-critic 匹配多目标奖励**：每个奖励通道独立 critic——击打/平衡/犯规的价值估计互不缠住；课程调度是实验层 `actor_weight` 控制而非框架钩子，是奖励对比实验（成果 #9）的结构基础。
- **复现性是契约**：每 run 生成 git 代码快照分支 + `REPRODUCE.md`；resume 等价性有测试契约；`DETERMINISM.md` 声明分层担保（保序采集 → worker 数不影响结果；GPU 位级不确定如实标注）。
- **实验即代码**：`exp_*.py` 注册表自动发现——实验是一个可 diff 的文件，ef 扫参/分布族 sweep 靠它低成本展开。
- **每 update 可评估**：`policy_exports/uNNNNN` 全是可打分产物 + best-of-run 导出。

### B. Debug 系统

- **per-update 取证**：`--dump-at` 定点捕获指定 update 的完整状态，`--dump-hypothesis` 记录当时假设，`--dump-full-grad` 抓全梯度——把"训练崩了看曲线猜"变成可复盘的取证。
- **分场景分析器**：dumpkit viewer 按问题类型组织（梯度信号/采样分布/价值诊断，见 `dumpkit/USAGE.md`）。
- **术中干预**：`--param KEY=VALUE@UPDATE` 热补丁，不重启改超参。
- **指标词典**：`metric_catalog` 统一字段口径，分析不用猜语义。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| `SamplingSpec`/`SamplingContext` 全链路记录 | rollout 采样与训练 `evaluate_actions` 共享同字段值 → importance ratio 不假 |
| ef 声明式程序（`ef_programs` 注册表） | 同一探索 spec 在 CPU 与 device collector 语义平价——device 路径逐帧回调会破坏 job-keyed 确定性与 D2H 同步 |
| `ReferenceSpec` 动作空间集成（非参数 EMA） | σ 地板可相对"历史策略集成"定义；选哪几代、权重几何全归实验层——机制通用、策略自由 |
| Gen0 排除 + warmup 门（`reference_horizon`） | 当前策略自身入集会把 Δ 机械钉到 0；H 门保证集成大小恒定语义不变 |
| `delta_mode` frozen | rollout 时记录 `det(Gen0)−a_ref` 载荷、训练原样回放——σ_eff 在 update 内对 θ 静止，消除 dynamic 参数化下的 PPO 不稳 |
| multi-critic per channel | 奖励通道价值估计独立 → 奖励设计实验可比、可消融 |
| git 快照 + `REPRODUCE.md` | 任意 run 可精确复现，实验间结论可比 |

## 🔑 关键点与边界

- **PPO 是成熟主路径；SAC 在飞**（`sac/PLAN.md`，有独立 DEBUG_PLAYBOOK）。
- `explore_factor` 与 `uncertainty_floor` 是**两条独立通道**：前者改采样分布，后者是训练侧损失——别混用。
- `delta_factor≠0` 必须配 `reference_horizon>0`；Δ 机制只**增宽** σ 不收缩（是地板不是混合）。
- 确定性担保到"统计等价+采集保序"，GPU 训练不承诺位级相同（`DETERMINISM.md`）。
- `obsolete/` 是 v1 遗留只读参考，勿依赖。
- 深入文档：`ppo/GUIDE.md`（完整用法）· `rollout/README.md`（数据管线）· `DETERMINISM.md` · `ppo/dumpkit/USAGE.md` · `policies/POLICY_SELECTION.md`（分布族选型）。

## 目录结构

```
baseline/framework/
├── train.py            # 统一训练 CLI（--experiment/--algo ppo|sac/…）
├── critic_mlp.py       # CriticMLP（PPO/SAC 共用）
├── code_snapshot.py    # git 代码快照 + REPRODUCE.md 生成（可复现性）
├── rollout/            # 回合采集与数据模型（Job/Episode/ParallelRollouter）
├── ppo/                # PPO 实现——见 ppo/GUIDE.md（完整使用指南）
├── sac/                # SAC 实现——开发中（见 sac/PLAN.md，审计范围外）
└── obsolete/           # v1 框架遗留代码（只读参考，勿依赖）
```

## 数据流

```
experiments_{ppo,sac}/exp_*.py          # 实验定义（注册表自动发现）
        │
        ▼
train.py --experiment X --algo ppo     # CLI 入口
        │
        ▼
ppo/loop.py + trainer.py               # 训练循环 + 更新
        │  collect
        ▼
rollout/  ParallelRollouter.collect(jobs) → List[Episode] → Trajectory
        │
        ▼
baseline/runs/<run_name>/              # checkpoints/policy_exports/videos/REPRODUCE.md
```

## 文档地图

| 要做什么 | 看 |
|---|---|
| 写一个新 PPO 实验 | `ppo/GUIDE.md`（完整指南）+ `experiments_ppo/README.md` |
| 理解采样/回合数据格式 | `rollout/README.md` |
| 理解策略族选型 | `ppo/policies/POLICY_SELECTION.md` |
| 诊断训练 | `ppo/dumpkit/`（CONTEXT + DATA_FLOW） |
| 批量/GPU 采样 | `envs/batchframework/`（设备端对等实现） |
| 训练 CLI 全参数 | `CLAUDE.md` §Training CLI / `train.py --help` |
