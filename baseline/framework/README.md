# baseline/framework — 统一训练框架

> 类型：指南

自研的 on-policy/off-policy 训练框架（**不用** Stable-Baselines3）。
核心设计：multi-critic（每个奖励通道一个 value/Q critic）、
课程由实验层在 `build_trajectories()` 里调 `actor_weight`、
实验注册表自动发现 `exp_*.py`。

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
