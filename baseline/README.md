# baseline — 训练路径地图

> 类型：指南

一页看懂"从实验想法到训练产物"的完整链路。每步给最短路径 + 深挖文档。

## 目录构成

```
baseline/
├── framework/        # 统一训练框架（train.py/ppo/sac/rollout）——见 framework/README.md
├── experiments_ppo/  # PPO 实验注册表（exp_*.py 自动发现）——见其中 README.md
├── experiments_sac/  # SAC 实验注册表（开发中）
├── humanoid21/       # humanoid21 专属训练资产（blueprints/rewards/plugins/curriculum 遗留）
├── runs/             # 训练输出（gitignored）
└── *.py 散件         # 一次性 sweep/诊断脚本（run_sweep_*, compare_lc4, bench_throughput）
```

## 训练全链路（五步）

### ① 定义实验

新建 `experiments_ppo/exp_<name>.py`，继承 `CombatExperimentPPOBase`
（或 `ExperimentPPO`），导出 `EXPERIMENT_CLASS`。**不用注册**——
`--experiment <name>` 自动发现。
→ 深入：`experiments_ppo/README.md`、`ppo/GUIDE.md`（ExperimentPPO API）

### ② 启动训练

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/framework/train.py \
  --experiment <name> --algo ppo --smoke      # 冒烟（2 update 8 eps）
PYTHONPATH=. python3 baseline/framework/train.py \
  --experiment <name> --algo ppo --background # 正式（后台，日志在 run_dir/train.log）
```

`--list-experiments` 列全部可跑实验；`--param KEY=VALUE[@UPDATE]` 打参数补丁。
→ 深入：`CLAUDE.md` §Training CLI（全参数表）、`framework/README.md`

### ③ 产物落盘

`baseline/runs/<run_name>/`：`config.json`、`train.log`、`checkpoints/`、
`policy_exports/`（每 update 策略）、`videos/`、`REPRODUCE.md`（复现命令）。
→ 深入：`CLAUDE.md` §Run Directory Structure

### ④ 诊断/调试

dumpkit 捕获某 update 的完整帧+梯度：`--dump-at N` 或事后
`dump_rollout.py`；viewer 起本地 web 查看。
→ 深入：`ppo/dumpkit/CONTEXT.md`、`DATA_FLOW.md`

### ⑤ 评估策略

单回合 `RoundRunner`、多回合 `MatchRunner`、吞吐基准 `bench_rollout.py`。
→ 深入：`envs/framework/RUNNERS.md`、`rollout/README.md`

## 关键心智模型

- **实验 = 配方**（exp_*.py），**框架 = 引擎**（framework/），
  **环境 = 蓝图**（`*_env.yaml`，见 `envs/framework/BLUEPRINTS.md`）——三层正交。
- **PPO 是成熟路径**，SAC 在飞（`sac/PLAN.md`）。
- `humanoid21/curriculum/` 是**遗留参考**（v1 框架已删，不可运行）。
