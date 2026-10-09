# experiments_sac — SAC 实验注册表

> 类型：指南

SAC 实验注册表：自动发现 `exp_sac_*.py`，每个文件导出
`EXPERIMENT_CLASS: type[ExperimentSAC]`。

- `base.py` — 共享默认参数基类
- `fact_providers.py` — 观测/奖励 provider 工厂
- 实验：`exp_sac_balance.py`、`exp_sac_standup.py`

**状态**：SAC 路径在开发中（见 `baseline/framework/sac/PLAN.md`），
接口未冻结。运行：

```bash
PYTHONPATH=. python3 baseline/framework/train.py --experiment sac_balance --algo sac
```
