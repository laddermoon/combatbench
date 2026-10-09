# policy/random — RandomCombatPolicy

> 类型：指南

参考实现：均匀随机策略。`policy.py` 实现 `Policy` ABC（`envs/framework/policy.py`），
`act()` 返回 `[-scale, scale]` 均匀采样动作。用于冒烟测试与基线下限。
蓝图：`policy/blueprints/random.yaml`。
