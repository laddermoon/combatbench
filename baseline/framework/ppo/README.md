# baseline/framework/ppo — Multi-Critic PPO 训练框架

面向 CombatBench humanoid21 的 PPO 实现：**每个 reward channel 一个独立 critic**
（独立 gamma/GAE-lambda），实验通过 hook 完全控制轨迹切分、课程权重、
探索调度与 per-update 指标；框架负责 rollout → buffer → update → 日志/截面的
全流程编排与可观测性。

## 目录结构

```
ppo/
├── experiment.py     # ExperimentPPO 接口 + CommonParams/PPOParams/LRSpec/ExplorationSpec
├── loop.py           # 训练循环编排（rollout→buffer→update→eval→log→dump→checkpoint）
├── trainer.py        # ppo_update + PPOBuffer + UpdateStats（框架指标定义）
├── trajectory.py     # Trajectory / ChannelData / RewardChannel
├── stochastic_policy.py  # 训练期随机包装（explore_factor 注入）
├── debug.py          # Debug 工具集 CLI（runs/summary/metrics/catalog/dump/render/delta/viewer）
├── algos/            # GAE / advantage 计算
├── policies/         # 生产策略（truncated_normal_mlp）+ todo/ 在建策略族
├── dumpkit/          # dump 捕获/渲染/delta + debug viewer（CONTEXT.md 在此）
└── tests/            # pytest 套件
```

## 文档索引 —— 想看什么去哪看

| 我想…… | 文档 |
|---|---|
| **写一个新 PPO 实验**（接口、数据流、hook 契约、示例） | [GUIDE.md](GUIDE.md) + [../../experiments_ppo/README.md](../../experiments_ppo/README.md) |
| **理解探索控制**（explore_factor / uncertainty_floor / 调度 hook） | [DESIGN_unified_exploration_control.md](DESIGN_unified_exploration_control.md) |
| **调试一个训练 run**（指标查询、dump 下钻、工具/端点速查） | [dumpkit/CONTEXT.md](dumpkit/CONTEXT.md) —— AI 友好的能力地图 |
| **指标是什么、怎么读**（hint/读法/诊断倾向） | [dumpkit/metric_catalog.py](dumpkit/metric_catalog.py)（或 `debug.py catalog`） |
| **dump 截面管线数据流** | [dumpkit/DATA_FLOW.md](dumpkit/DATA_FLOW.md) |
| **Debug viewer 设计** | [dumpkit/DESIGN_viewer_overview.md](dumpkit/DESIGN_viewer_overview.md) + scene1-4 |
| **策略族设计与选型决策** | [policies/todo/DESIGN_OVERVIEW.md](policies/todo/DESIGN_OVERVIEW.md)、[policies/todo/DECISIONS.md](policies/todo/DECISIONS.md)、[policies/DESIGN_truncated_normal.md](policies/DESIGN_truncated_normal.md) |
| **历史修复记录**（P0 已全部完成；P1/P2 部分未做） | [FIXPLAN.md](FIXPLAN.md) —— 附录 B「最容易犯的错」仍值得读 |
| **已主动搁置的想法**（不做，原因在文档里） | [TODO_reference_policy_delta_exploration.md](TODO_reference_policy_delta_exploration.md) |

## 快速上手

```bash
cd /data1/mono/things/combatbench && PYTHONPATH=.

# 冒烟：跑 2 个 update 验证实验能起来
python3 baseline/framework/train.py --experiment basic_balance --algo ppo --smoke

# Debug 工具集（全部离线，JSON 输出）：
python3 baseline/framework/ppo/debug.py runs --tail 5      # 有哪些 run
python3 baseline/framework/ppo/debug.py summary <run>      # 定向：指标摘要+语义
python3 baseline/framework/ppo/debug.py viewer baseline/runs  # 交互界面
```

测试：`PYTHONPATH=. python3 -m pytest baseline/framework/ppo/tests/ -x -q`
