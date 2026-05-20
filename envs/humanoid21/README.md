# Humanoid21 仿真场景

基于 MuJoCo 的 21 自由度人形机器人双人对抗仿真环境。

## 概述

本场景实现了两个 21-DOF 人形机器人的对抗战斗，包含：
- MuJoCo 物理引擎仿真
- PD 关节控制
- 碰撞检测与伤害系统
- 血量与胜负判定

## 快速开始

### 命令行运行（推荐）

使用 `rule_blueprint.yaml` + `round_runner.py` CLI 运行标准比赛规则：

```bash
# 基本用法：两个随机策略对战
python3 -m envs.framework.round_runner \
    --blueprint envs/humanoid21/rule_blueprint.yaml \
    --policy-a policy.random.policy:RandomCombatPolicy?scale=0.5 \
    --policy-b policy.random.policy:RandomCombatPolicy?scale=0.3 \
    --seed 42

# 录制视频
python3 -m envs.framework.round_runner \
    --blueprint envs/humanoid21/rule_blueprint.yaml \
    --policy-a policy.random.policy:RandomCombatPolicy \
    --policy-b policy.random.policy:RandomCombatPolicy \
    --video match.mp4

# 附加 Recorder（可重复）
python3 -m envs.framework.round_runner \
    --blueprint envs/humanoid21/rule_blueprint.yaml \
    --policy-a policy.random.policy:RandomCombatPolicy \
    --policy-b policy.random.policy:RandomCombatPolicy \
    --recorder some.module:MyRecorder?path=trace.jsonl
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--blueprint` | 蓝图文件路径（JSON 或 YAML） |
| `--policy-a` | 机器人 A 的策略：`module:ClassName?key=value` |
| `--policy-b` | 机器人 B 的策略：`module:ClassName?key=value` |
| `--video` | 视频保存路径（可选） |
| `--recorder` | 附加 Recorder（可重复） |
| `--seed` | 回合种子（可选） |

**策略指定格式：**
- `policy.random.policy:RandomCombatPolicy`
- `policy.random.policy:RandomCombatPolicy?scale=0.2&seed=42`

### 通过蓝图编程使用

```python
from envs.framework import EnvBlueprint, RoundRunner
from policy.random.policy import RandomCombatPolicy

blueprint = EnvBlueprint.load("envs/humanoid21/rule_blueprint.yaml")

with RoundRunner(
    blueprint=blueprint,
    policy_a=RandomCombatPolicy(scale=0.5),
    policy_b=RandomCombatPolicy(scale=0.3),
) as runner:
    result = runner.run(seed=42)

print(result)
# {'steps': 600, 'termination_reasons': ['timeout'], 'seed': 42}
```

### 录制视频

```python
from envs.framework import EnvBlueprint, RoundRunner
from envs.framework.common_plugins import VideoRecorderPlugin
from policy.random.policy import RandomCombatPolicy

blueprint = EnvBlueprint.load("envs/humanoid21/rule_blueprint.yaml")

with RoundRunner(
    blueprint=blueprint,
    policy_a=RandomCombatPolicy(),
    policy_b=RandomCombatPolicy(),
    video_plugin=VideoRecorderPlugin(fps=30, output_path="match.mp4"),
) as runner:
    result = runner.run(seed=42)
```

## 规则蓝图

`rule_blueprint.yaml` 声明了标准比赛规则的 `EnvBlueprint`，包含：

- **Simulator**：`MujocoCombatSimulator`（默认 2m 初始距离）
- **Plugins**：仅 `CombatScoringPlugin`（100 HP，damage scale 100）
- **Observer Plugins**：仅 `CombatScoringObserver`（同时追踪双方状态）

可以通过 Python 代码基于该蓝图构造运行时再附加观测、Recorder 或视频插件。

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 仿真器 | `simulator.py` | MuJoCo 物理引擎封装，PD 控制 |
| 战斗插件 | `plugins.py` | 伤害计算、KO 判定（`CombatScoringPlugin`） |
| 观测插件 | `observer_plugins.py` | 战斗状态观测（`CombatScoringObserver`） |
| 规则蓝图 | `rule_blueprint.yaml` | 标准比赛规则蓝图 |

### 插件系统

**CombatScoringPlugin** — 战斗计分
- 检测有效碰撞
- 计算伤害
- 判定 KO（血量 ≤ 0）
- 发布 `metrics`：health_a / health_b / damage_taken_a / damage_taken_b

**CombatScoringObserver** — 战斗状态观测
- 消费 `CombatScoringPlugin` 数据
- 输出双方血量、累计伤害、本步伤害、命中事件、KO 状态
- 同时观测 robot_a 和 robot_b

## 目录结构

```
humanoid21/
├── __init__.py           # 导出 MujocoCombatSimulator
├── simulator.py          # MuJoCo 仿真器
├── plugins.py            # 战斗插件（CombatScoringPlugin 等）
├── observer_plugins.py   # 观测插件（CombatScoringObserver 等）
├── rule_blueprint.yaml   # 标准比赛规则蓝图
└── tests/                # 测试
```

## 相关文档

- [SPEC.md](SPEC.md) — 观测空间、动作空间、控制模式详细说明
- [Policy 文档](../../policy/README.md) — 策略接口定义
