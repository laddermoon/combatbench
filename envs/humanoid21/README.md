# Humanoid21 仿真场景

基于 MuJoCo 的 21 自由度人形机器人双人对抗仿真环境。

## 概述

本场景实现了两个 21-DOF 人形机器人的对抗战斗，包含：
- MuJoCo 物理引擎仿真（500Hz 物理步、20Hz 决策步、每步 25 个物理子步）
- PD 关节控制（归一化动作 `[-1, 1]`，见 `CONTROLSPEC.md`）
- 96 维观测空间（见 `DATASPEC.md` / `OBSERVATION_zh.md`）
- 碰撞检测与伤害系统、血量与胜负判定（规则见 `docs/RULE.md`）

## 快速开始

### 命令行运行（推荐）

使用 `blueprint.yaml` + `round_runner.py` CLI 运行标准比赛规则
（策略以**蓝图 YAML 文件**指定，蓝图中再指向策略实现）：

```bash
# 基本用法：两个随机策略对战
PYTHONPATH=. python3 -m envs.framework.round_runner \
    --env-blueprint envs/humanoid21/blueprint.yaml \
    --policy-a-blueprint policy/blueprints/random.yaml \
    --policy-b-blueprint policy/blueprints/random.yaml \
    --seed 42

# 录制视频
PYTHONPATH=. python3 -m envs.framework.round_runner \
    --env-blueprint envs/humanoid21/blueprint.yaml \
    --policy-a-blueprint policy/blueprints/random.yaml \
    --policy-b-blueprint policy/blueprints/humanoid21/standing.yaml \
    --video match.mp4
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--env-blueprint` | 环境蓝图 YAML 路径（必需） |
| `--policy-a-blueprint` / `--policy-b-blueprint` | 策略蓝图 YAML 路径（必需） |
| `--video` | 视频保存路径（可选，MP4） |
| `--recorder` | 附加 Recorder，格式 `module:Class?key=value`（可重复） |
| `--seed` | 回合种子（可选；缺省解析为随机 int 并写回结果） |
| `--options-json` | per-episode 参数 JSON 文件（如 `initial_health_a`） |
| `--health-a` / `--health-b` | 直接指定本回合初始 HP |
| `--score-log-file` | 逐物理子步的伤害审计日志输出路径 |
| `--want-extras` | 让策略产出 extras（log_prob 等）并交给 recorder |

### 通过蓝图编程使用

```python
from envs.framework.blueprint import EnvBlueprint
from envs.framework.round_runner import RoundRunner
from policy.random.policy import RandomCombatPolicy

blueprint = EnvBlueprint.load("envs/humanoid21/blueprint.yaml")

with RoundRunner(
    blueprint=blueprint,
    policy_a=RandomCombatPolicy(scale=0.5),
    policy_b=RandomCombatPolicy(scale=0.3),
) as runner:
    result = runner.run(seed=42)

print(result)
# {'steps': 600, 'termination_reasons': [...], 'seed': 42,
#  'health_a': ..., 'health_b': ...}
```

### 录制视频

```python
from envs.framework.blueprint import EnvBlueprint
from envs.framework.round_runner import RoundRunner
from envs.framework.common_plugins import VideoRecorderPlugin
from policy.random.policy import RandomCombatPolicy

blueprint = EnvBlueprint.load("envs/humanoid21/blueprint.yaml")

with RoundRunner(
    blueprint=blueprint,
    policy_a=RandomCombatPolicy(),
    policy_b=RandomCombatPolicy(),
    video_plugin=VideoRecorderPlugin(fps=30, output_path="match.mp4"),
) as runner:
    result = runner.run(seed=42)
```

## 规则蓝图

`blueprint.yaml` 声明了标准比赛规则的 `EnvBlueprint`（参数化，支持
`initial_distance` / `max_steps` 覆盖），包含：

- **Simulator**：`Humanoid21Simulator`（`battle_v2.xml`，默认 2m 初始距离）
- **Plugins**：`CombatScoringPlugin`（100 HP，force_scale=100）
- **Observer Plugins**：`CombatScoringObserver`（输出双方血量/伤害/KO 状态）

可以通过 Python 代码基于该蓝图构造运行时再附加观测、Recorder 或视频插件。

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 仿真器 | `simulator.py` | MuJoCo 物理引擎封装，PD 控制，静态/核心/派生数据接口 |
| 元数据 | `meta.py` | XML 解析出的机器人结构表（joint/body/keypoint/affiliation） |
| 战斗插件 | `plugins.py` | 伤害计算、KO 判定（`CombatScoringPlugin`）、约束/冻结等 |
| 扰动插件 | `disturbance_plugins.py` | 外力扰动、初始状态扰动、随机倒伏等 |
| 观测插件 | `observer_plugins.py` | `CombatScoringObserver`、平衡分析观测器等 |
| 规则蓝图 | `blueprint.yaml` | 标准比赛规则蓝图 |

### 插件系统

**CombatScoringPlugin** — 战斗计分
- 逐物理子步检测有效碰撞并累计伤害（公式见 `docs/RULE.md`）
- 动作步边界判定 KO（血量 ≤ 0 时请求终止）
- 写 `metrics`：health_a / health_b / damage_taken_a / damage_taken_b 等

**CombatScoringObserver** — 战斗状态观测
- 输出 `{robot_a, robot_b}`：health / cumulative_damage_taken /
  step_damage_taken / step_hit_events / is_ko
- `events` / `step_hit_events` / `step_damage_taken` 经 `ctx.events`
  游标差分得出"本步事件"（事件日志只增、observer 记上次读到的位置）——
  也是消费共享事件 journal 的参考实现

## 目录结构

```
humanoid21/
├── __init__.py             # 导出 Humanoid21Simulator
├── simulator.py            # MuJoCo 仿真器
├── meta.py                 # XML 结构元数据
├── plugins.py              # 战斗插件（CombatScoringPlugin 等）
├── disturbance_plugins.py  # 扰动插件
├── observer_plugins.py     # 观测插件（CombatScoringObserver 等）
├── blueprint.yaml          # 标准比赛规则蓝图（参数化）
├── battle_v2.xml           # 场地模型（当前使用，圆形围栏）
├── battle_v1.xml           # 旧场地模型（历史保留）
├── DATASPEC.md             # 数据接口规范
├── CONTROLSPEC.md          # 控制接口规范
├── OBSERVATION_zh.md       # 观测空间设计
├── CONTACT_DESIGN.md       # 接触参数设计
├── ACCEPTANCE_CRITERIA.md  # 底层控制验收标准
└── tests/                  # 测试
```

## 相关文档

- [DATASPEC.md](DATASPEC.md) — 观测/数据接口详细规范
- [CONTROLSPEC.md](CONTROLSPEC.md) — 动作空间、PD 控制、频率约定
- [OBSERVATION_zh.md](OBSERVATION_zh.md) — 观测空间设计理由
- [Policy 文档](../../policy/README.md) — 策略接口定义
