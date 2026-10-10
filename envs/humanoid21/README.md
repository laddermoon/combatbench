# Humanoid21 仿真场景

> 类型：指南

基于 MuJoCo 的 21 自由度人形机器人双人对抗仿真环境。

## 概述

本场景实现了两个 21-DOF 人形机器人的对抗战斗，包含：
- MuJoCo 物理引擎仿真（500Hz 物理步、20Hz 决策步、每步 25 个物理子步）
- PD 关节控制（归一化动作 `[-1, 1]`，见 `CONTROLSPEC.md`）
- 96 维观测空间（见 `DATASPEC.md` / `OBSERVATION_zh.md`）
- 碰撞检测与伤害系统、血量与胜负判定（规则见 `docs/RULE.md`）

## 🏆 这项成果的价值

Humanoid21 是平台的**物理赛场**（成果 #0 的载体）——一个"能站着打赢"
比"能跑 benchmark 分数"难得多的对抗环境：

- **观测空间是论证过的工程设计**：96 维观测的每一条归一化规则都有
  实测依据（足底受力从牛顿改体重倍数，是因为原始值吃掉网络第一层
  99.9% 输入能量导致梯度自锁——不是经验，是测量，见 `OBSERVATION_zh.md`
  §2.3）。这在 sim 项目里很少见。
- **控制接口天然公平**：归一化 `[-1,1]` PD 位置控制（`CONTROLSPEC.md`）——
  所有策略写同一种关节目标语言，不同实现可比、可移植。
- **接触工程是实测调出来的**：碰撞/伤害可信度的物理基础（condim、
  solimp、solref）建立在 MuJoCo 三条合成规则的正确理解+实测之上
  （`CONTACT_DESIGN.md`），不是默认值凑合。
- **可审计、可度量**：逐物理子步伤害日志（`--score-log-file`）、
  确定性种子链、以及一套量化的**仿真性能基准**（`BENCHMARK.md`）。

## 💡 为什么有这样的价值（设计依据）

| 设计选择 | 使什么成立 |
| :--- | :--- |
| 全量纲归一化（96 维无因次） | 网络第一层不被任何单一物理单位淹没——实测证明梯度下降不会自愈量纲失衡（自锁），必须修在观测定义处 |
| egocentric 坐标系统一 | 除高度/足底力外全在机体系——yaw 不变性、值域有界，天然适合策略网络 |
| 对手关键点观测（头/手/脚 位置+速度） | "手臂晃动"与"重拳来袭"可区分——格斗策略所需的对手信息结构 |
| 归一化 PD 位置控制 | 策略与力矩整定解耦——换策略不用重调底层，排行榜可比性的物理基础 |
| 圆形围栏场地 | 无角落退化解——对抗策略必须正面解决，不能卡墙角 |
| 接触参数实测调参 | 打击判定与伤害结算有物理可信度，不是视觉特效 |

## 📊 性能基准

环境性能是可度量的——指标、测量方法与实测快照见 `BENCHMARK.md`：

| 指标 | 参考标线 | 最新快照 |
| :--- | :--- | :--- |
| 跟踪误差 | 承重<0.05rad / 末端<0.02rad | 未达标线（误差随幅度线性） |
| 响应延迟 | <0.2s，过冲<5% | ~120ms 等效滞后，未达标线 |
| 零震荡/控制努力 | rate<10，努力<30% | 静态力矩 66-90%，未达标线 |
| 绝对稳定性 | 无崩溃/发散 | **达标** |
| 仿真吞吐 | 信息项 | ≈5390 物理步/s ≈10.8×实时 |

复测：`PYTHONPATH=. python3 envs/humanoid21/benchmark.py`
（绝对稳定性达标 = 训练安全底线成立；跟踪保真度是已知改进方向）。

## 🔑 关键点与边界

- **频率分层**：500Hz 物理 / 20Hz 决策 / 每决策步 25 物理子步——
  物理稳定性与决策成本的最优折中。
- **CPU 参考路径**：单进程按回合执行，面向正确性与灵活性；
  批量/GPU 路径见 `envs/batchframework/`（语义对齐、实现不同）。
- **观测即契约**：96 维布局见 `OBSERVATION_zh.md` §三——改维度会使
  旧 checkpoint 失效（如 feet_forces 归一化变更，有迁移脚本先例）。
- **历史 arena**：`battle_v1.xml`/`battle_v2.xml` 保留仅供旧数据迁移，
  现役场地是 `battle_circular_v2.xml`。

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
# {'steps': 600,
#  'termination_reasons': {'robot_a': ['timeout'], 'robot_b': ['timeout']},
#  'seed': 42, 'health_a': ..., 'health_b': ...}
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

- **Simulator**：`Humanoid21Simulator`（`battle_circular_v2.xml`
  圆形围栏场地，默认 2m 初始距离）
- **Plugins**：`CombatScoringPlugin`（100 HP，force_scale=100）
- **Observer Plugins**：`CombatScoringObserver`（输出双方血量/伤害/KO 状态）

可以通过 Python 代码基于该蓝图构造运行时再附加观测、Recorder 或视频插件。

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 仿真器 | `simulator.py` | MuJoCo 物理引擎封装，PD 控制，静态/核心/派生数据接口 |
| 元数据 | `meta.py` | XML 解析出的机器人结构表（joint/body/keypoint/affiliation） |
| 战斗插件 | `plugins.py` | 伤害计算、KO 判定（`CombatScoringPlugin`）、冻结（`FrozenRobotPlugin`） |
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
├── battle_circular_v2.xml  # 场地模型（当前使用，圆形围栏）
├── battle_v1.xml           # 旧场地模型（历史保留）
├── battle_v2.xml           # 旧场地模型（历史保留）
├── DATASPEC.md             # 数据接口规范
├── CONTROLSPEC.md          # 控制接口规范
├── OBSERVATION_zh.md       # 观测空间设计
├── CONTACT_DESIGN.md       # 接触参数设计
├── BENCHMARK.md            # 仿真性能基准（指标+标线+快照）
├── generate_pose_images.py # 姿态参考图生成脚本（产物在 pose_images/）
├── pose_images/            # 姿态参考图（generate_pose_images.py 产物，可重生成）
├── obs_analysis/           # 观测空间分析产物（REPORT.md + npy/png，历史研究资产）
└── tests/                  # 测试
```

## 调试通道

以下调试开关均为**默认关闭**的环境变量 / 构造参数，仅供排障，
不参与正常训练/评测路径：

| 通道 | 开关 | 位置 | 产物 |
|------|------|------|------|
| 倒伏过程截图 | `COMBATBENCH_FALL_DEBUG=1`（输出目录 `COMBATBENCH_FALL_DEBUG_DIR`，默认 `/tmp/fall_debug`） | `disturbance_plugins.py::RandomFallenStatePlugin` | 倒伏过程 PNG + 高度日志 |
| 计分明细日志 | `COMBAT_SCORE_DEBUG_FILE=<path>` | `plugins.py::CombatScoringPlugin` | 逐物理步 JSONL（力、伤害、HP） |
| 力矩饱和打印 | 构造参数 `debug_torque=True` | `simulator.py::Humanoid21Simulator` | 力矩饱和时/前 1000 步每百步向 stdout 打印 |

## 相关文档

- [DATASPEC.md](DATASPEC.md) — 观测/数据接口详细规范
- [CONTROLSPEC.md](CONTROLSPEC.md) — 动作空间、PD 控制、频率约定
- [OBSERVATION_zh.md](OBSERVATION_zh.md) — 观测空间设计理由
- [Policy 文档](../../policy/README.md) — 策略接口定义
