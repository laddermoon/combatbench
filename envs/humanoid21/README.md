# Humanoid21 仿真场景

基于 MuJoCo 的 21 自由度人形机器人双人对抗仿真环境。

## 概述

本场景实现了两个 21-DOF 人形机器人的对抗战斗，包含：
- MuJoCo 物理引擎仿真
- PD 关节控制
- 碰撞检测与伤害系统
- 血量与胜负判定

## 快速开始

### 基本使用

```python
from combatbench.envs.humanoid21 import make_env
from combatbench.policy import RandomCombatPolicy, StandingCombatPolicy
from combatbench.envs.framework import RoundRunner

# 创建环境
env = make_env(
    match_duration=30.0,    # 单回合时长（秒）
    control_frequency=20,   # 控制频率（Hz）
    non_fall_mode=True,     # 启用防摔倒约束
)

# 创建策略
policy_a = RandomCombatPolicy(scale=0.1)
policy_b = StandingCombatPolicy()

# 运行比赛
runner = RoundRunner(policy_a, policy_b, env)
result = runner.run(seed=42)

print(f"Winner: {result['winner']}")
```

### 多回合比赛

```python
from combatbench.envs.humanoid21 import make_env
from combatbench.envs.framework import MatchRunner

def env_factory():
    return make_env(
        match_duration=30.0,
        non_fall_mode=True
    )

runner = MatchRunner(
    policy_a=policy_a,
    policy_b=policy_b,
    env_factory=env_factory,
    total_rounds=6
)
result = runner.run(seed=42, video_dir="videos")
```

### 录制视频

```python
from combatbench.envs.framework.common_plugins import VideoRecorderPlugin

env = make_env(
    plugins=[VideoRecorderPlugin(fps=30, output_path="match.mp4")],
    match_duration=30.0
)
```

## 命令行工具

### run_round.py - 单回合比赛脚本

运行单回合对战，支持策略加载和视频录制。

```bash
# 基本用法（站立策略）
python -m envs.humanoid21.run_round --duration 10

# 录制视频
python -m envs.humanoid21.run_round --duration 10 --video match.mp4

# 使用随机策略
python -m envs.humanoid21.run_round \
    --policy-a policy.RandomCombatPolicy?scale=0.2 \
    --duration 10

# 启用防摔倒模式
python -m envs.humanoid21.run_round \
    --non-fall-mode \
    --duration 10

# 静默模式
python -m envs.humanoid21.run_round --duration 10 --quiet
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy-a` | `None` | 机器人 A 的策略（默认：StandingCombatPolicy） |
| `--policy-b` | `None` | 机器人 B 的策略（默认：StandingCombatPolicy） |
| `--duration` | `30.0` | 回合时长（秒） |
| `--control-frequency` | `20` | 控制频率（Hz） |
| `--non-fall-mode` | `False` | 启用防摔倒约束 |
| `--non-fall-pitch-limit-deg` | `5.0` | Pitch 限制（度） |
| `--non-fall-roll-limit-deg` | `5.0` | Roll 限制（度） |
| `--damage-scale` | `100.0` | 伤害缩放系数 |
| `--video` | `None` | 视频保存路径 |
| `--quiet` | `False` | 静默模式 |

### run_match.py - 多回合比赛脚本

运行多回合比赛，支持血量延续和 KO 判定。

```bash
# 基本用法（6回合比赛）
python -m envs.humanoid21.run_match --duration 10

# 录制视频（每回合单独保存）
python -m envs.humanoid21.run_match \
    --duration 10 \
    --video-dir videos/

# 自定义回合数
python -m envs.humanoid21.run_match \
    --rounds 3 \
    --duration 15 \
    --video-dir videos/

# 使用随机策略
python -m envs.humanoid21.run_match \
    --policy-a "policy.RandomCombatPolicy?scale=0.2&seed=42" \
    --policy-b "policy.RandomCombatPolicy?scale=0.1&seed=43" \
    --duration 10 \
    --video-dir videos/

# 启用防摔倒模式
python -m envs.humanoid21.run_match \
    --non-fall-mode \
    --duration 10 \
    --video-dir videos/
```

**比赛规则：**
1. **初始血量**：双方各 100 点
2. **KO 条件**：将对方血量降至 0，立即结束比赛
3. **时间判决**：时间结束时血量高者获胜
4. **平局判定**：血量相同则为平局
5. **血量延续**：每回合血量延续上一回合
6. **视频保存**：每回合单独保存为 `round_1.mp4`, `round_2.mp4`, ...

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy-a` | `None` | 机器人 A 的策略（默认：StandingCombatPolicy） |
| `--policy-b` | `None` | 机器人 B 的策略（默认：StandingCombatPolicy） |
| `--rounds` | `6` | 总回合数 |
| `--duration` | `30.0` | 每回合时长（秒） |
| `--control-frequency` | `20` | 控制频率（Hz） |
| `--non-fall-mode` | `False` | 启用防摔倒约束 |
| `--non-fall-pitch-limit-deg` | `5.0` | Pitch 限制（度） |
| `--non-fall-roll-limit-deg` | `5.0` | Roll 限制（度） |
| `--damage-scale` | `100.0` | 伤害缩放系数 |
| `--video-dir` | `None` | 视频保存目录 |
| `--quiet` | `False` | 静默模式 |

**策略指定格式：**
- 模块路径：`policy.RandomCombatPolicy`
- 带参数：`policy.RandomCombatPolicy?scale=0.2&seed=42`
- 配置文件：`@configs/policy_a.json`

## 功能实现

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 仿真器 | `simulator.py` | MuJoCo 物理引擎封装，PD 控制 |
| RL 适配器 | `rl_adapter.py` | 观测/动作空间定义，数据提取 |
| 战斗插件 | `plugins.py` | 伤害计算、防摔倒约束 |
| 工厂函数 | `__init__.py` | 环境组装 |

### 插件系统

**CombatScoringPlugin** - 战斗计分
- 检测有效碰撞
- 计算伤害（头部 -3，躯干 -1）
- 判定 KO（血量 ≤ 0）

**NonFallConstraintPlugin** - 防摔倒约束
- 限制 pitch/roll 角度
- 保持机器人直立

**FrozenRobotPlugin** - 冻结机器人
- 冻结指定机器人位置
- 用于训练和调试

## 配置参数

### make_env 参数

```python
make_env(
    arena_xml=None,           # 场景 XML 文件路径
    dt=0.002,                 # 物理时间步（秒）
    control_frequency=20,     # 控制频率（Hz）
    match_duration=30.0,      # 回合时长（秒）
    non_fall_mode=False,      # 是否启用防摔倒
    non_fall_pitch_limit_deg=5.0,   # pitch 限制（度）
    non_fall_roll_limit_deg=5.0,    # roll 限制（度）
    damage_scale=100.0,       # 伤害缩放系数
    initial_health=100.0,     # 初始血量
    initial_health_a=None,    # 机器人 A 初始血量
    initial_health_b=None,    # 机器人 B 初始血量
    plugins=None,             # 额外插件列表
)
```

## 目录结构

```
humanoid21/
├── __init__.py       # 工厂函数 make_env
├── simulator.py      # MujoCombatSimulator
├── rl_adapter.py     # Humanoid21RLAdapter
├── plugins.py        # 业务插件
├── run_round.py      # 单回合比赛脚本
└── run_match.py      # 多回合比赛脚本
```

## 相关文档

- [SPEC.md](SPEC.md) - 观测空间、动作空间、控制模式详细说明
- [Policy 文档](../../policy/README.md) - 策略接口定义
- [run_round.py](run_round.py) - 单回合比赛脚本
- [run_match.py](run_match.py) - 多回合比赛脚本
