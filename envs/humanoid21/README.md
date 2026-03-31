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
└── run_round.py      # 回合运行脚本
```

## 相关文档

- [SPEC.md](SPEC.md) - 观测空间、动作空间、控制模式详细说明
- [Policy 文档](../../policy/README.md) - 策略接口定义
