"""
CombatBench 环境模块

提供强化学习环境和仿真框架。
"""

# 框架模块
from .framework import (
    SimRunner,
    BaseHook,
    InvokeType,
    SimpleCombatEnv,
    StepDataBuilder,
    OpenSimulator,
)

# 核心模块（与具体机器人无关）
from .core import (
    PhysicsEngine,
    BaseRobot,
)

# Humanoid21 特定实现
from .humanoid21 import (
    Humanoid21Simulator,
    HumanoidRobot,
    CollisionDetector,
    ScoreCalculator,
    DefaultStepDataBuilder,
    HealthTerminationHook,
    Humanoid21NonFallEnv,
    Humanoid21FallEnv,
)

__all__ = [
    # 框架模块
    "SimRunner",
    "BaseHook",
    "InvokeType",
    "SimpleCombatEnv",
    "StepDataBuilder",
    "OpenSimulator",

    # 核心模块
    "PhysicsEngine",
    "BaseRobot",

    # Humanoid21 特定实现
    "Humanoid21Simulator",
    "HumanoidRobot",
    "CollisionDetector",
    "ScoreCalculator",
    "DefaultStepDataBuilder",
    "HealthTerminationHook",

    # 预置环境
    "Humanoid21NonFallEnv",
    "Humanoid21FallEnv",
]
