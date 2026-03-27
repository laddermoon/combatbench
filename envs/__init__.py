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
    # 核心组件
    Humanoid21Simulator,
    HumanoidRobot,
    CollisionDetector,
    ScoreCalculator,
    DefaultStepDataBuilder,
    HealthTerminationHook,
    # 单智能体环境
    Humanoid21SingleAgentEnv,
    Humanoid21VsFrozenEnv,
    Humanoid21VsStandingEnv,
    Humanoid21VsPolicyEnv,
    Humanoid21NonFallEnv,
    Humanoid21FallEnv,
    # 双智能体环境
    Humanoid21DualAgentEnv,
    Humanoid21MatchEnv,
    # Hooks
    FallDetectionHook,
    UprightConstraintHook,
    FreezeRobotHook,
    OpponentPolicyHook,
    # 数据构建器
    SingleAgentStepDataBuilder,
    DualAgentStepDataBuilder,
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

    # Humanoid21 核心组件
    "Humanoid21Simulator",
    "HumanoidRobot",
    "CollisionDetector",
    "ScoreCalculator",
    "DefaultStepDataBuilder",
    "HealthTerminationHook",

    # 单智能体环境
    "Humanoid21SingleAgentEnv",
    "Humanoid21VsFrozenEnv",
    "Humanoid21VsStandingEnv",
    "Humanoid21VsPolicyEnv",
    "Humanoid21NonFallEnv",
    "Humanoid21FallEnv",

    # 双智能体环境
    "Humanoid21DualAgentEnv",
    "Humanoid21MatchEnv",

    # Hooks
    "FallDetectionHook",
    "UprightConstraintHook",
    "FreezeRobotHook",
    "OpponentPolicyHook",

    # 数据构建器
    "SingleAgentStepDataBuilder",
    "DualAgentStepDataBuilder",
]
