"""
CombatBench 环境模块

提供强化学习环境和仿真框架。
"""

from .combat_gym import CombatGymEnv
from .simrunner import SimRunner
from .hook.base_hook import BaseHook, InvokeType

# 通用框架
from .rl_env import (
    SimpleCombatEnv,
    StepDataBuilder,
)

# Humanoid21 特定实现
from .humanoid21 import (
    Humanoid21Simulator,
    DefaultStepDataBuilder,
    HealthTerminationHook,
    Humanoid21CombatEnv,
)

# 预置环境
from .preset_envs import (
    Humanoid21NonFallEnv,
    Humanoid21FallEnv,
)

__all__ = [
    # 原始 Gym 环境
    "CombatGymEnv",

    # 仿真框架
    "SimRunner",

    # Hook 支持
    "BaseHook",
    "InvokeType",

    # 通用 RL 环境框架
    "SimpleCombatEnv",
    "StepDataBuilder",

    # Humanoid21 特定实现
    "Humanoid21Simulator",
    "DefaultStepDataBuilder",
    "HealthTerminationHook",
    "Humanoid21CombatEnv",

    # 预置环境
    "Humanoid21NonFallEnv",
    "Humanoid21FallEnv",
]
