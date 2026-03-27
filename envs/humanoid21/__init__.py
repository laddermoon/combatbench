"""
Humanoid21 模块

21 自由度人形机器人格斗仿真器和相关组件。
"""

from .humanoid21 import Humanoid21Simulator
from .robot import HumanoidRobot
from .collision import CollisionDetector
from .scoring import ScoreCalculator
from .humanoid21_base_hook import DefaultStepDataBuilder, HealthTerminationHook

# 导入所有环境和 Hooks
from .envs import (
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
    # 核心组件
    'Humanoid21Simulator',
    'HumanoidRobot',
    'CollisionDetector',
    'ScoreCalculator',
    'DefaultStepDataBuilder',
    'HealthTerminationHook',

    # 单智能体环境
    'Humanoid21SingleAgentEnv',
    'Humanoid21VsFrozenEnv',
    'Humanoid21VsStandingEnv',
    'Humanoid21VsPolicyEnv',
    'Humanoid21NonFallEnv',
    'Humanoid21FallEnv',

    # 双智能体环境
    'Humanoid21DualAgentEnv',
    'Humanoid21MatchEnv',

    # Hooks
    'FallDetectionHook',
    'UprightConstraintHook',
    'FreezeRobotHook',
    'OpponentPolicyHook',

    # 数据构建器
    'SingleAgentStepDataBuilder',
    'DualAgentStepDataBuilder',
]
