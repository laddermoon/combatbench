"""
CombatBench 核心模块

通用的核心组件，与具体机器人实现无关。
"""

from .physics import PhysicsEngine
from .base_robot import BaseRobot

__all__ = [
    "PhysicsEngine",
    "BaseRobot",
]
