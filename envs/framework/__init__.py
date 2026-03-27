"""
CombatBench 框架模块

通用的仿真和强化学习框架代码，与具体机器人实现无关。
"""

from .base_hook import BaseHook, InvokeType, HookWrapper
from .open_simulator import OpenSimulator
from .rl_env import StepDataBuilder, CombatGymEnv
from .simrunner import SimRunner

__all__ = [
    # Hook 框架
    "BaseHook",
    "InvokeType",
    "HookWrapper",

    # 仿真器接口
    "OpenSimulator",

    # RL 环境框架
    "StepDataBuilder",
    "CombatGymEnv",

    # 仿真运行器
    "SimRunner",
]
