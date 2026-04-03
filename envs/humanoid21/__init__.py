from typing import Any, Dict, List, Optional
import os
import sys
from pathlib import Path

# 把框架加进路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import EnvRuntime

from .simulator import MujocoCombatSimulator
from .observer_plugins import Humanoid21Observer, Humanoid21Rewarder, build_shared_runtime_info

def make_env(
    control_frequency: int = 20,
    match_duration: float = 30.0,
    plugins: Optional[List[Any]] = None,
) -> EnvRuntime:
    """
    工厂函数，用于创建组装好的 Humanoid21 对战环境。
    """
    sim_frequency = 1.0 / MujocoCombatSimulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / control_frequency)))
    max_steps = int(match_duration * control_frequency)

    # 1. 创建底层物理仿真器
    simulator = MujocoCombatSimulator()

    # 2. 挂载插件（用户提供的插件）
    active_plugins = plugins if plugins else []

    runtime = EnvRuntime(
        simulator=simulator,
        plugins=active_plugins,
        observer_plugins={
            'robot_a_obs': Humanoid21Observer('robot_a'),
            'robot_b_obs': Humanoid21Observer('robot_b'),
            'robot_a_reward': Humanoid21Rewarder('robot_a'),
            'robot_b_reward': Humanoid21Rewarder('robot_b'),
        },
        phy_steps_per_action=phy_steps_per_action,
        max_steps=max_steps
    )

    runtime.action_space = Humanoid21Observer.get_action_space()
    runtime.observation_space = Humanoid21Observer.get_observation_space()
    runtime.shared_info_builder = build_shared_runtime_info
    return runtime

__all__ = [
    "MujocoCombatSimulator",
    "Humanoid21Observer",
    "Humanoid21Rewarder",
    "make_env"
]
