from typing import Any, Dict, List, Optional
import os
import sys
from pathlib import Path

# 把框架加进路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import PolicyRuntime

from .simulator import MujocoCombatSimulator
from .plugins import NonFallConstraintPlugin, CombatScoringPlugin, FrozenRobotPlugin
from .runtime_units import Humanoid21Observer, Humanoid21Rewarder, build_shared_runtime_info

def make_env(
    arena_xml: Optional[str] = None,
    dt: float = 0.002,
    control_frequency: int = 20,
    match_duration: float = 30.0,
    non_fall_mode: bool = False,
    non_fall_pitch_limit_deg: float = 5.0,
    non_fall_roll_limit_deg: float = 5.0,
    damage_scale: float = 100.0,
    initial_health: float = 100.0,
    initial_health_a: Optional[float] = None,
    initial_health_b: Optional[float] = None,
    plugins: Optional[List[Any]] = None,
) -> PolicyRuntime:
    """
    工厂函数，用于创建组装好的 Humanoid21 对战环境。
    """
    if arena_xml is None:
        arena_xml = os.path.join(os.path.dirname(__file__), '../../assets/battle_v1.xml')
        
    sim_frequency = 1.0 / dt
    phy_steps_per_action = max(1, int(round(sim_frequency / control_frequency)))
    max_steps = int(match_duration * control_frequency)
    
    # 1. 创建底层物理仿真器
    simulator = MujocoCombatSimulator(arena_xml=arena_xml, dt=dt)
    
    # 2. 挂载业务插件
    active_plugins = []
    
    # 算分插件（必选）
    active_plugins.append(CombatScoringPlugin(
        initial_health=initial_health,
        initial_health_a=initial_health_a,
        initial_health_b=initial_health_b,
        damage_scale=damage_scale
    ))
    
    # 防摔倒约束插件（可选）
    if non_fall_mode:
        active_plugins.append(NonFallConstraintPlugin(
            pitch_limit_deg=non_fall_pitch_limit_deg,
            roll_limit_deg=non_fall_roll_limit_deg
        ))
        
    # 添加用户额外传入的插件（比如 VideoRecorderPlugin 等）
    if plugins:
        active_plugins.extend(plugins)

    runtime = PolicyRuntime(
        simulator=simulator,
        plugins=active_plugins,
        observers={
            'robot_a': Humanoid21Observer('robot_a'),
            'robot_b': Humanoid21Observer('robot_b'),
        },
        rewarders={
            'robot_a': Humanoid21Rewarder('robot_a'),
            'robot_b': Humanoid21Rewarder('robot_b'),
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
    "NonFallConstraintPlugin",
    "CombatScoringPlugin",
    "FrozenRobotPlugin",
    "make_env"
]
