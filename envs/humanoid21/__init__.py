from typing import Any, Dict, List, Optional
import os
import sys
from pathlib import Path

# 把框架加进路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import CombatGymEnv

from .simulator import MujocoCombatSimulator
from .rl_adapter import Humanoid21RLAdapter
from .plugins import NonFallConstraintPlugin, CombatScoringPlugin
from .pd_controller import PDControllerPlugin

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
    plugins: Optional[List[Any]] = None,
) -> CombatGymEnv:
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
    
    # 2. 创建 RL 适配层
    rl_adapter = Humanoid21RLAdapter()
    
    # 3. 挂载业务插件
    active_plugins = []
    
    # PD 控制器插件（取代直接力量矩控制，用于稳定站立）
    active_plugins.append(PDControllerPlugin())
    
    # 算分插件（必选）
    active_plugins.append(CombatScoringPlugin(initial_health=initial_health, damage_scale=damage_scale))
    
    # 防摔倒约束插件（可选）
    if non_fall_mode:
        active_plugins.append(NonFallConstraintPlugin(
            pitch_limit_deg=non_fall_pitch_limit_deg,
            roll_limit_deg=non_fall_roll_limit_deg
        ))
        
    # 添加用户额外传入的插件（比如 VideoRecorderPlugin 等）
    if plugins:
        active_plugins.extend(plugins)
        
    # 4. 组装并返回 Env
    env = CombatGymEnv(
        simulator=simulator,
        rl_adapter=rl_adapter,
        plugins=active_plugins,
        phy_steps_per_action=phy_steps_per_action,
        max_steps=max_steps
    )
    
    return env

__all__ = [
    "MujocoCombatSimulator",
    "Humanoid21RLAdapter",
    "NonFallConstraintPlugin",
    "CombatScoringPlugin",
    "PDControllerPlugin",
    "make_env"
]
