from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# 把框架加进路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import EnvRuntime

from .simulator import T800Simulator
from .observer_plugins import T800Observer
from .plugins import T800CombatScoringPlugin, FrozenRobotPlugin


def make_env(
    control_frequency: int = 20,
    match_duration: float = 30.0,
    plugins: Optional[List[Any]] = None,
    observer_plugins: Optional[Dict[str, Any]] = None,
) -> EnvRuntime:
    """
    T800 对战环境工厂函数（参照 humanoid21/__init__.py 结构）。
    默认挂载 T800CombatScoringPlugin 实现 HP/KO 判定。
    """
    sim_frequency = 1.0 / T800Simulator.DT
    phy_steps_per_action = max(1, int(round(sim_frequency / control_frequency)))
    max_steps = int(match_duration * control_frequency)

    simulator = T800Simulator()

    # 默认插件：战斗计分（HP、伤害、KO）
    active_plugins = plugins if plugins is not None else [T800CombatScoringPlugin(damage_scale=50.0)]

    # observer plugins
    if observer_plugins is None:
        observer_plugins = {}
    default_observers = {
        'robot_a_obs': T800Observer('robot_a'),
        'robot_b_obs': T800Observer('robot_b'),
    }
    for key, value in default_observers.items():
        if key not in observer_plugins:
            observer_plugins[key] = value

    runtime = EnvRuntime(
        simulator=simulator,
        plugins=active_plugins,
        observer_plugins=observer_plugins,
        phy_steps_per_action=phy_steps_per_action,
        max_steps=max_steps,
    )

    runtime.action_space = T800Observer.get_action_space()
    runtime.observation_space = T800Observer.get_observation_space()
    return runtime


__all__ = [
    "T800Simulator",
    "T800Observer",
    "T800CombatScoringPlugin",
    "FrozenRobotPlugin",
    "make_env",
]
