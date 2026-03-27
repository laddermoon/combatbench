"""
Humanoid21 格斗环境

基于 SimpleCombatEnv 框架的 21 自由度人形机器人格斗环境。
"""

from typing import Any, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..framework import SimpleCombatEnv, StepDataBuilder
from .humanoid21 import Humanoid21Simulator
from .humanoid21_base_hook import DefaultStepDataBuilder, HealthTerminationHook
from .robot import HumanoidRobot


class Humanoid21CombatEnv(SimpleCombatEnv):
    """
    Humanoid21 格斗环境

    基于 SimpleCombatEnv 的具体实现，专门用于 21 自由度人形机器人格斗。

    终止条件：
    - 时间到：自动终止（通过 match_duration 参数控制）
    - 血量归零：当任一机器人血量归零时终止（如果启用 HealthTerminationHook）

    使用方式：
        env = Humanoid21CombatEnv(
            render_mode=None,
            match_duration=30.0,
            enable_health_termination=True,
        )
        obs, info = env.reset()

        for _ in range(1000):
            action = {
                'robot_a': env.action_space['robot_a'].sample(),
                'robot_b': env.action_space['robot_b'].sample(),
            }
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        initial_distance: float = 2.0,
        enable_health_termination: bool = True,
        score_calculator=None,
    ):
        """
        初始化 Humanoid21 格斗环境

        Args:
            render_mode: 渲染模式 ("human", "rgb_array", None)
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            initial_distance: 机器人初始距离（米）
            enable_health_termination: 是否启用血量终止
            score_calculator: 血量计算器（如果启用血量终止）
        """
        # 创建仿真器
        simulator = Humanoid21Simulator(
            gui=(render_mode == "human"),
            initial_distance=initial_distance,
        )

        # 创建 step_data_builder
        step_data_builder = DefaultStepDataBuilder(score_calculator=score_calculator)

        # 创建 Hooks
        hooks = []
        if enable_health_termination and score_calculator is not None:
            hooks.append(HealthTerminationHook(score_calculator))

        # 初始化父类
        super().__init__(
            simulator=simulator,
            step_data_builder=step_data_builder,
            match_duration=match_duration,
            control_frequency=control_frequency,
            hooks=hooks,
        )

        # 设置 action_space（Humanoid21 特定）
        action_dim = HumanoidRobot.ACTION_DIM
        self.action_space = spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        })

        self.render_mode = render_mode


# ==================== 导出 ====================

__all__ = [
    'Humanoid21CombatEnv',
    'DefaultStepDataBuilder',
    'HealthTerminationHook',
]
