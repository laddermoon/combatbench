"""
强化学习环境构建框架 - 简洁版

最简单的实现：
- StepDataBuilder: 构建 observation、reward 和 info（作为 Hook 实现）
- SimpleCombatEnv: 通用 Gym 环境
- 终止通过：时间限制（环境自动） + Hook 返回 True
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator.open_simulator import OpenSimulator
from .simrunner import SimRunner
from .hook.base_hook import BaseHook, InvokeType


# ==================== 核心接口 ====================

class StepDataBuilder(BaseHook, ABC):
    """
    Step 数据构建器（作为 Hook 实现）

    在 POST_ACTION_STEP 时被调用，构建观测、奖励和 info。
    """

    @abstractmethod
    def build_step_data(
        self,
        f_get_core_state: Callable[[], Dict[str, Any]],
        f_get_derived_state: Callable[[], Dict[str, Any]],
        f_get_sensor_data: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Any]]:
        """
        构建观测、奖励和 info

        Args:
            f_get_core_state: 获取核心状态的函数
            f_get_derived_state: 获取衍生状态的函数
            f_get_sensor_data: 获取传感器数据的函数

        Returns:
            (observation, reward, info)
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """返回观测空间"""
        pass

    def get_last_data(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """获取最近构建的数据"""
        return (
            getattr(self, '_last_observation', None),
            getattr(self, '_last_reward', None),
            getattr(self, '_last_info', None),
        )

    # Hook 接口实现
    @property
    def priority(self) -> int:
        return -50  # 在 POST_ACTION_STEP 时执行

    def invoke(self, invoke_type: InvokeType, *args, **kwargs) -> bool:
        if invoke_type == InvokeType.POST_ACTION_STEP:
            f_get_core_state = kwargs.get('f_get_core_state')
            f_get_derived_state = kwargs.get('f_get_derived_state')
            f_get_sensor_data = kwargs.get('f_get_sensor_data')

            if f_get_core_state and f_get_derived_state and f_get_sensor_data:
                observation, reward, info = self.build_step_data(
                    f_get_core_state,
                    f_get_derived_state,
                    f_get_sensor_data,
                )
                self._last_observation = observation
                self._last_reward = reward
                self._last_info = info

        return False  # 不终止


# ==================== 简化的 Gym 环境 ====================

class SimpleCombatEnv(gym.Env):
    """
    简化的格斗环境

    通用的 Gym 环境，适用于任何实现 OpenSimulator 接口的仿真器。

    终止条件：
    - 时间到：自动终止（通过 match_duration 参数控制）
    - 其他：Hook 返回 True

    使用方式：
        env = SimpleCombatEnv(
            simulator=MySimulator(),
            step_data_builder=MyStepDataBuilder(),
            match_duration=30.0,  # 比赛时长（秒）
            hooks=[...],  # 可选的自定义 Hooks
        )
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        simulator: OpenSimulator,
        step_data_builder: StepDataBuilder,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        hooks: Optional[list] = None,
    ):
        """
        初始化环境

        Args:
            simulator: 实现 OpenSimulator 接口的仿真器
            step_data_builder: Step 数据构建器
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            hooks: 可选的 Hook 列表
        """
        super().__init__()

        self.simulator = simulator
        self.step_data_builder = step_data_builder
        self.match_duration = match_duration
        self.control_frequency = control_frequency

        # 计算参数
        dt = simulator.dt
        sim_frequency = 1.0 / dt
        self.phy_steps_per_action = max(1, int(round(sim_frequency / control_frequency)))
        self.max_steps = int(match_duration * control_frequency)

        # 创建 SimRunner
        self.runner = SimRunner(
            simulator=simulator,
            phy_steps_per_action=self.phy_steps_per_action,
            video_fps=30,
            enable_video=False,
        )

        # 将 step_data_builder 作为 Hook 附加
        self.runner.attach_hook(step_data_builder)

        # 附加额外的 Hooks
        if hooks:
            for hook_spec in hooks:
                if isinstance(hook_spec, tuple):
                    hook, invoke_types = hook_spec
                    self.runner.attach_hook(hook, invoke_types=invoke_types)
                else:
                    self.runner.attach_hook(hook)

        # 空间由具体实现定义
        self.observation_space = step_data_builder.get_observation_space()
        # action_space 需要由子类或具体实现设置
        self.action_space = None

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.runner.reset()

        # 手动调用一次 step_data_builder 获取初始观测
        obs, reward, info = self.step_data_builder.get_last_data()
        if obs is None:
            obs, reward, info = self._get_initial_data()

        return obs, info

    def step(self, action):
        self.current_step += 1

        # 时间到，终止
        if self.current_step > self.max_steps:
            obs, reward, info = self.step_data_builder.get_last_data()
            if obs is None:
                obs, reward, info = self._get_data()
            return obs, reward, True, False, info

        # Hook 已终止
        if not self.runner.is_episode_active:
            obs, reward, info = self.step_data_builder.get_last_data()
            if obs is None:
                obs, reward, info = self._get_data()
            return obs, reward, True, False, info

        self.runner.step(action)

        # 获取数据（从 Hook 缓存）
        obs, reward, info = self.step_data_builder.get_last_data()

        if obs is None:
            obs, reward, info = self._get_data()

        # 检查是否由 Hook 终止
        terminated = not self.runner.is_episode_active

        return obs, reward, terminated, False, info

    def _get_data(self):
        """直接从 simulator 获取数据（备用方法）"""
        # 默认实现，子类可以覆盖
        obs_a = self.simulator.robot_a.get_observation(opponent_robot=self.simulator.robot_b)
        obs_b = self.simulator.robot_b.get_observation(opponent_robot=self.simulator.robot_a)
        observation = {
            'robot_a_obs': obs_a.astype(np.float32),
            'robot_b_obs': obs_b.astype(np.float32),
        }
        reward = {'robot_a': 0.0, 'robot_b': 0.0}
        info = {'step': self.current_step}
        return observation, reward, info

    def _get_initial_data(self):
        """获取初始数据"""
        return self._get_data()

    def render(self):
        return self.runner.get_broadcastview_image()

    def close(self):
        self.runner.close()


# ==================== 导出 ====================

__all__ = [
    # 核心接口
    'StepDataBuilder',

    # 环境
    'SimpleCombatEnv',
]
