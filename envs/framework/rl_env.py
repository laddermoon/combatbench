"""
强化学习环境构建框架

最简单的实现：
- StepDataBuilder: 构建 observation、reward 和 info（作为 Hook 实现）
- CombatGymEnv: 通用 Gym 环境（框架类，直接使用）
- 终止通过：时间限制（环境自动） + Hook 返回 True
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .open_simulator import OpenSimulator
from .simrunner import SimRunner
from .base_hook import BaseHook, InvokeType


# ==================== 核心接口 ====================

class StepDataBuilder(BaseHook, ABC):
    """
    Step 数据构建器（作为 Hook 实现）

    在 POST_ACTION_STEP 时被调用，构建观测、奖励和 info。

    子类需要实现 build_step_data() 和 get_observation_space() 方法。
    """

    def __init__(self):
        super().__init__()
        self._core_state = None
        self._derived_state = None
        self._sensor_data = None

    @abstractmethod
    def build_step_data(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        构建观测、奖励和 info

        通过 self._core_state, self._derived_state, self._sensor_data
        访问已存储的仿真状态数据。

        Returns:
            (observation, reward, info)
        """
        pass

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """返回观测空间"""
        pass

    def get_last_data(self) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
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
                # 获取并存储状态数据
                self._core_state = f_get_core_state()
                self._derived_state = f_get_derived_state()
                self._sensor_data = f_get_sensor_data()

                observation, reward, info = self.build_step_data()
                self._last_observation = observation
                self._last_reward = reward
                self._last_info = info

        return False  # 不终止


# ==================== 简化的 Gym 环境 ====================

class CombatGymEnv(gym.Env):
    """
    通用的格斗 Gym 环境框架类（直接使用，无需继承）

    这是一个框架类，封装了 Gym 环境的标准功能和仿真循环控制。
    适用于任何实现 OpenSimulator 接口的仿真器。

    特点：
    - 封装 Gym 接口（reset, step, render, close）
    - 管理仿真循环和物理步进
    - 支持 Hook 机制扩展功能
    - 支持视频录制

    终止条件：
    - 时间到：自动终止（通过 match_duration 参数控制）
    - Hook 返回 True

    使用方式：
        env = CombatGymEnv(
            simulator=MySimulator(),
            step_data_builder=MyStepDataBuilder(),
            match_duration=30.0,
            control_frequency=20.0,
            hooks=[...],
        )
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    注意：这是框架类，不是抽象基类。应该直接使用，而不是继承。
          如需自定义环境，请通过自定义 StepDataBuilder 和 Hook 实现。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        simulator: OpenSimulator,
        step_data_builder: StepDataBuilder,
        match_duration: float = 30.0,
        control_frequency: float = 20.0,
        hooks: Optional[list] = None,
        record_video: bool = False,
        video_fps: int = 30,
    ):
        """
        初始化环境

        Args:
            simulator: 实现 OpenSimulator 接口的仿真器
            step_data_builder: Step 数据构建器
            match_duration: 比赛时长（秒）
            control_frequency: 控制频率（Hz）
            hooks: 可选的 Hook 列表
            record_video: 是否录制视频
            video_fps: 视频帧率
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
            video_fps=video_fps,
            enable_video=record_video,
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
                    self.runner.attach_hook(hook_spec)

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
        # 尝试通过 step_data_builder 获取正确格式的数据
        try:
            # 获取并存储状态数据到 step_data_builder
            self.step_data_builder._core_state = self.simulator.get_core_state()
            self.step_data_builder._derived_state = self.simulator.get_derived_state()
            self.step_data_builder._sensor_data = self.simulator.get_sensor_data()
            obs, reward, info = self.step_data_builder.build_step_data()
            self.step_data_builder._last_observation = obs
            self.step_data_builder._last_reward = reward
            self.step_data_builder._last_info = info
            return obs, reward, info
        except:
            # 如果失败，使用默认实现
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
        # 尝试通过 step_data_builder 获取正确格式的数据
        try:
            # 获取并存储状态数据到 step_data_builder
            self.step_data_builder._core_state = self.simulator.get_core_state()
            self.step_data_builder._derived_state = self.simulator.get_derived_state()
            self.step_data_builder._sensor_data = self.simulator.get_sensor_data()
            obs, reward, info = self.step_data_builder.build_step_data()
            self.step_data_builder._last_observation = obs
            self.step_data_builder._last_reward = reward
            self.step_data_builder._last_info = info
            return obs, reward, info
        except:
            # 如果失败，使用备用方法
            return self._get_data()

    def render(self):
        return self.runner.get_broadcastview_image()

    def close(self):
        self.runner.close()

    # ==================== 视频录制相关方法 ====================

    def get_video_buffer(self) -> List[np.ndarray]:
        """获取视频缓冲区"""
        return self.runner.get_video_buffer()

    def clear_video_buffer(self) -> None:
        """清空视频缓冲区"""
        self.runner.clear_video_buffer()

    def save_video(self, filepath: str, fps: Optional[int] = None) -> bool:
        """
        保存视频到指定路径

        Args:
            filepath: 输出文件路径
            fps: 视频帧率，如果为 None 则使用当前设置的 video_fps

        Returns:
            是否成功保存
        """
        return self.runner.save_video(filepath, fps)

    @property
    def video_enabled(self) -> bool:
        """视频录制是否启用"""
        return self.runner.video_enabled

    @video_enabled.setter
    def video_enabled(self, value: bool) -> None:
        """设置视频录制开关"""
        self.runner.video_enabled = value


# ==================== 导出 ====================

__all__ = [
    # 核心接口
    'StepDataBuilder',

    # 环境
    'CombatGymEnv',
]
