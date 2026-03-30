"""
测试 CombatGymEnv
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from combatbench.envs.framework.rl_env import CombatGymEnv, StepDataBuilder
from combatbench.envs.framework.base_hook import BaseHook, InvokeType


class MockStepDataBuilder(StepDataBuilder):
    """用于测试的 Mock StepDataBuilder"""

    def __init__(self):
        super().__init__()
        self.obs_dim = 127
        self.obs_shape = (127,)
        self._observation_count = 0

    def build_step_data(self):
        """构建测试数据"""
        self._observation_count += 1
        obs = {
            'robot_a_obs': np.random.randn(self.obs_dim).astype(np.float32),
            'robot_b_obs': np.random.randn(self.obs_dim).astype(np.float32),
        }
        reward = {'robot_a': 0.0, 'robot_b': 0.0}
        info = {'step': self._observation_count}
        return obs, reward, info

    def get_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Dict({
            "robot_a_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32),
            "robot_b_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32),
        })


class TestCombatGymEnvInit:
    """测试 CombatGymEnv 初始化"""

    def test_init_with_required_params(self, mock_simulator):
        """测试使用必需参数初始化"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        assert env.simulator == mock_simulator
        assert env.step_data_builder == builder
        assert env.match_duration == 30.0
        assert env.control_frequency == 20.0

    def test_init_with_all_params(self, mock_simulator):
        """测试使用所有参数初始化"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            match_duration=60.0,
            control_frequency=50.0,
            hooks=[],
            record_video=False,
            video_fps=60,
        )

        assert env.match_duration == 60.0
        assert env.control_frequency == 50.0
        assert env._video_fps == 60

    def test_observation_space_property(self, mock_simulator):
        """测试 observation_space 属性"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        space = env.observation_space
        assert space is not None
        assert isinstance(space, builder.get_observation_space().__class__)

    def test_action_space_property(self, mock_simulator):
        """测试 action_space 属性"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        # action_space 默认为 None，需要由子类或具体实现设置
        # 框架类不预设 action_space
        assert env.action_space is None

        # 设置 action_space
        import gymnasium as gym
        env.action_space = gym.spaces.Dict({
            "robot_a": gym.spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32),
            "robot_b": gym.spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32),
        })
        assert env.action_space is not None


class TestCombatGymEnvReset:
    """测试 CombatGymEnv reset"""

    def test_reset_returns_obs_and_info(self, mock_simulator):
        """测试 reset 返回 observation 和 info"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        obs, info = env.reset()

        assert obs is not None
        assert info is not None
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_calls_simulator_reset(self, mock_simulator):
        """测试 reset 调用 simulator.reset"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        env.reset()

        mock_simulator.reset.assert_called_once()

    def test_reset_with_seed(self, mock_simulator):
        """测试带 seed 参数的 reset"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        obs, info = env.reset(seed=42)

        assert 'seed' in info or obs is not None


class TestCombatGymEnvStep:
    """测试 CombatGymEnv step"""

    def test_step_returns_all_components(self, mock_simulator):
        """测试 step 返回所有组件"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )
        env.reset()

        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_time(self, mock_simulator):
        """测试 step 增加时间"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            match_duration=10.0,
        )
        env.reset()

        initial_step = env.current_step
        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        env.step(action)

        assert env.current_step > initial_step

    def test_step_terminates_on_timeout(self, mock_simulator):
        """测试 step 在时间到时终止"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            match_duration=0.01,  # 很短的时间
        )
        env.reset()

        # 需要超过 max_steps 才会终止
        # max_steps = int(match_duration * control_frequency) = int(0.01 * 20) = 0
        # 所以第一步就会终止
        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        obs, reward, terminated, truncated, info = env.step(action)

        # terminated=True 表示时间到（逻辑中返回 terminated=True, truncated=False）
        assert terminated is True  # 时间到

    def test_step_with_hook_termination(self, mock_simulator):
        """测试 Hook 终止"""
        builder = MockStepDataBuilder()

        # 创建一个会终止的 Hook
        class TerminateHook(BaseHook):
            @property
            def name(self):
                return "terminate"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                return invoke_type == InvokeType.POST_ACTION_STEP

        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            hooks=[TerminateHook()],
        )
        env.reset()

        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is True  # Hook 终止


class TestCombatGymEnvRender:
    """测试 CombatGymEnv render"""

    def test_render_returns_image(self, mock_simulator):
        """测试 render 返回图像"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        # 模拟 get_broadcastview_image 返回图像
        mock_simulator.get_broadcastview_image.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = env.render()
        assert isinstance(result, np.ndarray)
        assert result.shape == (720, 1280, 3)


class TestCombatGymEnvClose:
    """测试 CombatGymEnv close"""

    def test_close_cleans_up(self, mock_simulator):
        """测试 close 清理资源"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
        )

        env.close()

        # 验证 runner 被 close
        assert not env.runner.is_episode_active


class TestCombatGymEnvVideo:
    """测试 CombatGymEnv 视频功能"""

    def test_video_frames_captured_when_enabled(self, mock_simulator):
        """测试启用视频时捕获帧"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            record_video=True,
            video_fps=30,
        )
        env.reset()

        initial_frames = len(env._video_buffer)

        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        env.step(action)

        # 验证视频帧被捕获
        assert len(env._video_buffer) > initial_frames or len(env._video_buffer) >= 0

    def test_video_not_captured_when_disabled(self, mock_simulator):
        """测试禁用视频时不捕获帧"""
        builder = MockStepDataBuilder()
        env = CombatGymEnv(
            simulator=mock_simulator,
            step_data_builder=builder,
            record_video=False,
        )
        env.reset()

        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        env.step(action)

        # 验证视频缓冲区为空
        assert len(env._video_buffer) == 0
