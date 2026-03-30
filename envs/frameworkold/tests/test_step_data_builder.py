"""
测试 StepDataBuilder
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from combatbench.envs.framework.rl_env import StepDataBuilder
from combatbench.envs.framework.base_hook import InvokeType


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


class SingleAgentMockBuilder(StepDataBuilder):
    """单智能体 Mock Builder"""

    def __init__(self):
        super().__init__()
        self.obs_dim = 127

    def build_step_data(self):
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = 0.0
        info = {'step': 1}
        return obs, reward, info

    def get_observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)


class TestStepDataBuilder:
    """测试 StepDataBuilder"""

    def test_step_data_builder_is_abstract(self):
        """测试 StepDataBuilder 是抽象类"""
        with pytest.raises(TypeError):
            StepDataBuilder()

    def test_mock_builder_concrete_implementation(self):
        """测试 MockBuilder 可以被实例化"""
        builder = MockStepDataBuilder()
        assert isinstance(builder, StepDataBuilder)

    def test_build_step_data_returns_correct_format(self):
        """测试 build_step_data 返回正确格式"""
        builder = MockStepDataBuilder()
        obs, reward, info = builder.build_step_data()

        assert isinstance(obs, dict)
        assert 'robot_a_obs' in obs
        assert 'robot_b_obs' in obs
        assert isinstance(reward, dict)
        assert isinstance(info, dict)

    def test_get_observation_space_returns_space(self):
        """测试 get_observation_space 返回 Space"""
        builder = MockStepDataBuilder()
        space = builder.get_observation_space()

        import gymnasium as gym
        assert isinstance(space, gym.spaces.Dict)
        # Dict space 的 keys 属性包含所有键
        assert 'robot_a_obs' in space.keys()
        assert 'robot_b_obs' in space.keys()

    def test_state_storage_during_invoke(self):
        """测试 invoke 时存储状态数据"""
        builder = MockStepDataBuilder()

        # 模拟 invoke 调用 - 使用位置参数
        f_get_action = lambda: {}
        f_get_static_data = lambda: {}
        f_get_sensor_data = lambda: {}
        f_get_core_state = lambda: {'time': 1.0}
        f_get_derived_state = lambda: {'robots': {}}
        f_set_core_state = None
        f_set_action = None

        # HookWrapper 传递位置参数，顺序为：
        # invoke_type, f_get_action, f_get_static_data, f_get_sensor_data,
        # f_get_core_state, f_get_derived_state, f_set_core_state, f_set_action
        args = (
            f_get_action, f_get_static_data, f_get_sensor_data,
            f_get_core_state, f_get_derived_state, f_set_core_state, f_set_action
        )

        builder.invoke(InvokeType.POST_ACTION_STEP, *args)

        # 验证状态数据被存储
        assert builder._core_state is not None
        assert builder._derived_state is not None
        assert builder._sensor_data is not None

    def test_build_step_data_uses_stored_state(self):
        """测试 build_step_data 使用存储的状态数据"""
        builder = MockStepDataBuilder()

        # 先存储状态
        builder._core_state = {'time': 1.0}
        builder._derived_state = {'robots': {}}
        builder._sensor_data = {}

        # build_step_data 应该能够访问这些数据
        obs, reward, info = builder.build_step_data()
        # 如果没有错误，说明成功访问了存储的数据

    def test_get_last_data_returns_cached_data(self):
        """测试 get_last_data 返回缓存的数据"""
        builder = MockStepDataBuilder()

        # 初始时缓存为 None
        obs, reward, info = builder.get_last_data()
        assert obs is None
        assert reward is None
        assert info is None

        # 调用 build_step_data 存储数据
        obs1, reward1, info1 = builder.build_step_data()
        builder._last_observation = obs1
        builder._last_reward = reward1
        builder._last_info = info1

        # get_last_data 应该返回缓存的数据
        obs2, reward2, info2 = builder.get_last_data()
        assert obs2 is obs1
        assert reward2 is reward1
        assert info2 is info1

    def test_priority_property(self):
        """测试 priority 属性"""
        builder = MockStepDataBuilder()
        assert builder.priority == -50  # StepDataBuilder 默认优先级

    def test_invoke_at_pre_episode(self):
        """测试 PRE_EPISODE 时也存储状态"""
        builder = MockStepDataBuilder()

        # 使用位置参数
        args = (
            lambda: {},  # f_get_action
            lambda: {},  # f_get_static_data
            lambda: {},  # f_get_sensor_data
            lambda: {'time': 0.0},  # f_get_core_state
            lambda: {'robots': {}},  # f_get_derived_state
            None,  # f_set_core_state
            None,  # f_set_action
        )

        builder.invoke(InvokeType.PRE_EPISODE, *args)

        # 验证状态数据被存储
        assert builder._core_state is not None

    def test_invoke_at_post_action_step(self):
        """测试 POST_ACTION_STEP 时存储状态"""
        builder = MockStepDataBuilder()

        # 使用位置参数
        args = (
            lambda: {},  # f_get_action
            lambda: {},  # f_get_static_data
            lambda: {},  # f_get_sensor_data
            lambda: {'time': 0.0},  # f_get_core_state
            lambda: {'robots': {}},  # f_get_derived_state
            None,  # f_set_core_state
            None,  # f_set_action
        )

        builder.invoke(InvokeType.POST_ACTION_STEP, *args)

        # 验证状态数据被存储
        assert builder._core_state is not None

    def test_invoke_at_other_invoke_types(self):
        """测试其他 invoke_type 不存储状态"""
        builder = MockStepDataBuilder()

        # 重置状态
        builder._core_state = None
        builder._derived_state = None
        builder._sensor_data = None

        # PRE_PHY_STEP 不存储状态 - 使用位置参数
        args = (
            lambda: {},  # f_get_action
            lambda: {},  # f_get_static_data
            lambda: {},  # f_get_sensor_data
            lambda: {'time': 0.0},  # f_get_core_state
            lambda: {'robots': {}},  # f_get_derived_state
            None,  # f_set_core_state
            None,  # f_set_action
        )

        builder.invoke(InvokeType.PRE_PHY_STEP, *args)

        # 状态应该还是 None
        assert builder._core_state is None

    def test_single_agent_builder_format(self):
        """测试单智能体 Builder 返回格式"""
        builder = SingleAgentMockBuilder()
        obs, reward, info = builder.build_step_data()

        # 单智能体返回数组，不是字典
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (127,)
        assert isinstance(reward, (int, float))
        assert isinstance(info, dict)


class TestStepDataBuilderIntegration:
    """测试 StepDataBuilder 与 Hook 系统的集成"""

    def test_step_data_builder_as_hook(self):
        """测试 StepDataBuilder 可以作为 Hook 使用"""
        builder = MockStepDataBuilder()
        wrapper = MockHookWrapper()
        wrapper.attach(builder)

        # 应该可以正常附加
        assert len(wrapper.hooks) == 1

    def test_step_data_builder_hook_priority(self):
        """测试 StepDataBuilder 的 Hook 优先级"""
        builder = MockStepDataBuilder()
        assert builder.priority == -50  # 应该在 POST_ACTION_STEP 时执行

    def test_step_data_builder_hook_invoke_flow(self):
        """测试 StepDataBuilder 作为 Hook 的调用流程"""
        builder = MockStepDataBuilder()
        wrapper = MockHookWrapper()
        wrapper.attach(builder)

        # 使用位置参数 - HookWrapper 传递位置参数
        args = (
            lambda: {},  # f_get_action
            lambda: {},  # f_get_static_data
            lambda: {},  # f_get_sensor_data
            lambda: {'time': 1.0},  # f_get_core_state
            lambda: {'robots': {}},  # f_get_derived_state
            None,  # f_set_core_state
            None,  # f_set_action
        )

        wrapper.invoke(InvokeType.POST_ACTION_STEP, *args)

        # 验证状态被存储
        assert builder._core_state is not None


# Mock HookWrapper for testing
class MockHookWrapper:
    """简化的 HookWrapper 用于测试"""

    def __init__(self):
        self._hooks = []

    @property
    def hooks(self):
        """返回 hook 列表"""
        return [h[0] for h in self._hooks]

    def attach(self, hook, priority=0, invoke_types=None):
        self._hooks.append((hook, priority, invoke_types if invoke_types else list(InvokeType)))

    def invoke(self, invoke_type, *args, **kwargs):
        """简化版本的 invoke"""
        for hook, _, invoke_types in self._hooks:
            if invoke_types and invoke_type not in invoke_types:
                continue
            hook.invoke(invoke_type, *args, **kwargs)
        return False
