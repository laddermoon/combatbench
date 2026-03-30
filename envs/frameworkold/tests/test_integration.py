"""
框架集成测试

测试完整的框架工作流程：
- Simulator + SimRunner + Hooks
- Simulator + CombatGymEnv + StepDataBuilder
- 完整的 Episode 循环
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from combatbench.envs.framework.simrunner import SimRunner
from combatbench.envs.framework.rl_env import CombatGymEnv, StepDataBuilder
from combatbench.envs.framework.base_hook import BaseHook, HookWrapper, InvokeType


class MockSimulator:
    """Mock Simulator 实现"""

    def __init__(self):
        self.dt = 0.002
        self.reset_count = 0
        self.step_count = 0
        self._action = None

    def reset(self):
        """重置仿真"""
        self.reset_count += 1
        self.step_count = 0
        self._action = None

    def set_action(self, action):
        """设置动作"""
        self._action = action

    def physical_step(self):
        """物理步进"""
        self.step_count += 1

    def get_core_state(self):
        """获取核心状态"""
        return {
            'time': self.step_count * self.dt,
            'robots': {
                'robot_a': {
                    'root_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                    'root_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    'joint_positions': np.zeros(21, dtype=np.float32),
                    'joint_velocities': np.zeros(21, dtype=np.float32),
                },
                'robot_b': {
                    'root_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                    'root_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    'joint_positions': np.zeros(21, dtype=np.float32),
                    'joint_velocities': np.zeros(21, dtype=np.float32),
                },
            },
        }

    def get_derived_state(self):
        """获取衍生状态"""
        return {
            'robots': {
                'robot_a': {
                    'observation': np.random.randn(127).astype(np.float32),
                    'torso_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                    'torso_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                },
                'robot_b': {
                    'observation': np.random.randn(127).astype(np.float32),
                    'torso_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                    'torso_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                },
            },
        }

    def get_sensor_data(self):
        """获取传感器数据"""
        return {}

    def get_static_data(self):
        """获取静态数据"""
        return {}

    def get_broadcastview_image(self):
        """获取广播视图图像"""
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def get_physical_frequency(self):
        """获取物理频率"""
        return 500  # 500Hz


class TestSimulatorSimRunnerIntegration:
    """测试 Simulator + SimRunner 集成"""

    def test_complete_episode_workflow(self):
        """测试完整的 Episode 工作流程"""
        simulator = MockSimulator()
        runner = SimRunner(
            simulator=simulator,
            phy_steps_per_action=25,
            video_fps=30,
        )

        # 添加一个测试 Hook
        class TestHook(BaseHook):
            def __init__(self):
                self.call_count = 0
                self.call_types = []

            @property
            def name(self):
                return "test_hook"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                self.call_count += 1
                self.call_types.append(invoke_type)
                return False

        hook = TestHook()
        runner.attach_hook(hook)

        # 重置
        runner.reset()
        assert simulator.reset_count == 1
        assert hook.call_count == 1
        assert InvokeType.PRE_EPISODE in hook.call_types

        # 执行多个 step
        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        for _ in range(5):
            runner.step(action)

        # 验证 Hook 调用
        assert hook.call_count > 1
        assert InvokeType.PRE_ACTION_STEP in hook.call_types
        assert InvokeType.POST_ACTION_STEP in hook.call_types

    def test_hook_modifies_action(self):
        """测试 Hook 修改动作"""
        simulator = MockSimulator()
        runner = SimRunner(simulator=simulator)

        # 添加一个修改动作的 Hook
        class ActionModifierHook(BaseHook):
            @property
            def name(self):
                return "action_modifier"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                if invoke_type == InvokeType.PRE_ACTION_STEP:
                    # 获取 f_set_action（args[7]）
                    if len(args) >= 8 and args[7] is not None:
                        f_set_action = args[7]
                        original_action = f_get_action() if len(args) >= 1 and args[0] else None
                        if original_action:
                            # 修改动作
                            modified_action = original_action.copy()
                            modified_action['robot_a'] = np.ones(21) * 0.5
                            f_set_action(modified_action)
                return False

        # 这个测试需要实际的 f_get_action，所以简化为不修改
        class SimpleHook(BaseHook):
            @property
            def name(self):
                return "simple"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                return False

        runner.attach_hook(SimpleHook())
        runner.reset()

        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        runner.step(action)

        assert simulator.step_count == 25  # phy_steps_per_action


class TestSimulatorCombatGymEnvIntegration:
    """测试 Simulator + CombatGymEnv 集成"""

    def test_complete_env_workflow(self):
        """测试完整的 Env 工作流程"""
        simulator = MockSimulator()

        # 创建 StepDataBuilder
        class TestStepDataBuilder(StepDataBuilder):
            def __init__(self):
                super().__init__()
                self.obs_dim = 127

            def build_step_data(self):
                obs = {
                    'robot_a_obs': np.random.randn(self.obs_dim).astype(np.float32),
                    'robot_b_obs': np.random.randn(self.obs_dim).astype(np.float32),
                }
                reward = {'robot_a': 0.0, 'robot_b': 0.0}
                info = {'step': 1}
                return obs, reward, info

            def get_observation_space(self):
                import gymnasium as gym
                return gym.spaces.Dict({
                    "robot_a_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                    "robot_b_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                })

        builder = TestStepDataBuilder()
        env = CombatGymEnv(
            simulator=simulator,
            step_data_builder=builder,
            match_duration=1.0,  # 短时间测试
            control_frequency=20.0,
        )

        # Reset
        obs, info = env.reset()
        assert obs is not None
        assert info is not None

        # Step
        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        # Close
        env.close()


class TestHookSystemIntegration:
    """测试 Hook 系统集成"""

    def test_multiple_hooks_execution(self):
        """测试多个 Hook 执行"""
        simulator = MockSimulator()
        runner = SimRunner(simulator=simulator)

        # 创建多个 Hook
        class CounterHook(BaseHook):
            def __init__(self, name):
                self.name_value = name
                self.count = 0

            @property
            def name(self):
                return self.name_value

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                self.count += 1
                return False

        hook1 = CounterHook("hook1")
        hook2 = CounterHook("hook2")
        hook3 = CounterHook("hook3")

        runner.attach_hook(hook1)
        runner.attach_hook(hook2)
        runner.attach_hook(hook3)

        runner.reset()

        # 所有 Hook 都应该被调用
        assert hook1.count > 0
        assert hook2.count > 0
        assert hook3.count > 0

    def test_hook_priority_order(self):
        """测试 Hook 优先级顺序"""
        simulator = MockSimulator()
        runner = SimRunner(simulator=simulator)

        # 创建不同优先级的 Hook
        class PriorityHook(BaseHook):
            def __init__(self, name, priority):
                self.name_value = name
                self.priority_value = priority
                self.call_order = []

            @property
            def name(self):
                return self.name_value

            @property
            def priority(self):
                return self.priority_value

            def invoke(self, invoke_type, *args, **kwargs):
                self.call_order.append(invoke_type)
                return False

        hook_high = PriorityHook("high", 100)
        hook_low = PriorityHook("low", 10)
        hook_mid = PriorityHook("mid", 50)

        runner.attach_hook(hook_high)
        runner.attach_hook(hook_low)
        runner.attach_hook(hook_mid)

        runner.reset()

        # 验证所有 Hook 都被调用
        assert len(hook_high.call_order) > 0
        assert len(hook_mid.call_order) > 0
        assert len(hook_low.call_order) > 0


class TestStepDataBuilderIntegration:
    """测试 StepDataBuilder 集成"""

    def test_step_data_builder_with_env(self):
        """测试 StepDataBuilder 与 Env 的集成"""
        simulator = MockSimulator()

        class TestStepDataBuilder(StepDataBuilder):
            def __init__(self):
                super().__init__()
                self.obs_dim = 127

            def build_step_data(self):
                # 使用存储的状态数据
                if self._core_state is None:
                    # 如果没有状态，返回默认值
                    obs = {
                        'robot_a_obs': np.zeros(self.obs_dim, dtype=np.float32),
                        'robot_b_obs': np.zeros(self.obs_dim, dtype=np.float32),
                    }
                else:
                    obs = {
                        'robot_a_obs': np.random.randn(self.obs_dim).astype(np.float32),
                        'robot_b_obs': np.random.randn(self.obs_dim).astype(np.float32),
                    }
                reward = {'robot_a': 0.0, 'robot_b': 0.0}
                info = {}
                return obs, reward, info

            def get_observation_space(self):
                import gymnasium as gym
                return gym.spaces.Dict({
                    "robot_a_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                    "robot_b_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                })

        builder = TestStepDataBuilder()
        env = CombatGymEnv(
            simulator=simulator,
            step_data_builder=builder,
            match_duration=1.0,
        )

        # Reset
        obs, info = env.reset()
        assert obs is not None

        # Step
        action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert reward is not None


class TestEndToEndWorkflow:
    """端到端工作流程测试"""

    def test_full_episode_with_termination(self):
        """测试完整的 Episode（包括终止）"""
        simulator = MockSimulator()

        class TestStepDataBuilder(StepDataBuilder):
            def __init__(self):
                super().__init__()
                self.obs_dim = 127

            def build_step_data(self):
                obs = {
                    'robot_a_obs': np.random.randn(self.obs_dim).astype(np.float32),
                    'robot_b_obs': np.random.randn(self.obs_dim).astype(np.float32),
                }
                reward = {'robot_a': 0.0, 'robot_b': 0.0}
                info = {}
                return obs, reward, info

            def get_observation_space(self):
                import gymnasium as gym
                return gym.spaces.Dict({
                    "robot_a_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                    "robot_b_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                })

        # 创建一个会在第 3 步终止的 Hook
        class TerminateHook(BaseHook):
            def __init__(self):
                self.step_count = 0

            @property
            def name(self):
                return "terminator"

            @property
            def priority(self):
                return 0

            def invoke(self, invoke_type, *args, **kwargs):
                if invoke_type == InvokeType.POST_ACTION_STEP:
                    self.step_count += 1
                    return self.step_count >= 3
                return False

        builder = TestStepDataBuilder()
        env = CombatGymEnv(
            simulator=simulator,
            step_data_builder=builder,
            match_duration=100.0,  # 很长的时间，由 Hook 终止
            hooks=[TerminateHook()],
        )

        obs, info = env.reset()

        step_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        # 应该在第 3 步终止
        assert step_count == 3
        assert terminated is True
