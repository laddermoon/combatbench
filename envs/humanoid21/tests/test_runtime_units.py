"""
Humanoid21 Runtime Units 测试 - 测试 Observer 和 Rewarder
"""
import pytest
import numpy as np

from .conftest import MockMuJoCoSimulator, humanoid_observer, humanoid_rewarder
from envs.framework.context import SimContext, ReadOnlySimContext
from envs.humanoid21.runtime_units import (
    Humanoid21Observer,
    Humanoid21Rewarder,
    build_shared_runtime_info,
)
from envs.framework.context import TerminationReason


class TestHumanoid21Observer:
    """测试 Humanoid21Observer"""

    def test_raises_error_for_invalid_agent_id(self):
        """
        场景：传入无效的 agent_id
        预期：抛出 ValueError
        """
        with pytest.raises(ValueError, match="Unsupported agent_id"):
            Humanoid21Observer('robot_c')

    def test_observation_dimension_is_127(self, mock_simulator, mock_mj_name2id):
        """
        场景：构建观测
        预期：观测维度为 127
        """
        observer = Humanoid21Observer('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        observer.on_reset(ctx)
        obs = observer.get_output()

        assert obs is not None
        assert obs.shape == (127,), f"Expected shape (127,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_observation_space_definition(self):
        """
        场景：获取观测空间定义
        预期：返回正确的 Gymnasium Space
        """
        obs_space = Humanoid21Observer.get_observation_space()

        assert 'robot_a' in obs_space.spaces
        assert 'robot_b' in obs_space.spaces
        assert obs_space.spaces['robot_a'].shape == (127,)
        assert obs_space.spaces['robot_b'].shape == (127,)

    def test_action_space_definition(self):
        """
        场景：获取动作空间定义
        预期：返回正确的 Gymnasium Space
        """
        action_space = Humanoid21Observer.get_action_space()

        assert 'robot_a' in action_space.spaces
        assert 'robot_b' in action_space.spaces
        assert action_space.spaces['robot_a'].shape == (21,)
        assert action_space.spaces['robot_b'].shape == (21,)
        assert action_space.spaces['robot_a'].low[0] == -1.0
        assert action_space.spaces['robot_a'].high[0] == 1.0

    def test_updates_observation_on_post_step(self, mock_simulator, mock_mj_name2id):
        """
        场景：执行 step 后
        预期：观测被更新
        """
        observer = Humanoid21Observer('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        observer.on_reset(ctx)
        obs1 = observer.get_output()

        # 模拟一些状态变化
        mock_simulator._qpos[7] = 0.5  # 改变关节位置

        observer.on_post_step(ctx)
        obs2 = observer.get_output()

        # 验证：观测被更新
        # obs2[0] 应该包含关节位置的变化
        assert obs2[0] == 0.5  # 第一个关节位置

    def test_observation_is_finite(self, mock_simulator, mock_mj_name2id):
        """
        场景：构建观测
        预期：所有值都是有限的（无 NaN 或 Inf）
        """
        observer = Humanoid21Observer('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        observer.on_reset(ctx)
        obs = observer.get_output()

        assert np.all(np.isfinite(obs)), "Observation should not contain NaN or Inf"

    def test_opponent_observer_sees_different_data(self, mock_simulator, mock_mj_name2id):
        """
        场景：robot_a 和 robot_b 的 observer
        预期：看到不同的观测（对手位置是相对的）
        """
        observer_a = Humanoid21Observer('robot_a')
        observer_b = Humanoid21Observer('robot_b')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        observer_a.on_reset(ctx)
        observer_b.on_reset(ctx)

        obs_a = observer_a.get_output()
        obs_b = observer_b.get_output()

        # 验证：两个观测不同（因为对手位置是相对的）
        # 对手相对位置部分（大约在观测的 42+13+8 = 63 之后）
        assert not np.array_equal(obs_a, obs_b)


class TestHumanoid21Rewarder:
    """测试 Humanoid21Rewarder"""

    def test_raises_error_for_invalid_agent_id(self):
        """
        场景：传入无效的 agent_id
        预期：抛出 ValueError
        """
        with pytest.raises(ValueError, match="Unsupported agent_id"):
            Humanoid21Rewarder('robot_c')

    def test_returns_zero_on_reset(self, mock_simulator):
        """
        场景：reset 时
        预期：奖励为 0
        """
        rewarder = Humanoid21Rewarder('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        rewarder.on_reset(ctx)
        assert rewarder.get_output() == 0.0

    def test_returns_zero_on_post_step(self, mock_simulator):
        """
        场景：step 后
        预期：奖励为 0（当前实现返回 0）
        """
        rewarder = Humanoid21Rewarder('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        rewarder.on_post_step(ctx)
        assert rewarder.get_output() == 0.0

    def test_returns_zero_on_post_episode(self, mock_simulator):
        """
        场景：episode 结束时
        预期：奖励为 0
        """
        rewarder = Humanoid21Rewarder('robot_a')
        ctx = ReadOnlySimContext.from_sim_context(SimContext(mock_simulator))

        rewarder.on_post_episode(ctx)
        assert rewarder.get_output() == 0.0


class TestBuildSharedRuntimeInfo:
    """测试 build_shared_runtime_info 函数"""

    def test_includes_health_metrics(self):
        """
        场景：构建共享信息
        预期：包含血量信息
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['health_a'] = 85.0
        ctx.metrics['health_b'] = 60.0

        info = build_shared_runtime_info(ctx)

        assert info['health']['robot_a'] == 85.0
        assert info['health']['robot_b'] == 60.0

    def test_includes_damage_taken(self):
        """
        场景：构建共享信息
        预期：包含承受伤害信息
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['damage_taken_a'] = 15.0
        ctx.metrics['damage_taken_b'] = 40.0

        info = build_shared_runtime_info(ctx)

        assert info['damage_taken']['robot_a'] == 15.0
        assert info['damage_taken']['robot_b'] == 40.0

    def test_defaults_to_100_health_if_not_set(self):
        """
        场景：血量未设置
        预期：默认为 100
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))

        info = build_shared_runtime_info(ctx)

        assert info['health']['robot_a'] == 100.0
        assert info['health']['robot_b'] == 100.0

    def test_winner_is_none_when_not_terminated(self):
        """
        场景：episode 未终止
        预期：winner 为 None
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))

        info = build_shared_runtime_info(ctx)

        assert info['winner'] is None

    def test_determines_winner_on_ko(self):
        """
        场景：KO 终止
        预期：根据血量判定获胜者
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['health_a'] = 0.0
        ctx.metrics['health_b'] = 80.0
        ctx._termination_proposals = (TerminationReason.KO,)

        info = build_shared_runtime_info(ctx)

        assert info['winner'] == 'robot_b'

    def test_determines_draw_on_double_ko(self):
        """
        场景：双方血量都为 0
        预期：平局
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['health_a'] = 0.0
        ctx.metrics['health_b'] = 0.0
        ctx._termination_proposals = (TerminationReason.KO,)

        info = build_shared_runtime_info(ctx)

        assert info['winner'] == 'draw'

    def test_determines_winner_on_timeout(self):
        """
        场景：超时终止
        预期：血量高的获胜
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['health_a'] = 90.0
        ctx.metrics['health_b'] = 70.0
        ctx._termination_proposals = (TerminationReason.TIMEOUT,)

        info = build_shared_runtime_info(ctx)

        assert info['winner'] == 'robot_a'

    def test_determines_draw_on_timeout_with_equal_health(self):
        """
        场景：超时且血量相同
        预期：平局
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))
        ctx.metrics['health_a'] = 75.0
        ctx.metrics['health_b'] = 75.0
        ctx._termination_proposals = (TerminationReason.TIMEOUT,)

        info = build_shared_runtime_info(ctx)

        assert info['winner'] == 'draw'

    def test_uses_default_values_for_damage(self):
        """
        场景：伤害未设置
        预期：默认为 0
        """
        ctx = ReadOnlySimContext.from_sim_context(SimContext(MockMuJoCoSimulator()))

        info = build_shared_runtime_info(ctx)

        assert info['damage_taken']['robot_a'] == 0.0
        assert info['damage_taken']['robot_b'] == 0.0
