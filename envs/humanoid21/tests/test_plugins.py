"""
Humanoid21 插件测试 - 测试业务插件逻辑
"""
import pytest
import numpy as np

from .conftest import (
    MockMuJoCoSimulator,
    non_fall_plugin,
    combat_scoring_plugin,
    frozen_robot_plugin,
)
from envs.humanoid21.plugins import (
    NonFallConstraintPlugin,
    CombatScoringPlugin,
    FrozenRobotPlugin,
)
from envs.framework.context import SimContext, TerminationReason


def _create_writable_context(simulator):
    """创建具有写入权限的 SimContext"""
    ctx = SimContext(simulator)
    ctx._grant_mutator()
    return ctx


class TestNonFallConstraintPlugin:
    """测试防摔倒约束插件"""

    def test_clamps_pitch_within_limit(self, mock_simulator, non_fall_plugin):
        """
        场景：机器人 pitch 超过限制
        预期：pitch 被裁剪到限制范围内，水平速度清零
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置 robot_a 的 pitch 为 10 度（超过 5 度限制）
        mock_simulator.set_robot_orientation('robot_a', roll=0, pitch=10, yaw=0)
        mock_simulator.set_robot_orientation('robot_b', roll=0, pitch=0, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：pitch 被裁剪
        state = mock_simulator.get_core_state()
        orientation = state['robot_a']['root_orientation']

        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        euler = rot.as_euler('xyz', degrees=True)
        pitch = euler[1]

        assert abs(pitch) <= 5.0, f"Pitch {pitch} should be clamped to ±5 degrees"

    def test_clamps_roll_within_limit(self, mock_simulator, non_fall_plugin):
        """
        场景：机器人 roll 超过限制
        预期：roll 被裁剪到限制范围内
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置 robot_a 的 roll 为 10 度（超过 5 度限制）
        mock_simulator.set_robot_orientation('robot_a', roll=10, pitch=0, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：roll 被裁剪
        state = mock_simulator.get_core_state()
        orientation = state['robot_a']['root_orientation']

        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        euler = rot.as_euler('xyz', degrees=True)
        roll = euler[0]

        assert abs(roll) <= 5.0, f"Roll {roll} should be clamped to ±5 degrees"

    def test_resets_horizontal_velocity_when_clamped(self, mock_simulator, non_fall_plugin):
        """
        场景：姿态被裁剪时
        预期：水平线速度 (vx, vy) 被清零
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置姿态超过限制，并设置速度
        mock_simulator.set_robot_orientation('robot_a', roll=10, pitch=0, yaw=0)

        # 设置水平速度
        state = mock_simulator.get_core_state()
        state['robot_a']['root_linear_velocity'] = [1.0, 2.0, 0.0]
        mock_simulator.set_core_state(state)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：水平速度被清零
        state = mock_simulator.get_core_state()
        linear_vel = state['robot_a']['root_linear_velocity']
        assert linear_vel[0] == 0.0, "vx should be reset to 0"
        assert linear_vel[1] == 0.0, "vy should be reset to 0"

    def test_records_clamp_count_in_metrics(self, mock_simulator, non_fall_plugin):
        """
        场景：姿态被裁剪
        预期：metrics 中记录 clamp_count
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置姿态超过限制
        mock_simulator.set_robot_orientation('robot_a', roll=10, pitch=0, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：clamp_count 被记录
        assert ctx.metrics.get('robot_a_clamp_count', 0) == 1

    def test_does_not_modify_within_limits(self, mock_simulator, non_fall_plugin):
        """
        场景：机器人在限制范围内
        预期：不修改状态
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置姿态在限制内（3 度）
        mock_simulator.set_robot_orientation('robot_a', roll=3, pitch=3, yaw=0)

        # 记录原始状态
        original_state = mock_simulator.get_core_state()
        original_orientation = original_state['robot_a']['root_orientation'].copy()

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：状态未被修改
        state = mock_simulator.get_core_state()
        orientation = state['robot_a']['root_orientation']
        np.testing.assert_array_almost_equal(orientation, original_orientation)


class TestCombatScoringPlugin:
    """测试战斗计分插件"""

    def test_initializes_health_on_pre_episode(self, mock_simulator, combat_scoring_plugin):
        """
        场景：episode 开始
        预期：初始化血量和伤害指标
        """
        ctx = SimContext(mock_simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0
        assert ctx.metrics['damage_taken_a'] == 0.0
        assert ctx.metrics['damage_taken_b'] == 0.0

    def test_supports_different_initial_health(self):
        """
        场景：设置不同的初始血量
        预期：使用各自的初始血量
        """
        plugin = CombatScoringPlugin(
            initial_health_a=80.0,
            initial_health_b=120.0
        )
        ctx = SimContext(MockMuJoCoSimulator())

        plugin.on_pre_episode(ctx)

        assert ctx.metrics['health_a'] == 80.0
        assert ctx.metrics['health_b'] == 120.0

    def test_detects_head_hit_and_deals_damage(self, mock_simulator, combat_scoring_plugin):
        """
        场景：robot_a 的手击中 robot_b 的头部
        预期：robot_b 血量减少，伤害被记录
        """
        ctx = SimContext(mock_simulator)

        # 初始化血量
        combat_scoring_plugin.on_pre_episode(ctx)

        # 添加碰撞：hand_red (robot_a) 击中 head_blue (robot_b)
        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='hand_right_red',
            geom2_name='head_blue',
            impulse=50.0  # 冲量
        )

        # 执行插件
        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：伤害计算
        # damage = (3.0 * 50.0) / 100.0 = 1.5
        expected_damage = (3.0 * 50.0) / 100.0
        assert ctx.metrics['health_b'] == 100.0 - expected_damage
        assert ctx.metrics['damage_taken_b'] == expected_damage

    def test_detects_torso_hit(self, mock_simulator, combat_scoring_plugin):
        """
        场景：击中躯干
        预期：躯干伤害权重为 1.0
        """
        ctx = SimContext(mock_simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        # 添加碰撞：击中 torso
        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='hand_right_red',
            geom2_name='torso_blue',
            impulse=100.0
        )

        combat_scoring_plugin.on_post_action_step(ctx)

        # damage = (1.0 * 100.0) / 100.0 = 1.0
        expected_damage = (1.0 * 100.0) / 100.0
        assert ctx.metrics['health_b'] == 100.0 - expected_damage

    def test_triggers_ko_when_health_depleted(self, mock_simulator, combat_scoring_plugin):
        """
        场景：血量降至 0 或以下
        预期：请求 KO 终止
        """
        ctx = SimContext(mock_simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        # 直接设置血量为 0
        ctx.metrics['health_b'] = 0.0

        # 执行插件
        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：KO 被触发
        assert TerminationReason.KO in ctx.termination_proposals

    def test_records_hit_events(self, mock_simulator, combat_scoring_plugin):
        """
        场景：发生有效击中
        预期：事件被记录到 ctx.events
        """
        ctx = SimContext(mock_simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='hand_right_red',
            geom2_name='head_blue',
            impulse=50.0
        )

        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：事件被记录
        assert len(ctx.events) == 1
        event = ctx.events[0]
        assert event['type'] == 'hit'
        assert event['attacker'] == 'robot_a'
        assert event['defender'] == 'robot_b'
        assert event['part'] == 'head'

    def test_ignores_non_attack_parts(self, mock_simulator, combat_scoring_plugin):
        """
        场景：非攻击部位接触
        预期：不计伤害
        """
        ctx = SimContext(mock_simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        # 添加碰撞：head 对 head（不是攻击部位）
        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='head_red',
            geom2_name='head_blue',
            impulse=50.0
        )

        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：没有伤害
        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0
        assert len(ctx.events) == 0


class TestFrozenRobotPlugin:
    """测试冻结机器人插件"""

    def test_captures_initial_state_on_pre_episode(self, mock_simulator, frozen_robot_plugin):
        """
        场景：episode 开始
        预期：记录 robot_b 的初始状态
        """
        ctx = _create_writable_context(mock_simulator)

        # 设置 robot_b 的初始位置
        state = mock_simulator.get_core_state()
        state['robot_b']['root_position'] = [1.0, 0.5, 1.282]
        mock_simulator.set_core_state(state)

        # 执行插件
        frozen_robot_plugin.on_pre_episode(ctx)

        # 验证：初始状态被保存
        assert frozen_robot_plugin.initial_state is not None
        np.testing.assert_array_almost_equal(
            frozen_robot_plugin.initial_state['root_position'],
            [1.0, 0.5, 1.282]
        )

    def test_restores_state_on_post_phy_step(self, mock_simulator, frozen_robot_plugin):
        """
        场景：物理步之后
        预期：robot_b 被重置到初始状态
        """
        ctx = _create_writable_context(mock_simulator)

        # 先捕获初始状态
        frozen_robot_plugin.on_pre_episode(ctx)

        # 模拟 robot_b 位置发生变化
        state = mock_simulator.get_core_state()
        state['robot_b']['root_position'] = [2.0, 1.0, 0.5]
        state['robot_b']['root_linear_velocity'] = [1.0, 1.0, 0.0]
        mock_simulator.set_core_state(state)

        # 执行插件
        frozen_robot_plugin.on_post_phy_step(ctx)

        # 验证：robot_b 被重置
        state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(
            state['robot_b']['root_position'],
            frozen_robot_plugin.initial_state['root_position']
        )
        np.testing.assert_array_almost_equal(
            state['robot_b']['root_linear_velocity'],
            [0.0, 0.0, 0.0]
        )

    def test_only_affects_specified_robot(self, mock_simulator):
        """
        场景：冻结 robot_b
        预期：robot_a 不受影响
        """
        plugin = FrozenRobotPlugin(frozen_robot_id='robot_b')
        ctx = _create_writable_context(mock_simulator)

        plugin.on_pre_episode(ctx)

        # 改变两个机器人的位置
        state = mock_simulator.get_core_state()
        state['robot_a']['root_position'] = [0.0, 0.0, 2.0]
        state['robot_b']['root_position'] = [2.0, 0.0, 0.5]
        mock_simulator.set_core_state(state)

        plugin.on_post_phy_step(ctx)

        # 验证：robot_b 被重置，robot_a 保持不变
        state = mock_simulator.get_core_state()
        assert state['robot_a']['root_position'][2] == 2.0  # 未被重置

    def test_does_nothing_without_initial_state(self, mock_simulator, frozen_robot_plugin):
        """
        场景：没有初始化就调用 on_post_phy_step
        预期：不执行任何操作
        """
        ctx = _create_writable_context(mock_simulator)

        # 没有调用 on_pre_episode，initial_state 为 None
        assert frozen_robot_plugin.initial_state is None

        # 记录原始状态
        original_state = mock_simulator.get_core_state()

        # 执行插件（应该不报错，也不修改状态）
        frozen_robot_plugin.on_post_phy_step(ctx)

        # 验证：状态未变
        current_state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(
            current_state['robot_b']['root_position'],
            original_state['robot_b']['root_position']
        )
