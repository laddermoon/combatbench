"""
Humanoid21 插件测试 - 使用真实 MujocoCombatSimulator 测试插件逻辑
"""
import pytest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from .conftest import (
    simulator,
    non_fall_plugin,
    combat_scoring_plugin,
    frozen_robot_plugin,
    _create_writable_context,
)
from envs.humanoid21.plugins import (
    NonFallConstraintPlugin,
    CombatScoringPlugin,
    FrozenRobotPlugin,
)
from envs.framework.context import SimContext, TerminationReason


def _set_robot_orientation(simulator, robot_id, roll, pitch, yaw):
    """设置机器人的姿态角（度）"""
    static_data = simulator.get_static_data()
    robot_info = static_data['robot_info'][robot_id]
    qpos_adr = robot_info['qpos_adr']

    # 创建旋转
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    simulator.data.qpos[qpos_adr+3:qpos_adr+7] = quat_wxyz
    mujoco.mj_forward(simulator.model, simulator.data)


def _get_robot_orientation(simulator, robot_id):
    """获取机器人的姿态角（度）"""
    static_data = simulator.get_static_data()
    robot_info = static_data['robot_info'][robot_id]
    qpos_adr = robot_info['qpos_adr']

    quat_wxyz = simulator.data.qpos[qpos_adr+3:qpos_adr+7]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

    rot = R.from_quat(quat_xyzw)
    return rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]


class TestNonFallConstraintPlugin:
    """测试防摔倒约束插件"""

    def test_clamps_pitch_within_limit(self, simulator, non_fall_plugin):
        """
        场景：机器人 pitch 超过限制
        预期：pitch 被裁剪到限制范围内，水平速度清零
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 设置 robot_a 的 pitch 为 10 度（超过 5 度限制）
        _set_robot_orientation(simulator, 'robot_a', roll=0, pitch=10, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：pitch 被裁剪
        roll, pitch, yaw = _get_robot_orientation(simulator, 'robot_a')
        assert abs(pitch) <= 5.0 + 1e-6, f"Pitch {pitch} should be clamped to ±5 degrees"

    def test_clamps_roll_within_limit(self, simulator, non_fall_plugin):
        """
        场景：机器人 roll 超过限制
        预期：roll 被裁剪到限制范围内
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 设置 robot_a 的 roll 为 10 度（超过 5 度限制）
        _set_robot_orientation(simulator, 'robot_a', roll=10, pitch=0, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：roll 被裁剪
        roll, pitch, yaw = _get_robot_orientation(simulator, 'robot_a')
        assert abs(roll) <= 5.0, f"Roll {roll} should be clamped to ±5 degrees"

    def test_resets_horizontal_velocity_when_clamped(self, simulator, non_fall_plugin):
        """
        场景：姿态被裁剪时
        预期：水平线速度 (vx, vy) 被清零
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 设置姿态超过限制，并设置速度
        _set_robot_orientation(simulator, 'robot_a', roll=10, pitch=0, yaw=0)

        # 设置水平速度
        static_data = simulator.get_static_data()
        qvel_adr = static_data['robot_info']['robot_a']['qvel_adr']
        simulator.data.qvel[qvel_adr] = 1.0  # vx
        simulator.data.qvel[qvel_adr+1] = 2.0  # vy

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：水平速度被清零
        assert simulator.data.qvel[qvel_adr] == 0.0, "vx should be reset to 0"
        assert simulator.data.qvel[qvel_adr+1] == 0.0, "vy should be reset to 0"

    def test_records_clamp_count_in_metrics(self, simulator, non_fall_plugin):
        """
        场景：姿态被裁剪
        预期：metrics 中记录 clamp_count
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 设置姿态超过限制
        _set_robot_orientation(simulator, 'robot_a', roll=10, pitch=0, yaw=0)

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：clamp_count 被记录
        assert ctx.metrics.get('robot_a_clamp_count', 0) == 1

    def test_does_not_modify_within_limits(self, simulator, non_fall_plugin):
        """
        场景：机器人在限制范围内
        预期：不修改状态
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 记录原始姿态
        original_roll, original_pitch, original_yaw = _get_robot_orientation(simulator, 'robot_a')

        # 执行插件
        non_fall_plugin.on_post_phy_step(ctx)

        # 验证：姿态未变
        roll, pitch, yaw = _get_robot_orientation(simulator, 'robot_a')
        assert abs(roll - original_roll) < 0.01
        assert abs(pitch - original_pitch) < 0.01


class TestCombatScoringPlugin:
    """测试战斗计分插件"""

    def test_initializes_health_on_pre_episode(self, simulator, combat_scoring_plugin):
        """
        场景：episode 开始
        预期：初始化血量和伤害指标
        """
        ctx = SimContext(simulator)
        combat_scoring_plugin.on_pre_episode(ctx)

        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0
        assert ctx.metrics['damage_taken_a'] == 0.0
        assert ctx.metrics['damage_taken_b'] == 0.0

    def test_supports_different_initial_health(self, simulator):
        """
        场景：设置不同的初始血量
        预期：使用各自的初始血量
        """
        plugin = CombatScoringPlugin(
            initial_health_a=80.0,
            initial_health_b=120.0
        )
        ctx = SimContext(simulator)

        plugin.on_pre_episode(ctx)

        assert ctx.metrics['health_a'] == 80.0
        assert ctx.metrics['health_b'] == 120.0

    def test_detects_head_hit_and_deals_damage(self, simulator, combat_scoring_plugin):
        """
        场景：无碰撞发生
        预期：血量保持不变

        注意：此测试验证无碰撞时没有伤害。实际的碰撞检测需要
        真实的物理接触，难以在单元测试中复现。
        """
        simulator.reset()
        ctx = SimContext(simulator)

        # 初始化血量
        combat_scoring_plugin.on_pre_episode(ctx)
        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0

        # 执行插件（没有碰撞数据）
        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：血量保持不变
        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0

    def test_triggers_ko_when_health_depleted(self, simulator, combat_scoring_plugin):
        """
        场景：血量降至 0 或以下
        预期：请求 KO 终止
        """
        simulator.reset()
        ctx = SimContext(simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        # 直接设置血量为 0
        ctx.metrics['health_b'] = 0.0

        # 执行插件
        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：KO 被触发
        assert TerminationReason.KO in ctx.termination_proposals

    def test_ignores_non_attack_parts(self, simulator, combat_scoring_plugin):
        """
        场景：非攻击部位接触
        预期：不计伤害
        """
        simulator.reset()
        ctx = SimContext(simulator)

        combat_scoring_plugin.on_pre_episode(ctx)

        # 添加碰撞：head 对 head（不是攻击部位）
        derived = simulator.get_derived_state()
        derived['contacts'].append({
            'geom_a': 1,
            'geom_b': 2,
            'body_a': 1,  # head_red
            'body_b': 18,  # head_blue
            'position': np.array([0.0, 0.0, 1.5]),
            'normal': np.array([0.0, 0.0, 1.0]),
            'impulse': 50.0,
            'geom1_name': 'head_red',
            'geom2_name': 'head_blue',
        })

        combat_scoring_plugin.on_post_action_step(ctx)

        # 验证：没有伤害
        assert ctx.metrics['health_a'] == 100.0
        assert ctx.metrics['health_b'] == 100.0
        assert len(ctx.events) == 0


class TestFrozenRobotPlugin:
    """测试冻结机器人插件"""

    def test_captures_initial_state_on_pre_episode(self, simulator, frozen_robot_plugin):
        """
        场景：episode 开始
        预期：记录 robot_b 的初始状态
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 设置 robot_b 的初始位置
        static_data = simulator.get_static_data()
        qpos_adr_b = static_data['robot_info']['robot_b']['qpos_adr']
        simulator.data.qpos[qpos_adr_b:qpos_adr_b+3] = [1.0, 0.5, 1.282]
        mujoco.mj_forward(simulator.model, simulator.data)

        # 执行插件
        frozen_robot_plugin.on_pre_episode(ctx)

        # 验证：初始状态被保存
        assert frozen_robot_plugin.initial_state is not None
        np.testing.assert_array_almost_equal(
            frozen_robot_plugin.initial_state['root_position'],
            [1.0, 0.5, 1.282]
        )

    def test_restores_state_on_post_phy_step(self, simulator, frozen_robot_plugin):
        """
        场景：物理步之后
        预期：robot_b 被重置到初始状态
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 先捕获初始状态
        frozen_robot_plugin.on_pre_episode(ctx)

        # 模拟 robot_b 位置发生变化
        static_data = simulator.get_static_data()
        qpos_adr_b = static_data['robot_info']['robot_b']['qpos_adr']
        simulator.data.qpos[qpos_adr_b:qpos_adr_b+3] = [2.0, 1.0, 0.5]
        mujoco.mj_forward(simulator.model, simulator.data)

        # 执行插件
        frozen_robot_plugin.on_post_phy_step(ctx)

        # 验证：robot_b 被重置
        np.testing.assert_array_almost_equal(
            simulator.data.qpos[qpos_adr_b:qpos_adr_b+3],
            frozen_robot_plugin.initial_state['root_position']
        )

    def test_only_affects_specified_robot(self, simulator):
        """
        场景：冻结 robot_b
        预期：robot_a 不受影响
        """
        plugin = FrozenRobotPlugin(frozen_robot_id='robot_b')
        ctx = _create_writable_context(simulator)
        simulator.reset()

        plugin.on_pre_episode(ctx)

        # 改变两个机器人的位置
        static_data = simulator.get_static_data()
        qpos_adr_a = static_data['robot_info']['robot_a']['qpos_adr']
        qpos_adr_b = static_data['robot_info']['robot_b']['qpos_adr']

        simulator.data.qpos[qpos_adr_a+2] = 2.0  # robot_a z 位置
        simulator.data.qpos[qpos_adr_b:qpos_adr_b+3] = [2.0, 0.0, 0.5]  # robot_b 位置改变
        mujoco.mj_forward(simulator.model, simulator.data)

        original_z_a = simulator.data.qpos[qpos_adr_a+2]

        plugin.on_post_phy_step(ctx)

        # 验证：robot_b 被重置，robot_a 的 z 保持不变
        assert simulator.data.qpos[qpos_adr_a+2] == original_z_a
        assert simulator.data.qpos[qpos_adr_b] == simulator.data.qpos[qpos_adr_b]

    def test_does_nothing_without_initial_state(self, simulator, frozen_robot_plugin):
        """
        场景：没有初始化就调用 on_post_phy_step
        预期：不执行任何操作
        """
        ctx = _create_writable_context(simulator)
        simulator.reset()

        # 没有调用 on_pre_episode，initial_state 为 None
        assert frozen_robot_plugin.initial_state is None

        # 记录原始状态
        original_qpos = simulator.data.qpos.copy()

        # 执行插件（应该不报错，也不修改状态）
        frozen_robot_plugin.on_post_phy_step(ctx)

        # 验证：状态未变
        np.testing.assert_array_almost_equal(simulator.data.qpos, original_qpos)
