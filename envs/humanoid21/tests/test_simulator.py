"""
Humanoid21 Simulator 测试 - 测试状态管理和物理仿真
"""
import pytest
import numpy as np

from .conftest import MockMuJoCoSimulator
from envs.framework.context import SimContext


class TestSimulatorStateManagement:
    """测试模拟器状态管理"""

    def test_get_core_state_returns_all_required_keys(self, mock_simulator):
        """
        场景：获取核心状态
        预期：返回包含所有必需键的字典
        """
        state = mock_simulator.get_core_state()

        assert 'qpos' in state
        assert 'qvel' in state
        assert 'time' in state
        assert 'robot_a' in state
        assert 'robot_b' in state

    def test_get_core_state_returns_copies(self, mock_simulator):
        """
        场景：获取核心状态后修改返回值
        预期：不影响模拟器内部状态
        """
        state = mock_simulator.get_core_state()

        # 修改返回的状态
        state['qpos'][0] = 999.0
        state['robot_a']['root_position'][0] = 888.0

        # 验证：模拟器内部状态未变
        new_state = mock_simulator.get_core_state()
        assert new_state['qpos'][0] != 999.0
        assert new_state['robot_a']['root_position'][0] != 888.0

    def test_set_core_state_updates_qpos_qvel(self, mock_simulator):
        """
        场景：设置核心状态
        预期：qpos 和 qvel 被更新
        """
        new_qpos = np.ones_like(mock_simulator._qpos) * 0.5
        new_qvel = np.ones_like(mock_simulator._qvel) * 0.3

        state = mock_simulator.get_core_state()
        state['qpos'] = new_qpos
        state['qvel'] = new_qvel
        mock_simulator.set_core_state(state)

        # 验证：状态被更新
        current_state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(current_state['qpos'], new_qpos)
        np.testing.assert_array_almost_equal(current_state['qvel'], new_qvel)

    def test_set_core_state_updates_robot_positions(self, mock_simulator):
        """
        场景：设置机器人位置
        预期：root_position 被更新
        """
        state = mock_simulator.get_core_state()

        # 设置 robot_a 新位置
        state['robot_a']['root_position'] = [5.0, 3.0, 2.0]
        mock_simulator.set_core_state(state)

        # 验证：位置被更新
        current_state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(
            current_state['robot_a']['root_position'],
            [5.0, 3.0, 2.0]
        )

    def test_set_core_state_updates_robot_orientations(self, mock_simulator):
        """
        场景：设置机器人姿态
        预期：root_orientation 被更新
        """
        state = mock_simulator.get_core_state()

        # 设置新的四元数
        new_quat = [0.0, 0.0, 0.0, 1.0]  # [w, x, y, z]
        state['robot_a']['root_orientation'] = new_quat
        mock_simulator.set_core_state(state)

        # 验证：姿态被更新
        current_state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(
            current_state['robot_a']['root_orientation'],
            new_quat
        )

    def test_set_core_state_updates_robot_velocities(self, mock_simulator):
        """
        场景：设置机器人速度
        预期：速度被更新
        """
        state = mock_simulator.get_core_state()

        # 设置新速度
        state['robot_a']['root_linear_velocity'] = [1.0, 2.0, 3.0]
        state['robot_a']['root_angular_velocity'] = [0.1, 0.2, 0.3]
        mock_simulator.set_core_state(state)

        # 验证：速度被更新
        current_state = mock_simulator.get_core_state()
        np.testing.assert_array_almost_equal(
            current_state['robot_a']['root_linear_velocity'],
            [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_almost_equal(
            current_state['robot_a']['root_angular_velocity'],
            [0.1, 0.2, 0.3]
        )

    def test_set_core_state_synchronizes_structured_to_array(self, mock_simulator):
        """
        场景：通过 structured data 设置状态
        预期：正确同步到 qpos/qvel 数组
        """
        state = mock_simulator.get_core_state()

        # 通过 structured data 设置
        original_qpos = state['qpos'].copy()
        state['robot_a']['root_position'] = [10.0, 20.0, 30.0]
        mock_simulator.set_core_state(state)

        # 验证：qpos 数组被正确更新
        current_state = mock_simulator.get_core_state()
        qpos_adr = mock_simulator.robot_info['robot_a']['qpos_adr']
        np.testing.assert_array_almost_equal(
            current_state['qpos'][qpos_adr:qpos_adr+3],
            [10.0, 20.0, 30.0]
        )

    def test_get_static_data_returns_robot_info(self, mock_simulator):
        """
        场景：获取静态数据
        预期：包含 robot_info
        """
        static_data = mock_simulator.get_static_data()

        assert 'dt' in static_data
        assert 'robot_info' in static_data
        assert 'robot_a' in static_data['robot_info']
        assert 'robot_b' in static_data['robot_info']

    def test_get_static_data_robot_info_contains_required_keys(self, mock_simulator):
        """
        场景：检查 robot_info 内容
        预期：包含所有必需的键
        """
        static_data = mock_simulator.get_static_data()
        robot_a_info = static_data['robot_info']['robot_a']

        required_keys = [
            'body_id', 'root_jnt_id', 'qpos_adr', 'qvel_adr',
            'suffix', 'qpos_indices', 'qvel_indices',
            'jnt_ranges', 'ctrl_ranges', 'qpos0', 'actuators'
        ]

        for key in required_keys:
            assert key in robot_a_info, f"Missing key: {key}"

    def test_get_derived_state_returns_contacts(self, mock_simulator):
        """
        场景：获取派生状态
        预期：包含 contacts 列表
        """
        # 添加一些碰撞
        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='hand_red', geom2_name='head_blue',
            impulse=50.0
        )

        derived_state = mock_simulator.get_derived_state()

        assert 'contacts' in derived_state
        assert len(derived_state['contacts']) == 1

    def test_physical_step_increments_time(self, mock_simulator):
        """
        场景：执行物理步
        预期：时间增加
        """
        initial_time = mock_simulator._time
        mock_simulator.physical_step()
        assert mock_simulator._time == initial_time + mock_simulator.dt

    def test_reset_clears_time(self, mock_simulator):
        """
        场景：重置模拟器
        预期：时间归零
        """
        mock_simulator.physical_step()
        mock_simulator.physical_step()
        assert mock_simulator._time > 0

        mock_simulator.reset()
        assert mock_simulator._time == 0.0

    def test_reset_clears_contacts(self, mock_simulator):
        """
        场景：重置模拟器
        预期：碰撞列表被清空
        """
        mock_simulator.add_contact(
            geom1=1, geom2=2, body1=10, body2=20,
            geom1_name='hand_red', geom2_name='head_blue'
        )

        mock_simulator.reset()

        derived_state = mock_simulator.get_derived_state()
        assert len(derived_state['contacts']) == 0

    def test_reset_respects_custom_initial_distance(self, mock_simulator):
        """
        场景：使用自定义初始距离重置
        预期：机器人位置相应调整
        """
        # 使用自定义距离重置
        mock_simulator.reset(options={'initial_distance': 4.0})

        state = mock_simulator.get_core_state()
        pos_a = state['robot_a']['root_position']
        pos_b = state['robot_b']['root_position']

        # 验证：距离为 4.0
        distance = np.linalg.norm(pos_b - pos_a)
        assert abs(distance - 4.0) < 0.01

    def test_get_physical_frequency_returns_correct_value(self, mock_simulator):
        """
        场景：获取物理频率
        预期：返回 1/dt
        """
        expected_freq = 1.0 / mock_simulator.dt
        assert mock_simulator.get_physical_frequency() == expected_freq

    def test_robot_initial_positions_are_opposite(self, mock_simulator):
        """
        场景：初始状态
        预期：两个机器人相向而立，x 坐标相反
        """
        state = mock_simulator.get_core_state()
        pos_a = state['robot_a']['root_position']
        pos_b = state['robot_b']['root_position']

        # 验证：y 和 z 相同，x 相反
        assert abs(pos_a[1] - pos_b[1]) < 0.01  # y 相同
        assert abs(pos_a[2] - pos_b[2]) < 0.01  # z 相同
        assert abs(pos_a[0] + pos_b[0]) < 0.01  # x 相反

    def test_robot_initial_orientations_are_facing_each_other(self, mock_simulator):
        """
        场景：初始姿态
        预期：两个机器人面朝对方
        """
        state = mock_simulator.get_core_state()
        quat_a = state['robot_a']['root_orientation']  # [w, x, y, z]
        quat_b = state['robot_b']['root_orientation']

        # robot_a: [1, 0, 0, 0] = 无旋转，面朝 +x
        # robot_b: [0, 0, 0, 1] = 180度绕 z轴，面朝 -x
        np.testing.assert_array_almost_equal(quat_a, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(quat_b, [0.0, 0.0, 0.0, 1.0])


class TestSimulatorDataProperties:
    """测试模拟器的 data 和 model 属性"""

    def test_data_property_provides_qpos_access(self, mock_simulator):
        """
        场景：访问 data.qpos
        预期：返回 qpos 数组
        """
        data = mock_simulator.data
        assert hasattr(data, 'qpos')
        assert data.qpos is mock_simulator._qpos

    def test_data_property_provides_xpos_access(self, mock_simulator):
        """
        场景：访问 data.xpos
        预期：返回 xpos 数组
        """
        data = mock_simulator.data
        assert hasattr(data, 'xpos')
        assert data.xpos is mock_simulator._xpos

    def test_data_property_provides_time_access(self, mock_simulator):
        """
        场景：访问 data.time
        预期：返回当前时间
        """
        data = mock_simulator.data
        assert hasattr(data, 'time')
        assert data.time == mock_simulator._time
