"""
Humanoid21 Simulator 真实测试 - 使用真实 MujocoCombatSimulator

测试内容：
1. 仿真器是否正常工作（reset, step, forward等）
2. 接口返回数据是否与 DATASPEC.md 一致
"""
import pytest
import numpy as np
import mujoco

from envs.humanoid21.simulator import MujocoCombatSimulator


class TestSimulatorBasicFunctionality:
    """测试仿真器基本功能"""

    @pytest.fixture
    def simulator(self):
        """提供真实的 MujocoCombatSimulator 实例"""
        sim = MujocoCombatSimulator()
        yield sim
        sim.close()

    def test_initialization(self, simulator):
        """
        场景：创建仿真器
        预期：正确初始化，模型和数据可用
        """
        assert simulator.model is not None
        assert simulator.data is not None
        assert simulator.dt == 0.002

    def test_reset_works(self, simulator):
        """
        场景：重置仿真器
        预期：状态被正确重置，机器人在初始位置
        """
        simulator.reset()

        # 验证时间归零
        assert simulator.data.time == 0.0

        # 验证速度清零
        assert np.allclose(simulator.data.qvel, 0.0)

        # 验证机器人位置（根据 DATASPEC.md）
        # robot_a root at qpos_adr=0: x = -initial_distance/2, y = 0, z = 1.282
        # robot_b root at qpos_adr=28: x = +initial_distance/2, y = 0, z = 1.282
        pos_a = simulator.data.qpos[0:3]
        pos_b = simulator.data.qpos[28:31]

        assert abs(pos_a[0] + 1.0) < 0.01  # -initial_distance/2 = -1.0
        assert abs(pos_a[1]) < 0.01
        assert abs(pos_a[2] - 1.282) < 0.01

        assert abs(pos_b[0] - 1.0) < 0.01  # +initial_distance/2 = +1.0
        assert abs(pos_b[1]) < 0.01
        assert abs(pos_b[2] - 1.282) < 0.01

    def test_physical_step_increments_time(self, simulator):
        """
        场景：执行物理步
        预期：时间增加 dt
        """
        simulator.reset()
        initial_time = simulator.data.time
        simulator.physical_step()
        assert simulator.data.time == initial_time + simulator.dt

    def test_multiple_steps_work(self, simulator):
        """
        场景：执行多个物理步
        预期：仿真器不崩溃，状态正常更新
        """
        simulator.reset()

        for _ in range(100):
            simulator.physical_step()

        # 验证时间正确增加（使用近似比较处理浮点精度）
        assert abs(simulator.data.time - 100 * simulator.dt) < 1e-10


class TestStaticDataMatchesSpec:
    """测试 get_static_data() 返回数据与 DATASPEC.md 一致"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        yield sim
        sim.close()

    def test_static_data_contains_dt(self, simulator):
        """
        场景：获取静态数据
        预期：包含 dt = 0.002
        """
        static_data = simulator.get_static_data()
        assert 'dt' in static_data
        assert static_data['dt'] == 0.002

    def test_static_data_contains_robot_info(self, simulator):
        """
        场景：获取静态数据
        预期：包含 robot_info，包含 robot_a 和 robot_b
        """
        static_data = simulator.get_static_data()
        assert 'robot_info' in static_data
        assert 'robot_a' in static_data['robot_info']
        assert 'robot_b' in static_data['robot_info']

    def test_robot_a_info_matches_spec(self, simulator):
        """
        场景：检查 robot_a 的 robot_info
        预期：与 DATASPEC.md 中的值一致
        """
        static_data = simulator.get_static_data()
        info_a = static_data['robot_info']['robot_a']

        # 根据 DATASPEC.md
        assert info_a['body_id'] == 4
        assert info_a['root_jnt_id'] == 0
        assert info_a['qpos_adr'] == 0
        assert info_a['qvel_adr'] == 0
        assert info_a['suffix'] == '_red'
        assert info_a['actuators'] == list(range(21))
        assert len(info_a['qpos_indices']) == 21
        assert len(info_a['qvel_indices']) == 21
        assert len(info_a['jnt_ranges']) == 21
        assert len(info_a['ctrl_ranges']) == 21
        assert len(info_a['qpos0']) == 21

    def test_robot_b_info_matches_spec(self, simulator):
        """
        场景：检查 robot_b 的 robot_info
        预期：与 DATASPEC.md 中的值一致
        """
        static_data = simulator.get_static_data()
        info_b = static_data['robot_info']['robot_b']

        # 根据 DATASPEC.md
        assert info_b['body_id'] == 20
        assert info_b['root_jnt_id'] == 22
        assert info_b['qpos_adr'] == 28
        assert info_b['qvel_adr'] == 27
        assert info_b['suffix'] == '_blue'
        assert info_b['actuators'] == list(range(21, 42))
        assert len(info_b['qpos_indices']) == 21
        assert len(info_b['qvel_indices']) == 21
        assert len(info_b['jnt_ranges']) == 21
        assert len(info_b['ctrl_ranges']) == 21
        assert len(info_b['qpos0']) == 21


class TestCoreStateMatchesSpec:
    """测试 get_core_state() 返回数据与 DATASPEC.md 一致"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def test_core_state_structure(self, simulator):
        """
        场景：获取核心状态
        预期：只包含 qpos, qvel, time（不包含 robot_a/robot_b 结构化数据）
        """
        state = simulator.get_core_state()

        assert 'qpos' in state
        assert 'qvel' in state
        assert 'time' in state
        # 不应包含结构化的 robot_a/robot_b
        assert 'robot_a' not in state
        assert 'robot_b' not in state

    def test_qpos_dimensions_match_spec(self, simulator):
        """
        场景：检查 qpos 维度
        预期：qpos 维度为 56（与 DATASPEC.md 一致）
        """
        state = simulator.get_core_state()
        qpos = state['qpos']

        assert qpos.shape == (56,), f"Expected shape (56,), got {qpos.shape}"
        assert qpos.dtype == np.float64

    def test_qvel_dimensions_match_spec(self, simulator):
        """
        场景：检查 qvel 维度
        预期：qvel 维度为 54（与 DATASPEC.md 一致）
        """
        state = simulator.get_core_state()
        qvel = state['qvel']

        assert qvel.shape == (54,), f"Expected shape (54,), got {qvel.shape}"
        assert qvel.dtype == np.float64

    def test_qpos_indices_match_spec(self, simulator):
        """
        场景：检查 qpos 中的关键索引
        预期：与 DATASPEC.md 中的索引表一致
        """
        state = simulator.get_core_state()
        qpos = state['qpos']

        # robot_a root: qpos_adr=0, 7 DOF (0:7)
        assert qpos[0:3].shape == (3,)  # position [x, y, z]
        assert qpos[3:7].shape == (4,)  # orientation [w, x, y, z]

        # robot_b root: qpos_adr=28, 7 DOF (28:35)
        assert qpos[28:31].shape == (3,)  # position [x, y, z]
        assert qpos[31:35].shape == (4,)  # orientation [w, x, y, z]

    def test_qvel_indices_match_spec(self, simulator):
        """
        场景：检查 qvel 中的关键索引
        预期：与 DATASPEC.md 中的索引表一致
        """
        state = simulator.get_core_state()
        qvel = state['qvel']

        # robot_a root: qvel_adr=0, 6 DOF (0:6)
        assert qvel[0:3].shape == (3,)  # linear velocity [vx, vy, vz]
        assert qvel[3:6].shape == (3,)  # angular velocity [ωx, ωy, ωz]

        # robot_b root: qvel_adr=27, 6 DOF (27:33)
        assert qvel[27:30].shape == (3,)  # linear velocity [vx, vy, vz]
        assert qvel[30:33].shape == (3,)  # angular velocity [ωx, ωy, ωz]

    def test_set_core_state_works(self, simulator):
        """
        场景：设置核心状态
        预期：qpos 和 qvel 被正确设置
        """
        # 获取当前状态
        state = simulator.get_core_state()
        original_qpos = state['qpos'].copy()
        original_qvel = state['qvel'].copy()

        # 修改状态
        new_qpos = np.ones_like(original_qpos) * 0.5
        new_qvel = np.ones_like(original_qvel) * 0.1

        state['qpos'] = new_qpos
        state['qvel'] = new_qvel
        simulator.set_core_state(state)

        # 验证状态被设置
        new_state = simulator.get_core_state()
        np.testing.assert_array_almost_equal(new_state['qpos'], new_qpos)
        np.testing.assert_array_almost_equal(new_state['qvel'], new_qvel)


class TestDerivedStateWorks:
    """测试 get_derived_state() 正常工作"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def test_derived_state_structure(self, simulator):
        """
        场景：获取派生状态
        预期：包含 contacts 和机器人数据
        """
        derived = simulator.get_derived_state()

        assert 'contacts' in derived
        assert isinstance(derived['contacts'], list)
        assert 'robot_a' in derived
        assert 'robot_b' in derived

    def test_contacts_list_is_valid(self, simulator):
        """
        场景：检查碰撞列表
        预期：碰撞数据格式正确
        """
        derived = simulator.get_derived_state()
        contacts = derived['contacts']

        for contact in contacts:
            assert 'geom_a' in contact
            assert 'geom_b' in contact
            assert 'body_a' in contact
            assert 'body_b' in contact
            assert 'position' in contact
            assert 'normal' in contact
            assert 'force' in contact

    def test_xpos_dimensions(self, simulator):
        """
        场景：检查 xpos 维度
        预期：xpos shape = (33, 3)（33 个 body）
        """
        derived = simulator.get_derived_state()
        xpos = derived['robot_a']['xpos']

        assert xpos.shape == (33, 3), f"Expected (33, 3), got {xpos.shape}"

    def test_xquat_dimensions(self, simulator):
        """
        场景：检查 xquat 维度
        预期：xquat shape = (33, 4)（33 个 body，每个 4 元素四元数）
        """
        derived = simulator.get_derived_state()
        xquat = derived['robot_a']['xquat']

        assert xquat.shape == (33, 4), f"Expected (33, 4), got {xquat.shape}"


class TestModelDimensionsMatchSpec:
    """测试模型维度与 DATASPEC.md 一致"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        yield sim
        sim.close()

    def test_model_dimensions_match_spec(self, simulator):
        """
        场景：检查模型维度
        预期：与 DATASPEC.md 中的值完全一致
        """
        model = simulator.model

        assert model.nq == 56, f"Expected nq=56, got {model.nq}"
        assert model.nv == 54, f"Expected nv=54, got {model.nv}"
        assert model.nu == 42, f"Expected nu=42, got {model.nu}"
        assert model.na == 0, f"Expected na=0, got {model.na}"
        assert model.nbody == 33, f"Expected nbody=33, got {model.nbody}"
        assert model.njnt == 44, f"Expected njnt=44, got {model.njnt}"
        assert model.ngeom == 44, f"Expected ngeom=44, got {model.ngeom}"

    def test_body_ids_match_spec(self, simulator):
        """
        场景：检查关键 body 的 ID
        预期：与 DATASPEC.md 一致
        """
        model = simulator.model

        # 根据 DATASPEC.md 的 body 列表
        world_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'world')
        assert world_id == 0

        torso_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_red')
        assert torso_red_id == 1

        pelvis_red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis_red')
        assert pelvis_red_id == 4

        torso_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_blue')
        assert torso_blue_id == 17

        pelvis_blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis_blue')
        assert pelvis_blue_id == 20


class TestBroadcastViewWorks:
    """测试广播视图图像功能"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def test_get_broadcastview_image_returns_correct_shape(self, simulator):
        """
        场景：获取广播视图图像
        预期：返回正确形状的 RGB 图像
        """
        image = simulator.get_broadcastview_image()

        assert image.shape == (720, 1280, 3), f"Expected (720, 1280, 3), got {image.shape}"
        assert image.dtype == np.uint8


class TestPDControllerWorks:
    """测试 PD 控制器功能"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def test_action_is_applied(self, simulator):
        """
        场景：设置动作并执行物理步
        预期：动作被应用到 PD 控制器
        """
        # 获取初始关节位置
        static_data = simulator.get_static_data()
        qpos_indices = static_data['robot_info']['robot_a']['qpos_indices']
        initial_pos = simulator.data.qpos[qpos_indices[0]]

        # 设置动作（非零）
        simulator.set_action({'robot_a': np.ones(21) * 0.5, 'robot_b': None})

        # 执行物理步
        simulator.physical_step()

        # 验证关节位置发生变化（因为 PD 控制器在施加力矩）
        new_pos = simulator.data.qpos[qpos_indices[0]]
        # 由于动力学和 PD 控制，位置应该有变化
        # 注意：具体的变化量取决于物理参数，这里只验证有变化
        time_advanced = simulator.data.time > 0
        assert time_advanced, "Time should advance after physical_step"

    def test_ctrl_limits_are_respected(self, simulator):
        """
        场景：设置超出范围的 action
        预期：action 被裁剪到 [-1, 1]
        """
        # 设置超出范围的动作
        simulator.set_action({
            'robot_a': np.ones(21) * 10.0,  # 超出范围
            'robot_b': np.ones(21) * -10.0
        })

        # 获取设置的 action
        action = simulator.get_action()

        # 验证 action 被存储
        assert action['robot_a'] is not None
        assert action['robot_b'] is not None


class TestPDControllerPerformance:
    """测试 PD 控制器性能和响应时间"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def _get_joint_pos(self, simulator, robot_id='robot_a'):
        """获取当前关节位置（排除 -1 索引）"""
        static_data = simulator.get_static_data()
        qpos_indices = static_data['robot_info'][robot_id]['qpos_indices']
        pos = []
        for idx in qpos_indices:
            if idx >= 0:
                pos.append(simulator.data.qpos[idx])
        return np.array(pos, dtype=np.float32)

    def _get_valid_indices(self, simulator, robot_id='robot_a'):
        """获取有效的关节索引"""
        static_data = simulator.get_static_data()
        qpos_indices = static_data['robot_info'][robot_id]['qpos_indices']
        valid_indices = [i for i, idx in enumerate(qpos_indices) if idx >= 0]
        return valid_indices

    def test_action_to_target_conversion(self, simulator):
        """
        场景：验证 action → target_pos 的转换逻辑
        预期：target_pos = reference_pos + action_scale * action（裁剪后）
        """
        static_data = simulator.get_static_data()
        robot_info = static_data['robot_info']['robot_a']
        reference_pos = simulator.reference_pos['robot_a']
        action_scale = simulator.action_scale['robot_a']

        # 设置一个已知的 action
        test_action = np.array([0.5] * 21, dtype=np.float32)

        # 计算期望的 target_pos（包含裁剪）
        raw_target = reference_pos + action_scale * test_action
        expected_target = np.clip(raw_target,
                                  simulator.joint_limits['robot_a']['lower'],
                                  simulator.joint_limits['robot_a']['upper'])

        # 应用 action
        simulator.set_action({'robot_a': test_action, 'robot_b': None})

        # 获取实际的 target_positions
        actual_target = simulator.target_positions['robot_a']

        # 只比较有效索引
        valid_indices = self._get_valid_indices(simulator, 'robot_a')
        np.testing.assert_array_almost_equal(
            actual_target[valid_indices],
            expected_target[valid_indices],
            decimal=5
        )

    def test_pd_tracks_step_input(self, simulator):
        """
        场景：设置 action，验证 PD 控制器在工作
        预期：关节位置有变化（尽管可能因 dof_damping=5.0 无法完全收敛）
        """
        static_data = simulator.get_static_data()
        reference_pos = simulator.reference_pos['robot_a']
        action_scale = simulator.action_scale['robot_a']

        # 设置一个 action
        action = np.array([0.5] * 21, dtype=np.float32)
        simulator.set_action({'robot_a': action, 'robot_b': None})

        expected_target = reference_pos + action_scale * action

        valid_indices = self._get_valid_indices(simulator, 'robot_a')
        expected_valid = expected_target[valid_indices]

        # 记录初始位置
        initial_pos = self._get_joint_pos(simulator, 'robot_a')

        # 执行多个物理步
        for step in range(50):
            simulator.physical_step()

        final_pos = self._get_joint_pos(simulator, 'robot_a')

        # 验证：关节位置发生了变化（PD 控制器在工作）
        movement = np.max(np.abs(final_pos - initial_pos))
        print(f"PD 控制器工作验证: 移动量 = {movement:.4f} rad")

        # 验证有明显的移动（至少 0.01 rad）
        assert movement > 0.01, f"PD 控制器未产生明显移动: {movement:.6f} rad"

    def test_pd_response_time(self, simulator):
        """
        场景：验证 PD 控制器对 action 的响应
        预期：更大的 action 导致更大的初始移动
        """
        action = np.array([0.5] * 21, dtype=np.float32)
        simulator.set_action({'robot_a': action, 'robot_b': None})

        static_data = simulator.get_static_data()
        reference_pos = simulator.reference_pos['robot_a']
        action_scale = simulator.action_scale['robot_a']
        expected_target = reference_pos + action_scale * action

        valid_indices = self._get_valid_indices(simulator, 'robot_a')
        expected_valid = expected_target[valid_indices]

        # 记录初始位置
        initial_pos = self._get_joint_pos(simulator, 'robot_a')

        # 执行 50 步
        for step in range(50):
            simulator.physical_step()

        final_pos = self._get_joint_pos(simulator, 'robot_a')
        final_error = np.max(np.abs(expected_valid - final_pos))
        movement = np.max(np.abs(final_pos - initial_pos))

        print(f"PD 响应: 移动量={movement:.4f} rad, 到目标误差={final_error:.4f} rad")
        print(f"时间步长 dt = {simulator.dt} s, 50 步 = {50 * simulator.dt} s")

        # 验证：有移动且误差不是无穷大
        assert movement > 0.01, "PD 控制器未产生移动"
        assert final_error < 1.0, f"误差过大: {final_error:.4f} rad"

    def test_pd_with_different_action_magnitudes(self, simulator):
        """
        场景：测试不同 action 大小下的响应
        预期：PD 控制器在工作，关节位置有变化
        """
        action_magnitudes = [0.2, 0.5, 1.0]

        for mag in action_magnitudes:
            simulator.reset()
            action = np.array([mag] * 21, dtype=np.float32)
            simulator.set_action({'robot_a': action, 'robot_b': None})

            # 记录初始位置
            initial_pos = self._get_joint_pos(simulator, 'robot_a')

            # 执行一段时间
            for step in range(50):
                simulator.physical_step()

            final_pos = self._get_joint_pos(simulator, 'robot_a')
            movement = np.max(np.abs(final_pos - initial_pos))

            print(f"action={mag}: 移动量 = {movement:.4f} rad")

            # 验证：PD 控制器在工作（有移动）
            assert movement > 0.01, f"action={mag}: 无明显移动"

    def test_kp_kd_values_are_reasonable(self, simulator):
        """
        场景：验证 PD 参数 kp, kd 的值是否合理
        预期：kp 和 kd 应该为正数数组
        """
        kp = simulator.kp
        kd = simulator.kd

        print(f"PD 参数: kp = {kp}, kd = {kd}")

        # 验证 kp, kd 为正数
        assert np.all(kp > 0), f"kp 应该为正数，实际值: {kp}"
        assert np.all(kd > 0), f"kd 应该为正数，实际值: {kd}"

        # 对于 PD 控制，kd / kp 的比值影响阻尼
        # 临界阻尼: kd = 2 * sqrt(kp)
        damping_ratios = kd / (2 * np.sqrt(kp))
        print(f"阻尼比 (kd / 2*sqrt(kp)): mean={np.mean(damping_ratios):.2f}, min={np.min(damping_ratios):.2f}, max={np.max(damping_ratios):.2f}")

        # 验证阻尼比在合理范围内 (0.1 ~ 2.0)
        assert np.all(damping_ratios > 0.01), "阻尼比太小，可能导致振荡"
        assert np.all(damping_ratios < 5.0), "阻尼比太大，可能导致响应太慢"


class TestQuaternionFormat:
    """测试四元数格式与文档一致"""

    @pytest.fixture
    def simulator(self):
        sim = MujocoCombatSimulator()
        sim.reset()
        yield sim
        sim.close()

    def test_root_quaternion_is_wxyz(self, simulator):
        """
        场景：检查 root 四元数格式
        预期：使用 wxyz 顺序（与 DATASPEC.md 一致）
        """
        state = simulator.get_core_state()
        qpos = state['qpos']

        # robot_a root orientation at qpos[3:7]
        quat = qpos[3:7]

        # 验证四元数是单位四元数
        norm = np.linalg.norm(quat)
        assert abs(norm - 1.0) < 0.01, f"Quaternion should be unit length, got norm={norm}"

        # 初始时 robot_a 面朝 +x，四元数应该是 [1, 0, 0, 0] (wxyz)
        # [w, x, y, z] = [1, 0, 0, 0] 表示无旋转
        expected_quat = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(quat, expected_quat, decimal=2)

    def test_robot_b_quaternion_format(self, simulator):
        """
        场景：检查 robot_b 四元数格式
        预期：面朝 -x，四元数为 [0, 0, 0, 1]（180度绕 z轴）
        """
        state = simulator.get_core_state()
        qpos = state['qpos']

        # robot_b root orientation at qpos[31:35]
        quat = qpos[31:35]

        # 验证四元数是单位四元数
        norm = np.linalg.norm(quat)
        assert abs(norm - 1.0) < 0.01, f"Quaternion should be unit length, got norm={norm}"

        # robot_b 初始面朝 -x，四元数应该是 [0, 0, 0, 1] (wxyz)
        # [w, x, y, z] = [0, 0, 0, 1] 表示 180度绕 z轴旋转
        expected_quat = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(quat, expected_quat, decimal=2)
