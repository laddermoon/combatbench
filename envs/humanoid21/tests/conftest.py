"""
Humanoid21 测试配置与共享 Fixtures
"""
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.framework.backend import BaseSimulator


class MockMuJoCoSimulator(BaseSimulator):
    """
    轻量级模拟器，用于测试 Humanoid21 组件

    模拟 MujocoCombatSimulator 的关键行为，但不依赖 MuJoCo
    """

    def __init__(self, initial_distance: float = 2.0):
        self.initial_distance = initial_distance
        self.dt = 0.002

        # 模拟 qpos/qvel (简化版，只包含关键部分)
        # 实际 Humanoid21 有更多 DOF，这里只模拟必要的部分
        self._qpos_size = 147  # 7 (root) + 21*2 + 其他
        self._qvel_size = 140  # 6 (root) + 21*2 + 其他

        self._qpos = np.zeros(self._qpos_size, dtype=np.float32)
        self._qvel = np.zeros(self._qvel_size, dtype=np.float32)

        # Robot info for both robots
        self.robot_info = {
            'robot_a': {
                'body_id': 0,
                'root_jnt_id': 0,
                'qpos_adr': 0,
                'qvel_adr': 0,
                'suffix': '_red',
                'qpos_indices': list(range(7, 28)),  # 模拟 21 个关节
                'qvel_indices': list(range(6, 27)),
                'jnt_ranges': [np.array([-np.pi, np.pi]) for _ in range(21)],
                'ctrl_ranges': [np.array([-1.0, 1.0]) for _ in range(21)],
                'qpos0': [0.0] * 21,
                'actuators': list(range(21)),
            },
            'robot_b': {
                'body_id': 1,
                'root_jnt_id': 1,
                'qpos_adr': 70,  # 假设 robot_b 从这里开始
                'qvel_adr': 70,
                'suffix': '_blue',
                'qpos_indices': list(range(77, 98)),
                'qvel_indices': list(range(76, 97)),
                'jnt_ranges': [np.array([-np.pi, np.pi]) for _ in range(21)],
                'ctrl_ranges': [np.array([-1.0, 1.0]) for _ in range(21)],
                'qpos0': [0.0] * 21,
                'actuators': list(range(21, 42)),
            }
        }

        # 模拟 Mujoco data
        self._time = 0.0
        self._xpos = np.zeros((100, 3), dtype=np.float32)
        self._xquat = np.zeros((100, 4), dtype=np.float32)
        self._cvel = np.zeros((100, 6), dtype=np.float32)
        self._cfrc_ext = np.zeros((100, 6), dtype=np.float32)

        # 设置初始位置
        self._reset_positions()

        # Contacts
        self._contacts = []

        # 模拟 geom/body 名称映射
        self._geom_names = {}
        self._body_names = {}

    def _reset_positions(self):
        """重置机器人位置"""
        dist = self.initial_distance

        # Robot A (red) at -dist/2
        self._qpos[0:3] = [-dist / 2.0, 0.0, 1.282]  # x, y, z
        self._qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w, x, y, z)

        # Robot B (blue) at +dist/2
        b_adr = self.robot_info['robot_b']['qpos_adr']
        self._qpos[b_adr:b_adr+3] = [dist / 2.0, 0.0, 1.282]
        self._qpos[b_adr+3:b_adr+7] = [0.0, 0.0, 0.0, 1.0]

        # 更新 xpos
        self._xpos[0] = [-dist / 2.0, 0.0, 1.282]
        self._xpos[1] = [dist / 2.0, 0.0, 1.282]
        self._xquat[0] = [1.0, 0.0, 0.0, 0.0]
        self._xquat[1] = [0.0, 0.0, 0.0, 1.0]

    # IDataAccessor 接口
    def get_static_data(self) -> Dict[str, Any]:
        return {
            'dt': self.dt,
            'robot_info': self.robot_info
        }

    def get_core_state(self) -> Dict[str, Any]:
        state = {
            'qpos': self._qpos.copy(),
            'qvel': self._qvel.copy(),
            'time': self._time,
            'robot_a': {},
            'robot_b': {}
        }

        for r_id in ['robot_a', 'robot_b']:
            qpos_adr = self.robot_info[r_id]['qpos_adr']
            qvel_adr = self.robot_info[r_id]['qvel_adr']
            state[r_id]['root_position'] = self._qpos[qpos_adr:qpos_adr+3].copy()
            state[r_id]['root_orientation'] = self._qpos[qpos_adr+3:qpos_adr+7].copy()
            state[r_id]['root_linear_velocity'] = self._qvel[qvel_adr:qvel_adr+3].copy()
            state[r_id]['root_angular_velocity'] = self._qvel[qvel_adr+3:qvel_adr+6].copy()

        return state

    def set_core_state(self, state: Dict[str, Any]) -> None:
        self._qpos[:] = state['qpos']
        self._qvel[:] = state['qvel']
        self._time = state.get('time', self._time)

        # 同步 structured data 回 qpos/qvel
        for r_id in ['robot_a', 'robot_b']:
            if r_id in state:
                r_state = state[r_id]
                qpos_adr = self.robot_info[r_id]['qpos_adr']
                qvel_adr = self.robot_info[r_id]['qvel_adr']
                if 'root_position' in r_state:
                    self._qpos[qpos_adr:qpos_adr+3] = r_state['root_position']
                if 'root_orientation' in r_state:
                    self._qpos[qpos_adr+3:qpos_adr+7] = r_state['root_orientation']
                if 'root_linear_velocity' in r_state:
                    self._qvel[qpos_adr:qpos_adr+3] = r_state['root_linear_velocity']
                if 'root_angular_velocity' in r_state:
                    self._qvel[qpos_adr+3:qvel_adr+6] = r_state['root_angular_velocity']

    def get_derived_state(self) -> Dict[str, Any]:
        return {
            'contacts': self._contacts.copy(),
            'robot_a': {
                'xpos': self._xpos.copy(),
                'xvelp': self._cvel[:, 3:].copy(),
                'xquat': self._xquat.copy()
            },
            'robot_b': {
                'xpos': self._xpos.copy(),
                'xvelp': self._cvel[:, 3:].copy(),
                'xquat': self._xquat.copy()
            },
        }

    def get_sensor_data(self) -> Dict[str, Any]:
        return {'sensordata': np.zeros(10)}

    def get_action(self) -> Dict[str, Any]:
        return {'robot_a': np.zeros(21), 'robot_b': np.zeros(21)}

    def set_action(self, action: Dict[str, Any]) -> None:
        pass  # 简化版本，不处理 action

    # BaseSimulator 接口
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        self._qpos[:] = 0
        self._qvel[:] = 0
        self._time = 0.0
        self._contacts.clear()
        self._reset_positions()

        # 应用自定义初始距离
        if options and 'initial_distance' in options:
            self.initial_distance = float(options['initial_distance'])
            self._reset_positions()

    def physical_step(self) -> None:
        self._time += self.dt

    def get_physical_frequency(self) -> float:
        return 1.0 / self.dt

    def close(self) -> None:
        pass

    def get_broadcastview_image(self) -> Any:
        """返回模拟的广播图像（黑色画面）"""
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    # 添加模拟数据访问属性
    @property
    def data(self):
        """模拟 mujoco.MjData"""
        class MockData:
            def __init__(self, sim):
                self.sim = sim
                self.qpos = sim._qpos
                self.qvel = sim._qvel
                self.time = sim._time
                self.xpos = sim._xpos
                self.xquat = sim._xquat
                self.cvel = sim._cvel
                self.cfrc_ext = sim._cfrc_ext
                self.ncon = len(sim._contacts)
                self.contact = sim._contacts
                self.sensordata = np.zeros(10)

            def __getitem__(self, key):
                # 支持访问 geom/body 名称
                return None

        return MockData(self)

    @property
    def model(self):
        """模拟 mujoco.MjModel"""
        class MockModel:
            def __init__(self, sim):
                self.sim = sim
                self.geom_type = np.zeros(100, dtype=int)
                self.geom_bodyid = np.zeros(100, dtype=int)

        return MockModel(self)

    # 辅助方法：添加测试碰撞
    def add_contact(self, geom1: int, geom2: int, body1: int, body2: int,
                    geom1_name: str, geom2_name: str, impulse: float = 1.0):
        """添加一个碰撞记录用于测试"""
        self._contacts.append({
            'geom_a': geom1,
            'geom_b': geom2,
            'body_a': body1,
            'body_b': body2,
            'geom1_name': geom1_name,
            'geom2_name': geom2_name,
            'position': np.array([0.0, 0.0, 0.0]),
            'normal': np.array([0.0, 0.0, 1.0]),
            'impulse': impulse
        })

    # 辅助方法：设置机器人姿态（用于测试防摔倒）
    def set_robot_orientation(self, robot_id: str, roll: float, pitch: float, yaw: float):
        """设置机器人的姿态角（度）"""
        from scipy.spatial.transform import Rotation as R

        r_id = 'robot_a' if robot_id == 'robot_a' else 'robot_b'
        qpos_adr = self.robot_info[r_id]['qpos_adr']

        # 创建旋转
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        self._qpos[qpos_adr+3:qpos_adr+7] = quat_wxyz


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_simulator():
    """提供 MockMuJoCoSimulator 实例"""
    sim = MockMuJoCoSimulator()
    yield sim
    sim.close()


@pytest.fixture
def mock_simulator_fresh():
    """每次测试都提供全新的 MockMuJoCoSimulator"""
    return MockMuJoCoSimulator()


# 导入 humanoid21 模块
from envs.humanoid21.plugins import (
    NonFallConstraintPlugin,
    CombatScoringPlugin,
    FrozenRobotPlugin,
)
from envs.humanoid21.runtime_units import (
    Humanoid21Observer,
    Humanoid21Rewarder,
    build_shared_runtime_info,
)
from envs.framework.context import SimContext, ReadOnlySimContext, TerminationReason


@pytest.fixture
def sim_context(mock_simulator):
    """提供 SimContext 实例"""
    return SimContext(mock_simulator)


@pytest.fixture
def non_fall_plugin():
    """提供 NonFallConstraintPlugin 实例"""
    return NonFallConstraintPlugin(pitch_limit_deg=5.0, roll_limit_deg=5.0)


@pytest.fixture
def combat_scoring_plugin():
    """提供 CombatScoringPlugin 实例"""
    return CombatScoringPlugin(
        initial_health=100.0,
        damage_scale=100.0
    )


@pytest.fixture
def frozen_robot_plugin():
    """提供 FrozenRobotPlugin 实例"""
    return FrozenRobotPlugin(frozen_robot_id='robot_b')


@pytest.fixture
def humanoid_observer():
    """提供 Humanoid21Observer 实例"""
    return Humanoid21Observer('robot_a')


@pytest.fixture
def humanoid_rewarder():
    """提供 Humanoid21Rewarder 实例"""
    return Humanoid21Rewarder('robot_a')
