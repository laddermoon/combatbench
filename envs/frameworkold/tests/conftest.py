"""
Pytest 配置和共享 Fixtures
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional, Callable


@pytest.fixture
def mock_simulator():
    """Mock OpenSimulator 实例"""
    simulator = Mock()
    simulator.dt = 0.002
    simulator.reset = Mock()
    simulator.physical_step = Mock()
    simulator.get_physical_frequency = Mock(return_value=500)  # 500Hz = 2ms per step
    simulator.set_action = Mock()
    simulator.get_core_state = Mock(return_value={
        'time': 0.0,
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
    })
    simulator.get_derived_state = Mock(return_value={
        'robots': {
            'robot_a': {
                'observation': np.zeros(127, dtype=np.float32),
                'torso_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                'torso_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
            'robot_b': {
                'observation': np.zeros(127, dtype=np.float32),
                'torso_position': np.array([0.0, 0.0, 1.4], dtype=np.float32),
                'torso_orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            },
        },
    })
    simulator.get_sensor_data = Mock(return_value={})
    simulator.get_static_data = Mock(return_value={})
    simulator.get_broadcastview_image = Mock(return_value=np.zeros((720, 1280, 3), dtype=np.uint8))
    return simulator


@pytest.fixture
def sample_actions():
    """示例动作数据"""
    return {
        'robot_a': np.random.uniform(-0.5, 0.5, 21).astype(np.float32),
        'robot_b': np.random.uniform(-0.5, 0.5, 21).astype(np.float32),
    }


@pytest.fixture
def sample_core_state():
    """示例核心状态"""
    return {
        'time': 1.5,
        'robots': {
            'robot_a': {
                'root_position': np.array([0.1, 0.2, 1.35], dtype=np.float32),
                'root_orientation': np.array([0.99, 0.0, 0.0, 0.1], dtype=np.float32),
                'joint_positions': np.ones(21, dtype=np.float32) * 0.1,
                'joint_velocities': np.ones(21, dtype=np.float32) * 0.05,
            },
            'robot_b': {
                'root_position': np.array([-0.1, -0.2, 1.45], dtype=np.float32),
                'root_orientation': np.array([0.98, 0.0, 0.0, 0.15], dtype=np.float32),
                'joint_positions': np.ones(21, dtype=np.float32) * -0.1,
                'joint_velocities': np.ones(21, dtype=np.float32) * -0.05,
            },
        },
    }


@pytest.fixture
def sample_derived_state():
    """示例衍生状态"""
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


@pytest.fixture
def sample_sensor_data():
    """示例传感器数据"""
    return {
        'touch': {},
        'force': {},
        'imu': {},
    }
