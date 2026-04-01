"""
Humanoid21 测试配置与共享 Fixtures
"""
import sys
from pathlib import Path
import pytest
import mujoco

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.humanoid21.simulator import MujocoCombatSimulator
from envs.humanoid21.plugins import (
    NonFallConstraintPlugin,
    CombatScoringPlugin,
    FrozenRobotPlugin,
)
from envs.humanoid21.runtime_units import (
    Humanoid21Observer,
    Humanoid21Rewarder,
)
from envs.framework.context import SimContext


# =============================================================================
# Simulator Fixtures
# =============================================================================

@pytest.fixture
def simulator():
    """提供 MujocoCombatSimulator 实例"""
    sim = MujocoCombatSimulator()
    yield sim
    sim.close()


@pytest.fixture
def reset_simulator(simulator):
    """提供已重置的 MujocoCombatSimulator 实例"""
    simulator.reset()
    yield simulator


# =============================================================================
# Plugin Fixtures
# =============================================================================

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


# =============================================================================
# Runtime Unit Fixtures
# =============================================================================

@pytest.fixture
def humanoid_observer():
    """提供 Humanoid21Observer 实例"""
    return Humanoid21Observer('robot_a')


@pytest.fixture
def humanoid_rewarder():
    """提供 Humanoid21Rewarder 实例"""
    return Humanoid21Rewarder('robot_a')


# =============================================================================
# Context Helpers
# =============================================================================

def _create_writable_context(simulator):
    """创建具有写入权限的 SimContext"""
    ctx = SimContext(simulator)
    ctx._grant_mutator()
    return ctx
