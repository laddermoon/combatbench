"""P1 centroidal-state extraction tests."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.centroidal_dynamics.model_spec import build_model_spec
from mocap.humanoid21.centroidal_dynamics.state import (
    CentroidalStateExtractor,
    extract_centroidal_state,
)


@pytest.fixture(scope="module")
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture(scope="module")
def extractor(sim):
    return CentroidalStateExtractor(sim, build_model_spec(sim))


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def test_manual_and_subtree_state_match_at_reset(extractor):
    for robot_id in ("robot_a", "robot_b"):
        manual = extractor.manual_state(robot_id)
        subtree = extractor.subtree_state(robot_id, refresh=False)
        checked = extractor.state(robot_id)
        assert manual.mass == pytest.approx(subtree.mass, abs=1e-10)
        assert np.allclose(manual.com_pos, subtree.com_pos,
                           atol=1e-10, rtol=0.0)
        assert np.allclose(manual.lin_mom, subtree.lin_mom,
                           atol=1e-9, rtol=0.0)
        assert np.allclose(manual.ang_mom, subtree.ang_mom,
                           atol=1e-8, rtol=0.0)
        assert checked.mass == pytest.approx(manual.mass)
        assert checked.metadata["consistency"]["com"] < 1e-10


def test_manual_state_includes_root_and_joint_velocities(sim, extractor):
    qvel = np.linspace(-0.6, 0.7, sim.model.nv)
    sim.data.qvel[:] = qvel
    for robot_id in ("robot_a", "robot_b"):
        st = extractor.state(robot_id)
        assert np.linalg.norm(st.lin_mom) > 1.0
        assert np.linalg.norm(st.ang_mom) > 1.0
        assert st.metadata["consistency"]["lin_mom"] < 1e-8
        assert st.metadata["consistency"]["ang_mom"] < 1e-7


def test_robot_subtrees_are_isolated(sim, extractor):
    sim.reset(seed=0)
    before_b = extractor.state("robot_b")
    sim.data.qvel[:] = 0.0
    sim.data.qvel[sim._meta['robots']['robot_a']['qvel_indices']] = 0.25
    after_b = extractor.state("robot_b")
    after_a = extractor.state("robot_a")
    assert np.allclose(after_b.com_pos, before_b.com_pos,
                       atol=1e-12, rtol=0.0)
    assert np.allclose(after_b.lin_mom, np.zeros(3), atol=1e-10)
    assert np.allclose(after_b.ang_mom, np.zeros(3), atol=1e-10)
    assert np.linalg.norm(after_a.lin_mom) > 1.0


def test_root_translation_moves_com_but_not_momentum(sim, extractor):
    sim.reset(seed=0)
    before = extractor.state("robot_a")
    shift = np.array([0.31, -0.17, 0.09])
    root_qpos = sim._meta['robots']['robot_a']['root_qpos_adr']
    sim.data.qpos[root_qpos:root_qpos + 3] += shift
    mujoco.mj_forward(sim.model, sim.data)
    after = extractor.state("robot_a")
    assert np.allclose(after.com_pos, before.com_pos + shift,
                       atol=1e-10, rtol=0.0)
    assert np.allclose(after.lin_mom, before.lin_mom,
                       atol=1e-10, rtol=0.0)
    assert np.allclose(after.ang_mom, before.ang_mom,
                       atol=1e-9, rtol=0.0)


def test_yaw_rotation_covariance_about_root_origin(sim, extractor):
    sim.reset(seed=0)
    before = extractor.state("robot_a")
    robot = extractor.model_spec.robots["robot_a"]
    root_origin = sim.data.xpos[robot.root_body_id].copy()
    yaw = 0.37
    Rz = R.from_euler("z", yaw).as_matrix()
    root_qpos = sim._meta['robots']['robot_a']['root_qpos_adr']
    q_old = sim.data.qpos[root_qpos + 3:root_qpos + 7].copy()
    q_rot = R.from_euler("z", yaw) * R.from_quat(_wxyz_to_xyzw(q_old))
    sim.data.qpos[root_qpos + 3:root_qpos + 7] = _xyzw_to_wxyz(
        q_rot.as_quat())
    mujoco.mj_forward(sim.model, sim.data)
    after = extractor.state("robot_a")
    expected_com = root_origin + Rz @ (before.com_pos - root_origin)
    assert np.allclose(after.com_pos, expected_com,
                       atol=1e-9, rtol=0.0)
    assert np.allclose(after.lin_mom, np.zeros(3),
                       atol=1e-10, rtol=0.0)
    assert np.allclose(after.ang_mom, np.zeros(3),
                       atol=1e-10, rtol=0.0)


def test_com_is_not_root_body_frame_origin(sim, extractor):
    sim.reset(seed=0)
    st = extractor.state("robot_a")
    root = extractor.model_spec.robots["robot_a"].root_body_id
    assert np.linalg.norm(st.com_pos - sim.data.xpos[root]) > 0.3
    assert np.linalg.norm(st.com_pos - sim.data.xipos[root]) > 0.05


def test_extract_convenience_rejects_unknown_robot(sim):
    with pytest.raises(ValueError, match="robot_id"):
        extract_centroidal_state(sim, "robot_c")
