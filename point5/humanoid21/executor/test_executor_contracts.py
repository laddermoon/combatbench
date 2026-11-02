"""E0 verification: contract constants match the real simulator."""
import numpy as np
import pytest

from point5.humanoid21.executor.contracts import (
    CASES, ExecutorConfig, RefFrame)
from point5.humanoid21.executor.sim_iface import (
    action_to_joint, joint_to_action, load_interface)


@pytest.fixture(scope="module")
def sim():
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator(initial_distance=6.0)
    s.reset(seed=0)
    yield s
    s.close()


def test_timing_matches_sim(sim):
    cfg = ExecutorConfig()
    iface = load_interface(sim)
    assert iface.dt_physics == pytest.approx(cfg.phy_dt)
    assert iface.steps_per_action == cfg.phy_steps == 25


def test_action_is_normalized_pd_target(sim):
    """action=0 targets joint mid-range, NOT the standing pose."""
    iface = load_interface(sim)
    tgt = action_to_joint(np.zeros(21), iface)
    assert not np.allclose(tgt, 0.0)  # mid-range != standing zeros
    np.testing.assert_allclose(
        action_to_joint(iface.standing_action, iface), 0.0, atol=1e-4)


def test_norm_roundtrip(sim):
    iface = load_interface(sim)
    q = np.linspace(-0.4, 0.4, 21)
    np.testing.assert_allclose(
        action_to_joint(joint_to_action(q, iface), iface), q, atol=1e-9)


def test_spawn_matches_meta(sim):
    """Standing spawn: joint_pos_norm == norm(0) == standing action;
    robot_a sits at x=-dist/2, meta root z."""
    from envs.humanoid21.meta import Humanoid21Meta
    from point5.humanoid21.executor.sim_iface import load_interface
    core = sim.get_core_state()["robot_a"]
    iface = load_interface(sim)
    np.testing.assert_allclose(
        core["joint_pos_norm"], iface.standing_action, atol=1e-3)
    rp = Humanoid21Meta.INITIAL_POSES["standing"]["root_pos"]
    assert core["root_pos"][2] == pytest.approx(rp[2], abs=1e-3)
    assert core["root_pos"][0] == pytest.approx(-3.0, abs=1e-3)


def test_cases_table():
    req = [c for c in CASES if c.required]
    ext = [c for c in CASES if not c.required]
    assert len(req) == 4 and len(ext) == 6
    assert {c.hand for c in req} == {"left", "right"}
    assert {c.seed for c in req} == {0, 1}


def test_refframe_fields():
    r = RefFrame(t=0.0, p5_world=np.zeros((5, 3)),
                 p5_vel_world=np.zeros((5, 3)),
                 com_pos=np.zeros(3), com_vel=np.zeros(3),
                 yaw=0.0, yaw_rate=0.0,
                 support=np.array([True, True]),
                 tol_point=np.full(5, 0.1))
    assert r.p5_world.shape == (5, 3) and r.support[0]
