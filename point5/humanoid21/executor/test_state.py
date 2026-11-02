"""E1 verification: snapshot determinism, sync, state extraction."""
import numpy as np
import pytest

from point5.humanoid21.executor.contracts import ExecutorConfig
from point5.humanoid21.executor.sim_iface import load_interface
from point5.humanoid21.executor.state import (
    capture_current, restore, snapshot, sync, fall_check)


@pytest.fixture(scope="module")
def pair():
    from envs.humanoid21.simulator import Humanoid21Simulator
    a = Humanoid21Simulator(initial_distance=6.0)
    b = Humanoid21Simulator(initial_distance=6.0)
    a.reset(seed=0)
    b.reset(seed=0)
    yield a, b
    a.close()
    b.close()


def _step(sim, a, n=1):
    for _ in range(n):
        sim.set_action({"robot_a": a, "robot_b": np.zeros(21)})
        for _ in range(25):
            sim.physical_step()


def test_restore_is_deterministic(pair):
    env, w = pair
    snap = snapshot(env)
    a = np.full(21, 0.3, np.float32)
    _step(env, a, 2)
    s1 = env.get_core_state()["robot_a"]
    restore(env, snap)
    _step(env, a, 2)
    s2 = env.get_core_state()["robot_a"]
    for k in s1:
        np.testing.assert_allclose(s1[k], s2[k], atol=1e-12,
                                   err_msg=k)


def test_sync_makes_identical(pair):
    env, w = pair
    _step(env, np.full(21, 0.2, np.float32), 3)   # perturb env
    sync(w, env)
    a = np.full(21, -0.4, np.float32)
    _step(env, a, 1)
    _step(w, a, 1)
    for k in env.get_core_state()["robot_a"]:
        np.testing.assert_allclose(
            env.get_core_state()["robot_a"][k],
            w.get_core_state()["robot_a"][k], atol=1e-12)


def test_current_fields(env_pair=None):
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator(initial_distance=6.0)
    s.reset(seed=0)
    try:
        cur = capture_current(s)
        assert cur.p5_world.shape == (5, 3)
        assert 0.8 < cur.com_pos[2] < 1.2
        assert cur.mass > 30
        assert cur.torso_tilt < 0.1
        assert cur.contacts["foot_right_force"] > 100
        assert fall_check(cur, ExecutorConfig()) is None
    finally:
        s.close()
