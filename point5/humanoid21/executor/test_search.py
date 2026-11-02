"""E5 verification: search keeps the robot standing vs open-loop fall."""
import numpy as np
import pytest

from point5.humanoid21.executor.contracts import ExecutorConfig, RefFrame
from point5.humanoid21.executor.search import search_window
from point5.humanoid21.executor.sim_iface import load_interface
from point5.humanoid21.executor.state import (
    capture_current, snapshot, sync)


@pytest.fixture(scope="module")
def sims():
    from envs.humanoid21.simulator import Humanoid21Simulator
    e = Humanoid21Simulator(initial_distance=6.0)
    w = Humanoid21Simulator(initial_distance=6.0)
    e.reset(seed=0); w.reset(seed=0)
    yield e, w
    e.close(); w.close()


def _hold_refs(cur, cfg, h):
    r = RefFrame(t=0.0, p5_world=cur.p5_world,
                 p5_vel_world=np.zeros((5, 3)),
                 com_pos=cur.com_pos, com_vel=np.zeros(3),
                 yaw=np.deg2rad(cur.frame_pose[1]), yaw_rate=0.0,
                 support=np.array([True, True]),
                 tol_point=np.full(5, cfg.tol_default))
    return [r] * h


def test_search_maintains_stance(sims):
    env, w = sims
    cfg = ExecutorConfig(n_samples=12, n_iters=1, horizon=4, seed=0)
    iface = load_interface(env)
    prev = iface.standing_action.astype(np.float32)
    seq = np.tile(prev, (cfg.horizon, 1))
    for k in range(8):
        snap = sync(w, env)
        cur = capture_current(env, with_snapshot=False)
        refs = _hold_refs(cur, cfg, cfg.horizon)
        sol = search_window(w, snap, refs, seq, prev,
                            iface.standing_action,
                            "robot_a", "robot_b", cfg, seed=k)
        a = sol["action"]
        env.set_action({"robot_a": a, "robot_b": np.zeros(21)})
        for _ in range(25):
            env.physical_step()
        post = capture_current(env, with_snapshot=False)
        assert post.p5_world[0, 2] > 1.0, f"fell at step {k}"
        prev = a
        seq = np.vstack([sol["seq"][1:], sol["seq"][-1:]])
    assert post.torso_tilt < 0.4
