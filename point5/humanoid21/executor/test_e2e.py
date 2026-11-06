"""End-to-end regression: ExecutorPolicy inside a real sim loop.

Asserts the current capability floor, not the dream:
  - survives N steps on the punch intent (no collapse),
  - produces arm reach (punch extension) and some foot travel,
  - stand clip holds posture within tolerance.

Not asserting full step completion (+0.35) yet — pose tracking can't
express a swing trajectory; that's the known v1 ceiling.
"""
import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from point5.contracts import Keyframe, P5Clip, Transfer
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.humanoid21.executor.tools.com_solver import com_estimate
from point5.humanoid21.executor.tools.pose_anchor import PoseAnchor
from point5.humanoid21.executor.e2e_policy import ExecutorPolicy


def _run(clip, n_steps, seed=0):
    sim = Humanoid21Simulator(initial_distance=6.0,
                              initial_pose_a="standing")
    sim.reset(seed=seed)
    pol = ExecutorPolicy(robot_id="robot_a", clip=clip)
    com_z, tilt, fl_x, reach = [], [], [], []
    for _ in range(n_steps):
        obs = sim.get_observation()["robot_a"]
        act, _ = pol.act(obs)
        sim.set_action({"robot_a": np.asarray(act, np.float32),
                        "robot_b": np.zeros(21, np.float32)})
        for _ in range(25):
            sim.physical_step()
        o = sim.get_observation()["robot_a"]
        p5 = PoseAnchor.from_obs(o).anchor('ankle_mid').p5().pos
        com_z.append(com_estimate(p5)[2])
        g = o[42:45]
        tilt.append(np.degrees(np.arccos(
            np.clip(-g[2] / (np.linalg.norm(g) + 1e-9), -1, 1))))
        fl_x.append(p5[4, 0])
        reach.append(p5[2, 0] - p5[0, 0])
    sim.close()
    return np.array(com_z), np.array(tilt), np.array(fl_x), np.array(reach)


def _stand_clip():
    p = stand_p5()
    return P5Clip(name="stand", keyframes=(
        Keyframe(pos=p), Keyframe(pos=p, transfer=Transfer(0, 0, 0, 5.0))))


def _punch_clip():
    return plan_stand_punch(np.array([1.05, 0, 1.15]), "left").task


@pytest.mark.slow
def test_stand_survives():
    com_z, tilt, _, _ = _run(_stand_clip(), 60)
    assert com_z[-1] > 0.70, f"collapsed to {com_z[-1]:.3f}"
    assert tilt[-1] < 25.0, f"leaning {tilt[-1]:.1f}deg at end"


@pytest.mark.slow
def test_punch_survives_and_reaches():
    com_z, tilt, fl_x, reach = _run(_punch_clip(), 60)
    assert com_z[-1] > 0.65, f"collapsed to {com_z[-1]:.3f}"
    assert reach.max() > 0.25, f"no arm extension ({reach.max():.3f})"
    # foot should at least START traveling forward sometimes
    assert fl_x.max() > 0.02, f"foot never moved ({fl_x.max():.3f})"
