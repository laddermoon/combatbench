"""Algorithm 1 tests: centroidal trajectory generation."""
import numpy as np
import pytest

from point5.contracts import Keyframe, P5Clip, Transfer
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.humanoid21.executor.capability_solver import capability
from point5.humanoid21.executor.centroidal import (
    MASS_KG, CentroidalState)
from point5.humanoid21.executor.centroidal_planner import (
    CTRL_DT, CentroidalPlanner)
from point5.humanoid21.executor.com_solver import com_estimate
from point5.humanoid21.executor.contracts import Context

DT = CTRL_DT


def _c0():
    return CentroidalState(com=np.array([0.03, 0.0, 0.87]),
                           lin_mom=np.zeros(3), ang_mom=np.zeros(3))


def _punch_clip(target=(1.05, 0, 1.15), hand="left",
                start_dur=0.5):
    plan = plan_stand_punch(np.asarray(target, np.float64), hand)
    assert plan.reachable, plan.reason
    raw = list(plan.task.keyframes)
    kfs = [Keyframe(pos=stand_p5()),
           Keyframe(pos=raw[0].pos,
                    transfer=Transfer(0, 0, 0, start_dur))] + raw[1:]
    return P5Clip(name="punch", keyframes=tuple(kfs))


def _plan(clip, c0=None):
    return CentroidalPlanner("robot_a").plan(
        Context(np.zeros(96), clip), c0 or _c0())


def test_traj_ends_at_target_at_rest():
    tr = _plan(_punch_clip())
    assert tr.n_steps > 0
    assert np.linalg.norm(tr[-1].com_vel) < 1e-6
    # final com == last keyframe's estimate (in frame0 coords)
    plan = plan_stand_punch(np.array([1.05, 0, 1.15]), "left")
    raw = list(plan.task.keyframes)
    from point5.contracts import compose_frame, to_world
    poses = [(np.zeros(2), 0.0), compose_frame(np.zeros(2), 0.0,
                                               Transfer(0, 0, 0, 0.5))]
    for k in range(2, len(raw) + 1):
        poses.append(compose_frame(*poses[-1], raw[k - 1].transfer))
    cl = com_estimate(raw[-1].pos)
    want = to_world(cl[None], poses[-1][0], poses[-1][1])[0]
    assert np.allclose(tr[-1].com, want, atol=1e-9)


def test_velocity_bounded_and_smooth():
    tr = _plan(_punch_clip())
    vs = np.array([s.com_vel for s in tr.states])
    cap = capability(stand_p5())
    # peaks stay under the standing lateral/forward envelope + slack
    assert np.linalg.norm(vs[:, :2], axis=1).max() < cap.v_xy_cap + 0.5
    # no jumps between consecutive states
    steps = np.linalg.norm(np.diff([s.com for s in tr.states],
                                   axis=0), axis=1)
    assert steps.max() < 0.12                      # <12 cm per 50ms


def test_first_segment_starts_from_c0():
    c0 = _c0()
    c0.lin_mom = MASS_KG * np.array([0.1, 0.0, 0.0])
    tr = _plan(_punch_clip(), c0)
    assert np.linalg.norm(tr[0].com - c0.com) < 0.05
    # initial velocity carried over (decaying blend)
    assert tr[0].com_vel[0] > 0.05


def test_capability_stretches_infeasible_dur():
    """An impossibly short segment gets stretched by capability."""
    plan = plan_stand_punch(np.array([1.05, 0, 1.15]), "left")
    raw = list(plan.task.keyframes)
    kfs = [Keyframe(pos=stand_p5()),
           Keyframe(pos=raw[0].pos,
                    transfer=Transfer(0, 0, 0, 0.05))] + raw[1:]
    tr = _plan(P5Clip(name="x", keyframes=tuple(kfs)))
    # requested 0.05s but coil->swing COM shift needs more -> stretched
    assert tr.n_steps * DT > 0.3


def test_feasible_dur_respected():
    """A generous duration is used as-is (no stretching)."""
    tr = _plan(_punch_clip(start_dur=5.0))
    # first segment alone ~= 5s -> n_steps ~100+ for seg1
    assert tr.n_steps * DT >= 5.0


def test_single_keyframe_settles():
    clip = P5Clip(name="hold", keyframes=(Keyframe(pos=stand_p5()),))
    tr = _plan(clip)
    assert 0 < tr.n_steps <= 20
    assert np.linalg.norm(tr[-1].com_vel) < 1e-6


def test_extract_then_plan_on_real_obs():
    """End-to-end: sim standing obs -> z0 -> plan over punch clip."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    s = Humanoid21Simulator(initial_distance=6.0,
                            initial_pose_a="standing")
    s.reset(0)
    obs = s.get_derived_state()["robot_a"]["observation"]
    pl = CentroidalPlanner("robot_a")
    z0 = pl.extract_centroidal_state(obs)
    tr = pl.plan(Context(obs, _punch_clip()), z0)
    assert tr.n_steps > 0 and np.all(np.isfinite(tr[-1].z))
    # trajectory begins near the extracted state
    assert np.linalg.norm(tr[0].com - z0.com) < 0.1


def test_lateral_target_plan():
    tr = _plan(_punch_clip(target=(1.05, 0.4, 1.15), hand="right"))
    assert np.all(np.isfinite(tr[-1].z))
