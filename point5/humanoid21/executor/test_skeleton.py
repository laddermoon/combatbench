"""Skeleton smoke tests: anchor contract + pipeline stub calls."""
import numpy as np
import pytest

from point5.contracts import Keyframe, P5Clip, Point5State
from point5.humanoid21.stand_punch import stand_p5
from point5.humanoid21.executor.contracts import (
    AnchorError, Context, Observation, PlanError)
from point5.humanoid21.executor.executor import Executor


def _ctx():
    pos = stand_p5()
    clip = P5Clip(name="t", keyframes=(Keyframe(pos=pos),))
    return Context(observation=np.zeros(96), clip=clip)


def _patched(ex, pos):
    ex._extract_p5 = lambda obs: np.asarray(pos)


def test_anchor_rejects_mismatch():
    ex = Executor.__new__(Executor)          # skip sim init for speed
    _patched(ex, stand_p5())
    ctx = _ctx()
    moved = ctx.clip.keyframes[0].pos.copy()
    moved[0, 2] += 0.05
    ctx.clip = P5Clip(name="t", keyframes=(
        Keyframe(pos=moved),))
    with pytest.raises(AnchorError):
        ex._verify_anchor(ctx)


def test_anchor_accepts_exact():
    ex = Executor.__new__(Executor)
    _patched(ex, stand_p5())
    ex._verify_anchor(_ctx())                # must not raise


def test_pipeline_stubs_raise():
    from point5.humanoid21.executor.centroidal_planner import (
        CentroidalPlanner)
    ex = Executor.__new__(Executor)
    ex.centroidal = CentroidalPlanner()
    ex.action_computer = None
    _patched(ex, stand_p5())
    with pytest.raises(PlanError):
        ex.gen_action(_ctx())


def test_extract_centroidal_standing():
    """obs(standing) -> c near local COM, l=L=0."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    from point5.humanoid21.executor.centroidal_planner import (
        CentroidalPlanner)
    s = Humanoid21Simulator(initial_distance=6.0,
                            initial_pose_a="standing")
    s.reset(0)
    obs = s.get_derived_state()["robot_a"]["observation"]
    z = CentroidalPlanner("robot_a").extract_centroidal_state(obs)
    assert 0.7 < z.com[2] < 1.0 and abs(z.com[1]) < 0.1
    assert np.linalg.norm(z.lin_mom) < 1.0
    assert np.linalg.norm(z.ang_mom) < 1.0


def test_capability_physics():
    from point5.humanoid21.executor.capability_solver import capability
    c = capability(stand_p5())
    assert c.static_ok and not c.single_support
    # lateral margin > forward (feet side by side)
    assert c.margin([0, 1]) > c.margin([1, 0]) > 0
    # authority positive and capped by friction cone
    assert 0 < c.a_max([0, 1]) <= 0.6 * 9.81 + 1e-9
    # lifted-foot pose: COM not over the support foot -> unstable
    p5 = stand_p5().copy(); p5[4, 2] = 0.3
    c2 = capability(p5)
    assert c2.single_support and not c2.static_ok
