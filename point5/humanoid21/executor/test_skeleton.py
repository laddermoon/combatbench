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
    p5 = Point5State(pos=pos)
    object.__setattr__(p5, "metadata", dict(p5.metadata) | {
        "frame_origin_xy": [0.0, 0.0], "frame_yaw_deg": 0.0})
    obs = Observation(
        time=0.0, robot_id="robot_a", core={},
        p5=p5, frame_pose=(np.zeros(2), 0.0),
        p5_world=pos, com_pos=np.array([0., 0., 1.0]),
        com_vel=np.zeros(3), ang_mom=np.zeros(3), mass=40.8,
        torso_tilt=0.0)
    clip = P5Clip(name="t", keyframes=(Keyframe(pos=pos),))
    return Context(observation=obs, clip=clip)


def test_anchor_rejects_mismatch():
    ex = Executor.__new__(Executor)          # skip sim init for speed
    ctx = _ctx()
    moved = ctx.clip.keyframes[0].pos.copy()
    moved[0, 2] += 0.05
    ctx.clip = P5Clip(name="t", keyframes=(
        Keyframe(pos=moved),))
    with pytest.raises(AnchorError):
        ex._verify_anchor(ctx)


def test_anchor_accepts_exact():
    ex = Executor.__new__(Executor)
    ex._verify_anchor(_ctx())                # must not raise


def test_pipeline_stubs_raise():
    ex = Executor.__new__(Executor)
    with pytest.raises(PlanError, match="algorithm 1"):
        ex.gen_action(_ctx())
