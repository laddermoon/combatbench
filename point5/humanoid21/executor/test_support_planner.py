"""Scenario tests for the support-planning sub-algorithm."""
import numpy as np
import pytest

from point5.contracts import Keyframe, P5Clip, Transfer
from point5.humanoid21.stand_punch import stand_p5, plan_stand_punch
from point5.humanoid21.executor.contracts import (FootSupport,
                                                  SupportLandMark)
from point5.humanoid21.executor.support_planner import (
    plan_footsteps, support_transitions)

P = stand_p5()


def _mk(com=(0.03, 0.0, 0.87), vel=(0, 0, 0),
        fr=0.5, fl=0.5):
    return SupportLandMark(
        com=np.asarray(com, float),
        feet={
            "right": FootSupport(fr, P[3, :2].copy()),
            "left": FootSupport(fl, P[4, :2].copy())})


def _punch(target=(1.05, 0, 1.15), hand="left"):
    plan = plan_stand_punch(np.asarray(target, float), hand)
    assert plan.reachable
    raw = list(plan.task.keyframes)
    return P5Clip(name="p", keyframes=tuple(
        [Keyframe(pos=stand_p5()),
         Keyframe(pos=raw[0].pos,
                  transfer=Transfer(0, 0, 0, 0.5))] + raw[1:]))


def _shift_clip(dy=0.3):
    """A clip requiring a right-foot step to the side."""
    p2 = P.copy()
    p2[3] += np.array([0, -dy, 0])          # right foot steps -y
    return P5Clip(name="s", keyframes=(
        Keyframe(pos=P),
        Keyframe(pos=p2, transfer=Transfer(0, 0, 0, 1.0))))


# -- baseline ---------------------------------------------------------------
def test_still_no_corrective():
    steps = plan_footsteps(_mk(), _punch())
    assert all(s.kind == "move" for s in steps)
    # swing is a mid-air waypoint (z>0) -> skipped; only the
    # LAND frame emits a footstep
    assert [(s.foot, s.kind) for s in steps] == [("left", "move")]


def test_no_transition_no_steps():
    clip = P5Clip(name="h", keyframes=(
        Keyframe(pos=P),
        Keyframe(pos=P, transfer=Transfer(0, 0, 0, 1.0))))
    assert support_transitions(clip) is None
    assert plan_footsteps(_mk(), clip) == []


# -- capture-point physics ----------------------------------------------------
def test_slow_drift_inside_hull_no_step():
    """v small enough that xi stays in polygon -> no corrective."""
    steps = plan_footsteps(_mk(), _punch(), vel=(0, 0.05, 0))
    assert all(s.kind == "move" for s in steps)


def test_strong_lateral_drift_capture_step():
    """v=+0.5y throws xi beyond +y edge -> nearest foot braces."""
    steps = plan_footsteps(_mk(), _punch(), vel=(0, 0.5, 0))
    cor = [s for s in steps if s.kind == "corrective"]
    assert cor and cor[0].foot == "left"
    assert cor[0].target[1] > P[4, 1]          # steps further +y


def test_backward_momentum_backward_step():
    """COM moving -x while intent is forward -> backward brace step."""
    steps = plan_footsteps(_mk(), _punch(), vel=(-0.4, 0, 0))
    cor = [s for s in steps if s.kind == "corrective"]
    assert cor
    # capture step lands BEHIND the current foot position
    assert cor[0].target[0] < 0.05


def test_step_length_clamped():
    steps = plan_footsteps(_mk(), _punch(), vel=(0, 2.0, 0))
    cor = [s for s in steps if s.kind == "corrective"]
    for s in cor:
        fp = _mk().feet[s.foot].pos
        assert np.linalg.norm(s.target - fp) <= 0.35 + 1e-9


# -- support transitions ------------------------------------------------------
def test_side_shift_detected():
    trans = support_transitions(_shift_clip())
    assert trans and trans[0].mover == "right"


def test_mover_target_matches_keyframe():
    steps = plan_footsteps(_mk(), _shift_clip())
    mv = [s for s in steps if s.kind == "move"]
    assert mv[0].foot == "right"
    assert np.allclose(mv[0].target, P[3, :2] + [0, -0.3],
                       atol=1e-9)


# -- airborne start -----------------------------------------------------------
def test_single_support_start():
    """Start with left foot unloaded: plan still emits steps."""
    lm = _mk(fr=1.0, fl=0.0)
    steps = plan_footsteps(lm, _punch())
    assert steps
