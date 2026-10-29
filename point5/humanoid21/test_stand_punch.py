"""Tests for point5.humanoid21.stand_punch (standing jab planner)."""
from __future__ import annotations

import numpy as np
import pytest

from point5.contracts import POINT_INDEX
from point5.trajectory import keyframe_world
from point5.humanoid21.stand_punch import (
    StandPunchConfig,
    plan_stand_punch,
    stand_p5,
)

CFG = StandPunchConfig()
FRONT = np.array([0.55, 0.0, 1.15])


def test_stand_p5_is_valid_local():
    p = stand_p5()
    assert p.shape == (5, 3)
    assert 1.2 < p[0, 2] < 1.6                     # head height
    assert p[3:, 2].max() < 0.15                   # ankles planted


def test_reachable_frontal_both_hands():
    for hand in ("left", "right"):
        p = plan_stand_punch(FRONT, hand, CFG)
        assert p.reachable, p.reason
        assert len(p.task.keyframes) == 3
        assert p.task.keyframes[0].transfer is None
        for kf in p.task.keyframes[1:]:
            assert kf.transfer is not None


def test_verdicts():
    assert plan_stand_punch(np.array([-0.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_sector"
    assert plan_stand_punch(np.array([1.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_reach"


def test_load_chambers_and_sinks():
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    assert w[1, 0, 2] < w[0, 0, 2]                 # head sinks on load
    i = POINT_INDEX["hand_right"]
    assert w[1, i, 0] < w[0, i, 0] + 1e-9          # fist pulled back
    assert w[1, i, 2] < w[1, 0, 2]                 # at/below chin


def test_feet_never_move_in_range():
    """In-range target: no stepping needed, feet stay planted."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    assert p.diag["step_len"] == 0.0
    for k in range(3):
        np.testing.assert_allclose(w[k, 3:], w[0, 3:], atol=1e-9)


def test_far_target_steps_and_sinks():
    """Far target: lead foot steps, COM drops + transfers, anchor stays."""
    far = np.array([0.95, 0.0, 1.15])
    p = plan_stand_punch(far, "right", CFG)
    assert p.reachable
    w = keyframe_world(p.task)
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[1, move] - w[0, move]) @ a
    assert adv == pytest.approx(p.diag["step_len"], abs=0.01)
    np.testing.assert_allclose(w[1, anchor], w[0, anchor], atol=1e-9)
    assert w[1, 0, 2] < w[0, 0, 2] - 0.05      # COM drops deeper
    # strike hand reaches target after stepping
    i = POINT_INDEX["hand_right"]
    assert np.linalg.norm(w[2, i] - far) < 0.15


def test_twist_schedule():
    p = plan_stand_punch(FRONT, "right", CFG)
    dys = [k.transfer.dyaw for k in p.task.keyframes[1:]]
    # right hand: coil + (counter), snap - (into punch); left mirrors
    assert dys[0] > 0 and dys[1] < 0
    p_l = plan_stand_punch(FRONT, "left", CFG)
    dys_l = [k.transfer.dyaw for k in p_l.task.keyframes[1:]]
    assert dys_l[0] < 0 and dys_l[1] > 0


def test_strike_hand_reaches_target_with_penetration():
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    i = POINT_INDEX["hand_right"]
    d = np.linalg.norm(w[2, i] - FRONT)
    assert d == pytest.approx(CFG.penetration, abs=0.03)


def test_initial_state_override():
    init = type("S", (), {"pos": stand_p5()})()
    p = plan_stand_punch(FRONT, "right", CFG, initial=init)
    assert p.reachable
