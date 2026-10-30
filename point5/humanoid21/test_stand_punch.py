"""Tests for point5.humanoid21.stand_punch (standing strike planner).

Choreography: coil(降重心) -> swing(抬脚上步+刃架) -> land(落地+抱架)
-> strike(转腰出拳).  Lead-hand mechanics only.
"""
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
        assert len(p.task.keyframes) == 4
        assert [k.label for k in p.task.keyframes] == \
            ["coil", "swing", "land", "strike"]
        assert p.task.keyframes[0].transfer is None
        for kf in p.task.keyframes[1:]:
            assert kf.transfer is not None


def test_verdicts():
    assert plan_stand_punch(np.array([-0.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_sector"
    assert plan_stand_punch(np.array([1.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_reach"


def test_coil_sinks_without_moving():
    """kf0: head sinks below standing, feet and hands stay at start."""
    from point5.contracts import ankle_mid_xy, to_local
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    s0 = stand_p5()
    s0_local = to_local(s0, ankle_mid_xy(s0), 0.0)  # F1 = identity
    assert w[0, 0, 2] < s0_local[0, 2] - 0.02       # head sinks
    np.testing.assert_allclose(w[0, 3:], s0_local[3:], atol=1e-9)


def test_swing_lifts_lead_foot():
    """kf1: lead foot is airborne mid-step along the attack axis."""
    far = np.array([1.0, 0.0, 1.15])
    p = plan_stand_punch(far, "right", CFG)
    w = keyframe_world(p.task)
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[1, move] - w[0, move]) @ a
    assert 0.2 * p.diag["step_len"] < adv < 0.8 * p.diag["step_len"]
    assert w[1, move, 2] > w[0, move, 2] + 0.02    # lifted
    np.testing.assert_allclose(w[1, anchor], w[0, anchor], atol=1e-9)


def test_feet_never_move_in_range():
    """In-range target: no stepping, feet planted through all frames."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    assert p.diag["step_len"] == 0.0
    for k in range(1, 4):
        # anchor + lead feet planted (swing lifts by clearance*0 too)
        np.testing.assert_allclose(
            w[k, 3:, :2], w[0, 3:, :2], atol=1e-9)
        np.testing.assert_allclose(
            w[k, 3:, 2], w[0, 3:, 2], atol=1e-9)


def test_far_target_lands_step_and_transfers():
    """kf2: lead foot lands a full parallel step, anchor stays, COM drops."""
    far = np.array([1.0, 0.0, 1.15])
    p = plan_stand_punch(far, "right", CFG)
    assert p.reachable
    w = keyframe_world(p.task)
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[2, move] - w[0, move]) @ a
    assert adv == pytest.approx(p.diag["step_len"], abs=0.01)
    np.testing.assert_allclose(w[2, anchor], w[0, anchor], atol=1e-9)
    assert w[2, 0, 2] < w[0, 0, 2] - 0.02          # deeper sink on land
    assert w[2, 0, 2] < stand_p5()[0, 2] - 0.08    # well below standing
    # COM has translated toward the target
    assert (w[3, 0] - w[0, 0]) @ a > 0.05
    # strike hand reaches target after stepping
    i = POINT_INDEX["hand_right"]
    assert np.linalg.norm(w[3, i] - far) < 0.15


def test_blade_and_strike_facing():
    """Left punch blades right (-30°) then keeps turning to -60°."""
    p = plan_stand_punch(FRONT, "left", CFG)
    assert p.diag["blade_deg"] == pytest.approx(-30.0)
    assert p.diag["strike_deg"] == pytest.approx(-60.0)
    dys = [k.transfer.dyaw for k in p.task.keyframes[1:]]
    assert dys[0] == pytest.approx(-30.0, abs=0.01)   # into blade
    assert dys[1] == pytest.approx(0.0, abs=0.01)     # hold on land
    assert dys[2] == pytest.approx(-30.0, abs=0.01)   # keep turning
    p_r = plan_stand_punch(FRONT, "right", CFG)
    assert p_r.diag["blade_deg"] == pytest.approx(30.0)
    assert p_r.diag["strike_deg"] == pytest.approx(60.0)


def test_guard_hands_at_chest_on_land():
    """kf2: both hands chambered forward of chest, below chin."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    a = p.diag["attack_axis"]
    for i in (POINT_INDEX["hand_right"], POINT_INDEX["hand_left"]):
        fwd = (w[2, i] - w[2, 0]) @ a
        assert fwd > 0                            # ahead of head line
        assert w[2, i, 2] < w[2, 0, 2] - 0.15     # chest, not chin-high


def test_strike_hand_reaches_target_with_penetration():
    p = plan_stand_punch(FRONT, "right", CFG)
    w = keyframe_world(p.task)
    i = POINT_INDEX["hand_right"]
    d = np.linalg.norm(w[3, i] - FRONT)
    assert d == pytest.approx(CFG.penetration, abs=0.03)


def test_initial_state_override():
    init = type("S", (), {"pos": stand_p5()})()
    p = plan_stand_punch(FRONT, "right", CFG, initial=init)
    assert p.reachable
