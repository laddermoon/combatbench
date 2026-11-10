"""Tests for point5.humanoid21.stand_punch (Base+Transfers model).

Choreography: coil(降重心) -> swing(抬脚上步+刃架) -> land(落地+抱架)
-> strike(转腰出拳).  Lead-hand mechanics only.

Frame indexing: raw[0] = Base (standing); raw[i] = Base.pos +
Transfers[i-1].delta_pos — the absolute pose in the base frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from point5.p5 import POINT_INDEX
from point5.humanoid21.stand_punch import (
    StandPunchConfig,
    plan_stand_punch,
    stand_p5,
)

CFG = StandPunchConfig()
FRONT = np.array([1.05, 0.0, 1.15])
# clip frame order: 0=base, 1=coil, 2=swing, 3=land, 4=strike
COIL, SWING, LAND, STRIKE = 1, 2, 3, 4


def _raw(clip):
    """(5,5,3) absolute poses in the base frame for all frames."""
    return np.stack([clip.Base.pos]
                    + [clip.Base.pos + t.delta_pos
                       for t in clip.Transfers])


def test_stand_p5_is_valid_local():
    p = stand_p5()
    assert p.shape == (5, 3)
    assert 1.2 < p[0, 2] < 1.6                     # head height
    assert p[3:, 2].max() < 0.15                   # ankles planted


def test_reachable_frontal_both_hands():
    for hand in ("left", "right"):
        p = plan_stand_punch(FRONT, hand, CFG)
        assert p.reachable, p.reason
        assert len(p.task.p5) == 5               # base + 4 targets
        assert [p.diag[f"label_{i}"] for i in (1, 2, 3, 4)] == \
            ["coil", "swing", "land", "strike"]
        assert len(p.task.transfers) == 4


def test_verdicts():
    assert plan_stand_punch(np.array([-0.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_sector"
    assert plan_stand_punch(np.array([1.5, 0., 1.1]), "right",
                            CFG).reason == "out_of_reach"
    assert plan_stand_punch(np.array([0.8, 0., 1.1]), "right",
                            CFG).reason == "too_close"


def test_coil_sinks_without_moving():
    """coil: head sinks below standing, feet and hands stay at start."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = _raw(p.task)
    s0 = stand_p5()
    assert w[COIL, 0, 2] < s0[0, 2] - 0.02        # head sinks
    np.testing.assert_allclose(w[COIL, 3:], s0[3:], atol=1e-9)


def test_swing_lifts_lead_foot():
    """swing: lead foot is airborne mid-step along the attack axis."""
    far = np.array([1.2, 0.0, 1.15])
    p = plan_stand_punch(far, "right", CFG)
    w = _raw(p.task)
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[SWING, move] - w[0, move]) @ a
    assert 0.2 * p.diag["step_len"] < adv < 0.8 * p.diag["step_len"]
    assert w[SWING, move, 2] > w[0, move, 2] + 0.02   # lifted
    np.testing.assert_allclose(w[SWING, anchor], w[0, anchor], atol=1e-9)


def test_min_step_commits_for_all_targets():
    """Every strike steps at least min_step — near targets included."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = _raw(p.task)
    assert p.diag["step_len"] >= CFG.min_step
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[LAND, move] - w[0, move]) @ a
    assert adv == pytest.approx(p.diag["step_len"], abs=0.01)
    np.testing.assert_allclose(w[LAND, anchor], w[0, anchor], atol=1e-9)


def test_far_target_lands_step_and_transfers():
    """land: lead foot lands a full parallel step, anchor stays,
    COM drops, strike hand reaches target."""
    far = np.array([1.2, 0.0, 1.15])
    p = plan_stand_punch(far, "right", CFG)
    assert p.reachable
    w = _raw(p.task)
    move = POINT_INDEX[p.diag["moving_foot"]]
    anchor = POINT_INDEX[p.diag["anchor_foot"]]
    a = p.diag["attack_axis"]
    adv = (w[LAND, move] - w[0, move]) @ a
    assert adv == pytest.approx(p.diag["step_len"], abs=0.01)
    np.testing.assert_allclose(w[LAND, anchor], w[0, anchor], atol=1e-9)
    assert w[LAND, 0, 2] < w[0, 0, 2] - 0.02      # deeper sink on land
    assert w[LAND, 0, 2] < stand_p5()[0, 2] - 0.08  # well below standing
    assert (w[STRIKE, 0] - w[0, 0]) @ a > 0.05    # COM drives toward target
    i = POINT_INDEX["hand_right"]
    assert np.linalg.norm(w[STRIKE, i] - far) < 0.35


def test_blade_and_strike_facing():
    """Left punch blades right (-30deg) then keeps turning to -60deg."""
    p = plan_stand_punch(FRONT, "left", CFG)
    assert p.diag["blade_deg"] == pytest.approx(-30.0)
    assert p.diag["strike_deg"] == pytest.approx(-60.0)
    # adjacent transitions in radians: coil->swing=-30deg,
    # swing->land=0, land->strike=-30deg (transfers[0]=base->coil≈0)
    dys = [np.degrees(t.delta_yaw) for t in p.task.transfers]
    assert dys[1] == pytest.approx(-30.0, abs=0.1)    # into blade
    assert dys[2] == pytest.approx(0.0, abs=0.1)      # hold on land
    assert dys[3] == pytest.approx(-30.0, abs=0.1)    # keep turning
    p_r = plan_stand_punch(FRONT, "right", CFG)
    assert p_r.diag["blade_deg"] == pytest.approx(30.0)
    assert p_r.diag["strike_deg"] == pytest.approx(60.0)


def test_guard_hands_at_chest_on_land():
    """land: both hands chambered forward of chest, below chin."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = _raw(p.task)
    a = p.diag["attack_axis"]
    for i in (POINT_INDEX["hand_right"], POINT_INDEX["hand_left"]):
        fwd = (w[LAND, i] - w[LAND, 0]) @ a
        assert fwd > 0                            # ahead of head line
        assert w[LAND, i, 2] < w[LAND, 0, 2] - 0.15   # chest, not chin


def test_strike_hand_fully_extends_through_target():
    """The fist travels arm_strike from the chamber — near-straight
    arm at impact with the target on its path."""
    p = plan_stand_punch(FRONT, "right", CFG)
    w = _raw(p.task)
    i = POINT_INDEX["hand_right"]
    ext = np.linalg.norm(w[STRIKE, i] - w[LAND, i])
    assert ext >= CFG.arm_strike - 0.05
    assert np.linalg.norm(w[STRIKE, i] - FRONT) < 0.35


def test_initial_state_override():
    init = type("S", (), {"pos": stand_p5()})()
    p = plan_stand_punch(FRONT, "right", CFG, initial=init)
    assert p.reachable
