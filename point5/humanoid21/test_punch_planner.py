"""Tests for point5.humanoid21.punch_planner (v2 local-frame planner)."""
from __future__ import annotations

import numpy as np
import pytest

from point5.contracts import POINT_INDEX, Point5State
from point5.trajectory import keyframe_world
from point5.humanoid21.punch_planner import (
    PunchPlannerConfig,
    plan_punch,
)


def _p5(head, hr, hl, fr, fl) -> Point5State:
    pos = np.zeros((5, 3), dtype=np.float64)
    pos[POINT_INDEX["head"]] = head
    pos[POINT_INDEX["hand_right"]] = hr
    pos[POINT_INDEX["hand_left"]] = hl
    pos[POINT_INDEX["foot_right"]] = fr
    pos[POINT_INDEX["foot_left"]] = fl
    return Point5State(pos=pos)


# A left-lead guard facing +X: left foot/hand forward, feet bladed.
# Positions are LOCAL-frame (ankle-mid origin, torso facing = +X) —
# foot XY midpoint must sit at the origin by v2 definition.
GUARD_L = _p5(
    head=[0.0, 0.0, 1.20],
    hr=[0.15, -0.10, 1.05],
    hl=[0.25, 0.10, 1.10],
    fr=[-0.125, -0.15, 0.02],
    fl=[0.125, 0.15, 0.01],
)

CFG = PunchPlannerConfig()


def _worlds(plan) -> np.ndarray:
    """Planned world positions per keyframe, shape (K,5,3)."""
    return keyframe_world(plan.task)


def _t(plan, k: int) -> np.ndarray:
    return _worlds(plan)[k]


def test_in_range_gets_power_step():
    """In-range punches still step the minimum drive distance."""
    plan = plan_punch(GUARD_L, np.array([0.55, 0.05, 1.15]), "left", CFG)
    assert plan.reachable
    assert plan.diag["step_len"] == pytest.approx(CFG.power_step_min)
    strike = _t(plan, -1)
    assert np.linalg.norm(
        strike[POINT_INDEX["hand_left"]] - np.array([0.55, 0.05, 1.15])
        ) > 0.03                              # penetration overshoot


def test_first_keyframe_has_no_transfer_rest_do():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    assert plan.task.keyframes[0].transfer is None
    for kf in plan.task.keyframes[1:]:
        assert kf.transfer is not None and kf.transfer.dur > 0


def test_start_keyframe_recovers_initial_pose():
    """keyframe_world at index 0 must equal the initial local pose —
    the clip's canonical world IS the initial frame."""
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    np.testing.assert_allclose(_t(plan, 0), GUARD_L.pos, atol=1e-9)


def test_step_translates_head_and_feet():
    """Slightly out-of-arm-reach target: the lead foot steps to cover
    the gap (step between power_step_min and max_step)."""
    tgt = np.array([0.95, 0.0, 1.1])
    plan = plan_punch(GUARD_L, tgt, "left", CFG)
    assert plan.reachable
    step = plan.diag["step_len"]
    assert step > CFG.power_step_min
    assert plan.diag["step_len"] == pytest.approx(
        min(plan.diag["reach_gap"] / CFG.head_step_share,
            CFG.max_step))
    assert plan.diag["moving_foot"] == "foot_left"


def test_out_of_reach_reports():
    plan = plan_punch(GUARD_L, np.array([2.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_reach"


def test_out_of_sector_reports():
    plan = plan_punch(GUARD_L, np.array([-1.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_sector"


def test_lead_foot_selection():
    plan = plan_punch(GUARD_L, np.array([0.30, -0.80, 1.10]), "right",
                      CFG)
    assert plan.reachable
    assert plan.diag["moving_foot"] == "foot_right"


def test_deeper_target_sinks_head_more():
    hi = plan_punch(GUARD_L, np.array([0.55, 0.0, 1.15]), "left", CFG)
    lo = plan_punch(GUARD_L, np.array([0.55, 0.0, 0.60]), "left", CFG)
    hz_hi = _t(hi, -1)[POINT_INDEX["head"]][2]
    hz_lo = _t(lo, -1)[POINT_INDEX["head"]][2]
    assert hz_lo < hz_hi


def test_phase_a_hands_ride_with_body():
    """Phase A (load): foot lands + head shifts; hands translate by the
    SAME body shift — the guard is carried by the torso."""
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    start, load = _t(plan, 0), _t(plan, 2)
    body_shift = load[POINT_INDEX["head"]] - start[POINT_INDEX["head"]]
    for hn in ("hand_left", "hand_right"):
        i = POINT_INDEX[hn]
        np.testing.assert_allclose(load[i], start[i] + body_shift,
                                   atol=1e-9)
    fl = GUARD_L.pos[POINT_INDEX["foot_left"]]
    assert np.linalg.norm(
        (load[POINT_INDEX["foot_left"]] - fl)[:2]) == pytest.approx(
        plan.diag["step_len"])
    assert load[POINT_INDEX["head"]][2] < start[POINT_INDEX["head"]][2]


def test_phase_b_guard_hand_tucks():
    """Phase B (strike): guard hand moves to the chin, not left hanging."""
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    strike = _t(plan, -1)
    hg = strike[POINT_INDEX["hand_right"]]
    hg0 = GUARD_L.pos[POINT_INDEX["hand_right"]]
    assert not np.allclose(hg, hg0)
    assert hg[2] < strike[POINT_INDEX["head"]][2]
    assert np.linalg.norm(hg - strike[POINT_INDEX["head"]]) < 0.25


def test_anchor_foot_identical_everywhere():
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    anchor = plan.diag["anchor_foot"]
    base = GUARD_L.pos[POINT_INDEX[anchor]]
    for k in range(len(plan.task.keyframes)):
        np.testing.assert_allclose(_t(plan, k)[POINT_INDEX[anchor]],
                                   base)


def test_rear_hand_gets_reach_bonus():
    """rear_hand = punching hand on the ANCHOR-foot side."""
    tgt = np.array([0.55, -0.05, 1.10])
    plan_r = plan_punch(GUARD_L, tgt, "right", CFG)
    assert plan_r.diag["rear_hand"] is True
    assert plan_r.diag["eff_arm_reach"] == pytest.approx(
        CFG.arm_reach + CFG.cross_reach_bonus)
    plan_l = plan_punch(GUARD_L, np.array([0.55, 0.05, 1.10]), "left",
                        CFG)
    assert plan_l.diag["rear_hand"] is False
    assert plan_l.diag["eff_arm_reach"] == pytest.approx(CFG.arm_reach)


def test_extend_midpoint_progresses_and_rear_stays_back():
    """The 'extend' frame sits between guard and impact along the path;
    rear-hand midpoints stay earlier (elbow leads, fist follows)."""
    tgt = np.array([0.9, 0.0, 1.1])
    for hand in ("left", "right"):
        plan = plan_punch(GUARD_L, tgt, hand, CFG)
        assert len(plan.task.keyframes) == 5
        hp_name = f"hand_{hand}"
        i = POINT_INDEX[hp_name]
        hp_load, mid, hit = _t(plan, 2)[i], _t(plan, 3)[i], _t(plan, 4)[i]
        f_mid = np.linalg.norm(mid - hp_load) \
            / np.linalg.norm(hit - hp_load)
        if plan.diag["rear_hand"]:
            assert f_mid < 0.45
        else:
            assert 0.35 < f_mid < 0.65


def test_durations_encode_slow_load_fast_extend():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    durs = [kf.transfer.dur for kf in plan.task.keyframes[1:]]
    assert sum(durs[:2]) > durs[2]
    assert plan.task.metadata["moving_foot"] == "foot_left"


def test_swing_foot_lifts_then_lands_at_measured_ankle_height():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    foot = plan.diag["moving_foot"]
    i = POINT_INDEX[foot]
    start, swing, load = _t(plan, 0)[i], _t(plan, 1)[i], _t(plan, 2)[i]
    axis = plan.diag["attack_axis"]
    assert swing[2] == pytest.approx(start[2] + CFG.foot_swing_clearance)
    assert load[2] == pytest.approx(start[2])
    assert 0 < (swing - start) @ axis < (load - start) @ axis


def test_strike_twist_goes_toward_punch_side():
    """Transfer dyaw on the strike frame: right hand twists CW
    (negative), left CCW (positive); magnitude = remaining twist share
    after extend (1 - twist_extend_share)."""
    tgt = np.array([0.9, 0.0, 1.1])
    for hand, sgn in (("right", -1.0), ("left", 1.0)):
        plan = plan_punch(GUARD_L, tgt, hand, CFG)
        tr = plan.task.keyframes[4].transfer
        tw = plan.diag["strike_yaw_twist_deg"]   # + = right side
        # world dyaw is opposite-signed to the punch-side convention;
        # extend consumed ~60% of the direction vector, so strike carries
        # ~40% of the magnitude (not exact: vector lerp, not angle lerp)
        assert np.sign(tr.dyaw) == -np.sign(tw)
        assert abs(tr.dyaw) == pytest.approx(
            abs(tw) * (1.0 - CFG.twist_extend_share), rel=0.05)

