"""Tests for point5.humanoid21.punch_planner (v2 two-phase planner)."""
from __future__ import annotations

import numpy as np
import pytest

from point5.contracts import POINT_INDEX, Point5State, WorldPoint
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
GUARD_L = _p5(
    head=[0.0, 0.0, 1.20],
    hr=[0.15, -0.10, 1.05],
    hl=[0.25, 0.10, 1.10],
    fr=[-0.05, -0.15, 0.02],
    fl=[0.20, 0.15, 0.01],
)

CFG = PunchPlannerConfig()


def _targets(kf):
    return {n: s.target.pos for n, s in kf.specs.items()}


def test_in_range_gets_power_step():
    """In-range punches still step the minimum drive distance."""
    plan = plan_punch(GUARD_L, np.array([0.55, 0.05, 1.15]), "left", CFG)
    assert plan.reachable
    assert plan.diag["step_len"] == pytest.approx(CFG.power_step_min)
    strike = _targets(plan.task.keyframes[-1])
    np.testing.assert_allclose(
        np.linalg.norm(strike["hand_left"] - np.array([0.55, 0.05, 1.15])),
        CFG.penetration, atol=1e-6)
    # lead foot stepped forward, anchor foot identical
    fl = GUARD_L.pos[POINT_INDEX["foot_left"]]
    np.testing.assert_allclose(
        strike["foot_left"][:2],
        fl[:2] + plan.diag["attack_axis"][:2] * CFG.power_step_min,
        atol=1e-9)
    np.testing.assert_allclose(
        strike["foot_right"], GUARD_L.pos[POINT_INDEX["foot_right"]])


def test_far_target_steps_beyond_minimum():
    tgt = np.array([1.00, 0.0, 1.10])
    gap = np.linalg.norm(tgt - GUARD_L.pos[POINT_INDEX["hand_left"]]) \
        - CFG.arm_reach
    plan = plan_punch(GUARD_L, tgt, "left", CFG)
    assert plan.reachable
    # the body follows only a share of the step, so the step itself
    # must cover gap / share
    assert plan.diag["step_len"] == pytest.approx(
        gap / CFG.head_step_share)
    assert plan.diag["moving_foot"] == "foot_left"


def test_out_of_reach_unreachable():
    plan = plan_punch(GUARD_L, np.array([2.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_reach"


def test_out_of_sector_unreachable():
    plan = plan_punch(GUARD_L, np.array([-1.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_sector"


def test_lateral_target_steps_side_foot():
    plan = plan_punch(GUARD_L, np.array([0.30, -0.80, 1.10]), "right", CFG)
    assert plan.reachable
    assert plan.diag["moving_foot"] == "foot_right"


def test_low_target_sinks_head():
    hi = plan_punch(GUARD_L, np.array([0.55, 0.0, 1.15]), "left", CFG)
    lo = plan_punch(GUARD_L, np.array([0.55, 0.0, 0.60]), "left", CFG)
    hz_hi = _targets(hi.task.keyframes[-1])["head"][2]
    hz_lo = _targets(lo.task.keyframes[-1])["head"][2]
    assert hz_lo < hz_hi


def test_phase_a_hands_ride_with_body():
    """Phase A (load): foot lands + head shifts; hands translate by the
    SAME body shift — the guard is carried by the torso, so relative
    to the head each hand keeps its offset."""
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    start = _targets(plan.task.keyframes[0])
    load = _targets(plan.task.keyframes[2])
    body_shift = load["head"] - start["head"]
    for hn in ("hand_left", "hand_right"):
        np.testing.assert_allclose(
            load[hn], start[hn] + body_shift, atol=1e-9)
        np.testing.assert_allclose(           # hand keeps offset from head
            load[hn] - load["head"], start[hn] - start["head"], atol=1e-9)
    fl = GUARD_L.pos[POINT_INDEX["foot_left"]]
    assert np.linalg.norm((load["foot_left"] - fl)[:2]) == pytest.approx(
        plan.diag["step_len"])
    assert load["head"][2] < GUARD_L.pos[POINT_INDEX["head"]][2]


def test_phase_b_guard_hand_tucks():
    """Phase B (strike): guard hand moves to the chin, not left hanging."""
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    strike = _targets(plan.task.keyframes[-1])
    hg = strike["hand_right"]
    hg0 = GUARD_L.pos[POINT_INDEX["hand_right"]]
    assert not np.allclose(hg, hg0)
    # tucked: below head, slightly off-centre to its own side
    assert hg[2] < strike["head"][2]
    assert np.linalg.norm(hg - strike["head"]) < 0.25


def test_anchor_foot_identical_everywhere():
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    anchor = plan.diag["anchor_foot"]
    base = GUARD_L.pos[POINT_INDEX[anchor]]
    for kf in plan.task.keyframes:
        np.testing.assert_allclose(kf.specs[anchor].target.pos, base)


def test_rear_hand_gets_reach_bonus():
    """rear_hand = punching hand on the ANCHOR-foot side; it gets the
    rotation reach bonus (frontal target: right hand is rear)."""
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
        # extension is measured from the LOADED guard position
        hp_load = _targets(plan.task.keyframes[2])[hp_name]
        mid = _targets(plan.task.keyframes[3])[hp_name]
        hit = _targets(plan.task.keyframes[4])[hp_name]
        f_mid = np.linalg.norm(mid - hp_load) \
            / np.linalg.norm(hit - hp_load)
        if plan.diag["rear_hand"]:
            assert f_mid < 0.45                 # rear hand stays back
        else:
            assert 0.35 < f_mid < 0.65          # lead ~halfway


def test_durations_encode_slow_load_fast_extend():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    d = plan.task.durations
    assert sum(d[:2]) > d[2]                 # load slower than extend
    assert plan.task.metadata["moving_foot"] == "foot_left"


def test_swing_foot_lifts_then_lands_at_measured_ankle_height():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    foot = plan.diag["moving_foot"]
    start, swing, load = (_targets(kf)[foot]
                          for kf in plan.task.keyframes[:3])
    axis = plan.diag["attack_axis"]
    assert swing[2] == pytest.approx(start[2] + CFG.foot_swing_clearance)
    assert load[2] == pytest.approx(start[2])
    assert 0 < (swing - start) @ axis < (load - start) @ axis


def test_all_anchors_are_world_points():
    plan = plan_punch(GUARD_L, np.array([0.8, 0.0, 1.1]), "left", CFG)
    for kf in plan.task.keyframes:
        for spec in kf.specs.values():
            assert isinstance(spec.target, WorldPoint)
