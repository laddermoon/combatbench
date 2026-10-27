"""Tests for mocap.point5.punch_planner (v2 two-phase planner)."""
from __future__ import annotations

import numpy as np
import pytest

from mocap.point5.contracts import POINT_INDEX, Point5State, WorldPoint
from mocap.point5.punch_planner import (
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
    assert plan.diag["step_len"] == pytest.approx(gap)
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


def test_phase_a_feet_move_hands_frozen():
    """Phase A (load): foot lands + head shifts, BOTH hands identical."""
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    load = _targets(plan.task.keyframes[1])
    np.testing.assert_allclose(
        load["hand_left"], GUARD_L.pos[POINT_INDEX["hand_left"]])
    np.testing.assert_allclose(
        load["hand_right"], GUARD_L.pos[POINT_INDEX["hand_right"]])
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
    """A rear-hand punch reaches a bit further than the same lead jab."""
    near = np.array([0.55, -0.05, 1.10])
    plan_r = plan_punch(GUARD_L, near, "right", CFG)
    assert plan_r.reachable and plan_r.diag["rear_hand"] is True
    # rear-hand target just beyond lead reach still reachable via bonus
    far = np.array([1.05, 0.0, 1.10])
    assert plan_punch(GUARD_L, far, "right", CFG).reachable


def test_durations_encode_slow_load_fast_extend():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    d = plan.task.durations
    assert d[0] > d[1]                       # load slower than extend
    assert plan.task.metadata["moving_foot"] == "foot_left"


def test_all_anchors_are_world_points():
    plan = plan_punch(GUARD_L, np.array([0.8, 0.0, 1.1]), "left", CFG)
    for kf in plan.task.keyframes:
        for spec in kf.specs.values():
            assert isinstance(spec.target, WorldPoint)
