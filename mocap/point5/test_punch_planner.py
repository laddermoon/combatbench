"""Tests for mocap.point5.punch_planner (v1 geometric strike planner)."""
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
    hr=[0.15, -0.10, 1.05],       # guard hand (right) slightly back
    hl=[0.25, 0.10, 1.10],        # lead hand (left) forward
    fr=[-0.05, -0.15, 0.02],      # rear foot (right)
    fl=[0.20, 0.15, 0.01],        # lead foot (left) ahead
)

CFG = PunchPlannerConfig()


def _targets(kf):
    return {n: s.target.pos for n, s in kf.specs.items()}


def test_in_range_no_step():
    tgt = np.array([0.55, 0.05, 1.15])     # within 0.55 arm reach
    plan = plan_punch(GUARD_L, tgt, "left", CFG)
    assert plan.reachable and plan.reason == ""
    assert plan.diag["step_len"] == 0.0
    strike = _targets(plan.task.keyframes[-1])
    np.testing.assert_allclose(
        np.linalg.norm(strike["hand_left"] - tgt),
        CFG.penetration, atol=1e-6)
    # both feet stay exactly where they started
    np.testing.assert_allclose(strike["foot_left"], GUARD_L.pos[POINT_INDEX["foot_left"]])
    np.testing.assert_allclose(strike["foot_right"], GUARD_L.pos[POINT_INDEX["foot_right"]])


def test_far_target_steps_lead_foot():
    tgt = np.array([1.00, 0.0, 1.10])
    gap = np.linalg.norm(tgt - GUARD_L.pos[POINT_INDEX["hand_left"]]) - CFG.arm_reach
    plan = plan_punch(GUARD_L, tgt, "left", CFG)
    assert plan.reachable
    assert plan.diag["step_len"] == pytest.approx(gap)
    assert plan.diag["moving_foot"] == "foot_left"      # lead foot along +X
    strike = _targets(plan.task.keyframes[-1])
    fl = GUARD_L.pos[POINT_INDEX["foot_left"]]
    np.testing.assert_allclose(
        strike["foot_left"], [fl[0] + gap, fl[1], 0.0], atol=1e-9)
    np.testing.assert_allclose(
        strike["foot_right"], GUARD_L.pos[POINT_INDEX["foot_right"]])


def test_out_of_reach_unreachable():
    plan = plan_punch(GUARD_L, np.array([2.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_reach"
    assert plan.task is None


def test_out_of_sector_unreachable():
    plan = plan_punch(GUARD_L, np.array([-1.0, 0.0, 1.2]), "left", CFG)
    assert not plan.reachable and plan.reason == "out_of_sector"


def test_lateral_target_steps_side_foot():
    # Target off to the right: right foot is the lead along the axis.
    tgt = np.array([0.30, -0.80, 1.10])
    plan = plan_punch(GUARD_L, tgt, "right", CFG)
    assert plan.reachable
    assert plan.diag["moving_foot"] == "foot_right"
    strike = _targets(plan.task.keyframes[-1])
    dxy = (strike["foot_right"] - GUARD_L.pos[POINT_INDEX["foot_right"]])[:2]
    assert np.linalg.norm(dxy) == pytest.approx(plan.diag["step_len"])
    np.testing.assert_allclose(
        strike["foot_left"], GUARD_L.pos[POINT_INDEX["foot_left"]])


def test_low_target_sinks_head():
    hi = plan_punch(GUARD_L, np.array([0.55, 0.0, 1.15]), "left", CFG)
    lo = plan_punch(GUARD_L, np.array([0.55, 0.0, 0.60]), "left", CFG)
    hz_hi = _targets(hi.task.keyframes[-1])["head"][2]
    hz_lo = _targets(lo.task.keyframes[-1])["head"][2]
    assert hz_lo < hz_hi                       # deeper target -> more sink
    assert hz_lo > GUARD_L.pos[POINT_INDEX["head"]][2] - CFG.head_sink_max - 1e-6


def test_anchor_foot_identical_everywhere():
    plan = plan_punch(GUARD_L, np.array([1.0, 0.0, 1.1]), "left", CFG)
    anchor = plan.diag["anchor_foot"]
    base = GUARD_L.pos[POINT_INDEX[anchor]]
    for kf in plan.task.keyframes:
        np.testing.assert_allclose(kf.specs[anchor].target.pos, base)


def test_prepare_chambers_and_keyframes_cover_five():
    plan = plan_punch(GUARD_L, np.array([0.9, 0.0, 1.1]), "left", CFG)
    assert len(plan.task.keyframes) == 3
    assert len(plan.task.durations) == 2
    prep = _targets(plan.task.keyframes[1])
    hp0 = GUARD_L.pos[POINT_INDEX["hand_left"]]
    assert prep["hand_left"][0] < hp0[0]       # chamber pulled back
    assert prep["head"][2] < GUARD_L.pos[POINT_INDEX["head"]][2]  # sink


def test_hand_switch_flips_roles():
    plan = plan_punch(GUARD_L, np.array([0.55, -0.05, 1.10]), "right", CFG)
    assert plan.reachable
    strike = _targets(plan.task.keyframes[-1])
    np.testing.assert_allclose(
        np.linalg.norm(strike["hand_right"] - np.array([0.55, -0.05, 1.10])),
        CFG.penetration, atol=1e-6)
    np.testing.assert_allclose(
        strike["hand_left"], GUARD_L.pos[POINT_INDEX["hand_left"]])


def test_all_anchors_are_world_points():
    plan = plan_punch(GUARD_L, np.array([0.8, 0.0, 1.1]), "left", CFG)
    for kf in plan.task.keyframes:
        for spec in kf.specs.values():
            assert isinstance(spec.target, WorldPoint)
