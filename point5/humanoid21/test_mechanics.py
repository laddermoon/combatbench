"""Tests for the pure mechanical model (mechanics.py)."""
from __future__ import annotations

import numpy as np
import pytest

from point5.humanoid21.mechanics import (
    JOINT_NAMES, LANDMARK_ORDER, Skeleton,
    norm_to_rad, rad_to_norm, spring_energy, solve_p5_pose)

_ANKLE_CENTER = ("ankle_y_right", "ankle_y_left")


@pytest.fixture(scope="module")
def sk():
    return Skeleton()


def _planted_zero_target(sk):
    """五点目标：无载姿态，脚槽放踝心（ankle_y 锚点），P5 语义。"""
    q0 = np.zeros(21)
    lo, hi = 0.2, 1.2
    for _ in range(25):
        m = (lo + hi) / 2
        a, _ = sk.anchors_jac([0, 0, m], [1, 0, 0, 0], q0, _ANKLE_CENTER)
        if a[:, 2].min() < 0.087:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    lm = sk.landmarks([0, 0, m], [1, 0, 0, 0], q0)
    a, _ = sk.anchors_jac([0, 0, m], [1, 0, 0, 0], q0, _ANKLE_CENTER)
    lm[3], lm[4] = a
    return lm


class TestStructure:
    def test_counts(self, sk):
        assert len(JOINT_NAMES) == 21
        assert len(LANDMARK_ORDER) == 5
        assert len(sk.order) == 16

    def test_norm_roundtrip(self):
        q = np.linspace(-1, 1, 21)
        np.testing.assert_allclose(rad_to_norm(norm_to_rad(q)), q)

    def test_zero_pose_is_unloaded(self, sk):
        assert spring_energy(np.zeros(21)) == pytest.approx(0.0)


class TestSprings:
    def test_laws(self):
        # spring table sanity: stiff linear core, quadratic absorbers
        from point5.humanoid21.mechanics import SPRINGS
        assert SPRINGS["abdomen_y"][0] == "linear"
        assert SPRINGS["knee_right"][0] == "quadratic"
        assert SPRINGS["elbow_right"][0] == "linear"

    def test_quadratic_soft_midrange(self):
        i = JOINT_NAMES.index("knee_right")
        q1 = np.zeros(21); q1[i] = 0.1
        q2 = np.zeros(21); q2[i] = 0.9
        assert spring_energy(q1) < spring_energy(q2) / 100.0


class TestP5Pose:
    def test_zero_target_recovers(self, sk):
        tgt = _planted_zero_target(sk)
        res = solve_p5_pose(tgt, skeleton=sk)
        assert res["satisfied"]
        assert res["point_errors"].max() < 0.04

    def test_deterministic(self, sk):
        tgt = _planted_zero_target(sk)
        a = solve_p5_pose(tgt, skeleton=sk)
        b = solve_p5_pose(tgt, skeleton=sk)
        np.testing.assert_allclose(a["q_norm"], b["q_norm"], atol=1e-6)

    def test_unreachable_reports(self, sk):
        tgt = _planted_zero_target(sk)
        tgt[0, 2] += 3.0                       # head 3m up: impossible
        res = solve_p5_pose(tgt, skeleton=sk)
        assert not res["satisfied"]
        assert not res["stage1"]["converged"]

    def test_crouch_legs_absorb(self, sk):
        """Lowering head: legs (hip/knee) should absorb, torso ~stays."""
        tgt = _planted_zero_target(sk)
        tgt[0, 2] -= 0.4
        res = solve_p5_pose(tgt, skeleton=sk)
        assert res["satisfied"]
        qn = res["q_norm"]
        i_abd = JOINT_NAMES.index("abdomen_y")
        i_knee = JOINT_NAMES.index("knee_right")
        i_hip = JOINT_NAMES.index("hip_y_right")
        # knee/hip flex substantially; abdomen stays near 0
        assert abs(qn[i_abd]) < 0.15
        assert abs(qn[i_knee]) > 0.3 or abs(qn[i_hip]) > 0.3
