"""Smoke tests for the staged pose solver."""
from __future__ import annotations

import numpy as np
import pytest

from mocap.point5.humanoid21.fk import Humanoid21FK
from mocap.point5.humanoid21.solver import PoseSolver
from mocap.point5.humanoid21.spring import load_spring_model


@pytest.fixture(scope="module")
def solver():
    fk = Humanoid21FK()
    return PoseSolver(fk=fk, spring=load_spring_model(fk))


def _target_of(solver, q=None, root_pos=(0, 0, 1.216)):
    fk = solver.fk
    q = solver.q0 if q is None else q
    fk.set_pose("robot_a", root_pos, (1, 0, 0, 0), q)
    return fk.landmark_positions("robot_a")


class TestSpringModel:
    def test_energy_zero_at_natural(self, solver):
        assert solver.spring.energy(solver.q0) == pytest.approx(0.0)

    def test_energy_positive_off_natural(self, solver):
        q = solver.q0.copy(); q[0] += 0.5
        assert solver.spring.energy(q) > 0.0

    def test_ankles_are_free_springs(self, solver):
        # ankles must NOT be fixed: locking them makes the leg a rigid
        # lever -> kneeling artifacts. They participate as springs.
        sm = solver.spring
        for n in ("ankle_y_right", "ankle_x_right",
                  "ankle_y_left", "ankle_x_left"):
            i = sm.joint_names.index(n)
            assert not sm.fixed_mask[i]
            assert sm.k[i] > 0.0


class TestSolve:
    def test_natural_pose_target_recovers(self, solver):
        target = _target_of(solver)
        res = solver.solve(target)
        assert res.satisfied
        assert res.point_errors.max() < solver.tolerance
        np.testing.assert_allclose(res.joint_pos, solver.q0, atol=0.15)

    def test_shifted_target_moves_pose(self, solver):
        target = _target_of(solver)
        target[1, 0] += 0.25            # hand_right 25cm forward
        res = solver.solve(target)
        assert res.point_errors[1] < 0.1
        assert res.landmark_pos[1, 0] > target[1, 0] - 0.05

    def test_impossible_target_reports_failure(self, solver):
        target = _target_of(solver)
        target[0, 2] += 2.0             # head 2m up: unreachable
        res = solver.solve(target)
        assert not res.satisfied
        assert res.failed_stage in ("head", "hands")

    def test_deterministic(self, solver):
        target = _target_of(solver)
        a = solver.solve(target)
        b = solver.solve(target)
        np.testing.assert_allclose(a.joint_pos, b.joint_pos, atol=1e-6)
