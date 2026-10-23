"""Tests for the second-stage continuous main-path representation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from .constraints import default_main_path_setup
from .contracts import (
    MainPathRequest,
    MainPathState,
    SupportSchedule,
)
from .support_sequence import SupportSequencePlanner
from .trajectory import (
    FootPointTrack,
    MainPath,
    MainPathTrajectoryPlanner,
    MainPathTrajectoryRequest,
    PathPiece,
)
from .validation import MainPathValidator
from .visualize import animate_main_path


def _setup():
    model, gait, constraints = default_main_path_setup()
    half_width = gait.stance_width * 0.5
    weight = gait.mass * gait.gravity
    state = MainPathState(
        time=0.0,
        com_pos=np.array([0.0, 0.0, gait.nominal_com_height]),
        lin_mom=np.zeros(3),
        ang_mom=np.zeros(3),
        heading_yaw=0.0,
        left_point_world=np.array([0.0, half_width, gait.foot_height]),
        right_point_world=np.array([0.0, -half_width, gait.foot_height]),
        left_force_world=np.array([0.0, 0.0, weight * 0.5]),
        right_force_world=np.array([0.0, 0.0, weight * 0.5]),
    )
    support_planner = SupportSequencePlanner(gait_model=gait)
    trajectory_planner = MainPathTrajectoryPlanner(
        gait_model=gait, model=model, constraints=constraints)
    return model, gait, constraints, state, support_planner, trajectory_planner


def _plan(target):
    _, _, _, state, support_planner, trajectory_planner = _setup()
    request = MainPathRequest(state, np.asarray(target, dtype=np.float64))
    schedule = support_planner.plan(request)
    return trajectory_planner.plan(
        MainPathTrajectoryRequest.from_support_result(request, schedule))


def test_state_at_returns_complete_continuous_state():
    result = _plan([0.0, 0.0, 0.0])
    assert result.success, result.report.issues
    path = result.path

    for time in (0.0, 0.17, 0.631, path.end_time):
        state = path.state_at(time)
        assert state.time == pytest.approx(time)
        assert state.com_pos.shape == (3,)
        assert state.lin_mom.shape == (3,)
        assert state.ang_mom.shape == (3,)
        assert state.heading_yaw == pytest.approx(0.0)
        assert state.left_support_active
        assert state.right_support_active
        assert state.left_force_world[2] > 0.0
        assert state.right_force_world[2] > 0.0

    states = path.states_at([0.11, 0.22])
    assert len(states) == 2
    assert states[0].time == pytest.approx(0.11)
    assert states[1].time == pytest.approx(0.22)

    with pytest.raises(ValueError, match="outside MainPath domain"):
        path.state_at(path.end_time + 0.01)


def test_static_hold_satisfies_arbitrary_interval_dynamics():
    result = _plan([0.0, 0.0, 0.0])
    assert result.success, result.report.issues
    validator = MainPathValidator(_setup()[2])

    report = validator.check_interval(result.path, 0.37, 1.23)
    assert report.passed, report.issues
    assert report.max_com_residual < 2e-3
    assert report.max_lin_mom_residual < 8e-2
    assert report.max_ang_mom_residual < 8e-2


def test_ballistic_piece_integrates_free_fall():
    model, gait, constraints, _, _, _ = _setup()
    state = MainPathState(
        time=0.0,
        com_pos=np.array([0.0, 0.0, gait.nominal_com_height]),
        lin_mom=np.zeros(3),
        ang_mom=np.zeros(3),
        heading_yaw=0.0,
        left_point_world=np.array([0.0, 0.09, gait.foot_height]),
        right_point_world=np.array([0.0, -0.09, gait.foot_height]),
        left_force_world=np.zeros(3),
        right_force_world=np.zeros(3),
        left_support_active=False,
        right_support_active=False,
    )
    schedule = SupportSchedule(intervals=(), horizon=0.12)
    tracks = {
        foot: FootPointTrack(
            foot=foot,
            initial_point=getattr(state, f"{foot}_point_world"),
            intervals=(),
            swing_segments=(),
        )
        for foot in ("left", "right")
    }
    piece = PathPiece(
        start_time=0.0,
        end_time=0.12,
        start_com=state.com_pos,
        start_lin_mom=state.lin_mom,
        start_ang_mom=state.ang_mom,
        left_active=False,
        right_active=False,
        left_point_start=np.zeros(3),
        left_point_end=np.zeros(3),
        right_point_start=np.zeros(3),
        right_point_end=np.zeros(3),
        left_force_start=np.zeros(3),
        left_force_end=np.zeros(3),
        right_force_start=np.zeros(3),
        right_force_end=np.zeros(3),
    )
    path = MainPath(
        initial_state=state,
        pieces=(piece,),
        foot_tracks=tracks,
        model=model,
        support_schedule=schedule,
    )

    final = path.state_at(0.12)
    assert final.left_support_active is False
    assert final.right_support_active is False
    assert np.linalg.norm(final.left_force_world) == pytest.approx(0.0)
    assert final.lin_mom[2] == pytest.approx(
        -gait.mass * gait.gravity * 0.12)
    assert final.com_pos[2] == pytest.approx(
        gait.nominal_com_height - 0.5 * gait.gravity * 0.12**2)

    report = MainPathValidator(constraints).check_interval(
        path, 0.033, 0.117)
    assert report.passed, report.issues


def test_short_forward_step_generates_valid_terminal_hold():
    result = _plan([0.4, 0.0, 0.0])
    assert result.success, [issue.code for issue in result.report.issues]
    path = result.path

    single = path.state_at(0.3)
    assert single.left_support_active
    assert not single.right_support_active
    assert np.linalg.norm(single.right_force_world) == pytest.approx(0.0)

    final = path.state_at(path.end_time)
    assert final.left_support_active
    assert final.right_support_active
    assert np.linalg.norm(final.com_pos[:2] - np.array([0.4, 0.0])) < 0.05
    assert np.linalg.norm(final.lin_mom) < 1.0
    assert np.linalg.norm(final.ang_mom) < 1.0

    validator = MainPathValidator(_setup()[2])
    report = validator.check_interval(path, 0.43, 1.73)
    assert report.passed, report.issues


def test_diagonal_step_generates_valid_terminal_hold():
    result = _plan([0.9, 0.3, 0.0])
    assert result.success, [issue.code for issue in result.report.issues]
    final = result.path.state_at(result.path.end_time)
    assert np.linalg.norm(final.com_pos[:2] - np.array([0.9, 0.3])) < 0.05
    assert np.linalg.norm(final.lin_mom) < 1.0
    assert np.linalg.norm(final.ang_mom) < 1.0


def test_validator_finds_interior_capability_violation():
    result = _plan([0.0, 0.0, 0.0])
    assert result.success, result.report.issues
    path = result.path
    index = len(path.pieces) // 2
    bad = replace(
        path.pieces[index],
        left_force_end=np.array([500.0, 0.0, 200.0]),
    )
    pieces = list(path.pieces)
    pieces[index] = bad
    bad_path = MainPath(
        initial_state=path.initial_state,
        pieces=tuple(pieces),
        foot_tracks=path.foot_tracks,
        model=path.model,
        support_schedule=path.support_schedule,
    )

    report = MainPathValidator(_setup()[2]).validate(bad_path)
    assert not report.passed
    assert {issue.code for issue in report.issues} & {
        "FRICTION_LIMIT", "FORCE_RATE_LIMIT", "FORCE_DISCONTINUITY"}
    assert any(issue.measured is not None for issue in report.issues)


def test_support_input_conflict_returns_structured_diagnostics():
    _, gait, _, state, support_planner, trajectory_planner = _setup()
    schedule = support_planner.plan(MainPathRequest(
        state, np.array([0.0, 0.0, 0.0])))
    incompatible = MainPathState(
        time=state.time,
        com_pos=state.com_pos,
        lin_mom=state.lin_mom,
        ang_mom=state.ang_mom,
        heading_yaw=state.heading_yaw,
        left_point_world=state.left_point_world,
        right_point_world=state.right_point_world,
        left_force_world=np.zeros(3),
        right_force_world=state.right_force_world,
        left_support_active=False,
        right_support_active=True,
    )
    result = trajectory_planner.plan(MainPathTrajectoryRequest(
        current_state=incompatible,
        support_schedule=schedule,
        target_pos_world=np.zeros(3),
    ))

    assert not result.success
    assert result.path is None
    assert result.report.issues
    assert result.report.issues[0].code == "INPUT_SUPPORT_CONFLICT"
    assert result.diagnostics["failure_stage"] == "precheck"
    assert gait.mass > 0.0


def test_main_path_visualization_writes_mp4(tmp_path):
    result = _plan([0.4, 0.0, 0.0])
    assert result.success, result.report.issues
    output = animate_main_path(
        result.path,
        np.array([0.4, 0.0, 0.0]),
        tmp_path / "main_path.mp4",
        title="test main path",
        fps=3,
    )
    assert output.exists()
    assert output.stat().st_size > 10_000
