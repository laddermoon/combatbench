"""Tests for the first main-path stage: support sequence generation."""
from __future__ import annotations

import numpy as np
import pytest

from .contracts import MainPathRequest, MainPathState, SupportInterval
from .gait_model import default_gait_model
from .support_sequence import SupportSequencePlanner
from .visualize import animate_support_schedule, plot_support_schedule


def _state(heading_yaw: float = 0.0) -> MainPathState:
    heading = np.array([np.cos(heading_yaw), np.sin(heading_yaw)])
    leftward = np.array([-heading[1], heading[0]])
    return MainPathState(
        time=0.0,
        com_pos=np.array([0.0, 0.0, 0.85]),
        lin_mom=np.zeros(3),
        ang_mom=np.zeros(3),
        heading_yaw=heading_yaw,
        left_point_world=np.array(
            [leftward[0] * 0.09, leftward[1] * 0.09, 0.027]),
        right_point_world=np.array(
            [-leftward[0] * 0.09, -leftward[1] * 0.09, 0.027]),
        left_force_world=np.array([0.0, 0.0, 200.0]),
        right_force_world=np.array([0.0, 0.0, 200.0]),
    )


def test_gait_model_extracts_model_derived_numbers():
    model = default_gait_model()
    assert model.mass == pytest.approx(40.844, rel=1e-4)
    assert model.stance_width == pytest.approx(0.18, rel=1e-6)
    assert model.foot_length == pytest.approx(0.265, rel=1e-3)
    assert model.foot_width == pytest.approx(0.094, rel=1e-3)
    assert model.leg_length == pytest.approx(0.83, rel=1e-6)
    assert model.nominal_step_length == pytest.approx(0.3444, rel=1e-3)
    assert model.max_step_length == pytest.approx(0.4769, rel=1e-3)
    assert model.double_support_duration == pytest.approx(0.15)
    assert model.swing_duration == pytest.approx(0.35)
    assert model.settle_duration == pytest.approx(0.60)


def test_state_rejects_force_on_inactive_support():
    state = _state()
    with pytest.raises(ValueError, match="inactive left support"):
        MainPathState(
            time=state.time,
            com_pos=state.com_pos,
            lin_mom=state.lin_mom,
            ang_mom=state.ang_mom,
            heading_yaw=state.heading_yaw,
            left_point_world=state.left_point_world,
            right_point_world=state.right_point_world,
            left_force_world=np.array([0.0, 0.0, 10.0]),
            right_force_world=state.right_force_world,
            left_support_active=False,
            right_support_active=True,
        )


def test_heading_defines_left_right_and_lateral_first_step():
    planner = SupportSequencePlanner()
    schedule = planner.plan(MainPathRequest(
        current_state=_state(heading_yaw=np.pi / 2.0),
        target_pos_world=np.array([0.0, 0.4, 0.0]),
    ))

    final = {
        foot: max(
            (x for x in schedule.intervals if x.foot == foot),
            key=lambda x: x.start_time,
        )
        for foot in ("left", "right")
    }
    assert final["left"].point_world[0] < final["right"].point_world[0]
    assert schedule.metadata["heading_yaw"] == pytest.approx(np.pi / 2.0)
    assert schedule.metadata["target_delta_body"] == pytest.approx([0.4, 0.0])

    lateral_schedule = planner.plan(MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.0, 0.4, 0.0]),
    ))
    assert lateral_schedule.metadata["first_foot"] == "left"
    assert lateral_schedule.metadata["target_delta_body"] == pytest.approx(
        [0.0, 0.4])


def test_near_target_generates_open_hold_intervals():
    planner = SupportSequencePlanner()
    schedule = planner.plan(MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.02, -0.01, 0.0]),
    ))

    assert schedule.metadata["mode"] == "hold"
    assert schedule.metadata["step_counts"] == {"left": 0, "right": 0}
    assert len(schedule.intervals) == 2
    assert all(item.end_time is None for item in schedule.intervals)
    assert schedule.active_feet(0.1) == ("left", "right")


def test_forward_target_generates_alternating_open_schedule():
    planner = SupportSequencePlanner()
    schedule = planner.plan(MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.4, 0.0, 0.0]),
    ))

    assert schedule.metadata["mode"] == "walk"
    assert schedule.metadata["first_foot"] == "right"
    assert schedule.metadata["step_counts"] == {"left": 1, "right": 1}
    assert len(schedule.intervals) == 4
    assert schedule.active_feet(0.3) == ("left",)
    assert schedule.active_feet(0.8) == ("right",)
    assert schedule.active_feet(1.1) == ("left", "right")
    assert all(
        item.end_time is None
        for item in schedule.intervals[-2:]
    )
    assert schedule.intervals[-2].start_time < schedule.horizon


def test_longer_target_scales_step_count_and_times():
    planner = SupportSequencePlanner()
    schedule = planner.plan(MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.9, 0.3, 0.0]),
    ))

    assert schedule.metadata["step_counts"] == {"left": 2, "right": 2}
    assert len(schedule.intervals) == 6
    assert schedule.intervals[-1].point_world[0] > 0.85
    assert schedule.intervals[-1].end_time is None
    assert schedule.intervals[-2].end_time is None
    assert schedule.horizon > schedule.metadata["last_landing_time"]


def test_support_interval_rejects_invalid_schema():
    with pytest.raises(ValueError, match="end_time"):
        SupportInterval(
            foot="left",
            start_time=1.0,
            end_time=0.5,
            point_world=np.zeros(3),
        )
    with pytest.raises(ValueError, match="point_world"):
        SupportInterval(
            foot="left",
            start_time=0.0,
            end_time=None,
            point_world=np.zeros(2),
        )


def test_visualization_writes_png(tmp_path):
    planner = SupportSequencePlanner()
    request = MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.7, 0.2, 0.0]),
    )
    schedule = planner.plan(request)
    out = plot_support_schedule(
        schedule,
        request.current_state,
        request.target_pos_world,
        tmp_path / "support_schedule.png",
        title="test schedule",
    )
    assert out.exists()
    assert out.stat().st_size > 10_000


def test_visualization_writes_generic_mp4(tmp_path):
    planner = SupportSequencePlanner()
    request = MainPathRequest(
        current_state=_state(),
        target_pos_world=np.array([0.4, 0.0, 0.0]),
    )
    schedule = planner.plan(request)
    out = animate_support_schedule(
        schedule,
        request.current_state,
        request.target_pos_world,
        tmp_path / "support_schedule.mp4",
        title="test playback",
        fps=5,
    )
    assert out.exists()
    assert out.stat().st_size > 10_000
