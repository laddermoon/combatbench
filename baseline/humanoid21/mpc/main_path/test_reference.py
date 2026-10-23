"""Tests for the third-stage parameterized motion reference."""
from __future__ import annotations

import numpy as np
import pytest

from .constraints import default_main_path_setup
from .contracts import MainPathRequest, MainPathState
from .reference import (
    MotionReference,
    MotionReferenceParams,
    SKELETON_BONES,
)
from .support_sequence import SupportSequencePlanner
from .trajectory import (
    MainPathTrajectoryPlanner,
    MainPathTrajectoryRequest,
)


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
    return gait, state, support_planner, trajectory_planner


def _plan(target):
    gait, state, support_planner, trajectory_planner = _setup()
    request = MainPathRequest(state, np.asarray(target, dtype=np.float64))
    schedule = support_planner.plan(request)
    result = trajectory_planner.plan(
        MainPathTrajectoryRequest.from_support_result(request, schedule))
    assert result.success, result.report.issues
    return gait, schedule, result


def _reference(target=(0.4, 0.0, 0.0), template="walk"):
    gait, schedule, result = _plan(list(target))
    params = MotionReferenceParams.for_template(template, gait)
    reference = MotionReference(result.path, params=params, gait=gait)
    return gait, schedule, result.path, reference


def _swing_mid_time(schedule, foot):
    own = [x for x in schedule.intervals if x.foot == foot]
    for prev, nxt in zip(own, own[1:]):
        if prev.end_time is not None:
            return 0.5 * (prev.end_time + nxt.start_time)
    return None


def test_motion_reference_state_at_returns_full_key_shape():
    _, _, path, reference = _reference()
    for time in (0.0, 0.37, 1.23, path.end_time):
        ref = reference.state_at(time)
        assert ref.time == pytest.approx(time)
        for name in (
            "pelvis_pos", "left_foot_pos", "right_foot_pos",
            "left_hand_hint", "right_hand_hint",
        ):
            assert getattr(ref, name).shape == (3,)
            assert np.all(np.isfinite(getattr(ref, name)))
        assert ref.left_elbow_angle == pytest.approx(
            reference.params.elbow_angle)
        assert 0.0 <= ref.left_arm_lift <= reference.params.arm_lift_max
        assert 0.0 <= ref.right_arm_lift <= reference.params.arm_lift_max


def test_contralateral_arm_swing_follows_stride():
    _, schedule, _, reference = _reference()
    mid = _swing_mid_time(schedule, "right")
    assert mid is not None
    ref = reference.state_at(mid)
    assert ref.right_foot_pos[0] > ref.left_foot_pos[0]
    # Right foot ahead -> contralateral (left) arm swings forward.
    assert ref.left_arm_swing > 0.0
    assert ref.right_arm_swing == pytest.approx(-ref.left_arm_swing)
    # Pelvis and trunk counter-twist in opposite directions.
    assert ref.pelvis_yaw_offset * ref.trunk_yaw_offset <= 0.0


def test_reference_is_continuous_over_time():
    _, _, _, reference = _reference()
    times = np.linspace(reference.start_time, reference.end_time, 120)
    states = reference.states_at(times)
    for prev, current in zip(states, states[1:]):
        assert abs(current.left_arm_swing - prev.left_arm_swing) < 0.15
        assert np.linalg.norm(
            current.pelvis_pos - prev.pelvis_pos) < 0.2
        assert np.linalg.norm(
            current.left_foot_pos - prev.left_foot_pos) < 0.2


def test_hold_case_keeps_neutral_reference():
    _, _, _, reference = _reference(target=(0.0, 0.0, 0.0))
    ref = reference.state_at(1.0)
    assert abs(ref.left_arm_swing) < 1e-6
    assert abs(ref.pelvis_yaw_offset) < 1e-6
    assert ref.left_support_active and ref.right_support_active


def test_boxing_guard_template_uses_guarded_arms():
    gait, _, _, reference = _reference(template="boxing_guard")
    ref = reference.state_at(0.0)
    assert ref.left_elbow_angle == pytest.approx(1.70)
    assert ref.left_arm_lift >= 0.55 - 1e-9
    params = reference.params
    assert params.template == "boxing_guard"


def test_sample_pose_returns_complete_consistent_skeleton():
    gait, schedule, _, reference = _reference()
    mid = _swing_mid_time(schedule, "right")
    pose = reference.sample_pose(mid)
    expected = {a for pair in SKELETON_BONES for a in pair}
    assert expected <= set(pose.keypoints)
    for point in pose.keypoints.values():
        assert np.all(np.isfinite(point))
    # Supporting foot ankle stays near ground height.
    ref = reference.state_at(mid)
    assert not ref.left_support_active or not ref.right_support_active
    left_ankle = pose.keypoints["ankle_left"]
    assert left_ankle[2] == pytest.approx(
        ref.left_foot_pos[2] + reference.params.ankle_height)
    # Two-link leg: hip-knee distance matches the upper-leg ratio.
    upper = reference.params.upper_leg_ratio * gait.leg_length
    for foot in ("left", "right"):
        hip = pose.keypoints[f"hip_{foot}"]
        knee = pose.keypoints[f"knee_{foot}"]
        assert np.linalg.norm(knee - hip) == pytest.approx(upper, abs=1e-9)


def test_sample_poses_and_motion_reference_video(tmp_path):
    from .visualize import animate_motion_reference

    _, _, _, reference = _reference()
    poses = reference.sample_poses([0.0, 0.5, 1.0])
    assert len(poses) == 3
    output = animate_motion_reference(
        reference,
        np.array([0.4, 0.0, 0.0]),
        tmp_path / "reference.mp4",
        title="test reference",
        fps=3,
    )
    assert output.exists()
    assert output.stat().st_size > 10_000


def test_sim_playback_solves_poses_and_renders(tmp_path):
    from envs.humanoid21.simulator import Humanoid21Simulator
    from .sim_playback import ReferenceIKSolver, render_reference_in_sim

    gait, schedule, _, reference = _reference(target=(0.0, 0.0, 0.0))
    sim = Humanoid21Simulator(
        initial_pose_a="standing", initial_pose_b="standing")
    sim.reset(seed=0)
    solver = ReferenceIKSolver(sim)
    cache = sim._robot("robot_a")
    stance_center = 0.5 * (
        sim.data.xipos[cache["keypoint_body_ids"]["foot_left"]][:2]
        + sim.data.xipos[cache["keypoint_body_ids"]["foot_right"]][:2])
    initial = reference.state_at(reference.start_time)
    ref_center = 0.5 * (
        initial.left_foot_pos[:2] + initial.right_foot_pos[:2])
    offset = np.array(
        [stance_center[0] - ref_center[0],
         stance_center[1] - ref_center[1], 0.0])
    q, err = solver.solve(
        reference.sample_pose(0.0),
        reference.state_at(0.0),
        world_offset=offset,
    )
    assert q.shape == sim.data.qpos.shape
    assert err < 0.03
    sim.close()

    output = render_reference_in_sim(
        reference, tmp_path / "reference_sim.mp4", fps=2)
    assert output.exists()
    assert output.stat().st_size > 10_000
