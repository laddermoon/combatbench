"""ObjectiveEvaluator unit tests."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as _R

from mocap.humanoid21.clip.core_clip import CoreClip
from mocap.humanoid21.clip.objective_evaluator import (
    ObjectiveEvaluator, ReclipObjectiveConfig)
from mocap.humanoid21.clip.target_path import TargetPath


def _state(root_pos=(0, 0, 1), yaw=0.0, joints=None,
           joint_vel=None, root_vel=None):
    q_xyzw = _R.from_euler("z", yaw).as_quat()
    return {
        "root_pos": np.array(root_pos, np.float32),
        "root_rot": np.roll(q_xyzw, 1).astype(np.float32),
        "root_vel_local": (np.zeros(3, np.float32) if root_vel is None
                           else np.asarray(root_vel, np.float32)),
        "root_angular_vel_local": np.zeros(3, np.float32),
        "joint_pos_norm": (np.zeros(21, np.float32) if joints is None
                           else np.asarray(joints, np.float32)),
        "joint_vel_norm": (np.zeros(21, np.float32) if joint_vel is None
                           else np.asarray(joint_vel, np.float32)),
    }


def _path(s_ext=0.1, root_pos_mode=None):
    clip = CoreClip.from_states(
        [_state((0, 0, 1), 0.0, np.zeros(21)),
         _state((1, 0, 1), 0.4, np.ones(21))],
        {"clip_id": "src"})
    return TargetPath.from_clip(
        clip, "endpoints", s_ext=s_ext,
        root_pos_mode=root_pos_mode)


def _with_pose(sample, **kwargs):
    st = _state(**kwargs)
    st["root_pos"] = sample.root_pos.copy()
    st["root_rot"] = sample.root_rot.copy()
    st["joint_pos_norm"] = sample.joint_pos_norm.copy()
    return st


def test_track_scores_pose_not_intermediate_velocity():
    path = _path()
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="track", time_budget_frames=1))
    slow = [_with_pose(path.sample(0.0), joint_vel=np.zeros(21)),
            _with_pose(path.sample(1.0), joint_vel=np.zeros(21))]
    fast = [dict(slow[0]), dict(slow[1])]
    fast[0]["joint_vel_norm"] = np.ones(21) * 20.0
    fast[0]["root_vel_local"] = np.ones(3) * 10.0
    actions = np.zeros((1, 21))
    r1 = ev.evaluate_rollout(slow, actions)
    r2 = ev.evaluate_rollout(fast, actions)
    assert r1.status == r2.status == "completed"
    assert r1.score == pytest.approx(r2.score)
    np.testing.assert_allclose(r1.progress, [0.0, 1.0])


def test_pass_detects_first_control_frame_at_s_end():
    path = _path()
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="pass", time_budget_frames=4, ds_max=0.4))
    states = [_with_pose(path.sample(0.0)),
              _with_pose(path.sample(0.40)),
              _with_pose(path.sample(0.75)),
              _with_pose(path.sample(1.00)),
              _with_pose(path.sample(1.05), root_pos=(1.05, 0, 0.1))]
    r = ev.evaluate_rollout(states, np.zeros((4, 21)))
    assert r.status == "completed"
    assert r.event_frame == 3
    assert r.metrics["event_time"] == pytest.approx(0.15)
    # A post-event fall does not change the pass result.
    assert r.metrics["fall_frame"] is None


def test_pass_budget_exhausted_reports_progress():
    path = _path()
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="pass", time_budget_frames=4, ds_max=0.5))
    states = [_with_pose(path.sample(0.0)),
              _with_pose(path.sample(0.5))]
    r = ev.evaluate_rollout(states, np.zeros((1, 21)))
    assert r.status == "budget_exhausted"
    assert r.event_frame is None
    assert r.metrics["s_final"] == pytest.approx(0.5)


def test_reach_stop_reports_zero_velocity_error():
    path = _path()
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="reach-stop", time_budget_frames=2, ds_max=1.0,
        terminal_vel_weight=0.1))
    states = [_with_pose(path.sample(0.0)),
              _with_pose(path.sample(1.0),
                         joint_vel=np.ones(21),
                         root_vel=np.ones(3))]
    r = ev.evaluate_rollout(states, np.zeros((1, 21)))
    assert r.status == "completed"
    assert r.event_frame == 1
    assert r.metrics["terminal_vel_error"] > 0.0
    assert r.metrics["terminal_pose_error"] == pytest.approx(0.0)


def test_free_root_position_does_not_change_pass_event():
    path = _path(root_pos_mode="free")
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="pass", time_budget_frames=2, ds_max=1.0))
    states = [_with_pose(path.sample(0.0)),
              _with_pose(path.sample(1.0))]
    for i, st in enumerate(states):
        st["root_pos"] += np.array(
            [100.0 * (i + 1), -20.0, 5.0], dtype=np.float32)
    r = ev.evaluate_rollout(states, np.zeros((1, 21)))
    assert r.status == "completed"
    assert r.event_frame == 1
    assert r.metrics["s_final"] == pytest.approx(1.0)
    assert r.metrics["terminal_pose_error"] == pytest.approx(0.0)
    assert (r.terminal_errors["pose"]["components"]["rp"]
            > 1.0)


def test_free_root_position_keeps_reach_stop_body_target():
    path = _path(root_pos_mode="free")
    ev = ObjectiveEvaluator(path, ReclipObjectiveConfig(
        kind="reach-stop", time_budget_frames=2, ds_max=1.0,
        terminal_vel_weight=0.1))
    states = [_with_pose(path.sample(0.0)),
              _with_pose(path.sample(1.0),
                         joint_vel=np.ones(21),
                         root_vel=np.ones(3))]
    states[1]["root_pos"] += np.array([10.0, 5.0, -1.0],
                                     dtype=np.float32)
    r = ev.evaluate_rollout(states, np.zeros((1, 21)))
    assert r.status == "completed"
    assert r.metrics["terminal_pose_error"] == pytest.approx(0.0)
    assert r.metrics["terminal_vel_error"] > 0.0


def test_track_rejects_free_root_position():
    with pytest.raises(ValueError, match="absolute root position"):
        ObjectiveEvaluator(_path(root_pos_mode="free"),
                           ReclipObjectiveConfig(
                               kind="track", time_budget_frames=1))


def test_reach_state_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError):
        ObjectiveEvaluator(_path(), ReclipObjectiveConfig(
            kind="reach-state", time_budget_frames=1))
