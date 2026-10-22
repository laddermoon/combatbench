"""TargetPath unit tests."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as _R

from mocap.humanoid21.clip.core_clip import CoreClip
from mocap.humanoid21.clip.target_path import (
    POSE_WEIGHTS, TargetPath, pose_error, resolve_pose_weights,
    velocity_error)


def _state(rng, root_pos=(0, 0, 1), yaw=0.0, joints=None):
    q_xyzw = _R.from_euler("z", yaw).as_quat()
    return {
        "root_pos": np.array(root_pos, np.float32),
        "root_rot": np.roll(q_xyzw, 1).astype(np.float32),
        "root_vel_local": rng.uniform(-1, 1, 3).astype(np.float32),
        "root_angular_vel_local": rng.uniform(-1, 1, 3).astype(np.float32),
        "joint_pos_norm": (rng.uniform(-1, 1, 21).astype(np.float32)
                           if joints is None else
                           np.asarray(joints, np.float32)),
        "joint_vel_norm": rng.uniform(-1, 1, 21).astype(np.float32),
    }


def _clip(states):
    return CoreClip.from_states(states, {"clip_id": "path-src"})


def test_endpoints_path_samples_bounds():
    rng = np.random.default_rng(10)
    states = [_state(rng, (0, 0, 1)), _state(rng, (1, 0, 1)),
              _state(rng, (2, 0, 1))]
    clip = _clip(states)
    path = TargetPath.from_clip(clip, "endpoints")

    p0 = path.sample(0.0)
    p1 = path.sample(1.0)
    np.testing.assert_allclose(p0.as_state()["joint_pos_norm"],
                               states[0]["joint_pos_norm"])
    np.testing.assert_allclose(p1.as_state()["joint_pos_norm"],
                               states[-1]["joint_pos_norm"])
    assert pose_error(states[0], p0.as_state())["total"] == pytest.approx(0)
    assert pose_error(states[-1], p1.as_state())["total"] == pytest.approx(0)


def test_all_path_hits_all_pose_waypoints():
    rng = np.random.default_rng(11)
    states = [
        _state(rng, (0, 0, 1), yaw=0.0),
        _state(rng, (0.4, 0.2, 1.1), yaw=0.2),
        _state(rng, (0.6, -0.1, 1.0), yaw=0.5),
        _state(rng, (1.0, 0.0, 1.0), yaw=0.8),
    ]
    clip = _clip(states)
    path = TargetPath.from_clip(clip, "all")

    assert np.all(path.knot_s[:-1] < path.knot_s[1:])
    for i, st in enumerate(states):
        s = path.frame_to_s(i)
        sample = path.sample(s).as_state()
        np.testing.assert_allclose(sample["root_pos"], st["root_pos"],
                                   atol=1e-6)
        np.testing.assert_allclose(sample["joint_pos_norm"],
                                   st["joint_pos_norm"], atol=1e-6)
        assert np.linalg.norm(sample["root_rot"]) == pytest.approx(1.0)


def test_projection_is_local_and_monotonic():
    rng = np.random.default_rng(12)
    states = [_state(rng, (0, 0, 1)),
              _state(rng, (0.5, 0, 1)),
              _state(rng, (1, 0, 1)),
              _state(rng, (1.5, 0, 1))]
    path = TargetPath.from_clip(_clip(states), "all")

    s_hint = 0.0
    for st in states:
        proj = path.project(st, s_hint, ds_max=0.4)
        assert proj.s >= s_hint
        assert proj.errors["total"] == pytest.approx(0.0, abs=1e-5)
        s_hint = proj.s

    # A limited local window cannot jump directly to the final waypoint
    # or to the bounded extension milestone.
    mid = path.project(states[0], path.frame_to_s(1), ds_max=0.05)
    assert mid.s <= path.frame_to_s(1) + 0.051
    end_like = path.project(states[-1], path.frame_to_s(1),
                            ds_max=0.02)
    assert end_like.s <= path.frame_to_s(1) + 0.021


def test_bounded_extension_and_s_range():
    rng = np.random.default_rng(13)
    states = [_state(rng, (0, 0, 1), yaw=0.0),
              _state(rng, (1, 0, 1), yaw=0.4)]
    path = TargetPath.from_clip(_clip(states), "endpoints", s_ext=0.1)
    p = path.sample(1.05)
    assert p.s == pytest.approx(1.05)
    assert np.linalg.norm(p.root_rot) == pytest.approx(1.0)
    assert p.root_pos[0] > 1.0
    with pytest.raises(ValueError, match="outside"):
        path.sample(1.11)


def test_pose_path_ignores_intermediate_velocity():
    rng = np.random.default_rng(14)
    s0 = _state(rng, (0, 0, 1))
    s1 = _state(rng, (1, 0, 1))
    different_velocity = dict(s0)
    different_velocity["joint_vel_norm"] = 100.0 * s0["joint_vel_norm"]
    different_velocity["root_vel_local"] = -10.0 * s0["root_vel_local"]

    assert pose_error(s0, different_velocity) == pose_error(s0, s0)
    assert velocity_error(s0, different_velocity)["total"] > 0.0


def test_pose_weight_overrides_merge_with_defaults():
    w, mode = resolve_pose_weights({"jp": 0.25}, "absolute")
    assert mode == "absolute"
    assert w["jp"] == pytest.approx(0.25)
    assert w["rp"] == pytest.approx(POSE_WEIGHTS["rp"])
    assert w["rq"] == pytest.approx(POSE_WEIGHTS["rq"])

    w, mode = resolve_pose_weights({"rp": 0.0})
    assert mode == "free"
    assert w["rp"] == 0.0
    assert w["jp"] == pytest.approx(POSE_WEIGHTS["jp"])
    assert w["rq"] == pytest.approx(POSE_WEIGHTS["rq"])

    with pytest.raises(ValueError, match="rp=0"):
        resolve_pose_weights({"rp": 0.0}, "absolute")
    with pytest.raises(ValueError, match="nonzero rp"):
        resolve_pose_weights({"rp": 1.0}, "free")
    with pytest.raises(ValueError, match="unknown pose weight"):
        pose_error(_state(np.random.default_rng(1)),
                   _state(np.random.default_rng(2)), {"bad": 1.0})


def test_free_root_position_excludes_root_motion_from_knot_spacing():
    rng = np.random.default_rng(16)
    s0 = _state(rng, (0, 0, 1), yaw=0.0,
                joints=np.zeros(21))
    moved_same_body = _state(rng, (5, -3, 0.7), yaw=0.0,
                             joints=np.zeros(21))
    s2 = _state(rng, (8, -4, 0.6), yaw=0.8,
                joints=np.ones(21))
    free_path = TargetPath.from_clip(
        _clip([s0, moved_same_body, s2]), "all",
        root_pos_mode="free")
    absolute_path = TargetPath.from_clip(
        _clip([s0, moved_same_body, s2]), "all")

    assert free_path.knot_s[1] < 1e-6
    assert absolute_path.knot_s[1] > 0.5


def test_free_root_position_makes_projection_translation_invariant():
    rng = np.random.default_rng(15)
    s0 = _state(rng, (0, 0, 1), yaw=0.0,
                joints=np.zeros(21))
    s1 = _state(rng, (4, -2, 0.8), yaw=0.6,
                joints=np.ones(21))
    path = TargetPath.from_pose_pair(
        s0, s1, root_pos_mode="free")
    assert path.root_pos_mode == "free"
    assert path.pose_weights["rp"] == 0.0

    translated = dict(s1)
    translated["root_pos"] = s1["root_pos"] + np.array(
        [100.0, -50.0, 10.0], dtype=np.float32)
    proj = path.project(translated, s_hint=0.9, ds_max=0.2)
    assert proj.s == pytest.approx(1.0)
    assert proj.errors["total"] == pytest.approx(0.0, abs=1e-7)
    assert proj.errors["components"]["rp"] > 1.0
