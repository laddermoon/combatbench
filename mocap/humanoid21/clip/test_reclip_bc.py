"""Reclip→BC pipeline tests."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as _R

from mocap.humanoid21.clip import reclip_bc
from mocap.humanoid21.clip.core_clip import CoreClip


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


def test_warmup_spec_is_deterministic_and_scheduled():
    a = reclip_bc.make_warmup_spec(7, base_seed=123,
                                  steps_min=0, steps_max=30)
    b = reclip_bc.make_warmup_spec(7, base_seed=123,
                                  steps_min=0, steps_max=30)
    assert a == b
    assert a["skill"] in reclip_bc.WARMUP_SKILLS
    assert len(a["schedule"]) == 1
    assert a["schedule"][0][0] == max(1, a["steps"])

    rows = reclip_bc.warmup_schedule("move_fwd", 4, 0.5)
    assert rows[0][0] == 4
    assert rows[0][1] > 0.0


def test_target_path_is_position_free():
    start = _state((0, 0, 1), 0.0, np.zeros(21))
    target = _state((10, -4, 0.8), 0.7, np.ones(21))
    path = reclip_bc.build_target_path(start, target, "target")
    assert path.mode == "endpoints"
    assert path.root_pos_mode == "free"
    assert path.pose_weights["rp"] == 0.0


def test_zero_policy_init_bank_and_record_loading(tmp_path):
    summary = reclip_bc.generate_init_state_bank(
        tmp_path, count=2, base_seed=5, policy=None,
        steps_min=0, steps_max=0, shard_size=1,
        command_scale_min=1.0, command_scale_max=1.0)
    assert summary["count"] == 2
    records = reclip_bc.load_init_records(tmp_path)
    assert len(records) == 2
    assert {r["init_id"] for r in records} == {
        "init_000000", "init_000001"}
    for i, r in enumerate(records):
        assert Path(r["shard_file"]).name == f"shard_{i:06d}.npz"
        assert r["frame_index"] == 0
        st = reclip_bc.load_bank_state(r)
        assert st["root_pos"].shape == (3,)
        assert st["joint_pos_norm"].shape == (21,)
        assert r["warmup"]["steps"] == 0

    # Resume at the same count is a no-op and keeps exactly two rows.
    summary = reclip_bc.generate_init_state_bank(
        tmp_path, count=2, base_seed=5, policy=None,
        steps_min=0, steps_max=0)
    assert summary["count"] == 2
    assert len(reclip_bc.load_init_records(tmp_path)) == 2


def test_solve_one_init_records_result_and_clip(tmp_path):
    record = {
        "init_id": "init_000000",
        "sample_index": 0,
        "warmup": {"skill": "step", "steps": 0},
    }
    start = _state((0, 0, 1), 0.0, np.zeros(21))
    target = _state((1, 0, 1), 0.3, np.ones(21) * 0.1)
    row = reclip_bc.solve_init_record(
        record, start, target, "target_clip", 6, tmp_path,
        budget_frames=1, ds_max=1.0, post_frames=0,
        sweeps=1, n_samples=2, n_iters=1, seeds=(0,),
        save_all=True)
    try:
        assert row["status"] in ("completed", "budget_exhausted")
        assert row["root_pos_mode"] == "free"
        assert row["clip_path"] is not None
        out = CoreClip.load(tmp_path / row["clip_path"])
        assert out.meta["reclip"]["path"]["root_pos_mode"] == "free"
        assert out.meta["reclip_bc"]["init_id"] == "init_000000"
    finally:
        if reclip_bc._WORKER_SIM is not None:
            reclip_bc._WORKER_SIM.close()
            reclip_bc._WORKER_SIM = None


def test_filter_results_accepts_replay_valid_completed(tmp_path):
    good_replay = {k: 0.0 for k in reclip_bc._REPLAY_KEYS}
    rows = [
        {"init_id": "a", "init_record": {"warmup": {"skill": "step"}},
         "status": "budget_exhausted", "event_frame": None,
         "fall_count": 0, "clip_path": None,
         "replay_error": good_replay},
        {"init_id": "a", "init_record": {"warmup": {"skill": "step"}},
         "status": "completed", "event_frame": 3,
         "fall_count": 0, "clip_path": "expert_clips/a.npz",
         "replay_error": good_replay, "terminal_pose_error": 0.1,
         "path_error_max": 0.2},
        {"init_id": "b", "init_record": {"warmup": {"skill": "step"}},
         "status": "budget_exhausted", "event_frame": None,
         "fall_count": 0, "clip_path": None,
         "replay_error": good_replay},
        {"init_id": "c", "init_record": {"warmup": {"skill": "step"}},
         "status": "completed", "event_frame": 4,
         "fall_count": 1, "clip_path": "expert_clips/c.npz",
         "replay_error": good_replay},
    ]
    results = tmp_path / "reclip_results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = reclip_bc.filter_reclip_results(
        results, tmp_path / "accepted.jsonl",
        tmp_path / "rejected.jsonl", tmp_path / "summary.json")
    assert summary["accepted"] == 1
    assert summary["rejected"] == 2
    assert summary["attempt_count"] == 4
    assert summary["attempt_status_counts"] == {
        "budget_exhausted": 2, "completed": 2}
    accepted = reclip_bc._read_jsonl(tmp_path / "accepted.jsonl")
    rejected = reclip_bc._read_jsonl(tmp_path / "rejected.jsonl")
    assert [r["init_id"] for r in accepted] == ["a"]
    assert {r["init_id"] for r in rejected} == {"b", "c"}
    assert rejected[1]["reject_reasons"] == ["fall"]
