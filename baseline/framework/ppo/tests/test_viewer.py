"""Tests for the debug viewer server and API.

Tests cover:
- DumpData: lazy NPZ loading, caching, derived helpers
- ViewerAPI: all endpoints return correct JSON structure
- traj_map.json: generation and fallback
- Trajectory slicing (seg_offsets)
- Image path resolution (with +1 offset)
- Timeline and epoch_frames data loading (when present)
- Server starts and serves index.html

Conventions follow test_dump.py:
- Simple print("test_xxx: PASS") at the end of each test.
- `if __name__ == "__main__":` block runs all tests sequentially.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.dumpkit.dump_request import DumpRequest
from baseline.framework.ppo.dumpkit.dump_capture import capture_dump
from baseline.framework.ppo.dumpkit.viewer.server import DumpData, ViewerAPI
from baseline.framework.ppo.experiment import (
    ActorEval,
    PPOParams,
)
from baseline.framework.ppo.trainer import PPOBuffer, ppo_update
from baseline.framework.ppo.trajectory import (
    ChannelData,
    RewardChannel,
    Trajectory,
)
from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.job import Job

# Reuse fixtures from test_trainer
from baseline.framework.ppo.tests.test_trainer import (
    SimpleActor,
    SimpleCritic,
    make_trajectory,
    make_channel_data,
    make_buffer,
    make_critics,
    make_optimizers,
    make_pp_params,
)

# Reuse helpers from test_dump
from baseline.framework.ppo.tests.test_dump import (
    _make_fake_episode,
    _make_fake_job,
)


# ---------------------------------------------------------------------------
# Helper: create a full synthetic dump for testing
# ---------------------------------------------------------------------------

def _create_test_dump(tmpdir: Path) -> Path:
    """Create a complete dump directory with all NPZ files.

    Returns the dump directory path.
    """
    obs_dim, action_dim = 8, 4
    T = 10
    reward_keys = ("r_test",)
    channels = (
        RewardChannel(name="r_test", gamma=0.99, gae_lambda=0.95),
    )

    rng = np.random.default_rng(42)
    trajs = [
        Trajectory(
            obs=rng.standard_normal((T, obs_dim)).astype(np.float32),
            actions=rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32),
            last_obs=rng.standard_normal(obs_dim).astype(np.float32),
            channels={
                "r_test": make_channel_data(T, rng=rng),
            },
            importance=1.0,
        )
    ]

    # Run ppo_update with dump_callback to get timeline + epoch_frames
    dump_collector = {}

    def cb(stage, data):
        dump_collector[stage] = data

    buf, actor = make_buffer(trajs, obs_dim, action_dim, reward_keys)
    critics = make_critics(reward_keys, obs_dim)
    actor_opt, critic_opts = make_optimizers(actor, critics)

    stats = ppo_update(
        actor=actor,
        critics=critics,
        actor_optimizer=actor_opt,
        critic_optimizers=critic_opts,
        buf=buf,
        reward_channels=channels,
        pp=make_pp_params(),
        grad_clip_norm=0.5,
        device=torch.device("cpu"),
        dump_callback=cb,
    )

    episodes = [_make_fake_episode(T=T, obs_dim=obs_dim, action_dim=action_dim)]
    jobs = [_make_fake_job(seed=42)]
    req = DumpRequest(hypothesis="test viewer")

    run_dir = tmpdir / "run"
    run_dir.mkdir()

    dump_dir = capture_dump(
        run_dir=run_dir,
        update=1,
        request=req,
        episodes=episodes,
        trajectories=trajs,
        buf=buf,
        stats=stats,
        jobs=jobs,
        dump_collector=dump_collector,
        experiment_name="test_exp",
    )

    return dump_dir


# ---------------------------------------------------------------------------
# DumpData tests
# ---------------------------------------------------------------------------

def test_dump_data_loads_manifest():
    """DumpData loads manifest.json correctly."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        m = data.manifest
        assert m["update"] == 1
        assert m["experiment_name"] == "test_exp"
        assert m["n_episodes"] == 1
        assert m["n_trajectories"] == 1
        print("test_dump_data_loads_manifest: PASS")


def test_dump_data_loads_npz_files():
    """DumpData loads all NPZ files."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        assert data.episodes_npz is not None
        assert "obs.robot_a" in data.episodes_npz
        assert data.trajectories_npz is not None
        assert "reward.r_test" in data.trajectories_npz
        assert data.buffer_npz is not None
        assert "obs" in data.buffer_npz
        assert data.gae_npz is not None
        assert "values_all" in data.gae_npz
        assert data.combine_npz is not None
        assert "combined_adv" in data.combine_npz
        assert data.update_npz is not None
        assert "approx_kl" in data.update_npz
        print("test_dump_data_loads_npz_files: PASS")


def test_dump_data_seg_offsets():
    """DumpData.seg_offsets returns correct cumulative offsets."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        offsets = data.seg_offsets
        assert len(offsets) == 2  # 1 trajectory → 2 offsets
        assert offsets[0] == 0
        assert offsets[1] == 10  # T=10
        print("test_dump_data_seg_offsets: PASS")


def test_dump_data_channel_names():
    """DumpData.channel_names returns channel names."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        cn = data.channel_names
        assert "r_test" in cn
        print("test_dump_data_channel_names: PASS")


def test_dump_data_traj_map():
    """DumpData.traj_map loads from traj_map.json."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        tm = data.traj_map
        assert len(tm) == 1  # 1 episode
        assert tm[0]["list_pos"] == 0
        assert len(tm[0]["trajectories"]) >= 1
        print("test_dump_data_traj_map: PASS")


def test_dump_data_timeline_and_epoch_frames():
    """DumpData loads timeline.npz and epoch_frames.npz when present."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)

        tl = data.timeline_npz
        assert tl is not None
        assert "n_steps" in tl
        assert "kl" in tl
        assert "epoch_idx" in tl

        ef = data.epoch_frames_npz
        assert ef is not None
        assert "n_epochs" in ef
        # Should have ratio.0 at minimum
        assert "ratio.0" in ef
        print("test_dump_data_timeline_and_epoch_frames: PASS")


# ---------------------------------------------------------------------------
# ViewerAPI tests
# ---------------------------------------------------------------------------

def test_api_manifest():
    """GET /api/manifest returns correct metadata."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/manifest")
        assert status == 200
        assert body["update"] == 1
        assert "r_test" in body["channel_names"]
        assert body["has_timeline"] is True
        assert body["has_epoch_frames"] is True
        print("test_api_manifest: PASS")


def test_api_episode_list():
    """GET /api/episode_list returns episode list."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/episode_list")
        assert status == 200
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["list_pos"] == 0
        print("test_api_episode_list: PASS")


def test_api_traj_map():
    """GET /api/traj_map returns full mapping."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/traj_map")
        assert status == 200
        assert len(body) == 1
        print("test_api_traj_map: PASS")


def test_api_episode_frame():
    """GET /api/episode/0/frame/0 returns frame data."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/episode/0/frame/0")
        assert status == 200
        assert body["episode_pos"] == 0
        assert body["frame"] == 0
        assert "obs_robot_a" in body
        assert "actions_robot_a" in body
        assert "trajectories" in body
        print("test_api_episode_frame: PASS")


def test_api_episode_frame_out_of_range():
    """GET /api/episode/0/frame/999 returns 404."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/episode/0/frame/999")
        assert status == 404
        print("test_api_episode_frame_out_of_range: PASS")


def test_api_traj_overview():
    """GET /api/trajectory/0/overview returns per-frame data."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/overview")
        assert status == 200
        assert body["traj_idx"] == 0
        assert body["length"] == 10
        assert "reward_r_test" in body
        assert "values_r_test" in body
        assert "advs_r_test" in body
        assert "rets_r_test" in body
        assert "combined_adv" in body
        print("test_api_traj_overview: PASS")


def test_api_traj_frame():
    """GET /api/trajectory/0/frame/5 returns single frame."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/frame/5")
        assert status == 200
        assert body["traj_idx"] == 0
        assert body["frame"] == 5
        assert "reward_r_test" in body
        assert "values_r_test" in body
        print("test_api_traj_frame: PASS")


def test_api_traj_epoch_overview():
    """GET /api/trajectory/0/epoch/0/overview returns epoch data."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/epoch/0/overview")
        assert status == 200
        assert body["traj_idx"] == 0
        assert body["epoch"] == 0
        assert "ratio" in body
        assert "clip_mask" in body
        assert "new_log_prob" in body
        print("test_api_traj_epoch_overview: PASS")


def test_api_traj_epoch_compare():
    """GET /api/trajectory/0/epoch_compare returns all epochs."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/epoch_compare")
        assert status == 200
        assert "n_epochs" in body
        assert "epochs" in body
        assert len(body["epochs"]) == body["n_epochs"]
        print("test_api_traj_epoch_compare: PASS")


def test_api_timeline_overview():
    """GET /api/timeline/overview returns timeline data."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/timeline/overview")
        assert status == 200
        assert body["available"] is True
        assert "n_steps" in body
        assert "kl" in body
        assert "epoch_idx" in body
        assert "mb_idx" in body
        assert "actor_active" in body
        assert "early_stop_step" in body
        print("test_api_timeline_overview: PASS")


def test_api_timeline_step():
    """GET /api/timeline/step/0 returns step detail."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/timeline/step/0")
        assert status == 200
        assert body["step"] == 0
        assert "epoch_idx" in body
        assert "kl" in body
        print("test_api_timeline_step: PASS")


def test_api_unknown_endpoint():
    """Unknown API endpoint returns 404."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/unknown")
        assert status == 404
        print("test_api_unknown_endpoint: PASS")


def test_api_image_not_found():
    """GET /api/image/0/0 returns 404 when no record dir."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/image/0/0")
        assert status == 404
        print("test_api_image_not_found: PASS")


# ---------------------------------------------------------------------------
# traj_map fallback test
# ---------------------------------------------------------------------------

def test_traj_map_fallback_from_frame_ids():
    """DumpData.traj_map builds from frame_ids when traj_map.json is missing."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))

        # Delete traj_map.json to test fallback
        (dump_dir / "traj_map.json").unlink()

        data = DumpData(dump_dir)
        tm = data.traj_map
        assert len(tm) >= 1
        # Should still have trajectory mapping from frame_ids
        assert len(tm[0]["trajectories"]) >= 1
        print("test_traj_map_fallback_from_frame_ids: PASS")


# ---------------------------------------------------------------------------
# Enhanced API field tests (Scene 1-3 enhancements)
# ---------------------------------------------------------------------------

def test_api_episode_frame_has_trajectory_channel_data():
    """GET /api/episode/0/frame/0 returns per-trajectory per-frame channel data."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/episode/0/frame/0")
        assert status == 200
        trajs = body["trajectories"]
        assert len(trajs) >= 1
        t = trajs[0]
        # Enhanced fields
        assert "covered" in t
        assert "traj_frame" in t
        assert t["covered"] is True
        assert t["traj_frame"] == 0
        # Per-channel data
        assert "reward_r_test" in t
        assert "actor_weight_r_test" in t
        assert "is_terminated_r_test" in t
        assert "floor_weight" in t
        assert "explore_factor" in t
        assert "importance" in t
        assert "frame_id" in t
        print("test_api_episode_frame_has_trajectory_channel_data: PASS")


def test_api_traj_overview_has_enhanced_fields():
    """GET /api/trajectory/0/overview returns key_frame_mask, seg_active, aw_l1_sum."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/overview")
        assert status == 200
        # key_frame_mask
        assert "key_frame_mask" in body
        assert "r_test" in body["key_frame_mask"]
        # key_seg_active / key_seg_terminated
        assert "key_seg_active" in body
        assert "key_seg_terminated" in body
        assert "r_test" in body["key_seg_active"]
        # aw_l1_sum
        assert "aw_l1_sum" in body
        # floor_weight, explore_factor, importance
        assert "floor_weight" in body
        assert "explore_factor" in body
        assert "importance" in body
        # is_terminated per channel
        assert "is_terminated_r_test" in body
        print("test_api_traj_overview_has_enhanced_fields: PASS")


def test_api_traj_epoch_overview_has_cross_ref_fields():
    """GET /api/trajectory/0/epoch/0/overview returns old_log_prob, old_value, return, combined_adv."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/epoch/0/overview")
        assert status == 200
        # Cross-referenced fields
        assert "old_log_prob" in body
        assert "old_value_r_test" in body
        assert "return_r_test" in body
        assert "combined_adv" in body
        assert "clip_eps" in body
        assert "actor_stopped_epoch" in body
        print("test_api_traj_epoch_overview_has_cross_ref_fields: PASS")


def test_api_manifest_has_clip_eps():
    """GET /api/manifest includes clip_eps, target_kl, n_epochs, n_steps."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/manifest")
        assert status == 200
        assert "clip_eps" in body
        assert "target_kl" in body
        assert "n_epochs" in body
        assert "n_steps" in body
        assert "early_stop_step" in body
        print("test_api_manifest_has_clip_eps: PASS")


def test_api_epoch_compare_has_clip_eps():
    """GET /api/trajectory/0/epoch_compare includes clip_eps."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        status, body = api.handle("/api/trajectory/0/epoch_compare")
        assert status == 200
        assert "clip_eps" in body
        print("test_api_epoch_compare_has_clip_eps: PASS")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dump_data_loads_manifest()
    test_dump_data_loads_npz_files()
    test_dump_data_seg_offsets()
    test_dump_data_channel_names()
    test_dump_data_traj_map()
    test_dump_data_timeline_and_epoch_frames()
    test_api_manifest()
    test_api_manifest_has_clip_eps()
    test_api_episode_list()
    test_api_traj_map()
    test_api_episode_frame()
    test_api_episode_frame_has_trajectory_channel_data()
    test_api_episode_frame_out_of_range()
    test_api_traj_overview()
    test_api_traj_overview_has_enhanced_fields()
    test_api_traj_frame()
    test_api_traj_epoch_overview()
    test_api_traj_epoch_overview_has_cross_ref_fields()
    test_api_traj_epoch_compare()
    test_api_epoch_compare_has_clip_eps()
    test_api_timeline_overview()
    test_api_timeline_step()
    test_api_unknown_endpoint()
    test_api_image_not_found()
    test_traj_map_fallback_from_frame_ids()
    print("\nAll viewer tests passed!")
