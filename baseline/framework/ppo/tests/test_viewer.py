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
import os
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
from baseline.framework.ppo.dumpkit.viewer.server import (
    DumpData, RunData, ViewerAPI, list_runs, query_runs_index, resolve_run,
    scan_experiments, experiments_index, experiment_detail,
)
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
        assert "kl_mean" in data.update_npz
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
# Rendered flag + RunData tests
# ---------------------------------------------------------------------------

def test_episode_rendered_flag():
    """episode_rendered / episode_list.rendered reflect record/episode_NNNNN PNGs."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        data = DumpData(dump_dir)
        api = ViewerAPI(data)

        # Not rendered yet
        assert data.episode_rendered(0) is False
        status, body = api.handle("/api/episode_list")
        assert status == 200
        assert body[0]["rendered"] is False

        # An empty episode dir without PNGs still counts as not rendered
        ep_dir = dump_dir / "record" / "episode_00000"
        ep_dir.mkdir(parents=True)
        assert data.episode_rendered(0) is False

        # With a PNG → rendered
        (ep_dir / "step_00001.png").write_bytes(b"\x89PNG fake")
        assert data.episode_rendered(0) is True
        status, body = api.handle("/api/episode_list")
        assert body[0]["rendered"] is True
        print("test_episode_rendered_flag: PASS")


def _write_train_log(run_dir: Path) -> Path:
    log = run_dir / "train.log"
    log.write_text(
        "human readable line\n"
        '__RAW_STATS__ {"update": 1, "episode_stats": {"ep_len_mean": 10.0}, '
        '"buffer_stats": {"per_channel": {"r_test": {"reward_mean": 0.5}}}, '
        '"stats": {"policy_loss_mean": -0.01, "ev_r_test": 0.8, '
        '"epoch_kl_stats": [{"kl": 0.1}]}, '
        '"timing": {"total": 1.2}}\n'
        "another line\n"
        '__RAW_STATS__ {"update": 2, "episode_stats": {"ep_len_mean": 11.0}, '
        '"buffer_stats": {"per_channel": {"r_test": {"reward_mean": 0.6}}}, '
        '"stats": {"policy_loss_mean": -0.02, "ev_r_test": 0.85}, '
        '"timing": {"total": 1.3}}\n',
        encoding="utf-8",
    )
    return log


def test_run_data_dumps():
    """RunData.dumps() lists dump dirs with manifest metadata."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent  # tmpdir/run

        rd = RunData(run_dir)
        dumps = rd.dumps()
        assert len(dumps) == 1
        assert dumps[0]["name"] == "u00001"
        assert dumps[0]["update"] == 1
        assert dumps[0]["n_episodes"] == 1
        assert "mtime" in dumps[0]
        print("test_run_data_dumps: PASS")


def test_run_data_metrics():
    """RunData.metrics() parses + flattens __RAW_STATS__ lines."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        _write_train_log(run_dir)

        rd = RunData(run_dir)
        metrics = rd.metrics()
        assert len(metrics) == 2
        m0 = metrics[0]
        assert m0["update"] == 1
        assert m0["stats.policy_loss_mean"] == -0.01
        assert m0["ep.ep_len_mean"] == 10.0
        assert m0["pc.reward_mean.r_test"] == 0.5
        # stats.ev_<ch> is regrouped into pc.ev.<ch>
        assert m0["pc.ev.r_test"] == 0.8
        assert "stats.ev_r_test" not in m0
        # Non-scalar fields are dropped
        assert "stats.epoch_kl_stats" not in m0
        assert m0["time.total"] == 1.2
        assert metrics[1]["update"] == 2
        print("test_run_data_metrics: PASS")


def test_run_data_metrics_experiment():
    """experiment dict in __RAW_STATS__ flattens to exp.*; invalid dropped."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        log = run_dir / "train.log"
        log.write_text(
            '__RAW_STATS__ {"update": 1, "stats": {"policy_loss_mean": -0.01}, '
            '"experiment": {"online_success": 0.25, "bad_nan": NaN, '
            '"bad_str": "x", "count": 3}}\n',
            encoding="utf-8",
        )
        rd = RunData(run_dir)
        m = rd.metrics()[0]
        assert m["exp.online_success"] == 0.25
        assert m["exp.count"] == 3.0
        assert "exp.bad_nan" not in m  # non-finite dropped
        assert "exp.bad_str" not in m  # non-scalar dropped
        # Old logs without "experiment" still parse — covered by
        # test_run_data_metrics (its log has no experiment key).
        print("test_run_data_metrics_experiment: PASS")


def test_run_data_metrics_param_overrides():
    """param_overrides dict in __RAW_STATS__ is carried through as
    row-level metadata (Update Detail config-change evidence) — not
    flattened into chartable stats.* keys."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        log = run_dir / "train.log"
        log.write_text(
            '__RAW_STATS__ {"update": 1, "stats": {"policy_loss_mean": -0.01}}\n'
            '__RAW_STATS__ {"update": 2, "stats": {"policy_loss_mean": -0.02}, '
            '"param_overrides": {"dual_clip_c": 3, "learning_rate": 0.0001}}\n',
            encoding="utf-8",
        )
        rd = RunData(run_dir)
        ms = rd.metrics()
        assert "param_overrides" not in ms[0]  # absent → no key at all
        assert ms[1]["param_overrides"] == {
            "dual_clip_c": 3, "learning_rate": 0.0001}
        # Never leaks into a chartable stats.* / flat key
        assert not any(k.startswith("stats.param_overrides")
                       or "param_overrides." in k for k in ms[1])
        print("test_run_data_metrics_param_overrides: PASS")


def test_metric_catalog_sections():
    """catalog() serves the six themed sections in fixed order; curated
    specs carry subtitle+guide; suppression covers grad_sig."""
    from baseline.framework.ppo.dumpkit.metric_catalog import catalog

    cat = catalog()
    ids = [s["id"] for s in cat["sections"]]
    assert ids == ["task", "policy_update", "signal",
                   "explore", "sampling", "cost"]
    for sec in cat["sections"]:
        assert sec["title"] and sec["question"], sec["id"]
        assert sec["curated"], f"section {sec['id']} has no curated charts"
        for spec in sec["curated"]:
            assert spec.get("subtitle"), f"curated spec missing subtitle: {spec}"
            assert spec.get("guide"), f"curated spec missing guide: {spec}"
    # gradsig scalars must not auto-chart — suppressed by prefix
    assert "stats.grad_sig_" in cat["suppress_prefixes"]
    print("test_metric_catalog_sections: PASS")


def test_run_data_grad_sig():
    """RunData.grad_sig() round-trips a gradsig/uNNNNN.npz artifact into
    a JSON-safe payload; missing artifacts return None."""
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        gs_dir = run_dir / "gradsig"
        gs_dir.mkdir(parents=True)

        cos_edges = np.linspace(-1.0, 1.0, 9)
        norm_edges = np.geomspace(1e-2, 1e2, 5)
        hist = np.zeros((4, 8), dtype=np.int64)
        hist[2, 6] = 7
        grad_norm = np.array([0.5, 1.0, 2.0, np.nan], dtype=np.float32)
        cos = np.array([0.1, -0.2, 0.9, np.nan], dtype=np.float32)
        proj = np.array([0.05, -0.2, 1.8, np.nan], dtype=np.float32)
        valid = np.array([True, True, True, False])
        np.savez_compressed(
            gs_dir / "u00042.npz",
            hist=hist,
            cos_edges=cos_edges,
            norm_edges=norm_edges,
            n_sampled=np.array(4),
            n_valid=np.array(3),
            n_excluded=np.array(1),
            n_nonfinite=np.array(1),
            n_frames_in_hist=np.array(3),
            gnorm=np.array(0.123),
            coherence=np.array(0.42),
            dir_cos=np.array(0.77),
            proj_mean=np.array(0.55),
            proj_std=np.array(0.9),
            frac_neg=np.array(1.0 / 3.0),
            norm_quantiles=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            norm_edges_derived=np.array(True),
            hist_under=np.array([0, 0, 0, 1, 2, 0, 0, 0], dtype=np.int64),
            hist_over=np.array([0, 0, 0, 0, 3, 4, 0, 0], dtype=np.int64),
            n_under=np.array(3),
            n_over=np.array(7),
            grad_norm=grad_norm,
            cos=cos,
            proj=proj,
            valid=valid,
            n_params=np.array(1234),
        )

        rd = RunData(run_dir)
        out = rd.grad_sig(42)
        assert out is not None
        assert out["available"] is True
        assert out["update"] == 42
        assert out["hist"][2][6] == 7
        assert len(out["cos_edges"]) == 9
        assert len(out["norm_edges"]) == 5
        assert out["n_sampled"] == 4
        assert out["n_valid"] == 3
        assert out["n_excluded"] == 1
        assert out["n_frames_in_hist"] == 3
        assert abs(out["gnorm"] - 0.123) < 1e-9
        assert abs(out["coherence"] - 0.42) < 1e-9
        assert abs(out["dir_cos"] - 0.77) < 1e-9
        assert abs(out["proj_mean"] - 0.55) < 1e-9
        assert abs(out["proj_std"] - 0.9) < 1e-9
        assert abs(out["frac_neg"] - 1.0 / 3.0) < 1e-9
        assert out["norm_edges_derived"] is True
        # Under/overflow edge rows come through when present.
        assert out["hist_under"] == [0, 0, 0, 1, 2, 0, 0, 0]
        assert out["hist_over"] == [0, 0, 0, 0, 3, 4, 0, 0]
        assert out["n_under"] == 3
        assert out["n_over"] == 7
        # Raw per-frame arrays; the invalid frame's non-finite values
        # serialize as null (literal NaN would break fetch().json()).
        assert out["grad_norm"][:3] == [0.5, 1.0, 2.0]
        assert out["grad_norm"][3] is None
        assert out["cos"][3] is None
        assert out["proj"][3] is None
        assert out["valid"] == [True, True, True, False]
        assert out["norm_quantile_levels"] == [0.05, 0.25, 0.5, 0.75, 0.95]
        # JSON-serializable
        json.dumps(out, allow_nan=False)

        # Old pairwise-format artifact missing the new keys → treated as
        # unavailable (KeyError caught → None → API reports no data).
        np.savez_compressed(
            gs_dir / "u00043.npz",
            hist=hist, cos_edges=cos_edges, norm_edges=norm_edges,
            n_sampled=np.array(50), n_valid=np.array(48),
            n_excluded=np.array(2), n_pairs=np.array(1128),
            n_pairs_in_hist=np.array(1128),
            pair_mean=np.array(0.1), pair_std=np.array(0.2),
            norm_quantiles=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            norm_edges_derived=np.array(True),
        )
        assert rd.grad_sig(43) is None

        # Missing artifact → None (→ API 404 + available: false)
        assert rd.grad_sig(44) is None
        # Malformed / corrupt npz → None, not an exception
        (gs_dir / "u00099.npz").write_bytes(b"not an npz")
        assert rd.grad_sig(99) is None
        print("test_run_data_grad_sig: PASS")


def test_sanitize_exp_metrics():
    """_sanitize_exp_metrics keeps finite scalars with safe keys only."""
    from baseline.framework.ppo.loop import _sanitize_exp_metrics

    out = _sanitize_exp_metrics({
        "ok": 1.5,
        "int_val": 2,
        "flag": True,
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "Bad Key": 1.0,
        "dotted.key": 1.0,
        "nested": {"a": 1},
        "arr": np.array([1.0]),
        "str_val": "x",
        5: 1.0,
    })
    assert out == {"ok": 1.5, "int_val": 2.0, "flag": 1.0}
    assert _sanitize_exp_metrics(None) == {}
    assert _sanitize_exp_metrics("not a dict") == {}
    assert _sanitize_exp_metrics({}) == {}
    print("test_sanitize_exp_metrics: PASS")


def test_metric_catalog_structure():
    """catalog() is JSON-safe, well-shaped, and covers the open zones."""
    import json as _json
    from baseline.framework.ppo.dumpkit.metric_catalog import catalog

    cat = catalog()
    assert cat["layout"] and isinstance(cat["layout"], list)
    for spec in cat["layout"]:
        kinds = [k for k in ("keys", "pc", "pcm", "timing",
                             "zone_pick", "zone_rest") if k in spec]
        # A spec can combine framework keys with a single pc overlay.
        # pc/pcm/timing/zone_pick/zone_rest are mutually exclusive;
        # keys can appear alongside pc.
        assert kinds, f"empty spec: {spec}"
        multi = [k for k in ("pc", "pcm", "timing",
                             "zone_pick", "zone_rest") if k in spec]
        assert len(multi) <= 1, f"conflicting spec shape: {spec}"
        if "keys" in spec:
            assert spec.get("hint") or spec.get("guide"), \
                f"keys spec missing doc: {spec}"
            for key in spec["keys"]:
                assert "." in key, f"unnamespaced layout key: {key}"
        if "pcm" in spec:
            assert spec["metrics"], f"pcm spec missing metrics: {spec}"
            assert spec.get("hint") or spec.get("guide"), \
                f"pcm spec missing doc: {spec}"
    prefixes = [z["prefix"] for z in cat["zones"]]
    assert prefixes == ["exp.", "eval.", "policy."]
    for z in cat["zones"]:
        assert z["title"] and z["color"] and z["hint"]
    # Round-trips through JSON — what /api/catalog actually serves.
    _json.loads(_json.dumps(cat))
    print("test_metric_catalog_structure: PASS")


def test_metric_doc():
    """metric_doc resolves flat keys to zone + hint."""
    from baseline.framework.ppo.dumpkit.metric_catalog import metric_doc

    d = metric_doc("stats.policy_loss_mean")
    assert d["zone"] == "framework" and "policy_loss_mean" in d["hint"]
    # Layout key falls back to its chart-group hint
    d = metric_doc("stats.post_kl_mean")
    assert d["zone"] == "framework" and "KL" in d["hint"]
    d = metric_doc("pc.ev.r_test")
    assert d["zone"] == "framework" and "explained variance" in d["hint"]
    d = metric_doc("exp.online_success")
    assert d["zone"] == "experiment" and "on_update" in d["hint"]
    d = metric_doc("eval.success")
    assert d["zone"] == "eval" and "on_eval" in d["hint"]
    d = metric_doc("policy.foo")
    assert d["zone"] == "policy" and "policy_stats" in d["hint"]
    # Unknown framework key → framework zone, empty-ish hint ok
    d = metric_doc("stats.no_such_metric")
    assert d["zone"] == "framework"
    print("test_metric_doc: PASS")


def test_run_summary():
    """RunData.summary(): per-key zone/hint + latest/min/max + updates."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        (run_dir / "train.log").write_text(
            '__RAW_STATS__ {"update": 1, "stats": {"policy_loss_mean": -0.01}, '
            '"experiment": {"online_success": 0.1}}\n'
            '__RAW_STATS__ {"update": 2, "stats": {"policy_loss_mean": -0.05}, '
            '"experiment": {"online_success": 0.3}, '
            '"eval_info": {"success": 0.5}}\n',
            encoding="utf-8",
        )
        s = RunData(run_dir).summary()
        assert s["n_updates"] == 2
        assert s["run"] == run_dir.name

        pl = s["metrics"]["stats.policy_loss_mean"]
        assert pl["n"] == 2 and pl["zone"] == "framework"
        assert pl["latest"] == -0.05 and pl["latest_update"] == 2
        assert pl["min"] == -0.05 and pl["min_update"] == 2
        assert pl["max"] == -0.01 and pl["max_update"] == 1
        assert pl["first_update"] == 1

        es = s["metrics"]["exp.online_success"]
        assert es["zone"] == "experiment" and es["n"] == 2
        assert "on_update" in es["hint"]

        ev = s["metrics"]["eval.success"]
        assert ev["zone"] == "eval" and ev["n"] == 1  # sparse, honest
        assert ev["first_update"] == 2 and ev["latest"] == 0.5

        # Drill-down availability is part of the digest
        assert s["n_dumps"] == 1 and s["dumps"] == ["u00001"]
        print("test_run_summary: PASS")


def test_debug_metrics_cmd(capsys):
    """debug.py metrics: JSON output, filters, name resolution, exit codes."""
    from baseline.framework.ppo import debug

    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        _write_train_log(run_dir)
        runs_root = run_dir.parent
        capsys.readouterr()  # drain fixture prints ([dump] captured ...)

        assert debug.main(["metrics", str(run_dir), "--tail", "1"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["n_updates"] == 1 and out["metrics"][0]["update"] == 2

        assert debug.main(
            ["metrics", str(run_dir), "--keys", "policy_loss_mean"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert set(out["metrics"][0]) == {"update", "stats.policy_loss_mean"}

        assert debug.main(
            ["metrics", str(run_dir), "--from-update", "2"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["n_updates"] == 1

        # Run-name resolution under --runs-root
        assert debug.main(
            ["metrics", run_dir.name, "--runs-root", str(runs_root),
             "--tail", "1"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["run"] == run_dir.name

        # Missing run → exit 2
        assert debug.main(
            ["metrics", "no_such_run", "--runs-root", str(runs_root)]) == 2
        capsys.readouterr()
        print("test_debug_metrics_cmd: PASS")


def test_debug_summary_cmd(capsys):
    """debug.py summary: digest JSON with zone+hint attached."""
    from baseline.framework.ppo import debug

    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        _write_train_log(run_dir)
        capsys.readouterr()  # drain fixture prints

        assert debug.main(["summary", str(run_dir)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["n_updates"] == 2
        assert out["metrics"]["stats.policy_loss_mean"]["latest"] == -0.02
        assert "zone" in out["metrics"]["stats.policy_loss_mean"]

        assert debug.main(
            ["summary", str(run_dir), "--keys", "ep_len"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert set(out["metrics"]) == {"ep.ep_len_mean"}
        print("test_debug_summary_cmd: PASS")


def test_debug_runs_cmd(capsys):
    """debug.py runs: list runs, filters, ordering, empty root."""
    from baseline.framework.ppo import debug

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Two minimal run dirs (config.json marks them as runs)
        for name in ("run_a", "run_b"):
            rd = root / name
            rd.mkdir()
            (rd / "config.json").write_text(
                '{"experiment": {"name": "exp_' + name + '"}}',
                encoding="utf-8",
            )
        (root / "not_a_run").mkdir()  # ignored: no config.json/train.log

        assert debug.main(["runs", "--runs-root", str(root)]) == 0
        out = json.loads(capsys.readouterr().out)
        names = [r["name"] for r in out["runs"]]
        assert sorted(names) == ["run_a", "run_b"]
        r0 = out["runs"][0]
        for f in ("experiment", "status", "update", "n_dumps",
                  "created", "activity"):
            assert f in r0, f"missing field {f}"

        # --status filter
        assert debug.main(
            ["runs", "--runs-root", str(root), "--status", "running"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert all(r["status"] == "running" for r in out["runs"])

        # --tail 1 keeps a single run
        assert debug.main(
            ["runs", "--runs-root", str(root), "--tail", "1"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert len(out["runs"]) == 1

        # Empty root → valid empty list; missing root → exit 2
        empty = root / "empty"
        empty.mkdir()
        assert debug.main(["runs", "--runs-root", str(empty)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["runs"] == []
        assert debug.main(
            ["runs", "--runs-root", str(root / "nope")]) == 2
        capsys.readouterr()
        print("test_debug_runs_cmd: PASS")


def test_debug_help_smoke(capsys):
    """Top-level -h prints the question-map epilog."""
    from baseline.framework.ppo import debug
    import pytest

    with pytest.raises(SystemExit) as exc:
        debug.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "by question:" in out
    for cmd in ("runs", "summary", "metrics", "catalog",
                "dump", "viewer", "render", "delta"):
        assert cmd in out
    print("test_debug_help_smoke: PASS")


def test_debug_catalog_cmd(capsys):
    """debug.py catalog: full catalog or single-key doc."""
    from baseline.framework.ppo import debug

    assert debug.main(["catalog", "--key", "stats.post_kl_mean"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["zone"] == "framework" and "KL" in out["hint"]

    assert debug.main(["catalog"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["layout"]) > 0
    assert [z["prefix"] for z in out["zones"]] == ["exp.", "eval.", "policy."]
    print("test_debug_catalog_cmd: PASS")


def test_run_data_metrics_incremental():
    """metrics() picks up appended lines and survives a partial tail line."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent
        log = _write_train_log(run_dir)

        rd = RunData(run_dir)
        assert len(rd.metrics()) == 2

        # Append a complete line → re-parse picks it up
        with open(log, "a", encoding="utf-8") as f:
            f.write('__RAW_STATS__ {"update": 3, "stats": {"policy_loss_mean": -0.03}}\n')
        metrics = rd.metrics()
        assert len(metrics) == 3
        assert metrics[2]["stats.policy_loss_mean"] == -0.03

        # A partial (newline-less) tail line is deferred, not dropped
        with open(log, "a", encoding="utf-8") as f:
            f.write('__RAW_STATS__ {"update": 4, "stats": {"policy_loss_mean": -0.04}}')
        assert len(rd.metrics()) == 3
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n")
        metrics = rd.metrics()
        assert len(metrics) == 4
        print("test_run_data_metrics_incremental: PASS")


def test_run_data_get_dump_api():
    """get_dump_api resolves valid dump names, rejects unknown ones."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent

        rd = RunData(run_dir)
        api = rd.get_dump_api("u00001")
        assert api is not None
        status, body = api.handle("/api/manifest")
        assert status == 200
        assert body["update"] == 1
        # Cached
        assert rd.get_dump_api("u00001") is api
        # Unknown dump
        assert rd.get_dump_api("u99999") is None
        assert rd.get_dump_api("../outside") is None
        print("test_run_data_get_dump_api: PASS")


def test_run_data_no_train_log():
    """RunData works without train.log — metrics() returns []."""
    with tempfile.TemporaryDirectory() as d:
        dump_dir = _create_test_dump(Path(d))
        run_dir = dump_dir.parent.parent

        rd = RunData(run_dir)
        assert rd.metrics() == []
        info = rd.run_info()
        assert info["has_train_log"] is False
        assert info["n_dumps"] == 1
        print("test_run_data_no_train_log: PASS")


def test_run_data_videos():
    """videos() lists mp4s newest-first, joining sidecar meta + eval."""
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        vdir = run_dir / "videos"
        vdir.mkdir(parents=True)
        (vdir / "u00025.mp4").write_bytes(b"x" * 10)
        (vdir / "u00025.log").write_text(json.dumps({
            "steps": 200, "seed": 7,
            "termination_reasons": {"robot_a": ["timeout"],
                                    "robot_b": ["ko"]},
            "health_a": 1.0, "health_b": 0.0,
        }), encoding="utf-8")
        (vdir / "u00050.mp4").write_bytes(b"y" * 20)   # no sidecar
        (vdir / "notes.txt").write_text("ignore me")  # non-mp4 skipped
        (run_dir / "train.log").write_text(
            '__RAW_STATS__ {"update": 25, "eval_info": '
            '{"success": 0.5, "final_pot": 0.9}}\n'
            '__RAW_STATS__ {"update": 50, "eval_info": '
            '{"success": 1.0, "final_pot": 1.0}}\n',
            encoding="utf-8",
        )
        rd = RunData(run_dir)
        vids = rd.videos()
        assert len(vids) == 2
        assert vids[0]["name"] == "u00050.mp4"      # newest update first
        assert vids[0]["eval_success"] == 1.0
        assert vids[0]["eval_pot"] == 1.0
        assert vids[0]["steps"] is None           # missing sidecar
        assert vids[1]["update"] == 25
        assert vids[1]["term_a"] == "timeout"
        assert vids[1]["term_b"] == "ko"
        assert vids[1]["health_a"] == 1.0
        assert vids[1]["seed"] == 7
        # video_path: whitelist + existence
        assert rd.video_path("u00025.mp4") is not None
        assert rd.video_path("../x.mp4") is None
        assert rd.video_path("u00099.mp4") is None
        assert rd.video_path("notes.txt") is None
        print("test_run_data_videos: PASS")


def test_run_data_metrics_has_eval_keys():
    """eval_info scalars flatten to eval.* keys in metrics()."""
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "run"
        run_dir.mkdir()
        (run_dir / "train.log").write_text(
            '__RAW_STATS__ {"update": 1, "stats": {"policy_loss_mean": -0.1}, '
            '"eval_info": {"success": 0.75, "final_pot": 0.8}}\n',
            encoding="utf-8",
        )
        m = RunData(run_dir).metrics()[0]
        assert m["eval.success"] == 0.75
        assert m["eval.final_pot"] == 0.8
        print("test_run_data_metrics_has_eval_keys: PASS")


# ---------------------------------------------------------------------------
# Runs-root index (list_runs / query_runs_index / resolve_run)
# ---------------------------------------------------------------------------


def _make_fake_run(
    root: Path, name: str, *, update=None, max_updates=100,
    eval_success=None, config=True, log=True, pid=None,
) -> Path:
    d = root / name
    d.mkdir()
    if config:
        cfg = {
            "algorithm": "ppo",
            "experiment": {
                "name": name,
                "common_params": {"max_updates": max_updates},
            },
        }
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if log:
        lines = ["human line"]
        if update is not None:
            ev = (
                f', "eval_info": {{"success": {eval_success}}}'
                if eval_success is not None else ""
            )
            lines.append(f'__RAW_STATS__ {{"update": {update}{ev}}}')
        (d / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if pid is not None:
        (d / "pid").write_text(str(pid), encoding="utf-8")
    return d


def test_list_runs_discovers_and_summarizes():
    """list_runs finds run dirs by markers, summarizes config + log tail."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_fake_run(root, "train_a_ppo_20260101_100000",
                       update=10, max_updates=10, eval_success=1.0)
        _make_fake_run(root, "train_b_ppo_20260102_100000",
                       update=5, max_updates=10, pid=99999999)
        (root / "not_a_run").mkdir()          # no markers → skipped
        (root / "loose.txt").write_text("x")  # non-dir → skipped

        runs = list_runs(root)
        assert len(runs) == 2
        a = next(r for r in runs if r["name"] == "train_a_ppo_20260101_100000")
        b = next(r for r in runs if r["name"] == "train_b_ppo_20260102_100000")
        assert a["status"] == "finished"        # update == max_updates
        assert a["eval_success"] == 1.0
        assert a["algo"] == "ppo"
        # name timestamp parsed for created
        assert a["created"] > 0
        assert b["status"] == "stopped"         # dead pid, incomplete
        assert b["update"] == 5
        assert b["eval_success"] is None        # last stats has no eval_info
        print("test_list_runs_discovers_and_summarizes: PASS")


def test_list_runs_status_rules():
    """running / finished / stopped / unknown classification."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_fake_run(root, "live", update=5, pid=os.getpid())
        _make_fake_run(root, "fresh", update=5)          # no pid, fresh log
        _make_fake_run(root, "dead", update=5, pid=99999999)
        _make_fake_run(root, "empty", update=None, pid=99999999)  # no stats
        runs = {r["name"]: r for r in list_runs(root)}
        assert runs["live"]["status"] == "running"
        assert runs["fresh"]["status"] == "running"      # mtime heuristic
        assert runs["dead"]["status"] == "stopped"
        assert runs["empty"]["status"] == "unknown"
        print("test_list_runs_status_rules: PASS")


def test_query_runs_index_filter_sort_paginate():
    """query_runs_index: search filter, created-desc default, pagination."""
    runs = [
        {"name": "train_floor_ppo_20260103_000000", "experiment": "floor",
         "algo": "ppo", "status": "running", "update": 3, "max_updates": 10,
         "eval_success": 0.5, "created": 300.0, "activity": 300.0,
         "n_dumps": 0},
        {"name": "train_ctrl_ppo_20260101_000000", "experiment": "ctrl",
         "algo": "ppo", "status": "finished", "update": 10, "max_updates": 10,
         "eval_success": 1.0, "created": 100.0, "activity": 100.0,
         "n_dumps": 2},
        {"name": "train_floor2_ppo_20260102_000000", "experiment": "floor2",
         "algo": "ppo", "status": "stopped", "update": 7, "max_updates": 10,
         "eval_success": 0.8, "created": 200.0, "activity": 200.0,
         "n_dumps": 1},
    ]
    # default: created desc
    res = query_runs_index(runs)
    assert [r["name"] for r in res["runs"]] == [
        "train_floor_ppo_20260103_000000",
        "train_floor2_ppo_20260102_000000",
        "train_ctrl_ppo_20260101_000000",
    ]
    # search over name + experiment
    res = query_runs_index(runs, q="floor")
    assert res["total"] == 2
    res = query_runs_index(runs, q="CTRL")
    assert res["total"] == 1
    # pagination
    res = query_runs_index(runs, size=2, page=1)
    assert res["total"] == 3 and len(res["runs"]) == 2
    res = query_runs_index(runs, size=2, page=2)
    assert len(res["runs"]) == 1
    # sort by update asc
    res = query_runs_index(runs, sort="update", order="asc")
    assert [r["update"] for r in res["runs"]] == [3, 7, 10]
    print("test_query_runs_index_filter_sort_paginate: PASS")


def test_resolve_run_validation_and_cache():
    """resolve_run accepts real run dirs, rejects traversal, caches."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_fake_run(root, "train_x", update=1)
        (root / "plain_dir").mkdir()
        cache = {}
        rd = resolve_run(root, "train_x", cache)
        assert isinstance(rd, RunData)
        assert resolve_run(root, "train_x", cache) is rd  # cached
        assert resolve_run(root, "..", cache) is None
        assert resolve_run(root, "a/b", cache) is None
        assert resolve_run(root, "nonexistent", cache) is None
        assert resolve_run(root, "plain_dir", cache) is None  # no markers
        print("test_resolve_run_validation_and_cache: PASS")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Experiments index / dashboard (runs-root mode)
# ---------------------------------------------------------------------------

def test_scan_experiments_discovers_registry():
    """scan_experiments() parses exp_*.py files into a metadata table."""
    exps = scan_experiments()
    names = {e["name"] for e in exps}
    assert "basic_balance" in names          # known PPO experiment
    ppo = [e for e in exps if e["algo"] == "ppo"]
    assert ppo, "expected at least one PPO experiment"
    for e in ppo:
        assert e["source"].endswith(".py")
        assert isinstance(e["tunables"], list)
        assert all("key" in t and "default" in t for t in e["tunables"])
        # 'name' is identity, never a tunable
        assert all(t["key"] != "name" for t in e["tunables"])
    print("test_scan_experiments_discovers_registry: PASS")


def _exp_run(root: Path, name: str, exp: str, *, created: float,
             params: dict, update=10, checkpoint=None) -> dict:
    """A run summary + config.json/checkpoint on disk for exp tests."""
    d = _make_fake_run(root, name, update=update)
    cfg = {
        "algorithm": "ppo",
        "experiment": {
            "name": exp,
            "common_params": {"max_updates": 100},
            "ppo_params": params,
        },
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if checkpoint is not None:
        ck = d / "checkpoints"
        ck.mkdir()
        (ck / f"checkpoint_u{checkpoint:05d}.pt").write_text("x")
    return {
        "name": name, "experiment": exp, "algo": "ppo",
        "status": "finished", "update": update, "max_updates": 100,
        "eval_success": 1.0, "eval_pot": 0.9,
        "created": created, "activity": created, "n_dumps": 0,
    }


def test_experiments_index_groups_runs():
    """experiments_index groups run stats by experiment + (unassigned)."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        runs = [
            _exp_run(root, "train_basic_balance_ppo_20260101_000000",
                     "basic_balance", created=100.0, params={}),
            _exp_run(root, "train_basic_balance_ppo_20260102_000000",
                     "basic_balance", created=200.0, params={}),
            _exp_run(root, "orphan_run", "unregistered_exp",
                     created=50.0, params={}),
        ]
        idx = experiments_index(runs)
        by_name = {e["name"]: e for e in idx}
        bb = by_name["basic_balance"]
        assert bb["n_runs"] == 2
        assert bb["latest_activity"] == 200.0
        assert bb["latest_eval_pot"] == 0.9
        un = by_name["(unassigned)"]
        assert un["n_runs"] == 1
        assert un["latest_run"] == "orphan_run"
        print("test_experiments_index_groups_runs: PASS")


def test_experiment_detail_diffs_and_checkpoints():
    """experiment_detail returns params diff columns + latest ckpt per run."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        runs = [
            _exp_run(root, "train_basic_balance_ppo_20260101_000000",
                     "basic_balance", created=100.0,
                     params={"learning_rate": 3e-4, "target_kl": 0.05},
                     checkpoint=500),
            _exp_run(root, "train_basic_balance_ppo_20260102_000000",
                     "basic_balance", created=200.0,
                     params={"learning_rate": 2e-4, "target_kl": 0.05},
                     checkpoint=800),
        ]
        det = experiment_detail("basic_balance", runs, root)
        assert det is not None
        assert det["name"] == "basic_balance"
        assert len(det["runs"]) == 2
        # learning_rate varies → diff column; target_kl same → not a diff
        assert "learning_rate" in det["diff_params"]
        assert "target_kl" not in det["diff_params"]
        assert "name" not in det["diff_params"]
        # each run carries only the diff keys
        assert set(det["runs"][0]["params"]) == {"learning_rate"}
        # latest checkpoint per run, newest update first
        assert [c["update"] for c in det["checkpoints"]] == [800, 500]
        assert det["schema"]["common_params"]["max_updates"] == 100
        assert det["cwd"]
        # unknown experiment → None → 404 upstream
        assert experiment_detail("no_such_exp", runs, root) is None
        print("test_experiment_detail_diffs_and_checkpoints: PASS")


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
    test_episode_rendered_flag()
    test_run_data_dumps()
    test_run_data_metrics()
    test_run_data_metrics_incremental()
    test_run_data_get_dump_api()
    test_run_data_no_train_log()
    test_list_runs_discovers_and_summarizes()
    test_list_runs_status_rules()
    test_query_runs_index_filter_sort_paginate()
    test_resolve_run_validation_and_cache()
    test_run_data_videos()
    test_run_data_metrics_has_eval_keys()
    test_run_data_grad_sig()
    test_scan_experiments_discovers_registry()
    test_experiments_index_groups_runs()
    test_experiment_detail_diffs_and_checkpoints()
    print("\nAll viewer tests passed!")
