"""S3 ``frame`` tool tests.

Tests cover:
- parse_frame_id parses ep0003:robot_a:137.
- parse_frame_id rejects malformed IDs.
- FrameIndex maps flat index ↔ frame ID.
- build_frame_index from synthetic episodes.
- inspect_frame extracts per-frame values.
- filter_frames with --where expression.
- render_frame produces human-readable output.
- render_frame_list produces a compact table.

Conventions follow test_s2_sink.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.frame import (
    FrameData, FrameIndex, parse_frame_id, build_frame_index,
    inspect_frame, filter_frames, render_frame, render_frame_list, frame,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_snapshot(tmp_path):
    """Create a minimal snapshot with episodes + replay arrays."""
    from baseline.framework.rollout import Episode, blueprint_hash
    from envs.framework.blueprint import EnvBlueprint, ClassSpec

    snapshot_dir = tmp_path / "debug" / "u00001"
    episodes_dir = snapshot_dir / "episodes"
    replay_dir = snapshot_dir / "replay"
    episodes_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)

    # Create 2 episodes, each 5 frames.
    bp = EnvBlueprint(simulator=ClassSpec(cls="test:TestSim", config={}))
    bp_hash = blueprint_hash(bp)
    episodes = []
    for ep_idx in range(2):
        T = 5
        obs_dim = 8
        rng = np.random.default_rng(ep_idx)
        ep = Episode(
            base_seed=42, episode_index=ep_idx, blueprint_hash=bp_hash,
            num_frames=T, episode_options={},
            agent_termination_proposal_records={"robot_a": (("timeout", T),)},
            observations={"robot_a": rng.standard_normal((T, obs_dim)).astype(np.float32)},
            actions={"robot_a": rng.uniform(-1, 1, (T, 3)).astype(np.float32)},
            action_extras={"robot_a": {}},
            explore_factors={"robot_a": np.zeros(T, dtype=np.float32)},
            observer_outputs={},
            final_observation={"robot_a": rng.standard_normal(obs_dim).astype(np.float32)},
            episode_metrics={},
        )
        episodes.append(ep)

    # Save episodes.
    from baseline.framework.rollout import EpisodeCollection
    coll = EpisodeCollection(blueprint=bp, episodes=episodes)
    coll.save(episodes_dir)

    # Create replay arrays (total 10 frames).
    total = 10
    np.savez(replay_dir / "buffer.npz",
             **{"explore_factor": np.zeros(total, dtype=np.float32)})
    np.savez(replay_dir / "gae.npz",
             **{"advantages.r_a": np.linspace(-1, 1, total, dtype=np.float32),
                "returns.r_a": np.full(total, 0.5, dtype=np.float32),
                "values.r_a": np.full(total, 0.3, dtype=np.float32)})
    np.savez(replay_dir / "combine.npz",
             **{"aw_frame.r_a": np.full(total, 1.0, dtype=np.float32),
                "aw_normed.r_a": np.full(total, 0.5, dtype=np.float32),
                "contribution.r_a": np.linspace(-0.1, 0.1, total, dtype=np.float32),
                "combined_adv": np.linspace(-0.05, 0.05, total, dtype=np.float32)})
    np.savez(replay_dir / "debug_arrays.npz",
             **{"h_left": np.ones(total, dtype=np.float32) * 0.1,
                "h_right": np.ones(total, dtype=np.float32) * 0.1})

    return snapshot_dir


# ---------------------------------------------------------------------------
# parse_frame_id tests
# ---------------------------------------------------------------------------

def test_parse_frame_id_valid():
    ep, agent, t = parse_frame_id("ep0003:robot_a:137")
    assert ep == 3
    assert agent == "robot_a"
    assert t == 137
    print("test_parse_frame_id_valid: PASS")


def test_parse_frame_id_malformed():
    with pytest.raises(ValueError):
        parse_frame_id("ep3:robot_a")
    with pytest.raises(ValueError):
        parse_frame_id("ep3:robot_a:abc")
    with pytest.raises(ValueError):
        parse_frame_id("xxx:robot_a:5")
    print("test_parse_frame_id_malformed: PASS")


# ---------------------------------------------------------------------------
# FrameIndex tests
# ---------------------------------------------------------------------------

def test_frame_index_mapping():
    """FrameIndex maps flat index ↔ frame ID."""
    idx = FrameIndex()
    idx.seg_episode = [0, 1]
    idx.seg_agent = ["robot_a", "robot_a"]
    idx.seg_t_start = [0, 0]
    idx.seg_length = [5, 5]
    idx.total_frames = 10

    # ep0:robot_a:0 → flat 0
    assert idx.flat_index(0, "robot_a", 0) == 0
    # ep0:robot_a:4 → flat 4
    assert idx.flat_index(0, "robot_a", 4) == 4
    # ep1:robot_a:0 → flat 5
    assert idx.flat_index(1, "robot_a", 0) == 5
    # ep1:robot_a:4 → flat 9
    assert idx.flat_index(1, "robot_a", 4) == 9

    # Reverse: flat 7 → ep1:robot_a:2
    assert idx.frame_id(7) == "ep0001:robot_a:2"
    print("test_frame_index_mapping: PASS")


def test_frame_index_not_found():
    idx = FrameIndex()
    idx.seg_episode = [0]
    idx.seg_agent = ["robot_a"]
    idx.seg_t_start = [0]
    idx.seg_length = [5]
    idx.total_frames = 5

    with pytest.raises(KeyError):
        idx.flat_index(99, "robot_a", 0)
    print("test_frame_index_not_found: PASS")


def test_build_frame_index(tmp_path):
    """build_frame_index from synthetic episodes."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    idx = build_frame_index(snapshot_dir)
    assert idx.total_frames == 10
    assert len(idx.seg_episode) == 2
    assert idx.seg_episode == [0, 1]
    print("test_build_frame_index: PASS")


# ---------------------------------------------------------------------------
# inspect_frame tests
# ---------------------------------------------------------------------------

def test_inspect_frame(tmp_path):
    """inspect_frame extracts per-frame values."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    data = inspect_frame(snapshot_dir, "ep0001:robot_a:2")
    assert data.episode_index == 1
    assert data.agent_id == "robot_a"
    assert data.t == 2
    assert data.flat_index == 7  # ep1 starts at flat 5, t=2 → flat 7
    # Should have values from replay arrays.
    assert "combine.combined_adv" in data.values
    assert "debug.h_left" in data.values
    print("test_inspect_frame: PASS")


def test_inspect_frame_observation(tmp_path):
    """inspect_frame includes observation from episode."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    data = inspect_frame(snapshot_dir, "ep0000:robot_a:0")
    assert data.observation is not None
    assert data.observation.shape == (8,)
    print("test_inspect_frame_observation: PASS")


# ---------------------------------------------------------------------------
# filter_frames tests
# ---------------------------------------------------------------------------

def test_filter_frames(tmp_path):
    """filter_frames returns matching frames."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    # combined_adv goes from -0.05 to +0.05.  Filter for > 0.
    frames = filter_frames(snapshot_dir, "combine.combined_adv > 0", limit=20)
    # About half the frames should match (the positive half).
    assert len(frames) > 0
    assert len(frames) <= 10
    # All should have positive combined_adv.
    for f in frames:
        assert f.values["combine.combined_adv"] > 0
    print("test_filter_frames: PASS")


def test_filter_frames_limit(tmp_path):
    """filter_frames respects limit."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    frames = filter_frames(snapshot_dir, "combine.aw_normed.r_a > 0", limit=3)
    assert len(frames) == 3
    print("test_filter_frames_limit: PASS")


def test_filter_frames_no_match(tmp_path):
    """filter_frames returns empty list when no match."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    frames = filter_frames(snapshot_dir, "combine.combined_adv > 100", limit=20)
    assert len(frames) == 0
    print("test_filter_frames_no_match: PASS")


# ---------------------------------------------------------------------------
# Rendering tests
# ---------------------------------------------------------------------------

def test_render_frame(tmp_path):
    """render_frame produces human-readable output."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    data = inspect_frame(snapshot_dir, "ep0000:robot_a:0")
    text = render_frame(data)
    assert "ep0000:robot_a:0" in text
    assert "combined_adv" in text
    print("test_render_frame: PASS")


def test_render_frame_list(tmp_path):
    """render_frame_list produces a compact table."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    frames = filter_frames(snapshot_dir, "combine.combined_adv > 0", limit=5)
    text = render_frame_list(frames)
    assert "frame_id" in text
    assert len(frames) > 0
    print("test_render_frame_list: PASS")


def test_render_frame_list_empty():
    """render_frame_list with no frames."""
    text = render_frame_list([])
    assert "无匹配" in text
    print("test_render_frame_list_empty: PASS")


# ---------------------------------------------------------------------------
# frame() public API tests
# ---------------------------------------------------------------------------

def test_frame_with_id(tmp_path):
    """frame() with --id returns rendered frame."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    text = frame(snapshot_dir, frame_id="ep0000:robot_a:0")
    assert "ep0000:robot_a:0" in text
    print("test_frame_with_id: PASS")


def test_frame_with_where(tmp_path):
    """frame() with --where returns rendered frame list."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    text = frame(snapshot_dir, where="combine.combined_adv > 0", limit=5)
    assert "frame_id" in text
    print("test_frame_with_where: PASS")


def test_frame_requires_id_or_where(tmp_path):
    """frame() raises when neither --id nor --where is given."""
    snapshot_dir = _make_synthetic_snapshot(tmp_path)
    with pytest.raises(ValueError):
        frame(snapshot_dir)
    print("test_frame_requires_id_or_where: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_parse_frame_id_valid()
    test_parse_frame_id_malformed()
    test_frame_index_mapping()
    test_frame_index_not_found()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_build_frame_index(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_inspect_frame(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_inspect_frame_observation(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_filter_frames(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_filter_frames_limit(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_filter_frames_no_match(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_render_frame(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_render_frame_list(Path(td))
    test_render_frame_list_empty()
    with tempfile.TemporaryDirectory() as td:
        test_frame_with_id(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_frame_with_where(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_frame_requires_id_or_where(Path(td))
    print("\nAll S3 frame tests passed.")
