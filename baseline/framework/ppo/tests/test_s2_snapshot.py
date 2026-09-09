"""S2 snapshot capture tests.

Tests cover:
- poll_request returns None when no sentinel.
- poll_request reads + atomic rename to u{u:05d}/request.json.
- poll_request rejects missing hypothesis.
- capture_snapshot writes manifest.json + episodes/ + actor/ + critics.pt + rng_state.pt.
- Subset selection: first N episodes, deterministic.
- --episodes all saves all episodes.
- critics.pt reloadable.
- rng_state.pt reloadable + restores RNG.

Conventions follow test_trainer.py / test_s1_provenance.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.debug.snapshot import (
    DebugRequest,
    poll_request,
    capture_snapshot,
    SENTINEL_FILENAME,
)
from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory, TrajectoryProvenance,
)
from baseline.framework.rollout import Episode, EpisodeCollection, blueprint_hash
from envs.framework.blueprint import EnvBlueprint, ClassSpec

from baseline.framework.ppo.experiment import ActorEval
import math


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class SimpleActor(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def evaluate_actions(self, obs, actions, explore_factor, *, want_stats=False):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        uncertainty_raw = dist.entropy().sum(dim=-1)
        H_max = 3 * (0.5 * math.log(2 * math.pi * math.e) + 1.0)
        H_min = 3 * (0.5 * math.log(2 * math.pi * math.e) + (-4.0))
        uncertainty_norm = (uncertainty_raw - H_min) / (H_max - H_min)
        return ActorEval(log_prob=log_prob, uncertainty=uncertainty_norm, stats=None)

    def to_blueprint(self, dest_path, *, stochastic=False):
        raise NotImplementedError


class SimpleCritic(nn.Module):
    def __init__(self, obs_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, obs):
        return self.net(obs)


def _make_episode(episode_index=0, T=5, obs_dim=8, action_dim=3, rng=None):
    """Build a minimal Episode for testing."""
    if rng is None:
        rng = np.random.default_rng()
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    rewards = rng.standard_normal(T).astype(np.float32)

    # Minimal EnvBlueprint for EpisodeCollection.
    bp = EnvBlueprint(
        simulator=ClassSpec(cls="test:TestSim", config={"obs_dim": obs_dim}),
    )
    bp_hash = blueprint_hash(bp)

    return Episode(
        base_seed=42,
        episode_index=episode_index,
        blueprint_hash=bp_hash,
        num_frames=T,
        episode_options={},
        agent_termination_proposal_records={"robot_a": (("timeout", T),)},
        observations={"robot_a": obs},
        actions={"robot_a": actions},
        action_extras={"robot_a": {}},
        explore_factors={"robot_a": np.zeros(T, dtype=np.float32)},
        observer_outputs={},
        final_observation={"robot_a": rng.standard_normal(obs_dim).astype(np.float32)},
        episode_metrics={},
    )


def _make_env_blueprint():
    return EnvBlueprint(
        simulator=ClassSpec(cls="test:TestSim", config={"obs_dim": 8}),
    )


# ---------------------------------------------------------------------------
# DebugRequest
# ---------------------------------------------------------------------------

def test_debug_request_valid():
    req = DebugRequest(hypothesis="test why KL is high")
    assert req.hypothesis == "test why KL is high"
    assert req.episodes_mode == "subset"
    assert req.episodes_n == 8
    assert req.include_full_grad is False
    print("test_debug_request_valid: PASS")


def test_debug_request_empty_hypothesis_rejected():
    with pytest.raises(ValueError):
        DebugRequest(hypothesis="")
    with pytest.raises(ValueError):
        DebugRequest(hypothesis="   ")
    print("test_debug_request_empty_hypothesis_rejected: PASS")


def test_debug_request_all_mode():
    req = DebugRequest(hypothesis="test", episodes_mode="all")
    assert req.episodes_mode == "all"
    print("test_debug_request_all_mode: PASS")


def test_debug_request_invalid_mode_rejected():
    with pytest.raises(ValueError):
        DebugRequest(hypothesis="test", episodes_mode="invalid")
    print("test_debug_request_invalid_mode_rejected: PASS")


# ---------------------------------------------------------------------------
# poll_request
# ---------------------------------------------------------------------------

def test_poll_request_no_sentinel(tmp_path):
    """poll_request returns None when no sentinel file."""
    result = poll_request(tmp_path, update=1)
    assert result is None
    print("test_poll_request_no_sentinel: PASS")


def test_poll_request_reads_and_moves(tmp_path):
    """poll_request reads sentinel and moves it to u{u:05d}/request.json."""
    sentinel = tmp_path / SENTINEL_FILENAME
    with open(sentinel, "w") as f:
        json.dump({
            "hypothesis": "test hypothesis",
            "episodes_mode": "subset",
            "episodes_n": 4,
            "include_full_grad": False,
        }, f)

    result = poll_request(tmp_path, update=5)
    assert result is not None
    assert result.hypothesis == "test hypothesis"
    assert result.episodes_n == 4

    # Sentinel should be gone.
    assert not sentinel.exists()
    # Request should be moved to debug/u00005/request.json.
    moved = tmp_path / "debug" / "u00005" / "request.json"
    assert moved.exists()

    print("test_poll_request_reads_and_moves: PASS")


def test_poll_request_rejects_missing_hypothesis(tmp_path):
    """poll_request rejects sentinel without hypothesis."""
    sentinel = tmp_path / SENTINEL_FILENAME
    with open(sentinel, "w") as f:
        json.dump({"episodes_mode": "all"}, f)

    result = poll_request(tmp_path, update=1)
    assert result is None  # rejected
    # Sentinel should be moved to rejected/.
    assert not sentinel.exists()
    rejected_dir = tmp_path / "debug" / "rejected"
    assert rejected_dir.exists()
    assert any(rejected_dir.iterdir())

    print("test_poll_request_rejects_missing_hypothesis: PASS")


def test_poll_request_corrupt_sentinel(tmp_path):
    """poll_request ignores corrupt sentinel (doesn't crash training)."""
    sentinel = tmp_path / SENTINEL_FILENAME
    sentinel.write_text("not json {{{")

    result = poll_request(tmp_path, update=1)
    assert result is None
    # Sentinel should still exist (corrupt file left for user to inspect).
    assert sentinel.exists()

    print("test_poll_request_corrupt_sentinel: PASS")


# ---------------------------------------------------------------------------
# capture_snapshot
# ---------------------------------------------------------------------------

def _setup_snapshot_dir(run_dir, update, request):
    """Create the request.json in the snapshot dir (simulates poll_request)."""
    snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(snapshot_dir / "request.json", "w") as f:
        json.dump({
            "hypothesis": request.hypothesis,
            "episodes_mode": request.episodes_mode,
            "episodes_n": request.episodes_n,
            "include_full_grad": request.include_full_grad,
        }, f)
    return snapshot_dir


def test_capture_snapshot_writes_all_files(tmp_path):
    """capture_snapshot writes manifest + episodes + actor + critics + rng."""
    episodes = [_make_episode(i, T=5, rng=np.random.default_rng(i)) for i in range(3)]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    req = DebugRequest(hypothesis="test snapshot", episodes_mode="subset", episodes_n=2)
    _setup_snapshot_dir(tmp_path, 1, req)
    # Create a fake actor export dir.
    export_dir = tmp_path / "policy_exports" / "u00001"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=1, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp",
        env_blueprint=bp,
    )

    assert (snapshot_dir / "manifest.json").exists()
    assert (snapshot_dir / "episodes").exists()
    assert (snapshot_dir / "actor").exists()
    assert (snapshot_dir / "actor" / "blueprint.yaml").exists()
    assert (snapshot_dir / "critics.pt").exists()
    assert (snapshot_dir / "actor_state.pt").exists()
    assert (snapshot_dir / "rng_state.pt").exists()
    assert (snapshot_dir / "replay").exists()
    assert (snapshot_dir / "request.json").exists()

    print("test_capture_snapshot_writes_all_files: PASS")


def test_capture_snapshot_subset_deterministic(tmp_path):
    """Subset mode saves first N episodes (deterministic)."""
    episodes = [_make_episode(i, T=3, rng=np.random.default_rng(i)) for i in range(5)]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    req = DebugRequest(hypothesis="test", episodes_mode="subset", episodes_n=2)
    _setup_snapshot_dir(tmp_path, 1, req)
    export_dir = tmp_path / "policy_exports" / "u00001"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=1, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp", env_blueprint=bp,
    )

    coll = EpisodeCollection.load(snapshot_dir / "episodes")
    loaded = list(coll)
    assert len(loaded) == 2
    # First 2 episodes (indices 0, 1).
    assert loaded[0].episode_index == 0
    assert loaded[1].episode_index == 1

    print("test_capture_snapshot_subset_deterministic: PASS")


def test_capture_snapshot_all_episodes(tmp_path):
    """--episodes all saves all episodes."""
    episodes = [_make_episode(i, T=3, rng=np.random.default_rng(i)) for i in range(4)]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    req = DebugRequest(hypothesis="test", episodes_mode="all")
    _setup_snapshot_dir(tmp_path, 1, req)
    export_dir = tmp_path / "policy_exports" / "u00001"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=1, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp", env_blueprint=bp,
    )

    coll = EpisodeCollection.load(snapshot_dir / "episodes")
    loaded = list(coll)
    assert len(loaded) == 4

    print("test_capture_snapshot_all_episodes: PASS")


def test_capture_snapshot_critics_reloadable(tmp_path):
    """critics.pt is reloadable with torch.load."""
    episodes = [_make_episode(0, T=3, rng=np.random.default_rng(0))]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    req = DebugRequest(hypothesis="test", episodes_mode="subset", episodes_n=1)
    _setup_snapshot_dir(tmp_path, 1, req)
    export_dir = tmp_path / "policy_exports" / "u00001"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=1, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp", env_blueprint=bp,
    )

    state = torch.load(snapshot_dir / "critics.pt", map_location="cpu", weights_only=False)
    assert "r_a" in state
    # Verify it can be loaded into a fresh critic.
    fresh = SimpleCritic()
    fresh.load_state_dict(state["r_a"])

    print("test_capture_snapshot_critics_reloadable: PASS")


def test_capture_snapshot_rng_state_reloadable(tmp_path):
    """rng_state.pt restores RNG state."""
    episodes = [_make_episode(0, T=3, rng=np.random.default_rng(0))]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    # Set a known RNG state, then capture snapshot (which saves RNG state).
    # Then generate random numbers, restore, and verify they match.
    torch.manual_seed(123)
    np.random.seed(123)

    req = DebugRequest(hypothesis="test", episodes_mode="subset", episodes_n=1)
    _setup_snapshot_dir(tmp_path, 1, req)
    export_dir = tmp_path / "policy_exports" / "u00001"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=1, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp", env_blueprint=bp,
    )

    # Generate random numbers AFTER snapshot capture.
    expected_torch = torch.randn(3)

    # Restore RNG state from snapshot (should be the state before expected_torch).
    rng_state = torch.load(snapshot_dir / "rng_state.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(rng_state["torch_cpu"])
    np.random.set_state(rng_state["numpy"])
    restored = torch.randn(3)

    assert torch.allclose(expected_torch, restored, atol=1e-6)

    print("test_capture_snapshot_rng_state_reloadable: PASS")


def test_capture_snapshot_manifest_contents(tmp_path):
    """manifest.json has correct fields."""
    episodes = [_make_episode(0, T=3, rng=np.random.default_rng(0))]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    req = DebugRequest(hypothesis="test manifest", episodes_mode="all", include_full_grad=True)
    _setup_snapshot_dir(tmp_path, 42, req)
    export_dir = tmp_path / "policy_exports" / "u00042"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=tmp_path, update=42, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="test_exp", env_blueprint=bp,
    )

    with open(snapshot_dir / "manifest.json") as f:
        manifest = json.load(f)

    assert manifest["update"] == 42
    assert manifest["experiment_name"] == "test_exp"
    assert manifest["episodes_mode"] == "all"
    assert manifest["episodes_n"] == 1
    assert manifest["include_full_grad"] is True
    assert manifest["hypothesis"] == "test manifest"

    print("test_capture_snapshot_manifest_contents: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    def _mk(p):
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    test_debug_request_valid()
    test_debug_request_empty_hypothesis_rejected()
    test_debug_request_all_mode()
    test_debug_request_invalid_mode_rejected()
    test_poll_request_no_sentinel(_mk("/tmp/test_s2_snap_none"))
    test_poll_request_reads_and_moves(_mk("/tmp/test_s2_snap_move"))
    test_poll_request_rejects_missing_hypothesis(_mk("/tmp/test_s2_snap_reject"))
    test_poll_request_corrupt_sentinel(_mk("/tmp/test_s2_snap_corrupt"))
    test_capture_snapshot_writes_all_files(_mk("/tmp/test_s2_snap_write"))
    test_capture_snapshot_subset_deterministic(_mk("/tmp/test_s2_snap_subset"))
    test_capture_snapshot_all_episodes(_mk("/tmp/test_s2_snap_all"))
    test_capture_snapshot_critics_reloadable(_mk("/tmp/test_s2_snap_critics"))
    test_capture_snapshot_rng_state_reloadable(_mk("/tmp/test_s2_snap_rng"))
    test_capture_snapshot_manifest_contents(_mk("/tmp/test_s2_snap_manifest"))
    print("\nAll S2 snapshot tests passed.")
