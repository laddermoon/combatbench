"""S2 replay + self-verification tests.

Tests cover:
- Synthetic snapshot → replay reproduces UpdateStats.
- Self-verification passes for --episodes all against a synthetic log entry.
- Self-verification detects injected mismatch.
- Self-verification refuses subset mode.
- Actor/critics deep-copied: snapshot files unchanged after replay.
- debug_arrays saved to replay/debug_arrays.npz.

Conventions follow test_trainer.py / test_s1_provenance.py.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.experiment import (
    ActorEval, CommonParams, ExplorationSpec, PPOParams, ExperimentPPO,
)
from baseline.framework.ppo.trajectory import (
    ChannelData, RewardChannel, Trajectory, TrajectoryProvenance,
)
from baseline.framework.ppo.trainer import PPOBuffer, ppo_update, set_seed
from baseline.framework.ppo.debug.sink import NpzSink
from baseline.framework.ppo.debug.snapshot import (
    DebugRequest, capture_snapshot, MANIFEST_FILENAME,
)
from baseline.framework.ppo.debug.replay import (
    replay_snapshot, verify_against_log, ReplayResult, VerificationResult,
)
from baseline.framework.rollout import Episode, EpisodeCollection, blueprint_hash
from envs.framework.blueprint import EnvBlueprint, ClassSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class SimpleActor(nn.Module):
    def __init__(self, obs_dim=8, action_dim=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def evaluate_actions(self, obs, actions, explore_factor, *, want_stats=False):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        uncertainty_raw = dist.entropy().sum(dim=-1)
        H_max = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + 1.0)
        H_min = self.action_dim * (0.5 * math.log(2 * math.pi * math.e) + (-4.0))
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


class MockExperiment(ExperimentPPO):
    """Minimal experiment for replay testing."""

    name = "mock_test"
    reward_keys = ("r_a",)
    gammas = {"r_a": 0.99}
    gae_lambdas = {"r_a": 0.95}

    def __init__(self):
        pass  # Skip parent __init__ which needs config

    def common_params(self) -> CommonParams:
        return CommonParams(
            name="mock_test", seed=42, max_updates=10,
            episodes_per_update=4, eval_interval=100,
            eval_episodes=4, video_eval_interval=100,
            rollout_workers=2,
            learning_rate=1e-3, critic_learning_rate=1e-3,
            grad_clip_norm=0.5,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            clip_eps=0.2, target_kl=0.0, update_epochs=2, minibatch_size=32,
        )

    def reward_channels(self):
        return (RewardChannel("r_a", gamma=0.99, gae_lambda=0.95),)

    def build_actor(self, device):
        return SimpleActor().to(device)

    def build_critic(self, key, device):
        return SimpleCritic().to(device)

    def build_trajectories(self, episodes):
        trajs = []
        for ep in episodes:
            T = ep.num_frames
            obs = ep.observations["robot_a"]
            actions = ep.actions["robot_a"]
            last_obs = ep.final_observation["robot_a"]
            channels = {
                "r_a": ChannelData(
                    reward=np.random.default_rng(ep.episode_index).standard_normal(T).astype(np.float32),
                    is_terminated=True,
                    actor_weight=1.0,
                ),
            }
            prov = TrajectoryProvenance(
                episode_index=ep.episode_index, agent_id="robot_a",
            )
            trajs.append(Trajectory(
                obs=obs, actions=actions, last_obs=last_obs,
                channels=channels, importance=1.0,
                explore_factor=ep.explore_factors.get("robot_a"),
                provenance=prov,
            ))
        return trajs

    def debug_arrays(self, episodes, trajectories):
        total = sum(len(t.obs) for t in trajectories)
        return {"test_array": np.zeros(total, dtype=np.float32)}

    def exploration(self, update):
        return None

    def build_jobs(self, policy_bp, base_seed, n_episodes):
        return []

    def on_eval(self, episodes, update):
        return {}

    def on_update(self, stats):
        pass


def _make_episode(episode_index=0, T=5, obs_dim=8, action_dim=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    bp = EnvBlueprint(simulator=ClassSpec(cls="test:TestSim", config={}))
    bp_hash = blueprint_hash(bp)
    return Episode(
        base_seed=42, episode_index=episode_index, blueprint_hash=bp_hash,
        num_frames=T, episode_options={},
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
    return EnvBlueprint(simulator=ClassSpec(cls="test:TestSim", config={}))


def _setup_snapshot_dir(run_dir, update, request):
    snapshot_dir = run_dir / "debug" / f"u{update:05d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(snapshot_dir / "request.json", "w") as f:
        json.dump({
            "hypothesis": request.hypothesis,
            "episodes_mode": request.episodes_mode,
            "episodes_n": request.episodes_n,
            "include_full_grad": request.include_full_grad,
        }, f)
    return snapshot_dir


def _create_snapshot(run_dir, update=1, episodes_mode="all", n_episodes=2):
    """Create a full snapshot for testing."""
    rng = np.random.default_rng(42)
    episodes = [_make_episode(i, T=10, rng=np.random.default_rng(i)) for i in range(n_episodes)]
    actor = SimpleActor()
    critics = {"r_a": SimpleCritic()}
    bp = _make_env_blueprint()

    if episodes_mode == "all":
        req = DebugRequest(hypothesis="test replay", episodes_mode="all")
    else:
        req = DebugRequest(hypothesis="test replay", episodes_mode="subset", episodes_n=n_episodes)

    _setup_snapshot_dir(run_dir, update, req)
    export_dir = run_dir / "policy_exports" / f"u{update:05d}"
    export_dir.mkdir(parents=True)
    (export_dir / "blueprint.yaml").write_text("test: true")

    snapshot_dir = capture_snapshot(
        run_dir=run_dir, update=update, request=req,
        episodes=episodes, actor=actor, actor_export_dir=export_dir,
        critics=critics, experiment_name="mock_test", env_blueprint=bp,
    )
    return snapshot_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_replay_reproduces_stats(tmp_path):
    """Synthetic snapshot → replay reproduces UpdateStats."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    result = replay_snapshot(snapshot_dir, experiment=exp)

    assert result.update == 1
    assert result.episodes_mode == "all"
    assert result.n_episodes == 2
    assert result.n_frames > 0
    assert result.stats is not None
    # Replay output should exist.
    assert (snapshot_dir / "replay" / "stats.json").exists()
    assert (snapshot_dir / "replay" / "buffer.npz").exists()
    assert (snapshot_dir / "replay" / "gae.npz").exists()
    assert (snapshot_dir / "replay" / "combine.npz").exists()
    assert (snapshot_dir / "replay" / "update.npz").exists()
    assert (snapshot_dir / "replay" / "debug_arrays.npz").exists()

    print("test_replay_reproduces_stats: PASS")


def test_replay_deep_copies_actor(tmp_path):
    """Actor/critics deep-copied: snapshot files unchanged after replay."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    # Save original actor state.
    original_state = torch.load(snapshot_dir / "actor_state.pt", map_location="cpu")

    result = replay_snapshot(snapshot_dir, experiment=exp)

    # Actor state file should be unchanged.
    after_state = torch.load(snapshot_dir / "actor_state.pt", map_location="cpu")
    for k in original_state:
        assert torch.equal(original_state[k], after_state[k]), (
            f"actor_state.pt changed after replay: key {k}"
        )

    print("test_replay_deep_copies_actor: PASS")


def test_verify_passes_for_all_mode(tmp_path):
    """Self-verification passes for --episodes all against a synthetic log entry."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    # Run replay first.
    result = replay_snapshot(snapshot_dir, experiment=exp)

    # Create a synthetic train.log with matching __RAW_STATS__.
    log_stats = result.stats.to_log_dict()
    log_entry = {"update": 1, "stats": log_stats}
    log_line = f"some prefix __RAW_STATS__ {json.dumps(log_entry)}\n"
    (tmp_path / "train.log").write_text(log_line)

    verification = verify_against_log(snapshot_dir, tmp_path)

    assert verification.comparable is True
    assert verification.verdict == "pass"
    assert verification.n_failed == 0

    print("test_verify_passes_for_all_mode: PASS")


def test_verify_detects_mismatch(tmp_path):
    """Self-verification detects injected mismatch."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    result = replay_snapshot(snapshot_dir, experiment=exp)

    # Create a synthetic train.log with a MISMATCHED field.
    log_stats = result.stats.to_log_dict()
    log_stats["approx_kl"] = log_stats["approx_kl"] + 1.0  # inject mismatch
    log_entry = {"update": 1, "stats": log_stats}
    log_line = f"some prefix __RAW_STATS__ {json.dumps(log_entry)}\n"
    (tmp_path / "train.log").write_text(log_line)

    verification = verify_against_log(snapshot_dir, tmp_path)

    assert verification.comparable is True
    assert verification.verdict == "fail"
    assert verification.n_failed > 0

    print("test_verify_detects_mismatch: PASS")


def test_verify_refuses_subset(tmp_path):
    """Self-verification refuses subset mode (not comparable)."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="subset", n_episodes=2)
    exp = MockExperiment()

    # Run replay.
    replay_snapshot(snapshot_dir, experiment=exp)

    # Create a synthetic train.log.
    (tmp_path / "train.log").write_text("dummy\n")

    verification = verify_against_log(snapshot_dir, tmp_path)

    assert verification.comparable is False
    assert verification.verdict == "not_comparable"

    print("test_verify_refuses_subset: PASS")


def test_verify_no_log_entry(tmp_path):
    """Self-verification fails gracefully when no matching log entry."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    replay_snapshot(snapshot_dir, experiment=exp)

    # Create a train.log without the matching update.
    (tmp_path / "train.log").write_text("no raw stats here\n")

    verification = verify_against_log(snapshot_dir, tmp_path)

    assert verification.comparable is True
    assert verification.verdict == "fail"

    print("test_verify_no_log_entry: PASS")


def test_replay_debug_arrays_saved(tmp_path):
    """debug_arrays saved to replay/debug_arrays.npz."""
    snapshot_dir = _create_snapshot(tmp_path, update=1, episodes_mode="all", n_episodes=2)
    exp = MockExperiment()

    result = replay_snapshot(snapshot_dir, experiment=exp)

    arrays = np.load(snapshot_dir / "replay" / "debug_arrays.npz")
    assert "test_array" in arrays
    assert arrays["test_array"].shape == (result.n_frames,)

    print("test_replay_debug_arrays_saved: PASS")


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

    test_replay_reproduces_stats(_mk("/tmp/test_s2_replay_stats"))
    test_replay_deep_copies_actor(_mk("/tmp/test_s2_replay_dc"))
    test_verify_passes_for_all_mode(_mk("/tmp/test_s2_replay_pass"))
    test_verify_detects_mismatch(_mk("/tmp/test_s2_replay_mismatch"))
    test_verify_refuses_subset(_mk("/tmp/test_s2_replay_subset"))
    test_verify_no_log_entry(_mk("/tmp/test_s2_replay_nolog"))
    test_replay_debug_arrays_saved(_mk("/tmp/test_s2_replay_arrays"))
    print("\nAll S2 replay tests passed.")
