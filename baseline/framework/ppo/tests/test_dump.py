"""Tests for the dump capture system (CLI + sentinel + capture).

Tests cover:
- DumpRequest: hypothesis required, frozen, defaults.
- poll_dump_request: no sentinel → None, valid → parsed + moved,
  corrupt → rejected, missing hypothesis → rejected.
- capture_dump: writes all NPZ files, frame_id present, manifest correct,
  RECORD_GUIDE.md contains round_runner command.
- CLI: dump subcommand writes sentinel, rejects empty hypothesis,
  rejects missing run_dir, refuses existing sentinel.
- ppo_update with dump_callback: callback receives correct stages,
  no callback → unchanged (regression).

Conventions follow test_trainer.py:
- Simple print("test_xxx: PASS") at the end of each test.
- `if __name__ == "__main__":` block runs all tests sequentially.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.dumpkit.dump_request import (
    DumpRequest,
    poll_dump_request,
    SENTINEL_FILENAME,
)
from baseline.framework.ppo.dumpkit.dump_capture import capture_dump
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


# ---------------------------------------------------------------------------
# DumpRequest tests
# ---------------------------------------------------------------------------

def test_dump_request_requires_hypothesis():
    """DumpRequest rejects empty/whitespace hypothesis."""
    try:
        DumpRequest(hypothesis="")
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        DumpRequest(hypothesis="   ")
        assert False, "should have raised"
    except ValueError:
        pass
    # Valid
    req = DumpRequest(hypothesis="test why KL is high")
    assert req.hypothesis == "test why KL is high"
    assert req.include_full_grad is False
    print("test_dump_request_requires_hypothesis: PASS")


def test_dump_request_frozen():
    """DumpRequest is frozen."""
    req = DumpRequest(hypothesis="test")
    try:
        req.hypothesis = "changed"
        assert False, "should have raised"
    except (AttributeError, Exception):
        pass
    print("test_dump_request_frozen: PASS")


# ---------------------------------------------------------------------------
# poll_dump_request tests
# ---------------------------------------------------------------------------

def test_poll_no_sentinel():
    """poll returns None when no sentinel exists."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = poll_dump_request(Path(d), update=0)
        assert result is None
    print("test_poll_no_sentinel: PASS")


def test_poll_valid_sentinel():
    """poll parses valid sentinel and moves it to dump dir."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        sentinel = run_dir / SENTINEL_FILENAME
        sentinel.write_text(json.dumps({
            "hypothesis": "why is KL high",
            "include_full_grad": True,
        }))
        result = poll_dump_request(run_dir, update=5)
        assert result is not None
        assert result.hypothesis == "why is KL high"
        assert result.include_full_grad is True
        # Sentinel moved
        assert not sentinel.exists()
        # Moved to dump dir
        moved = run_dir / "dumps" / "u00005" / "request.json"
        assert moved.exists()
    print("test_poll_valid_sentinel: PASS")


def test_poll_corrupt_sentinel():
    """poll returns None for corrupt JSON and leaves sentinel for inspection."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        sentinel = run_dir / SENTINEL_FILENAME
        sentinel.write_text("{not valid json")
        result = poll_dump_request(run_dir, update=0)
        assert result is None
        # Sentinel still exists (left for user to inspect)
        assert sentinel.exists()
    print("test_poll_corrupt_sentinel: PASS")


def test_poll_missing_hypothesis():
    """poll returns None when hypothesis is missing and moves sentinel aside."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        sentinel = run_dir / SENTINEL_FILENAME
        sentinel.write_text(json.dumps({"include_full_grad": True}))
        result = poll_dump_request(run_dir, update=0)
        assert result is None
        # Sentinel moved to rejected dir
        assert not sentinel.exists()
        rejected_dir = run_dir / "dumps" / "rejected"
        assert rejected_dir.exists()
    print("test_poll_missing_hypothesis: PASS")


def test_poll_one_shot():
    """poll is one-shot: second poll returns None after first consumed."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        sentinel = run_dir / SENTINEL_FILENAME
        sentinel.write_text(json.dumps({"hypothesis": "test"}))
        r1 = poll_dump_request(run_dir, update=0)
        assert r1 is not None
        r2 = poll_dump_request(run_dir, update=0)
        assert r2 is None
    print("test_poll_one_shot: PASS")


# ---------------------------------------------------------------------------
# capture_dump tests
# ---------------------------------------------------------------------------

def _make_fake_episode(T=10, obs_dim=8, action_dim=4, ep_idx=0, base_seed=42):
    """Build a minimal Episode for testing."""
    rng = np.random.default_rng(base_seed + ep_idx)
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.uniform(-0.9, 0.9, (T, action_dim)).astype(np.float32)
    final_obs = rng.standard_normal(obs_dim).astype(np.float32)
    explore_factors = rng.uniform(0.1, 0.5, T).astype(np.float32)
    return Episode(
        base_seed=base_seed + ep_idx,
        episode_index=ep_idx,
        blueprint_hash="test_hash",
        num_frames=T,
        episode_options={"initial_distance": 2.0},
        agent_termination_proposal_records={
            "robot_a": (("timeout", T),),
            "robot_b": (("timeout", T),),
        },
        observations={"robot_a": obs, "robot_b": obs.copy()},
        actions={"robot_a": actions, "robot_b": actions.copy()},
        action_extras={"robot_a": {}, "robot_b": {}},
        explore_factors={"robot_a": explore_factors, "robot_b": explore_factors.copy()},
        observer_outputs={
            "foot_state_a": {"value": rng.standard_normal(T).astype(np.float32)},
            "foot_state_b": {"value": rng.standard_normal(T).astype(np.float32)},
        },
        final_observation={"robot_a": final_obs, "robot_b": final_obs.copy()},
        episode_metrics={},
    )


def _make_fake_job(T=10, obs_dim=8, action_dim=4, seed=42):
    """Build a minimal Job for testing (with a fake env_bp)."""
    from envs.framework.blueprint import EnvBlueprint, ClassSpec
    from envs.framework.policy import PolicyBlueprint
    env_bp = EnvBlueprint(
        simulator=ClassSpec(cls="test:FakeSim", config={}),
        plugins=(),
        observer_plugins={},
    )
    # Minimal policy blueprint
    policy_bp = PolicyBlueprint(
        cls="test:FakePolicy",
        config={},
    )
    return Job(
        policy_a_bp=policy_bp,
        policy_b_bp=policy_bp,
        env_bp=env_bp,
        seed=seed,
        episode_options={"initial_distance": 2.0},
        stochastic=True,
    )


def test_capture_dump_writes_all_files():
    """capture_dump writes all expected files with frame_id correlation."""
    import tempfile
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
    buf, actor = make_buffer(trajs, obs_dim, action_dim, reward_keys)
    critics = make_critics(reward_keys, obs_dim)
    actor_opt, critic_opts = make_optimizers(actor, critics)

    pp = make_pp_params()
    stats = ppo_update(
        actor=actor,
        critics=critics,
        actor_optimizer=actor_opt,
        critic_optimizers=critic_opts,
        buf=buf,
        reward_channels=channels,
        pp=pp,
        grad_clip_norm=0.5,
        device=torch.device("cpu"),
    )

    # Collect dump data via callback
    dump_collector = {}

    def cb(stage, data):
        dump_collector[stage] = data

    # Rebuild buffer for second ppo_update (first consumed it)
    buf2, actor2 = make_buffer(trajs, obs_dim, action_dim, reward_keys)
    critics2 = make_critics(reward_keys, obs_dim)
    actor_opt2, critic_opts2 = make_optimizers(actor2, critics2)
    stats2 = ppo_update(
        actor=actor2,
        critics=critics2,
        actor_optimizer=actor_opt2,
        critic_optimizers=critic_opts2,
        buf=buf2,
        reward_channels=channels,
        pp=pp,
        grad_clip_norm=0.5,
        device=torch.device("cpu"),
        dump_callback=cb,
    )

    episodes = [_make_fake_episode(T=T, obs_dim=obs_dim, action_dim=action_dim)]
    jobs = [_make_fake_job(seed=42)]
    req = DumpRequest(hypothesis="test capture")

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        # Create config.json so it looks like a run dir
        (run_dir / "config.json").write_text(json.dumps({"name": "test_exp"}))
        # Create policy_exports dir (as training loop would)
        export_dir = run_dir / "policy_exports" / "u00001"
        export_dir.mkdir(parents=True)
        (export_dir / "policy_blueprint.yaml").write_text("test: true\n")

        dump_dir = capture_dump(
            run_dir=run_dir,
            update=1,
            request=req,
            episodes=episodes,
            trajectories=trajs,
            buf=buf,
            stats=stats2,
            jobs=jobs,
            dump_collector=dump_collector,
            experiment_name="test_exp",
        )

        # Check all files exist
        assert (dump_dir / "manifest.json").exists()
        assert (dump_dir / "env_blueprint.yaml").exists()
        assert (dump_dir / "episode_options.json").exists()
        assert (dump_dir / "episodes.npz").exists()
        assert (dump_dir / "trajectories.npz").exists()
        assert (dump_dir / "buffer.npz").exists()
        assert (dump_dir / "gae.npz").exists()
        assert (dump_dir / "combine.npz").exists()
        assert (dump_dir / "update.npz").exists()
        assert (dump_dir / "RECORD_GUIDE.md").exists()

        # Check manifest
        manifest = json.loads((dump_dir / "manifest.json").read_text())
        assert manifest["update"] == 1
        assert manifest["hypothesis"] == "test capture"
        assert manifest["experiment_name"] == "test_exp"

        # Check frame_id in buffer.npz
        buf_data = np.load(dump_dir / "buffer.npz", allow_pickle=True)
        assert "frame_id" in buf_data
        # Without provenance, frame_id is flat:{i}
        assert buf_data["frame_id"][0] == "flat:0"

        # Check frame_id in trajectories.npz
        traj_data = np.load(dump_dir / "trajectories.npz", allow_pickle=True)
        assert "frame_id" in traj_data

        # Check gae.npz has per-channel data
        gae_data = np.load(dump_dir / "gae.npz", allow_pickle=True)
        assert "advs_all" in gae_data
        assert "r_test" in gae_data["advs_all"].item()

        # Check combine.npz
        combine_data = np.load(dump_dir / "combine.npz", allow_pickle=True)
        assert "combined_adv" in combine_data

        # Check update.npz
        update_data = np.load(dump_dir / "update.npz", allow_pickle=True)
        assert "approx_kl" in update_data
        assert "grad_norm_actor_pre_clip" in update_data

        # Check RECORD_GUIDE.md contains round_runner command
        guide = (dump_dir / "RECORD_GUIDE.md").read_text()
        assert "round_runner" in guide
        assert "env_blueprint.yaml" in guide
        assert "--seed 42" in guide
        assert "--options-json" in guide

    print("test_capture_dump_writes_all_files: PASS")


# ---------------------------------------------------------------------------
# ppo_update dump_callback regression test
# ---------------------------------------------------------------------------

def test_ppo_update_no_callback_unchanged():
    """ppo_update with no dump_callback produces same stats structure."""
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
            channels={"r_test": make_channel_data(T, rng=rng)},
            importance=1.0,
        )
    ]
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
    )
    # Should produce valid stats
    assert hasattr(stats, "approx_kl")
    assert hasattr(stats, "policy_loss")
    assert hasattr(stats, "grad_norm_actor")
    print("test_ppo_update_no_callback_unchanged: PASS")


def test_ppo_update_callback_stages():
    """ppo_update with dump_callback invokes gae, combine, update stages."""
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
            channels={"r_test": make_channel_data(T, rng=rng)},
            importance=1.0,
        )
    ]
    buf, actor = make_buffer(trajs, obs_dim, action_dim, reward_keys)
    critics = make_critics(reward_keys, obs_dim)
    actor_opt, critic_opts = make_optimizers(actor, critics)

    stages_seen = []

    def cb(stage, data):
        stages_seen.append(stage)

    ppo_update(
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
    assert "gae" in stages_seen
    assert "combine" in stages_seen
    assert "update" in stages_seen
    print("test_ppo_update_callback_stages: PASS")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_dump_writes_sentinel():
    """CLI dump subcommand writes a valid sentinel file."""
    import tempfile
    from baseline.framework.ppo.debug import main
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "config.json").write_text(json.dumps({"name": "test"}))
        rc = main(["dump", str(run_dir), "--hypothesis", "test CLI"])
        assert rc == 0
        sentinel = run_dir / SENTINEL_FILENAME
        assert sentinel.exists()
        data = json.loads(sentinel.read_text())
        assert data["hypothesis"] == "test CLI"
    print("test_cli_dump_writes_sentinel: PASS")


def test_cli_dump_rejects_empty_hypothesis():
    """CLI rejects empty hypothesis."""
    import tempfile
    from baseline.framework.ppo.debug import main
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "config.json").write_text(json.dumps({"name": "test"}))
        rc = main(["dump", str(run_dir), "--hypothesis", ""])
        assert rc == 2
        assert not (run_dir / SENTINEL_FILENAME).exists()
    print("test_cli_dump_rejects_empty_hypothesis: PASS")


def test_cli_dump_rejects_missing_run_dir():
    """CLI rejects nonexistent run directory."""
    from baseline.framework.ppo.debug import main
    rc = main(["dump", "/nonexistent/path/xyz", "--hypothesis", "test"])
    assert rc == 2
    print("test_cli_dump_rejects_missing_run_dir: PASS")


def test_cli_dump_refuses_existing_sentinel():
    """CLI refuses when a sentinel already exists."""
    import tempfile
    from baseline.framework.ppo.debug import main
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "config.json").write_text(json.dumps({"name": "test"}))
        # Write an existing sentinel
        (run_dir / SENTINEL_FILENAME).write_text(json.dumps({"hypothesis": "old"}))
        rc = main(["dump", str(run_dir), "--hypothesis", "new"])
        assert rc == 3
        # Original sentinel unchanged
        data = json.loads((run_dir / SENTINEL_FILENAME).read_text())
        assert data["hypothesis"] == "old"
    print("test_cli_dump_refuses_existing_sentinel: PASS")


def test_cli_dump_full_grad_flag():
    """CLI --full-grad flag is written to sentinel."""
    import tempfile
    from baseline.framework.ppo.debug import main
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        (run_dir / "config.json").write_text(json.dumps({"name": "test"}))
        rc = main(["dump", str(run_dir), "--hypothesis", "test", "--full-grad"])
        assert rc == 0
        data = json.loads((run_dir / SENTINEL_FILENAME).read_text())
        assert data["include_full_grad"] is True
    print("test_cli_dump_full_grad_flag: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dump_request_requires_hypothesis()
    test_dump_request_frozen()
    test_poll_no_sentinel()
    test_poll_valid_sentinel()
    test_poll_corrupt_sentinel()
    test_poll_missing_hypothesis()
    test_poll_one_shot()
    test_capture_dump_writes_all_files()
    test_ppo_update_no_callback_unchanged()
    test_ppo_update_callback_stages()
    test_cli_dump_writes_sentinel()
    test_cli_dump_rejects_empty_hypothesis()
    test_cli_dump_rejects_missing_run_dir()
    test_cli_dump_refuses_existing_sentinel()
    test_cli_dump_full_grad_flag()
    print("\nAll dump tests passed!")
