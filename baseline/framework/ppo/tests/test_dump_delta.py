"""Tests for the policy-drift delta diagnostic (debug.py delta).

compute_delta replays a dumped episode's stored observations through the
deterministic act() of each exported policy generation and records
per-generation action vectors.  These tests cover:

- Self-play episodes: every trained agent gets an actions array.
- Non-self-play episodes: only agents with trajectories are analysed.
- Missing exports: recorded in meta, never silently dropped.
- gens bounds, episode bounds, empty traj_map: clear errors.
- ViewerAPI._episode_delta: JSON structure + unavailable fallback.

Conventions follow test_viewer.py: synthetic dump via capture_dump,
print("test_xxx: PASS"), __main__ runner.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.dumpkit.dump_capture import capture_dump
from baseline.framework.ppo.dumpkit.dump_delta import compute_delta, MAX_GENS
from baseline.framework.ppo.dumpkit.dump_request import DumpRequest
from baseline.framework.ppo.dumpkit.viewer.server import DumpData, ViewerAPI
from baseline.framework.ppo.experiment import PPOParams
from baseline.framework.ppo.policies import TruncatedNormalPolicy
from baseline.framework.ppo.trainer import ppo_update
from baseline.framework.ppo.trajectory import RewardChannel, Trajectory
from baseline.framework.rollout.episode import Episode

from baseline.framework.ppo.tests.test_trainer import (
    make_channel_data,
    make_buffer,
    make_critics,
    make_optimizers,
    make_pp_params,
)
from baseline.framework.ppo.tests.test_dump import _make_fake_job


OBS_DIM, ACT_DIM, T = 8, 4, 10
REWARD_KEYS = ("r_test",)
CHANNELS = (RewardChannel(name="r_test", gamma=0.99, gae_lambda=0.95),)


def _make_episode(agent_ids=("robot_a", "robot_b"), seed=7):
    """Episode with distinct per-agent obs so provenance is unambiguous."""
    rng = np.random.default_rng(seed)
    obs = {aid: rng.standard_normal((T, OBS_DIM)).astype(np.float32)
           for aid in agent_ids}
    return Episode(
        base_seed=seed,
        episode_index=0,
        blueprint_hash="delta_test",
        num_frames=T,
        episode_options={"initial_distance": 2.0},
        agent_termination_proposal_records={
            aid: (("timeout", T),) for aid in agent_ids
        },
        observations=obs,
        actions={
            aid: rng.uniform(-0.9, 0.9, (T, ACT_DIM)).astype(np.float32)
            for aid in agent_ids
        },
        action_extras={aid: {} for aid in agent_ids},
        explore_factors={
            aid: rng.uniform(0.1, 0.5, T).astype(np.float32)
            for aid in agent_ids
        },
        observer_outputs={},
        final_observation={
            aid: rng.standard_normal(OBS_DIM).astype(np.float32)
            for aid in agent_ids
        },
        episode_metrics={},
    )


def _traj_from_episode(ep: Episode, agent_id: str, seed=11):
    """Trajectory whose obs is the episode's obs for ``agent_id`` — so
    ``_make_frame_ids`` resolves provenance to (episode 0, agent_id, 0)."""
    rng = np.random.default_rng(seed)
    return Trajectory(
        obs=np.asarray(ep.observations[agent_id], dtype=np.float32),
        actions=rng.uniform(-0.9, 0.9, (T, ACT_DIM)).astype(np.float32),
        last_obs=np.asarray(
            ep.final_observation[agent_id], dtype=np.float32,
        ),
        channels={"r_test": make_channel_data(T, rng=rng)},
        importance=1.0,
    )


def _create_delta_run(tmpdir: Path, trained_agents=("robot_a", "robot_b"),
                      export_updates=(1, 2, 3, 4), update=4):
    """Build run_dir with a dump at u{update} + real policy exports.

    Returns (dump_dir, run_dir).
    """
    run_dir = tmpdir / "run"
    run_dir.mkdir()

    ep = _make_episode(agent_ids=("robot_a", "robot_b"))
    trajs = [_traj_from_episode(ep, aid) for aid in trained_agents]

    dump_collector = {}
    buf, actor = make_buffer(trajs, OBS_DIM, ACT_DIM, REWARD_KEYS)
    critics = make_critics(REWARD_KEYS, OBS_DIM)
    actor_opt, critic_opts = make_optimizers(actor, critics)
    stats = ppo_update(
        actor=actor,
        critics=critics,
        actor_optimizer=actor_opt,
        critic_optimizers=critic_opts,
        buf=buf,
        reward_channels=CHANNELS,
        pp=make_pp_params(),
        grad_clip_norm=0.5,
        device=torch.device("cpu"),
        dump_callback=lambda s, d: dump_collector.update({s: d}),
    )

    # Real policy exports — distinct weights per generation so Δ != 0.
    exports_root = run_dir / "policy_exports"
    for i, u in enumerate(export_updates):
        torch.manual_seed(1000 + u)
        pol = TruncatedNormalPolicy(
            obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=16,
        )
        pol.to_blueprint(dest_path=str(exports_root / f"u{u:05d}"))

    dump_dir = capture_dump(
        run_dir=run_dir,
        update=update,
        request=DumpRequest(hypothesis="delta test"),
        episodes=[ep],
        trajectories=trajs,
        buf=buf,
        stats=stats,
        jobs=[_make_fake_job()],
        dump_collector=dump_collector,
        experiment_name="delta_test",
    )
    return dump_dir, run_dir


# ---------------------------------------------------------------------------
# compute_delta
# ---------------------------------------------------------------------------

def test_delta_self_play_both_agents():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(Path(d))
        out_dir = compute_delta(dump_dir, episode_pos=0, gens=3, log=lambda *a: None)

        npz = np.load(out_dir / "delta.npz")
        meta = json.loads((out_dir / "meta.json").read_text())

        assert meta["update"] == 4
        assert meta["gen_updates"] == [4, 3, 2, 1]
        assert meta["missing_updates"] == []
        assert sorted(meta["agents"]) == ["robot_a", "robot_b"]

        for aid in ("robot_a", "robot_b"):
            acts = npz[f"actions.{aid}"]
            assert acts.shape == (4, T, ACT_DIM), acts.shape
            # generations differ → nonzero drift on at least some frames
            for g in range(1, 4):
                delta = acts[0] - acts[g]
                assert np.abs(delta).max() > 1e-6
    print("test_delta_self_play_both_agents: PASS")


def test_delta_non_self_play_single_agent():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(
            Path(d), trained_agents=("robot_a",),
        )
        out_dir = compute_delta(dump_dir, episode_pos=0, gens=2, log=lambda *a: None)

        npz = np.load(out_dir / "delta.npz")
        meta = json.loads((out_dir / "meta.json").read_text())

        assert meta["agents"] == ["robot_a"]
        assert "actions.robot_a" in npz
        assert "actions.robot_b" not in npz
        assert npz["actions.robot_a"].shape == (3, T, ACT_DIM)
    print("test_delta_non_self_play_single_agent: PASS")


def test_delta_missing_export_recorded():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, run_dir = _create_delta_run(Path(d))
        shutil.rmtree(run_dir / "policy_exports" / "u00002")
        out_dir = compute_delta(dump_dir, episode_pos=0, gens=3, log=lambda *a: None)

        meta = json.loads((out_dir / "meta.json").read_text())
        assert meta["missing_updates"] == [2]
        assert meta["gen_updates"] == [4, 3, 1]
        npz = np.load(out_dir / "delta.npz")
        assert npz["actions.robot_a"].shape == (3, T, ACT_DIM)
    print("test_delta_missing_export_recorded: PASS")


def test_delta_all_refs_missing_raises():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, run_dir = _create_delta_run(
            Path(d), export_updates=(4,), update=4,
        )
        try:
            compute_delta(dump_dir, episode_pos=0, gens=3, log=lambda *a: None)
        except FileNotFoundError as e:
            assert "at least 2 policy generations" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
    print("test_delta_all_refs_missing_raises: PASS")


def test_delta_gens_bounds():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(Path(d))
        for bad in (0, -1, MAX_GENS + 1):
            try:
                compute_delta(dump_dir, episode_pos=0, gens=bad, log=lambda *a: None)
            except ValueError:
                pass
            else:
                raise AssertionError(f"gens={bad} should raise ValueError")
    print("test_delta_gens_bounds: PASS")


def test_delta_episode_out_of_range():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(Path(d))
        try:
            compute_delta(dump_dir, episode_pos=99, gens=2, log=lambda *a: None)
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError")
    print("test_delta_episode_out_of_range: PASS")


def test_delta_no_trained_agents():
    with tempfile.TemporaryDirectory() as d:
        # Trajectory obs that matches no episode obs → flat: frame_ids →
        # traj_map has an empty trajectories list for episode 0.
        rng = np.random.default_rng(3)
        orphan = Trajectory(
            obs=rng.standard_normal((T, OBS_DIM)).astype(np.float32),
            actions=rng.uniform(-0.9, 0.9, (T, ACT_DIM)).astype(np.float32),
            last_obs=rng.standard_normal(OBS_DIM).astype(np.float32),
            channels={"r_test": make_channel_data(T, rng=rng)},
            importance=1.0,
        )
        run_dir = Path(d) / "run"
        run_dir.mkdir()
        ep = _make_episode()
        dump_collector = {}
        buf, actor = make_buffer([orphan], OBS_DIM, ACT_DIM, REWARD_KEYS)
        critics = make_critics(REWARD_KEYS, OBS_DIM)
        actor_opt, critic_opts = make_optimizers(actor, critics)
        stats = ppo_update(
            actor=actor, critics=critics, actor_optimizer=actor_opt,
            critic_optimizers=critic_opts, buf=buf,
            reward_channels=CHANNELS, pp=make_pp_params(),
            grad_clip_norm=0.5, device=torch.device("cpu"),
            dump_callback=lambda s, d2: dump_collector.update({s: d2}),
        )
        dump_dir = capture_dump(
            run_dir=run_dir, update=1,
            request=DumpRequest(hypothesis="delta test"),
            episodes=[ep], trajectories=[orphan], buf=buf, stats=stats,
            jobs=[_make_fake_job()], dump_collector=dump_collector,
            experiment_name="delta_test",
        )
        try:
            compute_delta(dump_dir, episode_pos=0, gens=1, log=lambda *a: None)
        except ValueError as e:
            assert "no trained agents" in str(e)
        else:
            raise AssertionError("expected ValueError")
    print("test_delta_no_trained_agents: PASS")


# ---------------------------------------------------------------------------
# ViewerAPI._episode_delta
# ---------------------------------------------------------------------------

def test_api_episode_delta_available():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(Path(d))
        compute_delta(dump_dir, episode_pos=0, gens=2, log=lambda *a: None)

        api = ViewerAPI(DumpData(dump_dir))
        status, body = api._episode_delta(0)
        assert status == 200
        assert body["available"] is True
        assert body["gen_updates"] == [4, 3, 2]
        assert set(body["agents"]) == {"robot_a", "robot_b"}
        acts = body["agents"]["robot_a"]["actions"]
        assert len(acts) == 3 and len(acts[0]) == T and len(acts[0][0]) == ACT_DIM

        # strict JSON serialisability (regression: bare NaN broke the UI)
        json.dumps(body, allow_nan=False)
    print("test_api_episode_delta_available: PASS")


def test_api_episode_delta_unavailable():
    with tempfile.TemporaryDirectory() as d:
        dump_dir, _ = _create_delta_run(Path(d))
        api = ViewerAPI(DumpData(dump_dir))
        status, body = api._episode_delta(0)
        assert status == 200
        assert body["available"] is False
    print("test_api_episode_delta_unavailable: PASS")


if __name__ == "__main__":
    test_delta_self_play_both_agents()
    test_delta_non_self_play_single_agent()
    test_delta_missing_export_recorded()
    test_delta_all_refs_missing_raises()
    test_delta_gens_bounds()
    test_delta_episode_out_of_range()
    test_delta_no_trained_agents()
    test_api_episode_delta_available()
    test_api_episode_delta_unavailable()
    print("\nall delta tests passed")
