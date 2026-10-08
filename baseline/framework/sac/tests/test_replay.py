"""Permanent tests for the ``sac_replay_v2`` contract."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline.framework.sac.experiment import ReplayPlan
from baseline.framework.sac.replay import SACReplayBuffer, SACReplayError
from baseline.framework.sac.transition import SAC_TRANSITION_SCHEMA, SACTransitionSlice


def _slice(
    *,
    slice_id: str = "slice-0",
    T: int = 4,
    obs_dim: int = 3,
    action_dim: int = 2,
    channels: int = 2,
    terminal: str = "none",
    task_facts: bool = True,
) -> SACTransitionSlice:
    obs = np.arange(T * obs_dim, dtype=np.float32).reshape(T, obs_dim)
    actions = np.full((T, action_dim), 0.25, dtype=np.float32)
    next_obs = np.concatenate([obs[1:], np.full((1, obs_dim), 9.0, dtype=np.float32)])
    rewards = np.ones((T, channels), dtype=np.float32)
    channel_valid = np.ones((T, channels), dtype=np.bool_)
    terminated = np.zeros(T, dtype=np.bool_)
    truncated = np.zeros(T, dtype=np.bool_)
    bootstrap = np.ones(T, dtype=np.float32)
    reason = [""] * T
    if terminal == "terminated":
        terminated[-1] = True
        bootstrap[-1] = 0.0
        reason[-1] = "imbalance"
    elif terminal == "timeout":
        truncated[-1] = True
        reason[-1] = "timeout"
    actor_gate = np.ones((T, channels), dtype=np.float32)
    actor_weight = np.full((T, channels), 1.0 / channels, dtype=np.float32)
    actor_gate_next = np.ones((T, channels), dtype=np.float32)
    actor_weight_next = np.full((T, channels), 1.0 / channels, dtype=np.float32)
    facts = {}
    if task_facts:
        facts["phi_pre"] = np.linspace(0.1, 0.4, T, dtype=np.float32)
    return SACTransitionSlice(
        schema_version=SAC_TRANSITION_SCHEMA,
        slice_id=slice_id,
        obs=obs,
        actions=actions,
        next_obs=next_obs,
        rewards=rewards,
        channel_valid=channel_valid,
        terminated=terminated,
        truncated=truncated,
        bootstrap=bootstrap,
        termination_reason=tuple(reason),
        physics_delta=np.ones(T, dtype=np.float32) * 0.01,
        actor_gate=actor_gate,
        actor_weight=actor_weight,
        actor_gate_next=actor_gate_next,
        actor_weight_next=actor_weight_next,
        sample_weight=np.ones(T, dtype=np.float32),
        task_facts=facts,
        reward_features={"feature": np.ones(T, dtype=np.float32)},
        source_keys=tuple(f"run:{slice_id}:agent:{t}" for t in range(T)),
        behavior={
            "mode": "stochastic",
            "explore_factor": 1.0,
            "parameters": {"alpha": 0.2},
        },
        collection={
            "run_id": "run",
            "collection_round": 1,
            "job_index": 0,
            "agent_id": "agent",
            "episode_seed": 11,
            "job_key": "job",
            "env_blueprint_hash": "env",
            "policy_fingerprint": "policy",
            "behavior_policy_version": "v1",
        },
        versions={
            "schema": SAC_TRANSITION_SCHEMA,
            "reward_semantics": "reward_v1",
            "objective_mode": "max_entropy",
            "regularizer_mode": "auto_alpha",
            "policy_arch": "s01",
        },
    )


def test_replay_admission_sampling_identity_and_payload() -> None:
    replay = SACReplayBuffer(
        capacity=8,
        obs_dim=3,
        action_dim=2,
        channel_names=("r_fall", "r_cross"),
        rng_seed=7,
    )
    assert replay.add_slices([_slice(T=4)]) == 4
    assert replay.size == 4

    batch = replay.sample(4, torch.device("cpu"))
    assert batch["obs"].shape == (4, 3)
    assert batch["rewards"].shape == (4, 2)
    assert batch["terminated"].shape == (4,)
    assert batch["bootstrap"].shape == (4,)
    assert batch["actor_gate_next"].shape == (4, 2)
    assert batch["actor_weight_next"].shape == (4, 2)
    np.testing.assert_allclose(
        batch["actor_weight_next"].sum(axis=1).cpu().numpy(), np.ones(4),
    )
    assert batch["task_facts"]["phi_pre"].shape == (4,)
    assert len(set(batch["sample_ids"].tolist())) == 4
    assert all(isinstance(key, str) and key for key in batch["source_keys"])
    assert batch["metadata"][0]["collection"]["collection_round"] == 1


def test_sample_is_deterministic_for_same_replay_seed() -> None:
    first = SACReplayBuffer(10, 3, 2, ("r",), rng_seed=123)
    second = SACReplayBuffer(10, 3, 2, ("r",), rng_seed=123)
    sl = _slice(T=8, channels=1)
    first.add_slices([sl])
    second.add_slices([sl])
    a = first.sample(4, torch.device("cpu"))
    b = second.sample(4, torch.device("cpu"))
    assert a["sample_ids"].tolist() == b["sample_ids"].tolist()
    assert torch.equal(a["obs"], b["obs"])


def test_fifo_overwrite_assigns_new_sample_id_and_preserves_traceability() -> None:
    replay = SACReplayBuffer(4, 3, 2, ("r",), rng_seed=0)
    replay.add_slices([_slice(slice_id="old", T=4, channels=1)])
    old_ids = replay.sample_ids[: replay.size].tolist()
    replay.add_slices([_slice(slice_id="new", T=4, channels=1)])
    assert replay.size == 4
    assert replay.total_inserted == 8
    assert replay.overwritten == 4
    assert set(replay.sample_ids[: replay.size]).isdisjoint(old_ids)
    stats = replay.buffer_stats()
    assert stats["total_inserted"] == 8
    assert stats["overwritten"] == 4
    assert stats["sample_id_min"] == 4
    assert stats["sample_id_max"] == 7


def test_duplicate_source_key_is_rejected_atomically() -> None:
    replay = SACReplayBuffer(8, 3, 2, ("r",))
    original = _slice(T=2, channels=1)
    replay.add_slices([original])
    before = replay.size
    with pytest.raises(SACReplayError, match="source_key"):
        replay.add_slices([original])
    assert replay.size == before


def test_malformed_slice_is_rejected_before_partial_insertion() -> None:
    replay = SACReplayBuffer(8, 3, 2, ("r",))
    good = _slice(slice_id="good", T=1, channels=1)
    bad = _slice(slice_id="bad", T=2, channels=1)
    object.__setattr__(bad, "source_keys", ("one",))
    with pytest.raises(ValueError, match="source_keys"):
        replay.add_slices([good, bad])
    assert replay.size == 0


def test_unsupported_replay_extensions_are_explicitly_rejected() -> None:
    with pytest.raises(SACReplayError, match="stratify"):
        SACReplayBuffer(
            8, 3, 2, ("r",),
            replay_plan=ReplayPlan(stratify_by="tag"),
        )
    with pytest.raises(SACReplayError, match="freshness"):
        SACReplayBuffer(
            8, 3, 2, ("r",),
            replay_plan=ReplayPlan(freshness_weight=0.1),
        )
    replay = SACReplayBuffer(8, 3, 2, ("r",))
    with pytest.raises(SACReplayError, match="1-step"):
        replay.sample_nstep(1, torch.device("cpu"), {"r": 2})
    with pytest.raises(SACReplayError, match="relabel"):
        replay.relabel(lambda *_: None, {})


def test_sample_unique_requirement_rejects_oversized_batch() -> None:
    replay = SACReplayBuffer(4, 3, 2, ("r",))
    replay.add_slices([_slice(T=2, channels=1)])
    with pytest.raises(SACReplayError, match="unique"):
        replay.sample(3, torch.device("cpu"))


def test_persistence_round_trip_restores_content_rng_and_identity(tmp_path) -> None:
    replay = SACReplayBuffer(8, 3, 2, ("r",), rng_seed=5)
    replay.add_slices([_slice(T=5, channels=1)])
    replay.sample(3, torch.device("cpu"))

    path = tmp_path / "replay.pt"
    replay.save(path)
    expected_next = replay.sample(3, torch.device("cpu"))
    restored = SACReplayBuffer.load(path)
    after = restored.sample(3, torch.device("cpu"))

    assert restored.size == replay.size
    assert restored.total_inserted == replay.total_inserted
    assert after["sample_ids"].tolist() == expected_next["sample_ids"].tolist()
    assert restored.source_keys[: restored.size] == replay.source_keys[: replay.size]
    np.testing.assert_allclose(restored.obs, replay.obs)
    np.testing.assert_allclose(
        restored.actor_weight_next, replay.actor_weight_next,
    )
    assert restored._seen_source_keys == replay._seen_source_keys


def test_get_by_sample_id_returns_self_contained_metadata() -> None:
    replay = SACReplayBuffer(8, 3, 2, ("r",), rng_seed=0)
    replay.add_slices([_slice(T=3, channels=1)])
    rows = replay.get_by_sample_ids([0, 2])
    assert rows["sample_ids"].tolist() == [0, 2]
    assert rows["metadata"][0]["slice_id"] == "slice-0"
    assert rows["task_facts"]["phi_pre"].shape == (2,)
    with pytest.raises(KeyError):
        replay.get_by_sample_ids([99])
