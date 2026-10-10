"""Permanent tests for ``sac_checkpoint_v1`` bundles."""
from __future__ import annotations

import torch
import pytest

from baseline.framework.sac.checkpoint import (
    SACCheckpointError,
    load_checkpoint_bundle,
    load_model_only,
    save_checkpoint_bundle,
)
from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.replay import SACReplayBuffer

from .test_replay import _slice


def _replay() -> SACReplayBuffer:
    replay = SACReplayBuffer(8, 3, 2, ("r",), rng_seed=19)
    replay.add_slices([_slice(T=4, channels=1)])
    replay.sample(2, torch.device("cpu"))
    return replay


def _config(lr: float = 3e-4) -> dict:
    return {
        "experiment": {"name": "unit", "seed": 7},
        "optimization": {"actor_lr": lr, "critic_lr": lr},
    }


def test_full_checkpoint_round_trip_restores_replay_and_runtime(tmp_path) -> None:
    replay = _replay()
    clocks = SACClockState(collection_round=2, env_step=40, agent_transition=80)
    path = save_checkpoint_bundle(
        tmp_path / "ckpt",
        trainer_state={"actor": {"w": torch.ones(2)}},
        replay=replay,
        runtime_state={
            "clocks": clocks.snapshot(),
            "numpy_rng": {"seed": 7},
        },
        experiment_state={"schedule": {"stage": 2}},
        config=_config(),
    )

    bundle = load_checkpoint_bundle(path, expected_config=_config())
    assert bundle.resume_mode == "full"
    assert bundle.replay is not None
    assert bundle.replay.size == replay.size
    assert bundle.replay.source_keys[:4] == replay.source_keys[:4]
    restored_next = bundle.replay.sample(2, torch.device("cpu"))
    expected_next = replay.sample(2, torch.device("cpu"))
    assert restored_next["sample_ids"].tolist() == expected_next["sample_ids"].tolist()
    assert SACClockState.from_mapping(bundle.runtime_state["clocks"]).env_step == 40
    assert bundle.experiment_state["schedule"]["stage"] == 2


def test_checkpoint_rejects_missing_artifact(tmp_path) -> None:
    path = save_checkpoint_bundle(
        tmp_path / "ckpt",
        trainer_state={},
        replay=_replay(),
        runtime_state={},
        experiment_state={},
        config=_config(),
    )
    (path / "replay.pt").unlink()
    with pytest.raises(SACCheckpointError, match="missing"):
        load_checkpoint_bundle(path)


def test_config_lock_rejects_optimization_override(tmp_path) -> None:
    path = save_checkpoint_bundle(
        tmp_path / "ckpt",
        trainer_state={},
        replay=_replay(),
        runtime_state={},
        experiment_state={},
        config=_config(lr=1e-3),
    )
    with pytest.raises(SACCheckpointError, match="config"):
        load_checkpoint_bundle(
            path,
            expected_config=_config(lr=2e-3),
            config_lock=True,
        )


def test_whitelisted_override_can_resume_without_config_lock(tmp_path) -> None:
    path = save_checkpoint_bundle(
        tmp_path / "ckpt",
        trainer_state={},
        replay=_replay(),
        runtime_state={},
        experiment_state={},
        config=_config(lr=1e-3),
        allowed_overrides=("optimization.actor_lr",),
    )
    expected = _config(lr=1e-3)
    expected["optimization"]["actor_lr"] = 2e-3
    bundle = load_checkpoint_bundle(
        path,
        expected_config=expected,
        allowed_overrides=("optimization.actor_lr",),
    )
    assert bundle.resume_mode == "full"


def test_warm_start_does_not_restore_replay_or_clocks(tmp_path) -> None:
    path = save_checkpoint_bundle(
        tmp_path / "ckpt",
        trainer_state={"actor": {"w": torch.ones(1)}},
        replay=_replay(),
        runtime_state={"clocks": SACClockState(env_step=99).snapshot()},
        experiment_state={},
        config=_config(),
    )
    bundle = load_checkpoint_bundle(path, warm_start=True)
    assert bundle.resume_mode == "warm_start"
    assert bundle.replay is None
    assert bundle.runtime_state is None
    assert bundle.trainer_state["actor"]["w"].item() == 1.0


def test_model_only_load_is_explicit_warm_start(tmp_path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    payload = load_model_only(path)
    assert payload["resume_mode"] == "warm_start"
    assert payload["replay"] is None
    assert payload["model_payload"]["weights"].shape == (2,)


def test_save_refuses_existing_checkpoint_directory(tmp_path) -> None:
    path = tmp_path / "ckpt"
    save_checkpoint_bundle(
        path,
        trainer_state={},
        replay=_replay(),
        runtime_state={},
        experiment_state={},
        config=_config(),
    )
    with pytest.raises(SACCheckpointError, match="already exists"):
        save_checkpoint_bundle(
            path,
            trainer_state={},
            replay=_replay(),
            runtime_state={},
            experiment_state={},
            config=_config(),
        )


def test_prune_checkpoints_keeps_latest_n_and_pinned(tmp_path) -> None:
    from baseline.framework.sac.checkpoint import prune_checkpoints

    root = tmp_path / "checkpoints"
    root.mkdir()
    names = [f"checkpoint_s{i:08d}" for i in (100, 200, 300, 400, 500)]
    for n in names:
        (root / n).mkdir()
    (root / "checkpoint_s00000200" / ".pinned").touch()
    (root / "checkpoint_s00000600.inflight").mkdir()  # atomic-save tmp dir
    (root / "notes.txt").touch()

    prune_checkpoints(root, keep_last=2)

    survivors = sorted(p.name for p in root.iterdir())
    assert survivors == [
        "checkpoint_s00000200",          # pinned
        "checkpoint_s00000400",
        "checkpoint_s00000500",
        "checkpoint_s00000600.inflight", # tmp dir never pruned
        "notes.txt",
    ]

    prune_checkpoints(root, keep_last=0)
    assert (root / "checkpoint_s00000500").exists()  # disabled


def test_prune_checkpoints_noop_on_missing_dir(tmp_path) -> None:
    from baseline.framework.sac.checkpoint import prune_checkpoints

    prune_checkpoints(tmp_path / "nonexistent", keep_last=3)
