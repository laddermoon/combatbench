"""Tests for WP3 episode reconstruction / cross-generation delta.

Full episode rerun is dogfooded on real dumps (heavy); here we cover
the blueprint-resolution priority chain, stale-path rewriting,
fingerprint provenance, and the explicit unavailable/error paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.debugkit import (
    begin_critic_tick_dump,
    finish_critic_tick_dump,
    load_dump,
)
from baseline.framework.sac.episode_recon import (
    _behavior_blueprint,
    cross_gen_delta,
    reconstruct_episode,
)
from baseline.framework.sac.experiment import SACParams
from baseline.framework.sac.tests.test_trainer import _batch, _models
from baseline.framework.sac.trainer import sac_update_v2, trainer_state_dict


def _make_dump(tmp_path: Path, batch=None) -> Path:
    """Synthetic sac_dump_v3 under <tmp>/run/debug_dumps/."""
    actor, critic, channels = _models(C=1)
    opt = torch.optim.Adam(actor.parameters())
    log_alpha = torch.tensor(0.0, requires_grad=True)
    batch = batch if batch is not None else _batch(B=4, C=1)
    sp = SACParams(use_grad_norm=False)
    run_dir = tmp_path / "run"
    tmp = begin_critic_tick_dump(
        dump_root=run_dir / "debug_dumps",
        critic_tick=1,
        clocks=SACClockState(),
        batch=batch,
        trainer_pre_state=trainer_state_dict(actor, critic, opt,
                                             log_alpha, None),
    )
    capture = {}
    stats = sac_update_v2(
        actor, critic, opt, log_alpha, None, batch, channels, sp,
        1.0, torch.device("cpu"), capture=capture,
    )
    return finish_critic_tick_dump(
        tmp,
        forward_capture=capture,
        trainer_post_state=trainer_state_dict(actor, critic, opt,
                                              log_alpha, None),
        update_stats=stats,
        actor=actor,
        critic=critic,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        critic_lr=1e-3,
    )


def _bp_dict(model_path: str) -> dict:
    return {
        "version": 1,
        "cls": "policy.humanoid21.standing.policy.StandingCombatPolicy",
        "config": {"model_path": model_path, "mode": "stochastic"},
    }


def test_blueprint_resolution_prefers_dump_snapshot(tmp_path) -> None:
    dump_dir = tmp_path / "dumps" / "critic_tick_00000001"
    snap = dump_dir / "policy_snapshot"
    snap.mkdir(parents=True)
    (snap / "model.pt").write_bytes(b"w")
    spec = {"behavior_policy": {
        "blueprint": _bp_dict("/stale/original.pt"),
        "snapshot_dir": "policy_snapshot",
    }}
    bp, source, fp = _behavior_blueprint(dump_dir, spec, tmp_path, 1)
    assert source == "dump_snapshot"
    assert bp.config["model_path"] == str(snap / "model.pt")
    assert fp is not None


def test_blueprint_resolution_rewrites_stale_export_path(tmp_path) -> None:
    run_dir = tmp_path / "run"
    exp_dir = run_dir / "policy_exports" / "r00007"
    exp_dir.mkdir(parents=True)
    local_model = exp_dir / "model.pt"
    local_model.write_bytes(b"w")
    exp_dir.joinpath("policy_blueprint.yaml").write_text(
        yaml.safe_dump(_bp_dict("/renamed/away/model.pt"))
    )
    bp, source, fp = _behavior_blueprint(
        tmp_path / "dumps" / "x", {}, run_dir, 7)
    assert source == "policy_exports"
    # stale absolute path rewritten to the self-contained neighbor file
    assert bp.config["model_path"] == str(local_model)
    # fingerprint hashes the un-rewritten dict — matches collection-time
    from baseline.framework.sac.collected_episode import mapping_hash
    assert fp == mapping_hash(_bp_dict("/renamed/away/model.pt"))


def test_blueprint_resolution_unavailable_and_fallback(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bp, source, fp = _behavior_blueprint(
        tmp_path / "d", {}, run_dir, 3)
    assert (bp, source, fp) == (None, "unavailable", None)
    # blueprint recorded but snapshot missing → original_path_fallback
    spec = {"behavior_policy": {
        "blueprint": _bp_dict("/orig/model.pt"),
        "snapshot_dir": "policy_snapshot",
    }}
    bp, source, _ = _behavior_blueprint(tmp_path / "d", spec, run_dir, 3)
    assert source == "original_path_fallback"
    assert bp.config["model_path"] == "/orig/model.pt"


def test_reconstruct_rejects_incomplete_metadata(tmp_path) -> None:
    dump_dir = _make_dump(tmp_path)
    with pytest.raises(KeyError, match="metadata incomplete"):
        reconstruct_episode(dump_dir, sample_id=0)
    with pytest.raises(KeyError, match="not in dump batch"):
        reconstruct_episode(dump_dir, sample_id=999)


def test_reconstruct_reports_unavailable_policy(tmp_path) -> None:
    meta = [{
        "collection": {
            "collection_round": 1, "job_index": 0,
            "episode_seed": 42, "agent_id": "robot_a",
            "run_id": "run", "policy_fingerprint": "x",
        },
        "frame_index": 0, "slice_id": "s0",
    }] + [{}] * 3
    batch = _batch(B=4, C=1)
    batch["metadata"] = meta
    dump_dir = _make_dump(tmp_path, batch=batch)
    run_dir = dump_dir.parent.parent
    (run_dir / "config.json").write_text(json.dumps({
        "experiment": {
            "name": "sac_balance", "knobs": {},
            "common_params": {"seed": 0, "episodes_per_update": 1},
        },
    }))
    # policy_exports absent and no snapshot → explicit unavailable error
    with pytest.raises(KeyError, match="no behavior policy"):
        reconstruct_episode(dump_dir, sample_id=0)


def test_cross_gen_delta_degrades_without_checkpoints(tmp_path) -> None:
    dump_dir = _make_dump(tmp_path)
    out = cross_gen_delta(dump_dir)
    assert out["n_checkpoints"] == 0
    assert "no checkpoints" in out["reason"]
    assert out["series"] == []


def test_cross_gen_delta_replays_saved_checkpoints(tmp_path) -> None:
    dump_dir = _make_dump(tmp_path)
    run_dir = dump_dir.parent.parent
    actor, critic, _ = _models(C=1)
    opt = torch.optim.Adam(actor.parameters())
    log_alpha = torch.tensor(0.0, requires_grad=True)
    state = trainer_state_dict(actor, critic, opt, log_alpha, None)
    for i in (1, 2):
        cdir = run_dir / "checkpoints" / f"checkpoint_s{i:08d}"
        cdir.mkdir(parents=True)
        torch.save(state, cdir / "trainer.pt")
    out = cross_gen_delta(dump_dir)
    assert out["n_checkpoints"] == 2
    assert [r["checkpoint"] for r in out["series"]] == [
        "checkpoint_s00000001", "checkpoint_s00000002",
    ]
    # identical checkpoints → zero drift, and self-series is exact
    assert out["series"][0]["drift_vs_first_det_l2"] == 0.0
    assert out["series"][1]["drift_vs_prev_det_l2"] == 0.0
    payload = load_dump(dump_dir)
    assert out["critic_tick"] == payload["manifest"]["critic_tick"]
