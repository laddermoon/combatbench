"""Permanent tests for SAC L2 critic-tick dumps and recomputation."""
from __future__ import annotations

import json

import pytest
import torch

from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.debugkit import (
    SACDumpError,
    begin_critic_tick_dump,
    find_sample,
    finish_critic_tick_dump,
    load_dump,
    recompute_dump,
    summarize_dump,
)
from baseline.framework.sac.experiment import SACParams
from baseline.framework.sac.tests.test_env_integration import (
    _FakeExperiment,
    _FakeRollouter,
)
from baseline.framework.sac.tests.test_trainer import _batch, _models
from baseline.framework.sac.trainer import sac_update_v2, trainer_state_dict
from baseline.framework.sac.loop import train_sac


def test_l2_dump_recomputes_standard_sac_losses(tmp_path) -> None:
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
    batch = _batch(B=16)
    sp = SACParams(use_grad_norm=False, reward_scale=2.0)
    clocks = SACClockState(collection_round=3, critic_tick=7)

    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps",
        critic_tick=8,
        clocks=clocks,
        batch=batch,
        trainer_pre_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        hypothesis="unit",
    )
    capture = {}
    stats = sac_update_v2(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        batch=batch,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
        capture=capture,
    )
    dump_dir = finish_critic_tick_dump(
        tmp,
        forward_capture=capture,
        trainer_post_state=trainer_state_dict(
            actor, critic, actor_optimizer, log_alpha, alpha_optimizer,
        ),
        update_stats=stats,
        actor=actor,
        critic=critic,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        critic_lr=1e-3,
    )

    payload = load_dump(dump_dir)
    assert payload["manifest"]["schema_version"] == "sac_dump_v2"
    assert payload["manifest"]["evidence_level"] == "recompute"
    assert payload["batch"]["sample_ids"].tolist() == list(range(16))

    result = recompute_dump(dump_dir)
    assert result["passed"], json.dumps(result["compared"], indent=2)
    assert result["compared"]["critic_loss"]["ok"]
    assert result["compared"]["actor_loss"]["ok"]
    assert result["compared"]["alpha_loss"]["ok"]

    sample = find_sample(dump_dir, sample_id=3)
    assert sample["source_key"] == "src:3"
    summary = summarize_dump(dump_dir)
    assert summary["critic_tick"] == 8
    assert summary["batch_size"] == 16


def test_loop_writes_dump_for_scheduled_critic_tick(tmp_path) -> None:
    run_dir = tmp_path / "run"
    train_sac(
        _FakeExperiment(),
        run_dir=run_dir,
        rollouter=_FakeRollouter(),
        dump_ticks={2},
        dump_hypothesis="phase2-test",
    )
    dump_dir = run_dir / "debug_dumps" / "critic_tick_00000002"
    result = recompute_dump(dump_dir)
    assert result["passed"]
    request = json.loads((dump_dir / "request.json").read_text())
    assert request["hypothesis"] == "phase2-test"


def test_dump_loader_rejects_corruption(tmp_path) -> None:
    actor, critic, channels = _models(C=1)
    opt = torch.optim.Adam(actor.parameters())
    log_alpha = torch.tensor(0.0, requires_grad=True)
    batch = _batch(B=4, C=1)
    sp = SACParams(use_grad_norm=False)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path,
        critic_tick=1,
        clocks=SACClockState(),
        batch=batch,
        trainer_pre_state=trainer_state_dict(actor, critic, opt, log_alpha, None),
    )
    capture = {}
    stats = sac_update_v2(
        actor, critic, opt, log_alpha, None, batch, channels, sp,
        1.0, torch.device("cpu"), capture=capture,
    )
    dump_dir = finish_critic_tick_dump(
        tmp,
        forward_capture=capture,
        trainer_post_state=trainer_state_dict(actor, critic, opt, log_alpha, None),
        update_stats=stats,
        actor=actor,
        critic=critic,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        critic_lr=1e-3,
    )
    (dump_dir / "batch.pt").write_bytes(b"corrupt")
    with pytest.raises(SACDumpError, match="hash mismatch"):
        load_dump(dump_dir)
