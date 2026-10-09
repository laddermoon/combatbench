"""P5-FAULT-1: injected faults must be locatable via SAC debug surfaces."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from baseline.framework.sac import analysis
from baseline.framework.sac.clocks import SACClockState
from baseline.framework.sac.debugkit import (
    begin_critic_tick_dump,
    finish_critic_tick_dump,
)
from baseline.framework.sac.experiment import SACParams
from baseline.framework.sac.metrics import SACMetricsWriter
from baseline.framework.sac.replay import SACReplayBuffer
from baseline.framework.sac.tests.test_replay import _slice
from baseline.framework.sac.tests.test_trainer import _batch, _models
from baseline.framework.sac.trainer import sac_update_v2, trainer_state_dict


def _dump_with_batch(tmp_path, batch, C=2):
    actor, critic, channels = _models(C=C)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    aopt = torch.optim.Adam([log_alpha], lr=1e-3)
    sp = SACParams(use_grad_norm=False)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps",
        critic_tick=1,
        clocks=SACClockState(),
        batch=batch,
        trainer_pre_state=trainer_state_dict(actor, critic, opt, log_alpha, aopt),
    )
    capture = {}
    stats = sac_update_v2(
        actor, critic, opt, log_alpha, aopt, batch, channels, sp,
        1.0, torch.device("cpu"), capture=capture,
    )
    return finish_critic_tick_dump(
        tmp, forward_capture=capture,
        trainer_post_state=trainer_state_dict(actor, critic, opt, log_alpha, aopt),
        update_stats=stats, actor=actor, critic=critic, channels=channels,
        sp=sp, grad_clip_norm=1.0, critic_lr=1e-3,
    )


def test_fault_termination_mask_violation_is_flagged(tmp_path):
    # sac_update itself refuses terminated+bootstrap>0 (fail loud), so a
    # real run can never dump such a batch; inject by finalizing a dump
    # without running the update — the consistency scanner is the guard
    # for dumps produced by older/external producers.
    batch = _batch(B=8)
    batch["bootstrap"] = torch.ones(8)  # terminated row wrongly bootstraps
    actor, critic, channels = _models(C=2)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    tmp = begin_critic_tick_dump(
        dump_root=tmp_path / "debug_dumps",
        critic_tick=1,
        clocks=SACClockState(),
        batch=batch,
        trainer_pre_state=trainer_state_dict(actor, critic, opt, log_alpha, None),
    )
    dump_dir = finish_critic_tick_dump(
        tmp, forward_capture={},
        trainer_post_state=trainer_state_dict(actor, critic, opt, log_alpha, None),
        update_stats={}, actor=actor, critic=critic, channels=channels,
        sp=SACParams(use_grad_norm=False), grad_clip_norm=1.0, critic_lr=1e-3,
    )
    out = analysis.dump_inspect(dump_dir)
    v = out["consistency"]["violations"]
    assert v.get("terminated_with_bootstrap") == 1


def test_fault_stale_replay_data_visible_in_demographics(tmp_path):
    replay = SACReplayBuffer(16, 3, 2, ("r",), rng_seed=0)
    stale = dataclasses.replace(
        _slice(T=8, channels=1),
        collection={
            "run_id": "run", "collection_round": 1, "job_index": 0,
            "agent_id": "agent", "episode_seed": 1, "job_key": "j",
            "env_blueprint_hash": "env", "policy_fingerprint": "policy_OLD",
            "behavior_policy_version": "v0",
        },
    )
    fresh = dataclasses.replace(
        _slice(slice_id="s2", T=8, channels=1),
        collection={
            "run_id": "run", "collection_round": 9, "job_index": 0,
            "agent_id": "agent", "episode_seed": 2, "job_key": "j",
            "env_blueprint_hash": "env", "policy_fingerprint": "policy_NEW",
            "behavior_policy_version": "v9",
        },
        source_keys=tuple(f"k2:{t}" for t in range(8)),
    )
    replay.add_slices([stale])
    replay.add_slices([fresh])
    replay.save(tmp_path / "replay.pt")
    rep = analysis.replay_report(tmp_path / "replay.pt")
    assert rep["policy_fingerprint_counts"]["policy_OLD"] == 8
    assert rep["policy_fingerprint_counts"]["policy_NEW"] == 8
    assert rep["collection_round_counts"]["1"] == 8
    assert rep["collection_round_counts"]["9"] == 8


def test_fault_alpha_runaway_visible_in_metric_series(tmp_path):
    run_dir = tmp_path / "run"
    writer = SACMetricsWriter(run_dir)
    for i in range(4):
        writer.emit_tick(
            SACClockState(critic_tick=i),
            {"temperature.alpha": 0.2 * (10 ** i)},
        )
    writer.close()
    series = analysis.metric_series(run_dir, "temperature.alpha")
    vals = [p["value"] for p in series["points"]]
    assert vals[0] < 1.0 and vals[-1] > 100.0
    assert vals == sorted(vals)  # monotone runaway is directly readable


def test_fault_channel_suppression_visible_in_inspect(tmp_path):
    # actor_weight=0 for r1: critic must still learn; only actor ignores it
    # (A3 semantics — suppression ≠ critic shutdown).
    batch = _batch(B=8, C=2)
    for key in ("actor_gate", "actor_weight", "actor_gate_next",
                "actor_weight_next"):
        batch[key][:, 1] = 0.0
        batch[key][:, 0] = 1.0
    dump_dir = _dump_with_batch(tmp_path, batch)
    out = analysis.dump_inspect(dump_dir)
    stats = out["update_stats"]
    assert stats["critic_updated_r1"] == 1.0
    assert stats["critic_valid_weight_r1"] == 8.0
    trace = analysis.dump_trace(dump_dir, sample_id=0)
    assert trace["per_channel"]["r1"]["actor_weight"] == 0.0
    # Partial channel_valid suppression: critic r1 trains on fewer rows and
    # actor rows drop; both facts are visible in stats.
    batch2 = _batch(B=8, C=2)
    batch2["channel_valid"][:4, 1] = False
    dump_dir2 = _dump_with_batch(tmp_path / "b", batch2)
    stats2 = analysis.dump_inspect(dump_dir2)["update_stats"]
    assert stats2["critic_valid_weight_r1"] == 4.0
    assert stats2["actor_valid_count"] == 4.0


def test_capture_on_off_does_not_change_training(tmp_path):
    """P5-INV-1: identical seed + config, dump capture on vs off must
    produce identical tick metrics and final trainer weights."""
    from baseline.framework.sac.checkpoint import load_checkpoint_bundle
    from baseline.framework.sac.tests.test_env_integration import (
        _FakeExperiment,
        _FakeRollouter,
    )
    from baseline.framework.sac.loop import train_sac

    out = {}
    for tag, ticks in (("off", None), ("on", {2, 4})):
        run_dir = tmp_path / tag
        torch.manual_seed(0)
        train_sac(
            _FakeExperiment(), run_dir=run_dir,
            rollouter=_FakeRollouter(), dump_ticks=ticks,
        )
        bundle = load_checkpoint_bundle(
            sorted((run_dir / "checkpoints").iterdir())[-1]
        )
        ticks_events = [
            {
                "event_type": e["event_type"],
                "clocks": e["clocks"],
                "metrics": {
                    k: v for k, v in (e.get("metrics") or {}).items()
                    if not k.startswith("timing.")
                    and k != "collection.wall_time_s"
                },
            }
            for e in analysis.iter_events(run_dir)
            if e["event_type"] in ("tick", "round")
        ]
        out[tag] = (bundle, ticks_events)
    a, b = out["off"], out["on"]
    assert a[1] == b[1]  # tick/round metrics identical (timing excluded)
    for key, ta in a[0].trainer_state["actor_state_dict"].items():
        assert torch.equal(
            ta, b[0].trainer_state["actor_state_dict"][key],
        )


def test_dump_capture_cost_is_recorded(tmp_path):
    """P5-INV-1: dump capture must emit a measurable cost metric."""
    from baseline.framework.sac.tests.test_env_integration import (
        _FakeExperiment,
        _FakeRollouter,
    )
    from baseline.framework.sac.loop import train_sac

    run_dir = tmp_path / "run"
    train_sac(
        _FakeExperiment(), run_dir=run_dir,
        rollouter=_FakeRollouter(), dump_ticks={2},
    )
    debug_events = [
        e for e in analysis.iter_events(run_dir)
        if e["event_type"] == "debug" and "debug.capture_s" in (
            e.get("metrics") or {}
        )
    ]
    assert debug_events
    for e in debug_events:
        assert e["metrics"]["debug.capture_s"] >= 0.0
        assert e["metrics"]["debug.bytes"] > 0.0


def test_fault_q_explosion_visible_in_inspect(tmp_path):
    batch = _batch(B=8)
    batch["rewards"] = batch["rewards"] * 0 + 1e5  # reward blowup
    dump_dir = _dump_with_batch(tmp_path, batch)
    out = analysis.dump_inspect(dump_dir)
    for ch in out["per_channel"].values():
        assert ch["targets_absmax"] > 1e4
        assert ch["td_abs_max"] > 1e4
