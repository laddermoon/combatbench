"""Tests for the canonical SAC metrics event surface."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from baseline.framework.sac.clocks import CLOCK_FIELDS, SACClockState
from baseline.framework.sac.metric_catalog import MetricCatalog
from baseline.framework.sac.metrics import (
    EVENTS_RELATIVE_PATH,
    MetricEvent,
    SACMetricsWriter,
    load_events,
)


def test_clock_state_snapshot_and_advances():
    clocks = SACClockState()
    clocks.advance_collection(env_steps=10, agent_transitions=20)
    clocks.tick_critic()
    clocks.tick_actor()
    clocks.tick_target()
    snap = clocks.snapshot()
    assert snap["collection_round"] == 1
    assert snap["env_step"] == 10
    assert snap["agent_transition"] == 20
    assert snap["critic_tick"] == 1
    assert snap["actor_tick"] == 1
    assert snap["target_tick"] == 1
    assert set(snap) == set(CLOCK_FIELDS)


def test_metrics_writer_jsonl_round_trip(tmp_path):
    clocks = SACClockState(env_step=4, agent_transition=8)
    with SACMetricsWriter(tmp_path) as writer:
        writer.emit_config(clocks, {"experiment": "fake", "seed": 7})
        writer.emit_round(
            clocks,
            {
                "collection.episodes": 2,
                "collection.env_steps": 4,
                "collection.agent_transitions": 8,
                "timing.collection_s": np.float32(0.01),
            },
            job_count=2,
        )
        writer.emit_tick(
            clocks,
            {
                "batch.size": 8,
                "critic.loss": 0.25,
                "actor.loss": -0.1,
                "temperature.alpha": 0.2,
            },
            batch_digest="abc",
        )
    path = tmp_path / EVENTS_RELATIVE_PATH
    assert path.exists()
    events = load_events(path)
    assert [e.event_type for e in events] == ["config", "round", "tick"]
    assert all(set(e.clocks) == set(CLOCK_FIELDS) for e in events)
    assert events[1].metrics["collection.agent_transitions"] == 8.0
    assert events[2].context["batch_digest"] == "abc"


def test_metric_catalog_rejects_bad_namespaces():
    catalog = MetricCatalog()
    with pytest.raises(ValueError, match="PPO-only"):
        catalog.validate_metric_name("tick", "ratio.mean")
    with pytest.raises(ValueError, match="not allowed"):
        catalog.validate_metric_name("tick", "episode.length")
    with pytest.raises(ValueError, match="unknown SAC metric event"):
        catalog.validate_metric_name("unknown", "debug.status")


def test_metric_event_rejects_missing_clock_and_nonfinite():
    clocks = SACClockState().snapshot()
    clocks.pop("critic_tick")
    with pytest.raises(ValueError, match="clocks missing"):
        MetricEvent(
            event_type="tick",
            clocks=clocks,
            metrics={"critic.loss": 1.0},
        ).validate()

    with pytest.raises(ValueError, match="non-finite"):
        MetricEvent(
            event_type="tick",
            clocks=SACClockState().snapshot(),
            metrics={"critic.loss": math.nan},
        ).validate()

    with pytest.raises(TypeError, match="numeric scalar"):
        MetricEvent(
            event_type="tick",
            clocks=SACClockState().snapshot(),
            metrics={"critic.loss": "bad"},
        ).validate()


def test_writer_closed_and_context_array_limit(tmp_path):
    writer = SACMetricsWriter(tmp_path)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.emit_debug(SACClockState(), {"debug.status": 1})

    writer = SACMetricsWriter(tmp_path / "second")
    with pytest.raises(ValueError, match="too large"):
        writer.emit_debug(
            SACClockState(),
            {"debug.status": 1},
            payload=np.zeros(20_000),
        )
    writer.close()


def test_load_events_rejects_corrupt_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"bad": True}) + "\n")
    with pytest.raises(ValueError, match="invalid SAC metrics event"):
        load_events(path)


def test_catalog_specs_match_emitted_keys():
    """R2-W03 (G8): catalog spec names must correspond to keys the
    framework actually emits — phantom entries break discovery, and
    unlisted emitted keys are invisible to the catalog."""
    from baseline.framework.sac.metric_catalog import DEFAULT_METRIC_SPECS

    specs = {(s.event_type, s.name) for s in DEFAULT_METRIC_SPECS}
    # standup emits eval.success (not eval.success_rate); balance emits
    # eval.survival_rate + eval.survived — verified against real
    # events.jsonl of acceptance runs.
    assert ("eval", "eval.success") in specs
    assert ("eval", "eval.success_rate") not in specs
    assert ("eval", "eval.survival_rate") in specs
    assert ("eval", "eval.survived") in specs
    # emitted tick/round keys previously missing from the catalog
    assert ("tick", "target.pair1_frac") in specs
    assert ("tick", "actor.pair1_frac") in specs
    assert ("tick", "actor.valid_count") in specs
    assert ("round", "behavior.explore_factor") in specs
    assert ("round", "behavior.random_start") in specs
    assert ("debug", "debug.request") in specs
    # u-mode regularizer visibility keys (R2-6 regression prerequisite)
    assert ("tick", "actor.reg_term") in specs
    assert ("tick", "regularizer.shannon") in specs


def test_tick_metrics_maps_u_mode_reg_keys():
    """U-mode reg terms in step_stats must reach the tick event stream —
    without this mapping u_floor training has no reg observability."""
    from baseline.framework.sac.loop import _tick_metrics

    m = _tick_metrics(
        {"reg_actor_term_mean": -0.12, "regularizer_shannon": 0.0},
        batch_size=4, replay_size=10, tau=0.05,
    )
    assert m["actor.reg_term"] == -0.12
    assert m["regularizer.shannon"] == 0.0
    # shannon path emits regularizer.shannon=1 but no actor.reg_term
    m2 = _tick_metrics(
        {"regularizer_shannon": 1.0}, batch_size=4, replay_size=10, tau=0.05,
    )
    assert "actor.reg_term" not in m2
    assert m2["regularizer.shannon"] == 1.0
