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
