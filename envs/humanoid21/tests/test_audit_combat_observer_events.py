"""Audit probe (P-H21-2): CombatScoringObserver reads ``metrics['events']``
but CombatScoringPlugin appends hit events to ``ctx.events`` — two
different containers. This test constructs a minimal ReadOnlySimContext-
shaped stub with a hit event on ``ctx.events`` and verifies whether the
observer surfaces it.

Run:
    PYTHONPATH=. python3 -m pytest envs/humanoid21/tests/test_audit_combat_observer_events.py -v
"""
from __future__ import annotations

import pytest

from envs.humanoid21.observer_plugins import CombatScoringObserver


class _StubCtx:
    """Minimal ctx stand-in matching what _build_output touches."""

    def __init__(self, metrics, events):
        self.metrics = metrics
        self.events = events


_HIT = {"attacker": "robot_b", "defender": "robot_a", "damage": 3.5}


def test_observer_misses_events_on_ctx_events():
    """Event lives on ctx.events (where the plugin writes) — metrics has none."""
    obs = CombatScoringObserver()
    ctx = _StubCtx(
        metrics={"health_a": 96.5, "health_b": 100.0, "damage_taken_a": 3.5},
        events=[_HIT],
    )
    out = obs._build_output(ctx)
    # Locks in the audited behavior: the observer misses ctx.events.
    # When the bug is fixed, THIS TEST SHOULD START FAILING — flip the
    # assertions and delete this comment.
    assert out["events"] == [], "P-H21-2 fixed? observer now reads ctx.events"
    assert out["robot_a"]["step_hit_events"] == []
    assert out["robot_a"]["step_damage_taken"] == 0.0


def test_observer_reads_metrics_events_if_present():
    """And conversely: if events WERE in metrics, observer would see them —
    confirming the container mismatch is the only issue."""
    obs = CombatScoringObserver()
    ctx = _StubCtx(
        metrics={"health_a": 96.5, "health_b": 100.0,
                 "damage_taken_a": 3.5, "events": [_HIT]},
        events=[],
    )
    out = obs._build_output(ctx)
    assert out["events"] == [_HIT]
    assert out["robot_a"]["step_hit_events"] == [_HIT]
    assert out["robot_a"]["step_damage_taken"] == pytest.approx(3.5)
