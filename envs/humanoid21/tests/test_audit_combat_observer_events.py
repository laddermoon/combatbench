"""Audit regression (P-H21-2, fixed): CombatScoringObserver consumed
``metrics['events']`` — a key nothing ever wrote — while hit events live
on ``ctx.events``. The observer now cursor-diffs the append-only event
journal: ``step_*`` keys reflect events appended since the previous
refresh, ``ctx.events`` itself is never mutated.

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


_HIT_A = {"type": "hit", "defender": "robot_a", "damage": 3.5}
_HIT_B = {"type": "hit", "defender": "robot_b", "damage": 1.25}


def _ctx(events):
    return _StubCtx(
        metrics={"health_a": 96.5, "health_b": 98.75,
                 "damage_taken_a": 3.5, "damage_taken_b": 1.25},
        events=list(events),
    )


def test_step_events_via_cursor_diff():
    """Events appended since the last refresh surface as this step's."""
    obs = CombatScoringObserver()
    journal = []

    out = obs._build_output(_ctx(journal))
    assert out["events"] == []
    assert out["robot_a"]["step_hit_events"] == []
    assert out["robot_a"]["step_damage_taken"] == 0.0

    journal.extend([_HIT_A, _HIT_B])
    out = obs._build_output(_ctx(journal))
    assert out["events"] == [_HIT_A, _HIT_B]
    assert out["robot_a"]["step_hit_events"] == [_HIT_A]
    assert out["robot_a"]["step_damage_taken"] == pytest.approx(3.5)
    assert out["robot_b"]["step_hit_events"] == [_HIT_B]
    assert out["robot_b"]["step_damage_taken"] == pytest.approx(1.25)

    # Same journal again → no new events (cumulative journal, not per-step).
    out = obs._build_output(_ctx(journal))
    assert out["events"] == []


def test_health_and_cumulative_from_metrics():
    """metrics-sourced keys are unaffected by the cursor mechanics."""
    obs = CombatScoringObserver()
    out = obs._build_output(_ctx([_HIT_A]))
    assert out["robot_a"]["health"] == pytest.approx(96.5)
    assert out["robot_a"]["cumulative_damage_taken"] == pytest.approx(3.5)
    assert out["robot_a"]["is_ko"] is False
    out = obs._build_output(
        _StubCtx(metrics={"health_a": 0.0}, events=[]))
    assert out["robot_a"]["is_ko"] is True


def test_pre_episode_baseline_resync():
    """on_pre_episode takes the current journal length as baseline — a
    leftover journal from a previous episode is not double-counted."""
    obs = CombatScoringObserver()
    journal = [_HIT_A]
    obs.on_pre_episode(_ctx(journal))
    out = obs.get_output()
    assert out["events"] == []

    journal.append(_HIT_B)
    out = obs._build_output(_ctx(journal))
    assert out["events"] == [_HIT_B]


def test_mid_episode_clear_resyncs():
    """Contract guard: if the journal is ever cleared mid-episode, the
    cursor resyncs instead of slicing garbage (dropped, not duplicated)."""
    obs = CombatScoringObserver()
    journal = [_HIT_A, _HIT_B]
    obs._build_output(_ctx(journal))
    journal.clear()
    journal.append(_HIT_B)
    out = obs._build_output(_ctx(journal))
    assert out["events"] == [_HIT_B]
