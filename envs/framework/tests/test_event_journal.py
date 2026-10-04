"""EventJournal contract tests — append-only within an episode.

The journal is the shared event bus: producers may only ``append``;
removal/reordering APIs do not exist. Episode-boundary clearing is
framework-owned (``SimContext.clear_episode_state`` → ``_reset``),
and ``epoch`` lets cursor-diff consumers detect resets exactly.
"""
from __future__ import annotations

import pytest

from envs.framework.context import EventJournal, SimContext


def test_append_and_read_sequence():
    j = EventJournal()
    j.append({"type": "hit", "damage": 1.0})
    j.append({"type": "hit", "damage": 2.0})
    assert len(j) == 2
    assert list(j) == [{"type": "hit", "damage": 1.0},
                       {"type": "hit", "damage": 2.0}]
    assert j[0]["damage"] == 1.0
    assert tuple(j) == ({"type": "hit", "damage": 1.0},
                        {"type": "hit", "damage": 2.0})


def test_no_removal_or_mutation_api():
    """pop/remove/clear/insert/__setitem__/__delitem__ do not exist —
    misuse fails loudly at the call site."""
    j = EventJournal()
    j.append("e")
    for op in ("pop", "remove", "clear", "insert",
               "__setitem__", "__delitem__", "__iadd__", "extend"):
        assert not hasattr(j, op), op
    with pytest.raises(AttributeError):
        j.pop()
    with pytest.raises(TypeError):
        j[0] = "rewritten"


def test_since_cursor():
    j = EventJournal()
    j.append("a")
    mark = len(j)
    j.append("b")
    j.append("c")
    assert j.since(mark) == ["b", "c"]
    assert j.since(len(j)) == []


def test_reset_is_epoch_marked_and_private():
    j = EventJournal()
    j.append("a")
    assert j.epoch == 0
    j._reset()
    assert len(j) == 0
    assert j.epoch == 1
    j.append("b")
    # (epoch, len) cursor semantics: epoch changed → retake all, not
    # continue from the stale offset.
    epoch, mark = 1, 0
    assert j.since(mark) == ["b"]


def test_simcontext_reset_uses_journal_reset():
    """clear_episode_state resets the journal (framework-owned) and
    bumps epoch — plugins never clear events themselves."""
    ctx = SimContext.__new__(SimContext)
    # Avoid needing a simulator: only the fields clear_episode_state
    # touches are exercised.
    ctx.episode_step = 5
    ctx.physics_step = 7
    ctx.metrics = {"hp": 1}
    ctx.events = EventJournal()
    ctx.events.append({"type": "hit"})
    ctx.agent_termination_proposals = {"robot_a": ["ko"], "robot_b": []}
    ctx.agent_terminated = {"robot_a": True, "robot_b": False}
    ctx.episode_options = {"x": 1}
    ctx.base_seed = 42

    ctx.clear_episode_state()

    assert len(ctx.events) == 0
    assert ctx.events.epoch == 1
    ctx.events.append({"type": "next_episode_hit"})
    assert ctx.events.since(0) == [{"type": "next_episode_hit"}]
