"""Regression test (P3-1 fix) — abandoned/closed episodes reach recorders.

``EnvRuntime.reset`` now runs the abandon sequence at the EnvRuntime
layer (before delegating to ``core.reset``), so a ``PostActionRecorder``
sees ``on_post_episode`` — with the still-populated ctx carrying the
``"abandoned"`` termination proposal — for episodes abandoned by a
subsequent ``reset()``. ``EnvRuntime.close`` likewise terminates an
open episode with reason ``"closed"``, notifying plugins and recorders.

Run: PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_reset_recorder_gap.py -q
"""
from __future__ import annotations

from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin
from envs.framework.recorder import PostActionRecorder

from .conftest import MockSimulator


class _CountingRecorder(PostActionRecorder):
    def __init__(self) -> None:
        self.pre_episodes = 0
        self.post_episodes = 0
        self.post_steps = 0
        self.post_episode_reasons = []

    @property
    def name(self) -> str:
        return "counting_recorder"

    def on_pre_episode(self, ctx) -> None:
        self.pre_episodes += 1

    def on_post_action_step(self, ctx, observation, action,
                            observer_outputs, action_extras) -> None:
        self.post_steps += 1

    def on_post_episode(self, ctx) -> None:
        self.post_episodes += 1
        # Record the per-agent termination proposals visible at flush time.
        self.post_episode_reasons.append(
            dict(ctx.agent_termination_proposals))


class _PostEpisodeCounter(BasePlugin):
    def __init__(self) -> None:
        self.post_episodes = 0

    @property
    def name(self) -> str:
        return "post_episode_counter"

    def on_post_episode(self, ctx) -> None:
        self.post_episodes += 1


def test_abandoned_episode_reaches_recorder_post_episode():
    """reset()-abandoned episode flushes recorder with 'abandoned' ctx."""
    sim = MockSimulator()
    rt = EnvRuntime(simulator=sim)
    rec = _CountingRecorder()
    rt.attach_recorder(rec)

    rt.reset(seed=1)
    rt.step(None, None)
    rt.reset(seed=2)   # abandons episode 1; episode 2 begins

    assert rec.pre_episodes == 2
    assert rec.post_episodes == 1
    # The recorder saw the abandoned ctx, including the proposal.
    assert any(
        "abandoned" in proposals
        for proposals in rec.post_episode_reasons[0].values()
    )

    rt.close()
    # close() flushes the still-open episode 2 with reason "closed".
    assert rec.post_episodes == 2
    assert any(
        "closed" in proposals
        for proposals in rec.post_episode_reasons[1].values()
    )


def test_close_terminates_open_episode_for_plugins_and_recorders():
    """close() on an active episode fires on_post_episode on both layers."""
    sim = MockSimulator()
    plugin = _PostEpisodeCounter()
    rt = EnvRuntime(simulator=sim, plugins=[plugin])
    rec = _CountingRecorder()
    rt.attach_recorder(rec)

    rt.reset(seed=3)
    rt.step(None, None)
    rt.close()

    assert plugin.post_episodes == 1
    assert rec.post_episodes == 1
