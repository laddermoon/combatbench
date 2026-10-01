"""Audit probe (Phase 3) — NOT a fix.

Question under audit: ``_RuntimeCore.reset()`` abandons an in-flight
episode via ``request_termination("abandoned")`` + ``_handle_termination()``,
whose comment says "so recorder manifests and observer state are flushed".
But ``_handle_termination`` only invokes **plugin** hooks — recorders live
in ``EnvRuntime._recorders`` and are invoked via ``_invoke_recorders``,
which ``EnvRuntime.reset`` never calls for the abandoned episode.

Expected consequence: a PostActionRecorder's ``on_post_episode`` fires for
a normally-terminated episode but is silently skipped when the episode is
abandoned by a subsequent ``reset()`` — unflushed manifests / dangling
episode dirs.

Run: PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_reset_recorder_gap.py -q
"""
from __future__ import annotations

from envs.framework.env_runtime import EnvRuntime
from envs.framework.recorder import PostActionRecorder

from .conftest import MockSimulator


class _CountingRecorder(PostActionRecorder):
    def __init__(self) -> None:
        self.pre_episodes = 0
        self.post_episodes = 0
        self.post_steps = 0

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


def test_abandoned_episode_skips_recorder_post_episode():
    """Locks in audited behavior: abandoned episode never reaches recorders.

    When EnvRuntime.reset forwards the abandoned-episode termination to
    recorders, this test SHOULD FAIL — flip to ``== 2`` and update the note.
    """
    sim = MockSimulator()
    rt = EnvRuntime(simulator=sim)
    rec = _CountingRecorder()
    rt.attach_recorder(rec)

    # Episode 1: reset, take one step, then abandon via a second reset.
    rt.reset(seed=1)
    rt.step(None, None)
    rt.reset(seed=2)   # abandons episode 1; episode 2 begins
    rt.close()

    # Plugin-side on_post_episode ran for the abandoned episode
    # (ctx termination reason "abandoned"), but the recorder only ever saw
    # one pre_episode boundary pair close: episode 1's post_episode is
    # missing entirely.
    assert rec.pre_episodes == 2   # both resets reach recorders' pre_episode
    # The abandoned episode 1 AND the still-open episode 2 (close() only
    # calls recorder.on_detach, never on_post_episode) are both missing —
    # recorders see 2 episode starts and zero episode ends.
    assert rec.post_episodes == 0, (
        "recorder saw post_episode for the abandoned episode — "
        "gap may be fixed; flip this test")
