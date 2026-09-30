"""Audit probe (regularization Phase 2) — NOT a fix.

Question under audit: ``SimContext._revoke_mutator`` only sets
``ctx.mutator = None``; the underlying ``_MutatorView`` object has no
lifecycle/validity check. A plugin that stashes the reference inside a
*writable* hook keeps a permanently-working write channel and can mutate
physics from a *read-only* hook (e.g. ``on_post_action_step``) — or from
``require_mutator=False`` context entirely.

If this probe passes, the capability sandbox is advisory, not enforced.

Run: ``PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_mutator_leak.py -q``
"""
from __future__ import annotations

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin

from .conftest import MockSimulator


class _MutatorStash(BasePlugin):
    """Grabs ctx.mutator during on_pre_action_step (writable) and reuses
    the stash inside on_post_action_step (read-only per hook table)."""

    def __init__(self) -> None:
        self.stashed = None
        self.post_step_write_succeeded = False

    @property
    def name(self) -> str:
        return "mutator_stash"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_action_step(self, ctx) -> None:
        if self.stashed is None and ctx.mutator is not None:
            self.stashed = ctx.mutator  # stash the view for later

    def on_post_action_step(self, ctx) -> None:
        assert ctx.mutator is None, "post_action_step should not grant mutator"
        if self.stashed is not None:
            try:
                self.stashed.set_action(
                    {"robot_a": np.zeros(21), "robot_b": np.zeros(21)})
                self.post_step_write_succeeded = True
            except Exception:
                pass


def test_stashed_mutator_still_writes_in_readonly_hook():
    """Locks in audited behavior: stash survives revocation.

    When the sandbox is fixed (e.g. _MutatorView gains a validity flag),
    this test SHOULD FAIL — flip to assert not succeeded and delete this
    docstring note.
    """
    p = _MutatorStash()
    rt = EnvRuntime(simulator=MockSimulator(), plugins=[p],
                    phy_steps_per_action=2)
    rt.reset(seed=0)
    rt.step(np.zeros(21), np.zeros(21))
    assert p.post_step_write_succeeded, (
        "stashed mutator did NOT write — sandbox may have been fixed")
