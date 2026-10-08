"""Regression test (P-FW-9 fix) — stashed mutator dies with the grant.

``SimContext._revoke_mutator`` invalidates ``_MutatorView``: a plugin that
stashes ``ctx.mutator`` inside a *writable* hook can no longer write with
the stash inside a *read-only* hook — the call raises ``RuntimeError``
and never reaches the simulator.

Run: ``PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_mutator_leak.py -q``
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin

from .conftest import MockSimulator


class _MutatorStash(BasePlugin):
    """Grabs ctx.mutator during on_pre_action_step (writable) and reuses
    the stash inside on_post_action_step (read-only per hook table)."""

    def __init__(self) -> None:
        self.stashed = None
        self.stash_error = None
        self.writable_hook_write_ok = False

    @property
    def name(self) -> str:
        return "mutator_stash"

    @property
    def require_mutator(self) -> bool:
        return True

    def on_pre_action_step(self, ctx) -> None:
        if self.stashed is None and ctx.mutator is not None:
            self.stashed = ctx.mutator  # stash the view for later
            # In-hook use must keep working.
            self.stashed.set_action(
                {"robot_a": np.zeros(21), "robot_b": np.zeros(21)})
            self.writable_hook_write_ok = True

    def on_post_action_step(self, ctx) -> None:
        assert ctx.mutator is None, "post_action_step should not grant mutator"
        if self.stashed is not None:
            try:
                self.stashed.set_action(
                    {"robot_a": np.zeros(21), "robot_b": np.zeros(21)})
            except RuntimeError as e:
                self.stash_error = e


def test_stashed_mutator_is_revoked_in_readonly_hook():
    """Stashed _MutatorView raises RuntimeError after revocation."""
    p = _MutatorStash()
    rt = EnvRuntime(simulator=MockSimulator(), plugins=[p],
                    phy_steps_per_action=2)
    rt.reset(seed=0)
    rt.step(np.zeros(21), np.zeros(21))
    assert p.writable_hook_write_ok, "in-hook mutator write should work"
    assert p.stash_error is not None and "revoked" in str(p.stash_error)
