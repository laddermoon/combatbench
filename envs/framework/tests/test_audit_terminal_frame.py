"""Audit probe (regularization Phase 1) — NOT a fix.

Question under audit: when a world plugin requests termination from a
*physics-step* hook (``on_pre_phy_step`` / ``on_post_phy_step``),
``_RuntimeCore.step`` returns early and never invokes the plugin-level
``on_post_action_step`` hook — so the observer dispatcher never refreshes
observers for the terminal transition. ``EnvRuntime.step`` still fires
recorders' ``on_post_action_step`` for that step, meaning the recorded
``observer_outputs`` for the terminating frame are STALE (one step behind).

Run: ``PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_terminal_frame.py -q``
"""
from __future__ import annotations

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.observer_plugin import BaseObserverPlugin
from envs.framework.plugin import BasePlugin
from envs.framework.recorder import PostActionRecorder

from .conftest import MockSimulator


class _MidPhysicsKiller(BasePlugin):
    """Terminates the whole episode from ``on_post_phy_step`` once the
    physics step counter reaches ``kill_at``."""

    def __init__(self, kill_at: int):
        self._kill_at = kill_at

    @property
    def name(self) -> str:
        return "mid_physics_killer"

    def on_post_phy_step(self, ctx) -> None:
        if ctx.physics_step >= self._kill_at:
            ctx.request_termination("audit_ko")


class _StepCountingObserver(BaseObserverPlugin):
    """Output = how many times on_post_action_step ran."""

    def __init__(self) -> None:
        self.refresh_count = 0

    def on_post_action_step(self, ctx) -> None:
        self.refresh_count += 1

    def get_output(self):
        return self.refresh_count


class _CaptureRecorder(PostActionRecorder):
    """Records the observer_outputs snapshot seen at each recorder frame."""

    def __init__(self) -> None:
        self.frames = []

    def on_post_action_step(self, ctx, observation, action, observer_outputs, action_extras=None) -> None:
        self.frames.append(dict(observer_outputs))


def _build(kill_at: int, phy_steps_per_action: int = 10):
    obs = _StepCountingObserver()
    rec = _CaptureRecorder()
    runtime = EnvRuntime(
        simulator=MockSimulator(),
        plugins=[_MidPhysicsKiller(kill_at)],
        observer_plugins={"probe": obs},
        recorders=[rec],
        phy_steps_per_action=phy_steps_per_action,
    )
    return runtime, obs, rec


def test_terminal_frame_observer_output_is_stale():
    """phy_steps=10, kill at physics_step=15 ⇒ termination occurs inside the
    SECOND action step's physics loop. The recorder still gets a second
    ``on_post_action_step`` frame — but the observer only refreshed once."""
    runtime, obs, rec = _build(kill_at=15)
    runtime.reset(seed=0)

    runtime.step(np.zeros(21), np.zeros(21))   # step 1: 10 phy steps, alive
    assert runtime.is_episode_active
    runtime.step(np.zeros(21), np.zeros(21))   # step 2: dies at phy step 15
    assert not runtime.is_episode_active

    # The recorder saw TWO post-action frames.
    assert len(rec.frames) == 2
    # But the observer only refreshed once — its on_post_action_step never
    # ran for the terminal transition (early return skipped the hook).
    assert obs.refresh_count == 1
    # Frame 2's recorded observer output is the stale pre-refresh value.
    assert rec.frames[1]["probe"] == 1  # stale — a fresh read would be 2


def test_post_action_termination_refreshes_normally():
    """Control case: termination requested from ``on_post_action_step``
    (e.g. TimeoutPlugin) does NOT skip the observer refresh — the hook runs
    first, then termination is handled. Shows the gap is specific to
    physics-step termination."""
    from envs.framework.common_plugins import TimeoutPlugin

    obs = _StepCountingObserver()
    rec = _CaptureRecorder()
    runtime = EnvRuntime(
        simulator=MockSimulator(),
        plugins=[TimeoutPlugin(max_steps=2)],
        observer_plugins={"probe": obs},
        recorders=[rec],
        phy_steps_per_action=10,
    )
    runtime.reset(seed=0)
    runtime.step(np.zeros(21), np.zeros(21))
    runtime.step(np.zeros(21), np.zeros(21))
    assert not runtime.is_episode_active

    assert len(rec.frames) == 2
    assert obs.refresh_count == 2
    assert rec.frames[1]["probe"] == 2  # fresh
