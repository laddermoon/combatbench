"""Terminal-frame contract tests (inclusive-boundary semantics).

Contract under test:

- Every entered ``step()`` emits exactly one recorder frame and fires
  ``on_post_action_step`` exactly once — including the frame where a
  world plugin requests termination mid-physics. The terminal frame is
  a valid transition and its observer outputs reflect the post-action
  refresh on the terminal state.
- ``ctx.episode_step`` counts entered ``step()`` calls unconditionally;
  whether a frame's action actually executed physics is decided by the
  ``physics_step`` delta, not by the step counter.

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


def test_terminal_frame_observer_output_is_fresh():
    """phy_steps=10, kill at physics_step=15 ⇒ termination inside the
    SECOND action step's physics loop. ``on_post_action_step`` still fires
    once for that entered step, so the terminal frame carries the
    observer value refreshed on the terminal state."""
    runtime, obs, rec = _build(kill_at=15)
    runtime.reset(seed=0)

    runtime.step(np.zeros(21), np.zeros(21))   # step 1: 10 phy steps, alive
    assert runtime.is_episode_active
    runtime.step(np.zeros(21), np.zeros(21))   # step 2: dies at phy step 15
    assert not runtime.is_episode_active

    # The recorder saw TWO post-action frames — the terminal frame is kept.
    assert len(rec.frames) == 2
    # The observer refreshed on every entered step, including the terminal one.
    assert obs.refresh_count == 2
    assert rec.frames[1]["probe"] == 2  # refreshed on the terminal state


def test_degenerate_frame_zero_physics():
    """Termination proposed in ``on_pre_action_step`` (before any physics):
    the step still counts ``episode_step`` and emits a recorder frame, but
    ``physics_step`` does not advance — consumers use the delta to exclude
    the degenerate frame from trajectories."""
    killed = {"fired": False}

    class _PreActionKiller(BasePlugin):
        @property
        def name(self) -> str:
            return "pre_action_killer"

        def on_pre_action_step(self, ctx) -> None:
            if not killed["fired"]:
                killed["fired"] = True
                ctx.request_termination("audit_pre_action")

    obs = _StepCountingObserver()
    rec = _CaptureRecorder()
    runtime = EnvRuntime(
        simulator=MockSimulator(),
        plugins=[_PreActionKiller()],
        observer_plugins={"probe": obs},
        recorders=[rec],
        phy_steps_per_action=10,
    )
    runtime.reset(seed=0)
    phys0 = runtime.ctx.physics_step

    runtime.step(np.zeros(21), np.zeros(21))
    assert not runtime.is_episode_active
    # episode_step counted the entered call even though no physics ran.
    assert runtime.ctx.episode_step == 1
    assert runtime.ctx.physics_step == phys0
    # The frame was still recorded and post_action still fired once.
    assert len(rec.frames) == 1
    assert obs.refresh_count == 1


def test_post_action_termination_refreshes_normally():
    """Control case: termination requested from ``on_post_action_step``
    (e.g. TimeoutPlugin) refreshes the observer as usual — the hook runs
    first, then termination is handled."""
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
