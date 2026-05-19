"""Observer dispatcher ordering + read-only enforcement.

Two contracts under test:

1. ``_ObserverDispatcherPlugin`` runs **first** in every hook it subscribes
   to, so downstream termination / reward plugins on the same hook always
   read observer outputs that reflect the *current* step's state.

2. The dispatcher is **read-only** regardless of whether the hook is writable.
   This is enforced by the generic ``require_mutator=False`` path in
   ``_PluginManager.invoke`` — not by the hook's read-only-ness. We pin this
   by placing the dispatcher on a writable hook (``on_pre_episode``) and
   asserting it still receives ``ctx.mutator is None``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin
from envs.framework.runtime_plugin import (
    BaseObserverPlugin,
    _ObserverDispatcherPlugin,
)


class _OrderTracker(BasePlugin):
    """Records the order in which plugins are invoked within each hook."""

    def __init__(self, label: str, priority: int = 0, require_mutator: bool = False):
        self._label = label
        self._priority = priority
        self._require_mutator = require_mutator
        self.pre_episode_log: List[str] = []
        self.post_action_log: List[str] = []
        self.pre_episode_mutator_states: List[bool] = []

    @property
    def name(self) -> str:
        return self._label

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def require_mutator(self) -> bool:
        return self._require_mutator

    def on_pre_episode(self, ctx) -> None:
        self.pre_episode_log.append(self._label)
        self.pre_episode_mutator_states.append(ctx.mutator is not None)

    def on_post_action_step(self, ctx) -> None:
        self.post_action_log.append(self._label)


class _StepStampObserver(BaseObserverPlugin):
    """Observer whose output is the last ``episode_step`` it saw.

    Downstream plugins on the same ``on_post_action_step`` hook will be able
    to read this stamp; if the dispatcher runs FIRST, they read the *current*
    step. If it runs last, they read the previous step.
    """

    def __init__(self) -> None:
        self._last_step: int = -1

    def on_pre_episode(self, ctx) -> None:
        self._last_step = ctx.episode_step

    def on_post_action_step(self, ctx) -> None:
        self._last_step = ctx.episode_step

    def get_output(self) -> int:
        return self._last_step


class _StepReader(BasePlugin):
    """Plugin that records what the observer reports during post-action."""

    def __init__(self, dispatcher: _ObserverDispatcherPlugin) -> None:
        self._dispatcher = dispatcher
        self.observed_vs_ctx: List[tuple[int, int]] = []

    @property
    def name(self) -> str:
        return "step_reader"

    @property
    def priority(self) -> int:
        # Lower than the dispatcher's +1_000_000 so the dispatcher wins.
        return 0

    @property
    def require_mutator(self) -> bool:
        return False

    def on_post_action_step(self, ctx) -> None:
        self.observed_vs_ctx.append(
            (self._dispatcher.get_output("step_stamp"), ctx.episode_step)
        )


def _build_runtime(
    mock_simulator,
    extra_plugins: Optional[List[BasePlugin]] = None,
) -> tuple[EnvRuntime, _ObserverDispatcherPlugin, _StepStampObserver]:
    dispatcher = _ObserverDispatcherPlugin()
    observer = _StepStampObserver()
    dispatcher.set_observer_plugin("step_stamp", observer)
    plugins: List[BasePlugin] = [dispatcher]
    if extra_plugins:
        plugins.extend(extra_plugins)
    runtime = EnvRuntime(
        simulator=mock_simulator,
        plugins=plugins,
        phy_steps_per_action=1,
    )
    return runtime, dispatcher, observer


class TestObserverDispatcherOrdering:
    def test_priority_is_positive_maximum(self):
        """The dispatcher's priority must be strictly greater than any
        reasonable user plugin priority so it sorts to the front."""
        dispatcher = _ObserverDispatcherPlugin()
        assert dispatcher.priority == 1_000_000

    def test_dispatcher_runs_before_default_priority_on_post_action_step(
        self, mock_simulator
    ):
        """Step reader (priority 0) must see the observer's fresh output."""
        dispatcher = _ObserverDispatcherPlugin()
        observer = _StepStampObserver()
        dispatcher.set_observer_plugin("step_stamp", observer)
        reader = _StepReader(dispatcher)
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[dispatcher, reader],
            phy_steps_per_action=1,
        )
        runtime.reset()
        for _ in range(3):
            runtime.step(np.zeros(21), np.zeros(21))

        assert len(reader.observed_vs_ctx) == 3
        for observed, current in reader.observed_vs_ctx:
            assert observed == current, (
                f"Step reader saw stale observer output: "
                f"observed={observed} current={current}"
            )

    def test_dispatcher_runs_before_default_priority_on_pre_episode(
        self, mock_simulator
    ):
        tracker = _OrderTracker("user_plugin", priority=0)
        runtime, dispatcher, _ = _build_runtime(mock_simulator, [tracker])
        # Sanity: sort result puts dispatcher before tracker
        plugin_order = [p.name for p in runtime._core.plugin_manager.iter_plugins()]
        assert plugin_order.index("observer_dispatcher") < plugin_order.index(
            "user_plugin"
        )

        runtime.reset()
        assert tracker.pre_episode_log == ["user_plugin"]
        # Dispatcher has no per-hook tracker; its execution is inferred from
        # the sorted plugin_order above + the next test verifying its
        # read-only contract.


class TestObserverDispatcherIsReadOnly:
    def test_observer_dispatcher_declares_no_mutator(self):
        dispatcher = _ObserverDispatcherPlugin()
        assert dispatcher.require_mutator is False

    def test_require_mutator_false_blocks_writes_even_in_writable_hook(
        self, mock_simulator
    ):
        """A plugin with require_mutator=False placed on ``on_pre_episode``
        (a writable hook) must still see ``ctx.mutator is None``."""
        tracker = _OrderTracker(
            "readonly_by_choice", priority=0, require_mutator=False
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[tracker],
            phy_steps_per_action=1,
        )
        runtime.reset()
        assert tracker.pre_episode_mutator_states == [False]

    def test_require_mutator_true_receives_mutator_in_writable_hook(
        self, mock_simulator
    ):
        """Control: the same scenario with require_mutator=True must hand
        over a live mutator. This pins the selectivity of the mechanism."""
        tracker = _OrderTracker(
            "writable", priority=0, require_mutator=True
        )
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[tracker],
            phy_steps_per_action=1,
        )
        runtime.reset()
        assert tracker.pre_episode_mutator_states == [True]

    def test_observer_dispatcher_never_sees_mutator_through_full_step(
        self, mock_simulator
    ):
        """End-to-end: run a full reset + step; the dispatcher's internal
        hooks (``on_pre_episode``, ``on_post_action_step``, ``on_post_episode``)
        must all have received ``ctx.mutator is None``.

        We patch the dispatcher to record the mutator state each time a
        hook fires.
        """
        dispatcher = _ObserverDispatcherPlugin()
        observer = _StepStampObserver()
        dispatcher.set_observer_plugin("step_stamp", observer)

        seen_mutator_states: List[tuple[str, bool]] = []
        for hook in (
            "on_pre_episode",
            "on_post_action_step",
            "on_post_episode",
        ):
            original = getattr(dispatcher, hook)

            def make_spy(hook_name=hook, wrapped=original):
                def spy(ctx):
                    seen_mutator_states.append(
                        (hook_name, ctx.mutator is not None)
                    )
                    return wrapped(ctx)
                return spy

            setattr(dispatcher, hook, make_spy())

        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[dispatcher],
            phy_steps_per_action=1,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        # Force termination so on_post_episode also fires.
        runtime._core.ctx.request_termination("test")
        runtime.step(np.zeros(21), np.zeros(21))

        hooks_invoked = {name for name, _ in seen_mutator_states}
        assert {"on_pre_episode", "on_post_action_step"}.issubset(hooks_invoked)
        for hook_name, had_mutator in seen_mutator_states:
            assert not had_mutator, (
                f"observer_dispatcher saw a live mutator at {hook_name}; "
                f"require_mutator=False was not honored"
            )
