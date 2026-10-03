"""Sandbox seal test — the raw simulator must be unreachable from ctx.

P-FW-4: ``SimContext`` used to carry ``self._simulator`` (the bare
BaseSimulator), so any plugin — including in read-only hooks where
``ctx.mutator is None`` — could do ``ctx._simulator.set_core_state(...)``
and bypass the whole accessor/mutator grant mechanism. The attribute was
deleted; this test pins the seal:

* plugin hooks (writable AND read-only) see no ``ctx._simulator``;
* observer hooks see no ``ctx._simulator`` / ``ctx.accessor._simulator``;
* recorder hooks see no ``ctx._simulator``;
* ``ctx.accessor`` and ``ctx.mutator`` never expose ``_simulator`` (the
  name-mangled ``__sim`` slots stay the documented opt-out convention —
  ``_AccessorView__sim`` / ``_MutatorView__sim`` — and are NOT covered by
  the ``_simulator`` name check).

Run: ``PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_simulator_reach.py -q``
"""
from __future__ import annotations

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin
from envs.framework.observer_plugin import BaseObserverPlugin
from envs.framework.recorder import PostActionRecorder


def _probe(ctx, sink: list, hook: str) -> None:
    """Record whether the raw simulator is reachable from this ctx.

    ``ReadOnlySimContext`` (observer ctx) has no ``mutator`` field at all,
    so probe it via ``getattr`` rather than assuming the SimContext shape.
    """
    mutator = getattr(ctx, "mutator", None)
    sink.append(
        (
            hook,
            getattr(ctx, "_simulator", "ABSENT"),
            getattr(ctx.accessor, "_simulator", "ABSENT"),
            getattr(mutator, "_simulator", "ABSENT") if mutator is not None else "no-mutator",
        )
    )


class _ReachProbingPlugin(BasePlugin):
    """Probes every plugin hook for ``_simulator`` reachability."""

    def __init__(self, sink: list):
        self._sink = sink

    @property
    def name(self) -> str:
        return "reach_probe"

    @property
    def require_mutator(self) -> bool:
        return True  # gets mutator on writable hooks

    def on_pre_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "on_pre_episode")

    def on_pre_action_step(self, ctx) -> None:
        _probe(ctx, self._sink, "on_pre_action_step")

    def on_pre_phy_step(self, ctx) -> None:
        _probe(ctx, self._sink, "on_pre_phy_step")

    def on_post_phy_step(self, ctx) -> None:
        _probe(ctx, self._sink, "on_post_phy_step")

    def on_post_action_step(self, ctx) -> None:
        _probe(ctx, self._sink, "on_post_action_step")

    def on_post_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "on_post_episode")


class _ReachProbingObserver(BaseObserverPlugin):
    def __init__(self, sink: list):
        self._sink = sink

    def on_pre_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "obs:on_pre_episode")

    def on_post_action_step(self, ctx) -> None:
        _probe(ctx, self._sink, "obs:on_post_action_step")

    def on_post_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "obs:on_post_episode")

    def get_output(self):
        return 0.0


class _ReachProbingRecorder(PostActionRecorder):
    def __init__(self, sink: list):
        self._sink = sink

    def on_pre_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "rec:on_pre_episode")

    def on_post_action_step(
        self, ctx, observation, action, observer_outputs, action_extras=None
    ) -> None:
        _probe(ctx, self._sink, "rec:on_post_action_step")

    def on_post_episode(self, ctx) -> None:
        _probe(ctx, self._sink, "rec:on_post_episode")


def test_simulator_unreachable_from_any_hook(mock_simulator):
    sink: list = []
    runtime = EnvRuntime(
        simulator=mock_simulator,
        plugins=[_ReachProbingPlugin(sink)],
        observer_plugins={"probe": _ReachProbingObserver(sink)},
        recorders=[_ReachProbingRecorder(sink)],
        max_steps=1,
    )
    runtime.reset(seed=0)
    runtime.step(np.zeros(21), np.zeros(21))
    # Second reset exercises the abandoned-episode path hooks.
    runtime.reset(seed=1)

    assert sink, "no hooks ran — test is vacuous"
    hooks_seen = {entry[0] for entry in sink}
    # Both a writable and a read-only plugin hook must have been probed.
    assert "on_pre_action_step" in hooks_seen
    assert "on_post_action_step" in hooks_seen
    assert "on_post_episode" in hooks_seen

    for hook, sim_attr, accessor_attr, mutator_attr in sink:
        assert sim_attr == "ABSENT", (
            f"{hook}: ctx._simulator reachable — sandbox seal broken "
            f"(got {sim_attr!r})"
        )
        assert accessor_attr == "ABSENT", (
            f"{hook}: ctx.accessor._simulator reachable — sandbox seal broken"
        )
        assert mutator_attr in ("ABSENT", "no-mutator"), (
            f"{hook}: ctx.mutator._simulator reachable — sandbox seal broken"
        )


def test_simulator_cannot_be_mutated_via_ctx_in_readonly_hook(mock_simulator):
    """Before the seal, a plugin could mutate physics in a read-only hook
    via ``ctx._simulator.set_core_state`` — now there is no path at all."""
    evidence: list = []

    class _SneakyPlugin(BasePlugin):
        @property
        def name(self) -> str:
            return "sneaky"

        @property
        def require_mutator(self) -> bool:
            return True

        def on_post_action_step(self, ctx) -> None:
            # Read-only hook: mutator must be withheld AND there must be
            # no back-door through ctx.
            evidence.append(("mutator", ctx.mutator))
            if ctx.mutator is not None:  # pragma: no cover - would be a bug
                evidence.append(("write_attempt", "mutator-leaked"))
            if hasattr(ctx, "_simulator"):  # pragma: no cover - would be a bug
                evidence.append(("write_attempt", "ctx._simulator-leaked"))

    runtime = EnvRuntime(
        simulator=mock_simulator,
        plugins=[_SneakyPlugin()],
        max_steps=1,
    )
    runtime.reset(seed=0)
    runtime.step(np.zeros(21), np.zeros(21))

    assert ("mutator", None) in evidence
    assert not any(tag == "write_attempt" for tag, _ in evidence)
