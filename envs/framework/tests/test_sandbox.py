"""Strict accessor/mutator sandbox (A3, reinstated after DATASPEC extension).

Plugins must be unable to reach backend internals through ``ctx.accessor``.
After the humanoid21 simulator's ``get_static_data()`` / ``get_derived_state()``
were extended to expose all physical quantities observers need, the proxies in
``envs/framework/context.py`` switched to a **strict allowlist**: only the
methods declared in :class:`IDataAccessor` (plus ``get_physical_frequency``)
and :class:`IDataMutator` are forwarded. Everything else raises
``AttributeError``.
"""
from __future__ import annotations

import numpy as np
import pytest

from envs.framework.context import (
    SimContext,
    _AccessorView,
    _MutatorView,
    _ACCESSOR_ALLOWED,
    _MUTATOR_ALLOWED,
)
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin


class TestAccessorViewSandbox:
    def test_accessor_is_proxy_not_simulator(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        assert isinstance(ctx.accessor, _AccessorView)
        assert ctx.accessor is not mock_simulator

    def test_accessor_exposes_exactly_allowlist(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        for name in _ACCESSOR_ALLOWED:
            assert hasattr(ctx.accessor, name), f"accessor missing allowed method {name!r}"

    def test_accessor_blocks_mutator_methods(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        for blocked in ("set_core_state", "set_action", "apply_external_force",
                        "reset", "physical_step", "close"):
            assert not hasattr(ctx.accessor, blocked), \
                f"accessor must not expose {blocked!r}"

    def test_accessor_blocks_backend_attributes(self, mock_simulator):
        """Backend escape hatches (model/data/_robot_cache etc.) are unreachable."""
        ctx = SimContext(mock_simulator)
        for forbidden in ("model", "data", "_robot_cache", "_state", "_simulator"):
            assert not hasattr(ctx.accessor, forbidden), \
                f"accessor must not expose backend field {forbidden!r}"

    def test_accessor_raises_with_helpful_message(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        with pytest.raises(AttributeError, match="Only"):
            _ = ctx.accessor.model  # type: ignore[attr-defined]

    def test_accessor_forwards_read_methods(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        core = ctx.accessor.get_core_state()
        assert "qpos" in core and "robot_a" in core
        static = ctx.accessor.get_static_data()
        assert "dt" in static

    def test_accessor_is_immutable(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        with pytest.raises(AttributeError, match="immutable"):
            ctx.accessor.injected_attr = 42  # type: ignore[attr-defined]


class TestMutatorViewSandbox:
    def test_mutator_is_none_outside_writable_hooks(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        assert ctx.mutator is None

    def test_mutator_proxy_when_granted(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        ctx._grant_mutator()
        assert isinstance(ctx.mutator, _MutatorView)
        for name in _MUTATOR_ALLOWED:
            assert hasattr(ctx.mutator, name)
        # Accessor-side reads unreachable through the mutator view
        assert not hasattr(ctx.mutator, "get_core_state")
        assert not hasattr(ctx.mutator, "model")

    def test_mutator_proxy_forwards_writes(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        ctx._grant_mutator()
        ctx.mutator.set_core_state({"custom_key": "sentinel"})
        assert mock_simulator.get_core_state()["custom_key"] == "sentinel"

    def test_revoke_clears_reference(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        ctx._grant_mutator()
        ctx._revoke_mutator()
        assert ctx.mutator is None

    def test_mutator_is_immutable(self, mock_simulator):
        ctx = SimContext(mock_simulator)
        ctx._grant_mutator()
        with pytest.raises(AttributeError, match="immutable"):
            ctx.mutator.extra = 99  # type: ignore[attr-defined]


class TestSandboxEndToEnd:
    def test_sneaky_plugin_cannot_reach_backend(self, mock_simulator):
        """A plugin that tries to write through ctx.accessor must fail."""

        class Sneaky(BasePlugin):
            def __init__(self):
                self.attempt_error: Exception | None = None

            @property
            def name(self):
                return "sneaky"

            def on_post_phy_step(self, ctx):
                try:
                    ctx.accessor.set_core_state({"hacked": True})
                except Exception as exc:
                    self.attempt_error = exc

        sneaky = Sneaky()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[sneaky],
            phy_steps_per_action=1,
            strict=False,  # keep runtime going so we can inspect attempt_error
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert isinstance(sneaky.attempt_error, AttributeError)
        assert "hacked" not in mock_simulator.get_core_state()

    def test_sneaky_plugin_cannot_reach_backend_attribute(self, mock_simulator):
        """Reaching for ``ctx.accessor._state`` (MockSimulator internal) is blocked."""

        class Peeker(BasePlugin):
            def __init__(self):
                self.err: Exception | None = None

            @property
            def name(self):
                return "peeker"

            def on_post_phy_step(self, ctx):
                try:
                    _ = ctx.accessor._state  # type: ignore[attr-defined]
                except Exception as exc:
                    self.err = exc

        peeker = Peeker()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[peeker],
            phy_steps_per_action=1,
            strict=False,
        )
        runtime.reset()
        runtime.step(np.zeros(21), np.zeros(21))
        assert isinstance(peeker.err, AttributeError)
