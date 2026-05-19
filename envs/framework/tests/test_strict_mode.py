"""Strict-mode exception handling (A2).

EnvRuntime defaults to ``strict=True``: any exception raised inside a plugin /
observer / recorder hook propagates and stops the runtime. ``strict=False``
swaps to best-effort mode where the exception is logged (with traceback) but
the runtime continues.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from envs.framework.context import ReadOnlySimContext
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin
from envs.framework.recorder import PostActionRecorder
from envs.framework.runtime_plugin import BaseObserverPlugin


class _ExplodingPlugin(BasePlugin):
    def __init__(self, hook: str):
        self._hook = hook

    @property
    def name(self) -> str:
        return "exploder"

    def on_pre_episode(self, ctx):
        if self._hook == "on_pre_episode":
            raise RuntimeError("boom@on_pre_episode")

    def on_post_action_step(self, ctx):
        if self._hook == "on_post_action_step":
            raise RuntimeError("boom@on_post_action_step")


class _ExplodingObserver(BaseObserverPlugin):
    def process_data(self, ctx):
        raise RuntimeError("boom@observer")

    def get_output(self):
        return 0.0


class _ExplodingRecorder(PostActionRecorder):
    def on_post_action_step(self, ctx: ReadOnlySimContext, observation, action, observer_outputs, action_extras=None):
        raise RuntimeError("boom@recorder")


class TestStrictDefault:
    """Default behaviour should re-raise."""

    def test_plugin_exception_reraised_on_pre_episode(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ExplodingPlugin("on_pre_episode")],
        )
        with pytest.raises(RuntimeError, match="boom@on_pre_episode"):
            runtime.reset()

    def test_plugin_exception_reraised_on_post_action_step(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ExplodingPlugin("on_post_action_step")],
        )
        runtime.reset()
        with pytest.raises(RuntimeError, match="boom@on_post_action_step"):
            runtime.step(np.zeros(21), np.zeros(21))

    def test_observer_exception_reraised(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            observer_plugins={"bad": _ExplodingObserver()},
        )
        with pytest.raises(RuntimeError, match="boom@observer"):
            runtime.reset()

    def test_recorder_exception_reraised(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[_ExplodingRecorder()],
        )
        runtime.reset()
        with pytest.raises(RuntimeError, match="boom@recorder"):
            runtime.step(np.zeros(21), np.zeros(21))


class TestNonStrictBestEffort:
    """``strict=False`` must log traceback (not just warn) and continue."""

    def _assert_logged(self, caplog, pattern: str):
        matched = any(
            pattern in rec.message or pattern in (rec.exc_text or "")
            for rec in caplog.records
        )
        assert matched, f"expected log containing {pattern!r}, got {[rec.message for rec in caplog.records]}"

    def test_plugin_exception_logged(self, mock_simulator, caplog):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ExplodingPlugin("on_post_action_step")],
            strict=False,
        )
        with caplog.at_level(logging.ERROR, logger="combatbench.envs.framework"):
            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))  # must not raise
        self._assert_logged(caplog, "boom@on_post_action_step")

    def test_recorder_exception_logged(self, mock_simulator, caplog):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[_ExplodingRecorder()],
            strict=False,
        )
        with caplog.at_level(logging.ERROR, logger="combatbench.envs.framework"):
            runtime.reset()
            runtime.step(np.zeros(21), np.zeros(21))
        self._assert_logged(caplog, "boom@recorder")

    def test_logged_records_contain_traceback(self, mock_simulator, caplog):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ExplodingPlugin("on_pre_episode")],
            strict=False,
        )
        with caplog.at_level(logging.ERROR, logger="combatbench.envs.framework"):
            runtime.reset()
        assert any(rec.exc_info is not None for rec in caplog.records), \
            "expected at least one log record to carry exception info (traceback)"
