"""Tests for :mod:`envs.framework.blueprint`.

Covers:
* Round-trip ``EnvRuntime -> EnvBlueprint -> EnvRuntime``.
* YAML / JSON serialization stability.
* Filtering of ``BLUEPRINT_EXCLUDE`` plugins (``VideoRecorderPlugin``)
  and the internal ``TimeoutPlugin``.
* Component opt-in via ``to_blueprint`` / ``from_blueprint``.
* ``build(debug_plugins=...)`` rejects non-excluded plugins and accepts
  excluded ones.
* Recorders are accepted via ``build(recorders=...)`` and never appear
  in the blueprint.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from envs.framework.blueprint import ClassSpec, EnvBlueprint, _resolve_class
from envs.framework.common_plugins import TimeoutPlugin, VideoRecorderPlugin
from envs.framework.env_runtime import EnvRuntime
from envs.framework.plugin import BasePlugin
from envs.framework.recorder import EpisodeBufferRecorder
from envs.framework.runtime_plugin import BaseObserverPlugin

# Re-import MockSimulator from conftest indirectly via fixture.


# ---------------------------------------------------------------------------
# Test components (module-level so _resolve_class can find them)
# ---------------------------------------------------------------------------
class _ConfigurablePlugin(BasePlugin):
    """Plugin that opts into the blueprint protocol."""

    def __init__(self, threshold: float = 0.5, label: str = "x"):
        self.threshold = threshold
        self.label = label

    @property
    def name(self) -> str:
        return f"configurable[{self.label}]"

    def to_blueprint(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "label": self.label}


class _StatelessPlugin(BasePlugin):
    """Plugin without ``to_blueprint`` -> default empty config."""

    @property
    def name(self) -> str:
        return "stateless"


class _NamedObserver(BaseObserverPlugin):
    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._step = 0

    def on_post_action_step(self, ctx) -> None:
        self._step += 1

    def get_output(self):
        return self._step

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_minimal_runtime_round_trip(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            phy_steps_per_action=4,
            max_steps=12,
        )
        blueprint = runtime.to_blueprint()
        assert blueprint.phy_steps_per_action == 4
        assert blueprint.max_steps == 12
        assert blueprint.plugins == ()
        assert blueprint.observer_plugins == {}

        # Build a fresh runtime and verify scalar fields match.
        from envs.framework.tests.conftest import MockSimulator  # noqa: WPS433
        new_runtime = blueprint.build()
        assert new_runtime._core.phy_steps_per_action == 4
        assert isinstance(new_runtime.simulator, MockSimulator)

    def test_round_trip_preserves_plugin_config(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ConfigurablePlugin(threshold=0.7, label="hello")],
            observer_plugins={"obs": _NamedObserver(agent_id="robot_b")},
            phy_steps_per_action=2,
            max_steps=5,
        )
        blueprint = runtime.to_blueprint()
        assert len(blueprint.plugins) == 1
        plugin_spec = blueprint.plugins[0]
        assert plugin_spec.config == {"threshold": 0.7, "label": "hello"}
        assert blueprint.observer_plugins["obs"].config == {"agent_id": "robot_b"}

        new_runtime = blueprint.build()
        plugins = [p for p in new_runtime.plugins if isinstance(p, _ConfigurablePlugin)]
        assert len(plugins) == 1
        assert plugins[0].threshold == 0.7
        assert plugins[0].label == "hello"
        assert isinstance(new_runtime.observer_plugins["obs"], _NamedObserver)
        assert new_runtime.observer_plugins["obs"].agent_id == "robot_b"

    def test_stateless_plugin_round_trip(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_StatelessPlugin()],
        )
        blueprint = runtime.to_blueprint()
        assert blueprint.plugins[0].config == {}
        new_runtime = blueprint.build()
        assert any(isinstance(p, _StatelessPlugin) for p in new_runtime.plugins)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
class TestFiltering:
    def test_video_recorder_excluded(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[VideoRecorderPlugin(fps=30, output_path="x.mp4")],
        )
        blueprint = runtime.to_blueprint()
        assert blueprint.plugins == (), (
            "VideoRecorderPlugin must be filtered out of the blueprint"
        )

    def test_timeout_plugin_round_trips_via_max_steps(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            max_steps=42,
        )
        blueprint = runtime.to_blueprint()
        # TimeoutPlugin is auto-attached by EnvRuntime when max_steps is set,
        # but the blueprint stores it as ``max_steps``, not as a plugin entry.
        assert blueprint.max_steps == 42
        assert all(spec.cls != "envs.framework.common_plugins:TimeoutPlugin"
                   for spec in blueprint.plugins)

        new_runtime = blueprint.build()
        timeout_plugins = [
            p for p in new_runtime.plugins if isinstance(p, TimeoutPlugin)
        ]
        assert len(timeout_plugins) == 1
        assert timeout_plugins[0].max_steps == 42

    def test_recorders_never_appear_in_blueprint(self, mock_simulator):
        recorder = EpisodeBufferRecorder()
        runtime = EnvRuntime(
            simulator=mock_simulator,
            recorders=[recorder],
        )
        blueprint = runtime.to_blueprint()
        # ``EnvBlueprint`` has no field for recorders at all.
        assert not hasattr(blueprint, "recorders")
        # And nothing in plugins matches.
        assert blueprint.plugins == ()


# ---------------------------------------------------------------------------
# build() debug_plugins / recorders
# ---------------------------------------------------------------------------
class TestBuildExtras:
    def test_build_accepts_recorders(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        blueprint = runtime.to_blueprint()
        recorder = EpisodeBufferRecorder()
        new_runtime = blueprint.build(recorders=[recorder])
        assert recorder in new_runtime.recorders

    def test_build_accepts_blueprint_excluded_debug_plugins(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        blueprint = runtime.to_blueprint()
        video = VideoRecorderPlugin(fps=30, output_path="dbg.mp4")
        new_runtime = blueprint.build(debug_plugins=[video])
        assert video in new_runtime.plugins

    def test_build_rejects_non_excluded_debug_plugin(self, mock_simulator):
        runtime = EnvRuntime(simulator=mock_simulator)
        blueprint = runtime.to_blueprint()
        with pytest.raises(ValueError, match="BLUEPRINT_EXCLUDE"):
            blueprint.build(debug_plugins=[_StatelessPlugin()])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_dict_round_trip(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ConfigurablePlugin(threshold=0.3, label="abc")],
            observer_plugins={"obs": _NamedObserver()},
            phy_steps_per_action=3,
            max_steps=9,
        )
        blueprint = runtime.to_blueprint()
        data = blueprint.to_dict()
        restored = EnvBlueprint.from_dict(data)
        assert restored == blueprint

    def test_text_round_trip(self, mock_simulator):
        runtime = EnvRuntime(
            simulator=mock_simulator,
            plugins=[_ConfigurablePlugin(threshold=0.1, label="t")],
        )
        blueprint = runtime.to_blueprint()
        text = blueprint.to_yaml()
        assert "_ConfigurablePlugin" in text
        assert "threshold" in text
        restored = EnvBlueprint.from_yaml(text)
        assert restored == blueprint

    def test_save_load_round_trip(self, mock_simulator, tmp_path):
        runtime = EnvRuntime(simulator=mock_simulator, max_steps=7)
        blueprint = runtime.to_blueprint()
        path = tmp_path / "env.blueprint.yaml"
        blueprint.save(path)
        assert path.exists()
        restored = EnvBlueprint.load(path)
        assert restored == blueprint


# ---------------------------------------------------------------------------
# ClassSpec resolution
# ---------------------------------------------------------------------------
class TestClassSpec:
    def test_resolve_class_colon_form(self):
        cls = _resolve_class(
            "envs.framework.common_plugins:VideoRecorderPlugin"
        )
        assert cls is VideoRecorderPlugin

    def test_resolve_class_dotted_form_compat(self):
        cls = _resolve_class(
            "envs.framework.common_plugins.VideoRecorderPlugin"
        )
        assert cls is VideoRecorderPlugin

    def test_class_spec_dict_round_trip(self):
        spec = ClassSpec(cls="m:Foo", config={"a": 1})
        assert ClassSpec.from_dict(spec.to_dict()) == spec
