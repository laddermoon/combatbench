"""VideoRecorderPlugin: instance-scoped output path with per-episode
options override.

History
-------
* Originally a class-level mutable ``videosave_path`` variable + class
  setter — cross-polluted multiple runtimes.
* Then replaced by a pure instance attribute + ``set_output_path()``
  instance method ("A5").
* Now: ctor default + per-episode override via
  ``EnvRuntime.reset(options={VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: ...})``
  picked up in ``on_pre_episode``. ``set_output_path`` is gone — routing
  through ``ctx.episode_options`` shares one paradigm with ``base_seed``
  / HP carry-over and is naturally isolated across runtimes.
"""
from __future__ import annotations

import numpy as np

from envs.framework.common_plugins import VideoRecorderPlugin


class TestVideoRecorderPluginIsolation:
    def test_no_shared_class_variable(self):
        """The legacy class-level override must be gone."""
        assert not hasattr(VideoRecorderPlugin, "videosave_path")
        assert not hasattr(VideoRecorderPlugin, "set_videosave_path")

    def test_no_set_output_path_setter(self):
        """The instance setter is also gone — use ``options`` instead."""
        assert not hasattr(VideoRecorderPlugin, "set_output_path")

    def test_init_sets_instance_output_path(self):
        plugin = VideoRecorderPlugin(fps=30, output_path="a.mp4")
        assert str(plugin.output_path) == "a.mp4"

    def test_two_instances_do_not_share_path(self):
        plugin_a = VideoRecorderPlugin(output_path="round_a.mp4")
        plugin_b = VideoRecorderPlugin(output_path="round_b.mp4")
        assert str(plugin_a.output_path) == "round_a.mp4"
        assert str(plugin_b.output_path) == "round_b.mp4"


class TestPerEpisodeOptionsOverride:
    """``ctx.episode_options[VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY]``
    overrides the ctor default for that one episode."""

    def test_options_key_overrides_ctor_default(self, mock_simulator):
        from envs.framework.env_runtime import EnvRuntime

        plugin = VideoRecorderPlugin(output_path="default.mp4")
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        runtime.reset(
            options={VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: "override.mp4"}
        )
        assert str(plugin.output_path) == "override.mp4"

    def test_no_options_keeps_ctor_default(self, mock_simulator):
        from envs.framework.env_runtime import EnvRuntime

        plugin = VideoRecorderPlugin(output_path="default.mp4")
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        runtime.reset(options={})
        assert str(plugin.output_path) == "default.mp4"

    def test_override_is_per_episode_only(self, mock_simulator):
        """After an override episode, the next episode without the option
        keeps the *previous override* as its in-memory default — the option
        only changes the path the next episode will save to. Verify the
        plugin honors whatever options the **current** episode carries."""
        from envs.framework.env_runtime import EnvRuntime

        plugin = VideoRecorderPlugin(output_path="default.mp4")
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])

        runtime.reset(
            options={VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: "ep1.mp4"}
        )
        assert str(plugin.output_path) == "ep1.mp4"
        # Drive the episode to terminal so the next reset is clean.
        runtime._core.ctx.request_termination("test")
        runtime.step(np.zeros(21), np.zeros(21))

        runtime.reset(
            options={VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: "ep2.mp4"}
        )
        assert str(plugin.output_path) == "ep2.mp4"


class TestRoundRunnerVideoSavePath:
    """``RoundRunner.run(videosave_path=...)`` must route the path through
    ``options`` rather than mutating the plugin directly."""

    def test_videosave_path_merges_into_options(self):
        from envs.framework.round_runner import RoundRunner

        merged = RoundRunner._merge_video_path_into_options(
            options=None, videosave_path="shot.mp4"
        )
        assert merged == {
            VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: "shot.mp4"
        }

    def test_caller_options_win_over_videosave_path(self):
        from envs.framework.round_runner import RoundRunner

        merged = RoundRunner._merge_video_path_into_options(
            options={VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY: "caller.mp4"},
            videosave_path="shot.mp4",
        )
        assert merged[VideoRecorderPlugin.OPTIONS_OUTPUT_PATH_KEY] == "caller.mp4"

    def test_videosave_path_none_passes_options_through(self):
        from envs.framework.round_runner import RoundRunner

        opts = {"hp_a": 100.0}
        assert RoundRunner._merge_video_path_into_options(opts, None) is opts
        assert RoundRunner._merge_video_path_into_options(None, None) is None

    def test_find_plugins_returns_video_recorder(self, mock_simulator):
        from envs.framework.env_runtime import EnvRuntime

        plugin = VideoRecorderPlugin(output_path="initial.mp4")
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        found = runtime.find_plugins(VideoRecorderPlugin)
        assert found == (plugin,)
