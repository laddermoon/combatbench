"""VideoRecorderPlugin (A5): instance-scoped output path.

The old implementation used a class-level mutable variable
``VideoRecorderPlugin.videosave_path`` plus a classmethod setter, which
cross-polluted multiple runtimes. A5 replaces it with a pure instance
attribute + ``set_output_path()`` instance method.
"""
from __future__ import annotations

from envs.framework.common_plugins import VideoRecorderPlugin


class TestVideoRecorderPluginIsolation:
    def test_no_shared_class_variable(self):
        """The legacy class-level override must be gone."""
        assert not hasattr(VideoRecorderPlugin, "videosave_path")
        assert not hasattr(VideoRecorderPlugin, "set_videosave_path")

    def test_init_sets_instance_output_path(self):
        plugin = VideoRecorderPlugin(fps=30, output_path="a.mp4")
        assert str(plugin.output_path) == "a.mp4"

    def test_two_instances_do_not_share_path(self):
        plugin_a = VideoRecorderPlugin(output_path="round_a.mp4")
        plugin_b = VideoRecorderPlugin(output_path="round_b.mp4")
        plugin_a.set_output_path("round_a_new.mp4")
        assert str(plugin_b.output_path) == "round_b.mp4"
        assert str(plugin_a.output_path) == "round_a_new.mp4"

    def test_set_output_path_none_is_noop(self):
        plugin = VideoRecorderPlugin(output_path="keep.mp4")
        plugin.set_output_path(None)
        assert str(plugin.output_path) == "keep.mp4"


class TestRoundRunnerSetsOutputPath:
    """RoundRunner.run(videosave_path=...) should locate attached
    VideoRecorderPlugin instances via runtime.find_plugins and update them.
    """

    def test_find_plugins_returns_video_recorder(self, mock_simulator):
        from envs.framework.env_runtime import EnvRuntime

        plugin = VideoRecorderPlugin(output_path="initial.mp4")
        runtime = EnvRuntime(simulator=mock_simulator, plugins=[plugin])
        found = runtime.find_plugins(VideoRecorderPlugin)
        assert found == (plugin,)
