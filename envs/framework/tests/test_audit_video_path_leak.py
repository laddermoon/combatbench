"""Regression test (P3-2 fix) — per-episode output_path no longer sticky.

``VideoRecorderPlugin.on_pre_episode`` now restores the ctor default when
an episode carries no ``video_output_path`` override — the override covers
only the episode that passed it, as the class docstring promises.

Run: PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_video_path_leak.py -q
"""
from __future__ import annotations

from envs.framework.common_plugins import VideoRecorderPlugin


class _MiniCtx:
    def __init__(self, options):
        self.episode_options = options
        self.accessor = self
        self._f = 0

    def get_physical_frequency(self):
        return 500.0

    def get_broadcastview_image(self):
        return None


def test_episode_path_override_scoped_to_one_episode():
    """An override episode is followed by ctor-default episodes."""
    p = VideoRecorderPlugin(fps=30, output_path="/tmp/ctor_default.mp4")

    # Episode 1: no override — ctor default
    p.on_pre_episode(_MiniCtx({}))
    assert str(p.output_path) == "/tmp/ctor_default.mp4"

    # Episode 2: override applies to this episode
    p.on_pre_episode(_MiniCtx({"video_output_path": "/tmp/override.mp4"}))
    assert str(p.output_path) == "/tmp/override.mp4"

    # Episode 3: no override — ctor default restored
    p.on_pre_episode(_MiniCtx({}))
    assert str(p.output_path) == "/tmp/ctor_default.mp4"
