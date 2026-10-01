"""Audit probe (Phase 2) — NOT a fix.

Question under audit: VideoRecorderPlugin's per-episode
``video_output_path`` override (ctx.episode_options) is documented as
covering *this episode only*, but the implementation assigns straight
into ``self.output_path`` — permanently replacing the ctor default for
all subsequent episodes that don't pass an override.

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


def test_episode_path_override_leaks_into_next_episode():
    """Locks in audited behavior: override is sticky.

    When the plugin is fixed (save/restore or a per-episode variable),
    this test SHOULD FAIL — flip the final assert.
    """
    p = VideoRecorderPlugin(fps=30, output_path="/tmp/ctor_default.mp4")

    # Episode 1: no override — uses ctor default
    p.on_pre_episode(_MiniCtx({}))
    assert str(p.output_path) == "/tmp/ctor_default.mp4"

    # Episode 2: override to a different path
    p.on_pre_episode(_MiniCtx({"video_output_path": "/tmp/override.mp4"}))
    assert str(p.output_path) == "/tmp/override.mp4"

    # Episode 3: no override — documented behavior says ctor default
    # returns; actual behavior keeps the override.
    p.on_pre_episode(_MiniCtx({}))
    assert str(p.output_path) == "/tmp/override.mp4", (
        "override no longer sticky — bug may be fixed; flip this test")
