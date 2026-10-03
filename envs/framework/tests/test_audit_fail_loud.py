"""Fail-loud regression tests — P-FW-5.

Two silent-failure holes were sealed:

1. ``IDataMutator.apply_external_force`` used to default to ``pass`` —
   a disturbance plugin calling it on a backend that never implemented
   the method "succeeded" silently and the experiment ran with **no
   forces applied**. The default now raises ``NotImplementedError``.
   (Same fix applied to the batch hierarchy's
   ``IBatchDataMutator.apply_external_force``.)

2. ``VideoRecorderPlugin.on_post_episode`` used to swallow every failure
   with ``print`` — ffmpeg missing → cv2 missing → "Warning: cannot save
   video", or any write error → "Error saving video". Recorded frames
   were silently dropped. Failures now raise; the ffmpeg→cv2 fallback is
   kept only for the genuine "ffmpeg binary missing" case.

Run: ``PYTHONPATH=. python3 -m pytest envs/framework/tests/test_audit_fail_loud.py -q``
"""
from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pytest

from envs.framework.common_plugins import VideoRecorderPlugin


class TestApplyExternalForceFailLoud:
    def test_default_raises_notimplementederror(self, mock_simulator):
        # conftest MockSimulator does not implement apply_external_force —
        # calling it must raise, not silently no-op.
        with pytest.raises(NotImplementedError, match="apply_external_force"):
            mock_simulator.apply_external_force("torso", np.zeros(3))

    def test_mutator_view_propagates(self, mock_simulator):
        # Through the sandbox mutator view the same call must still raise —
        # the view forwards to the backend, it does not swallow.
        from envs.framework.context import _MutatorView

        view = _MutatorView(mock_simulator)
        with pytest.raises(NotImplementedError, match="apply_external_force"):
            view.apply_external_force("torso", np.zeros(3))


class TestVideoRecorderFailLoud:
    def _plugin_with_frames(self, tmp_path):
        plugin = VideoRecorderPlugin(output_path=str(tmp_path / "out.mp4"))
        plugin._frames.append(np.zeros((8, 8, 3), dtype=np.uint8))
        return plugin

    def test_no_encoder_raises(self, tmp_path, monkeypatch):
        """ffmpeg missing AND cv2 unimportable → RuntimeError (was: warn+drop)."""

        def _no_ffmpeg(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(subprocess, "Popen", _no_ffmpeg)
        # sys.modules['cv2'] = None makes `import cv2` raise ImportError.
        monkeypatch.setitem(sys.modules, "cv2", None)

        plugin = self._plugin_with_frames(tmp_path)
        with pytest.raises(RuntimeError, match="ffmpeg.*opencv|无法保存视频"):
            plugin.on_post_episode(ctx=None)

    def test_ffmpeg_failure_propagates(self, tmp_path, monkeypatch):
        """ffmpeg present but exits nonzero → RuntimeError (was: caught by
        the blanket ``except Exception`` and printed)."""

        class _DeadProc:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.returncode = 1

            def wait(self):
                return self.returncode

        monkeypatch.setattr(
            subprocess, "Popen", lambda *args, **kwargs: _DeadProc()
        )

        plugin = self._plugin_with_frames(tmp_path)
        with pytest.raises(RuntimeError, match="ffmpeg exited"):
            plugin.on_post_episode(ctx=None)

    def test_cv2_writer_open_failure_raises(self, tmp_path, monkeypatch):
        """cv2 fallback where VideoWriter can't open the file → raise
        (was: silent — VideoWriter fails open without any error)."""
        import types

        def _no_ffmpeg(*args, **kwargs):
            raise FileNotFoundError("ffmpeg")

        class _Writer:
            def isOpened(self):
                return False

            def write(self, frame):
                pass

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(
            VideoWriter_fourcc=lambda *a: 0,
            VideoWriter=lambda *a, **k: _Writer(),
            cvtColor=lambda f, c: f,
            COLOR_RGB2BGR=0,
        )
        monkeypatch.setattr(subprocess, "Popen", _no_ffmpeg)
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        plugin = self._plugin_with_frames(tmp_path)
        with pytest.raises(RuntimeError, match="VideoWriter"):
            plugin.on_post_episode(ctx=None)
