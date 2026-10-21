"""Kinematic-replay renderer: CoreClip -> MP4.

This plays the clip by OVERWRITING the simulator state each frame
(set_core_state + mj_forward).  It is a visualization of the recorded
kinematic motion — NOT a physical execution check.  Whether the robot
can actually produce this motion with actions is a separate question
(Phase-2 verification work).

Usage:
    PYTHONPATH=. python3 -m mocap.humanoid21.clip play \
        --clip <id>.npz --out /data1/dev/replay/x.mp4 [--fps 20]
        [--slow 4] [--burn-frame] [--distance 6.0]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .core_clip import CONTROL_FPS, CoreClip


def _write_mp4(frames: List[np.ndarray], path, fps: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "faststart", str(path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        proc = None
    if proc is not None:
        for f in frames:
            proc.stdin.write(f.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
    else:
        import cv2
        wr = cv2.VideoWriter(str(path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError(f"cannot open {path} for writing")
        for f in frames:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
    print(f"[play] wrote {path} ({len(frames)} frames @ {fps}fps)")


def render(clip: CoreClip, out, fps: float = CONTROL_FPS,
           slow: float = 1.0, burn_frame: bool = False,
           distance: float = 6.0, robot: str = "robot_a",
           state_field: str = "state") -> Path:
    """Kinematic-replay a CoreClip into an MP4 file."""
    from envs.humanoid21.simulator import Humanoid21Simulator
    sim = Humanoid21Simulator(initial_distance=distance,
                              initial_pose_b="standing")
    frames: List[np.ndarray] = []
    burn = None
    if burn_frame:
        import cv2
        burn = cv2
    for k in range(clip.T):
        st = clip.state(k) if state_field == "state" \
            else clip.state_at(k * 0.05)
        sim.set_core_state({robot: st})
        img = sim.get_broadcastview_image()
        if img is None:
            raise RuntimeError(f"render failed at frame {k}")
        img = img.copy()
        if burn is not None:
            burn.putText(img, f"{k}  t={k * 0.05:.2f}s",
                         (10, 30), burn.FONT_HERSHEY_SIMPLEX,
                         0.8, (255, 255, 255), 2, burn.LINE_AA)
        frames.append(img)
    sim.close()
    _write_mp4(frames, out, fps / slow)
    return Path(out)


