"""Point5 clip sampling (v2).

A clip only stores local poses + per-keyframe transfers.  To get world
positions the consumer supplies the first frame's world pose
``f1 = (origin_xy, yaw_deg)``; subsequent frames follow by transfer
composition ``F_k = F_{k-1} ∘ T_k``.

Sampling is change-adaptive: between consecutive emitted frames no
point moves more than ``max_step`` metres in world space — per segment
``n = ceil(D / max_step)`` where ``D`` is the max point world
displacement between the two keyframes (world displacement includes
both local motion and frame motion).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import (
    FRAMES_FORMAT,
    Keyframe,
    P5Clip,
    SampleOutOfRange,
    TaskValidationError,
    compose_frame,
    to_world,
)

FramePose = Tuple[np.ndarray, float]     # (origin_xy(2,), yaw_deg)
DEFAULT_F1: FramePose = (np.zeros(2), 0.0)


def keyframe_frames(clip: P5Clip,
                    f1: FramePose = DEFAULT_F1) -> List[FramePose]:
    """World pose of every keyframe's local frame, by transfer chain."""
    out: List[FramePose] = [f1]
    for k in clip.keyframes[1:]:
        assert k.transfer is not None
        out.append(compose_frame(*out[-1], k.transfer))
    return out


def keyframe_world(clip: P5Clip,
                   f1: FramePose = DEFAULT_F1) -> np.ndarray:
    """World positions of every keyframe, shape (K, 5, 3)."""
    frames = keyframe_frames(clip, f1)
    return np.stack([to_world(k.pos, o, y)
                     for k, (o, y) in zip(clip.keyframes, frames)])


@dataclass(frozen=True)
class FrameSample:
    """One sampled clip frame: local pose + its world frame pose."""

    pos: np.ndarray            # local_pos (5,3), already segment-blended
    origin_xy: np.ndarray      # world origin of this frame
    yaw_deg: float
    time: float
    seg: int
    s: float

    def world_pos(self) -> np.ndarray:
        return to_world(self.pos, self.origin_xy, self.yaw_deg)


def _blend(p0: np.ndarray, p1: np.ndarray, s: float) -> np.ndarray:
    return p0 + (p1 - p0) * s


def sample_clip(clip: P5Clip,
                f1: FramePose = DEFAULT_F1,
                max_step: float = 0.01) -> List[FrameSample]:
    """Change-adaptive sampling in world space.

    n_seg subdivisions per segment bound the max per-point world
    displacement between emitted frames by ``max_step`` (linear
    interpolation of both local positions and the frame transfer).
    """
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise TaskValidationError("max_step must be positive")
    frames = keyframe_frames(clip, f1)
    times = np.concatenate([[0.0],
                            np.cumsum([k.transfer.dur
                                       for k in clip.keyframes[1:]])])
    out: List[FrameSample] = [FrameSample(clip.keyframes[0].pos,
                                          np.asarray(frames[0][0]),
                                          frames[0][1], 0.0, -1, 0.0)]
    for seg in range(clip.n_segments):
        k0, k1 = clip.keyframes[seg], clip.keyframes[seg + 1]
        tr = k1.transfer
        assert tr is not None
        o0, y0 = frames[seg]
        # max per-point world displacement over the whole segment
        w0 = to_world(k0.pos, o0, y0)
        w1 = to_world(k1.pos, *frames[seg + 1])
        d = float(np.linalg.norm(w1 - w0, axis=1).max())
        n = max(1, int(np.ceil(d / max_step)))
        for j in range(1, n + 1):
            s = j / n
            pos = _blend(k0.pos, k1.pos, s)
            # progressive frame motion: the transfer displacement is in
            # F_{seg} coords, so rotate it by y0 before adding
            do = np.array([tr.dx, tr.dy]) * s
            o_s = o0 + np.array(
                [np.cos(np.deg2rad(y0)) * do[0]
                 - np.sin(np.deg2rad(y0)) * do[1],
                 np.sin(np.deg2rad(y0)) * do[0]
                 + np.cos(np.deg2rad(y0)) * do[1]])
            out.append(FrameSample(pos, o_s, y0 + tr.dyaw * s,
                                   float(times[seg] + tr.dur * s), seg, s))
    return out


def frames_payload(clip: P5Clip, samples: List[FrameSample]) -> Dict[str, Any]:
    """JSON-safe sampled-clip payload."""
    return {
        "format": FRAMES_FORMAT,
        "clip": clip.name,
        "n_frames": len(samples),
        "frames": [{"pos": sm.pos.tolist(),
                    "origin": list(map(float, sm.origin_xy)),
                    "yaw": float(sm.yaw_deg),
                    "t": sm.time, "seg": sm.seg, "s": sm.s}
                   for sm in samples],
    }
