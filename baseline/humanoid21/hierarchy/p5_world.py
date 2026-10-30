"""Experiment-side helpers to resolve v2 P5 clips into the world frame.

P5 v2 clips are purely local: each keyframe stores a local (5,3) pose,
and every keyframe after the first carries a ``Transfer(dx, dy, dyaw,
dur)`` describing how its local frame moved relative to the previous
one.  The library intentionally has no world frame and no clock.

The experiment side supplies exactly one thing: ``f1 = (origin_xy,
yaw_deg)``, the world pose of the clip's first local frame, computed
once from the injected start state (the adapter reports it in
``Point5State.metadata`` as ``frame_origin_xy`` / ``frame_yaw_deg``).
Everything downstream — world waypoints for command guidance, world
reference poses for reward scoring — is derived here.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from point5.contracts import P5Clip, to_world
from point5.trajectory import FramePose, keyframe_frames

__all__ = ["f1_from_state", "clip_world_samples", "p5_to_world"]


def f1_from_state(state) -> FramePose:
    """World pose of a measured ``Point5State``'s local frame."""
    origin = np.asarray(state.metadata["frame_origin_xy"],
                        dtype=np.float64)
    return origin, float(state.metadata["frame_yaw_deg"])


def p5_to_world(pos_local: np.ndarray, f1: FramePose) -> np.ndarray:
    """(5,3) local-frame points -> world, under frame pose ``f1``."""
    return to_world(np.asarray(pos_local, dtype=np.float64), *f1)


def clip_world_samples(clip: P5Clip, f1: FramePose,
                       dt: float) -> np.ndarray:
    """Uniform-dt world samples of a clip, shape ``(T, 5, 3)``.

    Sample ``i`` is the world pose at time ``min((i+1)*dt, duration)``
    — i.e. the pose the robot should reach after action ``i``.  Local
    positions blend linearly between keyframes while the carrying frame
    advances through the segment transfer proportionally (same scheme
    as ``point5.trajectory.sample_clip`` but at fixed time steps).
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    frames = keyframe_frames(clip, f1)
    durs = [k.transfer.dur for k in clip.keyframes[1:]]
    times = np.concatenate([[0.0], np.cumsum(durs)])
    duration = float(times[-1])
    n_out = max(1, int(np.ceil((duration - 1e-9) / dt)))
    out = np.empty((n_out, 5, 3), dtype=np.float64)
    for i in range(n_out):
        t = min((i + 1) * dt, duration)
        seg = int(np.searchsorted(times, t, side="right") - 1)
        seg = min(seg, len(durs) - 1)
        s = 1.0 if durs[seg] <= 0.0 else min(
            (t - times[seg]) / durs[seg], 1.0)
        k0, k1 = clip.keyframes[seg], clip.keyframes[seg + 1]
        tr = k1.transfer
        pos = k0.pos + (k1.pos - k0.pos) * s
        o0, y0 = frames[seg]
        rad = np.deg2rad(y0)
        do = np.array([tr.dx, tr.dy]) * s
        o_s = np.asarray(o0, dtype=np.float64) + np.array(
            [np.cos(rad) * do[0] - np.sin(rad) * do[1],
             np.sin(rad) * do[0] + np.cos(rad) * do[1]])
        out[i] = to_world(pos, o_s, y0 + tr.dyaw * s)
    return out


def world_track_for_point(clip: P5Clip, f1: FramePose, dt: float,
                          point_index: int) -> np.ndarray:
    """(T,3) world trajectory of one P5 point — the command guidance
    track handed to ``set_effector_world_track``."""
    return clip_world_samples(clip, f1, dt)[:, point_index]
