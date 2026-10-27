"""Reference synthesis: resolve anchors, linearly interpolate, sample.

The whole computation is a per-point piecewise-linear polyline over
segments. No velocities anywhere (see DESIGN.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .contracts import (
    AnchorResolver,
    N_POINTS,
    POINT_INDEX,
    POINT_NAMES,
    Point5State,
    Point5Task,
    SampleOutOfRange,
    TaskValidationError,
)

_EPS = 1e-9


@dataclass(frozen=True)
class ResolvedTask:
    """A task whose anchors are all resolved to world-frame positions."""

    task: Point5Task
    positions: np.ndarray    # (K, N_POINTS, 3) resolved targets
    tolerances: np.ndarray   # (K, N_POINTS)
    weights: np.ndarray      # (K, N_POINTS)
    seg_starts: np.ndarray   # (K,) cumulative start times; [-1] == duration

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=np.float64)
        k = len(self.task.keyframes)
        if pos.shape != (k, N_POINTS, 3) or not np.all(np.isfinite(pos)):
            raise TaskValidationError(
                f"positions must be finite ({k},{N_POINTS},3), got {pos.shape}")


def resolve_task(task: Point5Task, resolver: AnchorResolver) -> ResolvedTask:
    """Resolve every spec's anchor into world-frame target positions."""
    k = len(task.keyframes)
    positions = np.zeros((k, N_POINTS, 3), dtype=np.float64)
    tolerances = np.zeros((k, N_POINTS), dtype=np.float64)
    weights = np.zeros((k, N_POINTS), dtype=np.float64)
    for ki, kf in enumerate(task.keyframes):
        for name, spec in kf.specs.items():
            j = POINT_INDEX[name]
            positions[ki, j] = np.asarray(resolver.resolve(spec.target),
                                          dtype=np.float64)
            tolerances[ki, j] = spec.tolerance
            weights[ki, j] = spec.weight
    if not np.all(np.isfinite(positions)):
        raise TaskValidationError("resolver returned non-finite positions")
    seg_starts = np.concatenate(
        [[0.0], np.cumsum(np.asarray(task.durations))])
    return ResolvedTask(task=task, positions=positions,
                        tolerances=tolerances, weights=weights,
                        seg_starts=seg_starts)


@dataclass(frozen=True)
class ReferenceFrames:
    """Dense sampling of a reference along time."""

    pos: np.ndarray       # (T, N_POINTS, 3)
    times: np.ndarray     # (T,)
    segs: np.ndarray      # (T,) int
    s: np.ndarray         # (T,)
    states: tuple         # (T,) Point5State, one per sample


class Point5Reference:
    """Sampleable linear reference built from a resolved task."""

    def __init__(self, resolved: ResolvedTask):
        self.resolved = resolved
        self.task = resolved.task
        self.duration = self.task.duration

    # -- sampling ----------------------------------------------------------

    def sample(self, seg: int, s: float) -> Point5State:
        """Sample normalized position ``s`` within segment ``seg``."""
        n_seg = self.task.n_segments
        if not (0 <= seg < n_seg):
            raise SampleOutOfRange(
                f"seg {seg} out of range [0,{n_seg - 1}]")
        if not (0.0 - _EPS <= s <= 1.0 + _EPS) or not np.isfinite(s):
            raise SampleOutOfRange(f"s {s} out of range [0,1]")
        s = min(max(float(s), 0.0), 1.0)
        p0 = self.resolved.positions[seg]
        p1 = self.resolved.positions[seg + 1]
        pos = (1.0 - s) * p0 + s * p1
        t = float(self.resolved.seg_starts[seg]
                  + s * self.task.durations[seg])
        label = (self.task.keyframes[seg].label if s < 0.5
                 else self.task.keyframes[seg + 1].label)
        return Point5State(
            pos=pos, time=t, vel=None, frame="world",
            metadata={"task": self.task.name, "seg": int(seg), "s": s,
                      "t": t, "keyframe_label": label})

    def sample_time(self, t: float) -> Point5State:
        """Sample absolute time. Out of range raises, never clamps."""
        if not np.isfinite(t):
            raise SampleOutOfRange(f"t {t} is not finite")
        if t < -_EPS or t > self.duration + _EPS:
            raise SampleOutOfRange(
                f"t {t} out of range [0,{self.duration}] for task "
                f"{self.task.name!r}")
        t = min(max(float(t), 0.0), self.duration)
        seg = int(np.searchsorted(self.resolved.seg_starts[1:], t,
                                  side="right"))
        seg = min(seg, self.task.n_segments - 1)
        d = self.task.durations[seg]
        s = (t - self.resolved.seg_starts[seg]) / d if d > 0 else 0.0
        return self.sample(seg, s)

    def frames(self, dt: float) -> ReferenceFrames:
        """Dense sampling at fixed ``dt``; last sample lands on duration."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise TaskValidationError("dt must be positive")
        n = int(np.floor(self.duration / dt)) + 1
        times = np.arange(n, dtype=np.float64) * dt
        if times[-1] < self.duration - _EPS:
            times = np.append(times, self.duration)
        else:
            times[-1] = self.duration
        states = tuple(self.sample_time(t) for t in times)
        return ReferenceFrames(
            pos=np.stack([st.pos for st in states]),
            times=times,
            segs=np.asarray([st.metadata["seg"] for st in states],
                            dtype=np.int32),
            s=np.asarray([st.metadata["s"] for st in states]),
            states=states)

    # -- convenience -------------------------------------------------------

    def target_at(self, keyframe: int) -> np.ndarray:
        """Resolved world targets at keyframe boundary (K,5,3) row."""
        return self.resolved.positions[keyframe]


def build_reference(task: Point5Task,
                    resolver: AnchorResolver) -> Point5Reference:
    """Resolve + wrap. ``initial`` (if needed) lives inside the resolver."""
    return Point5Reference(resolve_task(task, resolver))
