"""Data contracts for the reduced-order main-path planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

FootName = Literal["left", "right"]
FOOT_NAMES: Tuple[FootName, FootName] = ("left", "right")


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite (3,) vector, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class MainPathState:
    """Minimal instantaneous state owned by the main-path layer."""

    time: float
    com_pos: np.ndarray
    lin_mom: np.ndarray
    ang_mom: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        object.__setattr__(self, "com_pos", _vec3(self.com_pos, "com_pos"))
        object.__setattr__(self, "lin_mom", _vec3(self.lin_mom, "lin_mom"))
        object.__setattr__(self, "ang_mom", _vec3(self.ang_mom, "ang_mom"))


@dataclass(frozen=True)
class MainPathRequest:
    """Default request: walk to a horizontal position and stop."""

    current_state: MainPathState
    target_pos_world: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_pos_world",
            _vec3(self.target_pos_world, "target_pos_world"))


@dataclass(frozen=True)
class SupportInterval:
    """One per-foot support placement over a time interval."""

    foot: FootName
    start_time: float
    end_time: Optional[float]
    point_world: np.ndarray

    def __post_init__(self) -> None:
        if self.foot not in FOOT_NAMES:
            raise ValueError(f"unknown foot {self.foot!r}")
        if not np.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if self.end_time is not None:
            if not np.isfinite(self.end_time):
                raise ValueError("end_time must be finite or None")
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be greater than start_time")
        object.__setattr__(
            self, "point_world", _vec3(self.point_world, "point_world"))


@dataclass(frozen=True)
class SupportSchedule:
    """Planner-owned sequence of per-foot support intervals."""

    intervals: Tuple[SupportInterval, ...]
    horizon: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intervals = tuple(sorted(
            self.intervals,
            key=lambda x: (x.start_time, 0 if x.foot == "left" else 1)))
        object.__setattr__(self, "intervals", intervals)
        if not np.isfinite(self.horizon):
            raise ValueError("horizon must be finite")
        self._validate_intervals(intervals)

    @staticmethod
    def _validate_intervals(intervals: Tuple[SupportInterval, ...]) -> None:
        for foot in FOOT_NAMES:
            own = [x for x in intervals if x.foot == foot]
            for i, item in enumerate(own):
                if item.end_time is None and i != len(own) - 1:
                    raise ValueError(
                        f"open-ended {foot} interval must be the last interval")
                if i and item.start_time < float(own[i - 1].end_time or np.inf):
                    raise ValueError(f"overlapping {foot} support intervals")

    def active_feet(self, time: float) -> Tuple[FootName, ...]:
        """Return feet whose interval covers ``time`` in left/right order."""
        t = float(time)
        active = set()
        for item in self.intervals:
            if item.start_time <= t and (
                    item.end_time is None or t < item.end_time):
                active.add(item.foot)
        return tuple(foot for foot in FOOT_NAMES if foot in active)

    def active_point(self, foot: FootName, time: float) -> Optional[np.ndarray]:
        """Return the active support point for ``foot`` at ``time``."""
        t = float(time)
        for item in self.intervals:
            if item.foot == foot and item.start_time <= t and (
                    item.end_time is None or t < item.end_time):
                return item.point_world
        return None
