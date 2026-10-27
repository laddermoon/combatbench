"""Point5 business-language contracts.

Simulator-unaware: this module must not import mujoco, envs.*, baseline.*,
or any simulator binding. See DESIGN.md for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Union

import numpy as np

POINT_NAMES = ("head", "hand_right", "hand_left", "foot_right", "foot_left")
N_POINTS = len(POINT_NAMES)
POINT_INDEX = {name: i for i, name in enumerate(POINT_NAMES)}

GROUND_Z = 0.0
PLANT_HEIGHT_TOL = 0.02  # foot z below this counts as planted

TASK_FORMAT = "point5.task.v1"    # serialized Point5Task format tag
FRAMES_FORMAT = "point5.frames.v1"  # serialized sampled-frames payload tag


class Point5Error(ValueError):
    """Base class for point5 contract violations."""


class TaskValidationError(Point5Error):
    """Task/keyframe/spec structure is invalid."""


class AnchorResolutionError(Point5Error):
    """An anchor cannot be resolved to a world point."""


class SampleOutOfRange(Point5Error):
    """Sampling outside the task's declared domain."""


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise TaskValidationError(
            f"{name} must be a finite (3,) vector, got {arr.shape}")
    return arr


def is_planted(pos: np.ndarray, tol: float = PLANT_HEIGHT_TOL) -> bool:
    """Foot contact convention: pos z at/below ground + tol counts as planted."""
    return bool(np.asarray(pos, dtype=np.float64)[2] <= GROUND_Z + tol)


# ---------------------------------------------------------------------------
# Anchors (all static; resolved once at build time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldPoint:
    """Fixed world-frame point."""
    pos: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _vec3(self.pos, "pos"))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "world", "pos": self.pos.tolist()}


@dataclass(frozen=True)
class RootOffset:
    """Offset in the subject's root frame at task start (rotation included)."""
    offset: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset",
                         _vec3(self.offset, "offset"))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "root_offset", "offset": self.offset.tolist()}


@dataclass(frozen=True)
class SelfPointOffset:
    """World-frame offset from one of the subject's own keypoints at task start."""
    point: str
    offset: np.ndarray

    def __post_init__(self) -> None:
        if self.point not in POINT_INDEX:
            raise TaskValidationError(f"unknown point {self.point!r}")
        object.__setattr__(self, "offset",
                         _vec3(self.offset, "offset"))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "self_point", "point": self.point,
                "offset": self.offset.tolist()}


Anchor = Union[WorldPoint, RootOffset, SelfPointOffset]

_ANCHOR_TYPES = {
    "world": WorldPoint,
    "root_offset": RootOffset,
    "self_point": SelfPointOffset,
}


def anchor_from_dict(d: Mapping[str, Any]) -> Anchor:
    t = d.get("type")
    if t == "world":
        return WorldPoint(np.asarray(d["pos"], dtype=np.float64))
    if t == "root_offset":
        return RootOffset(np.asarray(d["offset"], dtype=np.float64))
    if t == "self_point":
        return SelfPointOffset(str(d["point"]),
                               np.asarray(d["offset"], dtype=np.float64))
    raise TaskValidationError(f"unknown anchor type {t!r}")


class AnchorResolver(Protocol):
    """Injected boundary: resolves a declared anchor to a world point."""

    def resolve(self, anchor: Anchor) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Point5State (shared by measured and reference)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point5State:
    """Positions of the five business points in a declared frame.

    ``vel`` is optional: v1 references never carry it, but a measured adapter
    may fill it for downstream use.
    """

    pos: np.ndarray
    time: float = 0.0
    vel: Optional[np.ndarray] = None
    frame: str = "world"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pos = np.asarray(self.pos, dtype=np.float64)
        if pos.shape != (N_POINTS, 3) or not np.all(np.isfinite(pos)):
            raise TaskValidationError(
                f"pos must be a finite ({N_POINTS},3) array, got {pos.shape}")
        object.__setattr__(self, "pos", pos)
        if self.vel is not None:
            vel = np.asarray(self.vel, dtype=np.float64)
            if vel.shape != (N_POINTS, 3) or not np.all(np.isfinite(vel)):
                raise TaskValidationError(
                    f"vel must be a finite ({N_POINTS},3) array or None")
            object.__setattr__(self, "vel", vel)
        if not np.isfinite(self.time):
            raise TaskValidationError("time must be finite")

    def point_pos(self, name: str) -> np.ndarray:
        return self.pos[_idx(name)]

    def point_vel(self, name: str) -> Optional[np.ndarray]:
        return None if self.vel is None else self.vel[_idx(name)]


def _idx(name: str) -> int:
    if name not in POINT_INDEX:
        raise TaskValidationError(f"unknown point {name!r}")
    return POINT_INDEX[name]


# ---------------------------------------------------------------------------
# Task language
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointSpec:
    """What one business point should be at a keyframe.

    ``tolerance``/``weight`` are annotations for downstream scoring/harness;
    the synthesizer does not consume them.
    """

    target: Anchor
    tolerance: float = 0.05
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.target, (WorldPoint, RootOffset,
                                        SelfPointOffset)):
            raise TaskValidationError(
                f"unknown anchor object {type(self.target).__name__}")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise TaskValidationError("tolerance must be positive")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise TaskValidationError("weight must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target.to_dict(),
                "tolerance": float(self.tolerance),
                "weight": float(self.weight)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PointSpec":
        return cls(target=anchor_from_dict(d["target"]),
                   tolerance=float(d.get("tolerance", 0.05)),
                   weight=float(d.get("weight", 1.0)))


@dataclass(frozen=True)
class Keyframe:
    """One boundary pose: exactly the five business points."""

    specs: Mapping[str, PointSpec]
    label: str = ""

    def __post_init__(self) -> None:
        names = set(self.specs.keys())
        if names != set(POINT_NAMES):
            missing = set(POINT_NAMES) - names
            extra = names - set(POINT_NAMES)
            raise TaskValidationError(
                f"keyframe must cover exactly {POINT_NAMES}; "
                f"missing={sorted(missing)} extra={sorted(extra)}")
        object.__setattr__(self, "specs", dict(self.specs))

    def spec(self, point: str) -> PointSpec:
        _idx(point)  # validates the name
        return self.specs[point]

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label,
                "specs": {n: s.to_dict() for n, s in self.specs.items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Keyframe":
        return cls(specs={n: PointSpec.from_dict(s)
                          for n, s in d["specs"].items()},
                   label=str(d.get("label", "")))


@dataclass(frozen=True)
class Point5Task:
    """A point5 trajectory intent: K keyframes + K-1 segment durations."""

    name: str
    keyframes: Tuple[Keyframe, ...]
    durations: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kfs = tuple(self.keyframes)
        durs = tuple(float(d) for d in self.durations)
        if len(kfs) < 2:
            raise TaskValidationError("task needs at least 2 keyframes")
        if len(durs) != len(kfs) - 1:
            raise TaskValidationError(
                f"durations must have len(keyframes)-1 = {len(kfs)-1}, "
                f"got {len(durs)}")
        if any(not np.isfinite(d) or d <= 0.0 for d in durs):
            raise TaskValidationError("durations must all be positive")
        object.__setattr__(self, "keyframes", kfs)
        object.__setattr__(self, "durations", durs)

    @property
    def duration(self) -> float:
        return float(sum(self.durations))

    @property
    def n_segments(self) -> int:
        return len(self.durations)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": TASK_FORMAT,
                "name": self.name,
                "keyframes": [k.to_dict() for k in self.keyframes],
                "durations": list(self.durations),
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Point5Task":
        fmt = d.get("format", TASK_FORMAT)
        if fmt != TASK_FORMAT:
            raise TaskValidationError(
                f"unsupported task format {fmt!r}; expected {TASK_FORMAT!r}")
        return cls(name=str(d.get("name", "")),
                   keyframes=tuple(Keyframe.from_dict(k)
                                   for k in d["keyframes"]),
                   durations=tuple(d["durations"]),
                   metadata=dict(d.get("metadata", {})))


# ---------------------------------------------------------------------------
# Static resolver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticStartResolver:
    """Resolves all v1 anchors against the task-start state.

    ``root_rot`` maps root-frame vectors to world-frame vectors
    (world = root_pos + root_rot @ offset).
    """

    initial: Point5State
    root_pos: np.ndarray
    root_rot: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_pos",
                         _vec3(self.root_pos, "root_pos"))
        rot = np.asarray(self.root_rot, dtype=np.float64)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            raise TaskValidationError("root_rot must be a finite (3,3) matrix")
        object.__setattr__(self, "root_rot", rot)

    def resolve(self, anchor: Anchor) -> np.ndarray:
        if isinstance(anchor, WorldPoint):
            return anchor.pos.copy()
        if isinstance(anchor, RootOffset):
            return self.root_pos + self.root_rot @ anchor.offset
        if isinstance(anchor, SelfPointOffset):
            return self.initial.point_pos(anchor.point) + anchor.offset
        raise AnchorResolutionError(
            f"cannot resolve anchor of type {type(anchor).__name__}")
