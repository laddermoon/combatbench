"""Point5 business-language contracts (v2: torso-facing local frame).

Simulator-unaware: this module must not import mujoco, envs.*, baseline.*,
or any simulator binding.  See DESIGN.md for the full contract.

Semantics recap:

- A P5 state is ``local_pos[5x3]`` expressed in the torso-facing local
  frame: origin = ankle XY midpoint, X = torso forward, Y = torso left,
  Z = up (ground z = 0).  No absolute position, no absolute facing, no
  yaw field.
- A clip is a keyframe chain; every keyframe after the first carries a
  ``transfer`` describing its own local frame inside the previous
  frame's local frame: ``F_k = F_{k-1} ∘ T(dx, dy, dyaw)``.
- World placement is the consumer's choice, not part of the pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

POINT_NAMES = ("head", "hand_right", "hand_left", "foot_right", "foot_left")
N_POINTS = len(POINT_NAMES)
POINT_INDEX = {name: i for i, name in enumerate(POINT_NAMES)}

GROUND_Z = 0.0
PLANT_HEIGHT_TOL = 0.02   # foot z below this counts as planted
MIN_FOOT_SEP_XY = 0.02    # feet XY closer than this = degenerate origin

STATE_FORMAT = "p5.v2"          # serialized single-pose tag
CLIP_FORMAT = "p5clip.v2"       # serialized clip tag
FRAMES_FORMAT = "point5.frames.v2"


class Point5Error(ValueError):
    """Base class for point5 contract violations."""


class TaskValidationError(Point5Error):
    """Clip/keyframe/state structure is invalid."""


class SampleOutOfRange(Point5Error):
    """Sampling outside the clip's declared domain."""


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise TaskValidationError(
            f"{name} must be a finite (3,) vector, got {arr.shape}")
    return arr


def _pos5(value: Any, name: str = "pos") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (N_POINTS, 3) or not np.all(np.isfinite(arr)):
        raise TaskValidationError(
            f"{name} must be a finite ({N_POINTS},3) array, got {arr.shape}")
    return arr


def _idx(name: str) -> int:
    if name not in POINT_INDEX:
        raise TaskValidationError(f"unknown point {name!r}")
    return POINT_INDEX[name]


def is_planted(pos: np.ndarray, tol: float = PLANT_HEIGHT_TOL) -> bool:
    """Foot contact convention: pos z at/below ground + tol counts planted."""
    return bool(np.asarray(pos, dtype=np.float64)[2] <= GROUND_Z + tol)


# ---------------------------------------------------------------------------
# Local-frame geometry derivations (all from stored positions only)
# ---------------------------------------------------------------------------

def ankle_mid_xy(pos: np.ndarray) -> np.ndarray:
    """Local-frame origin: XY midpoint of the two ankles, shape (2,)."""
    p = _pos5(pos)
    return 0.5 * (p[POINT_INDEX["foot_right"], :2]
                  + p[POINT_INDEX["foot_left"], :2])


def ankle_line_dir(pos: np.ndarray) -> Optional[np.ndarray]:
    """Unit XY direction left-ankle -> right-ankle, or None if degenerate."""
    p = _pos5(pos)
    d = p[POINT_INDEX["foot_right"], :2] - p[POINT_INDEX["foot_left"], :2]
    n = float(np.linalg.norm(d))
    return d / n if n >= MIN_FOOT_SEP_XY else None


def is_degenerate(pos: np.ndarray,
                  min_sep: float = MIN_FOOT_SEP_XY) -> bool:
    """Feet XY coincide: the local origin loses meaning -> illegal."""
    return ankle_line_dir(pos) is None


def twist_deg(pos: np.ndarray) -> Optional[float]:
    """Torso twist relative to the ankle line (degrees).

    The ankle-line normal ``n = (-b_y, b_x)`` (b = left->right ankle)
    points along +X for a square stance; its signed angle from +X is the
    twist.  None if feet degenerate.
    """
    b = ankle_line_dir(pos)
    if b is None:
        return None
    n = np.array([-b[1], b[0]])
    return float(np.degrees(np.arctan2(n[1], n[0])))


# ---------------------------------------------------------------------------
# Frame transforms (SE(2)-in-3D: XY rotation + translation, z passes through)
# ---------------------------------------------------------------------------

def _rot2(yaw_deg: float) -> np.ndarray:
    r = np.deg2rad(yaw_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])


def to_world(pos_local: np.ndarray, origin_xy: np.ndarray,
             yaw_deg: float) -> np.ndarray:
    """local -> world under frame pose (origin_xy, yaw_deg)."""
    p = np.asarray(pos_local, dtype=np.float64).copy()
    p[..., :2] = p[..., :2] @ _rot2(yaw_deg).T + np.asarray(origin_xy)
    return p


def to_local(pos_world: np.ndarray, origin_xy: np.ndarray,
             yaw_deg: float) -> np.ndarray:
    """world -> local under frame pose (origin_xy, yaw_deg)."""
    p = np.asarray(pos_world, dtype=np.float64).copy()
    p[..., :2] = (p[..., :2] - np.asarray(origin_xy)) @ _rot2(yaw_deg)
    return p


def compose_frame(origin: np.ndarray, yaw: float,
                  transfer: "Transfer") -> Tuple[np.ndarray, float]:
    """F_k = F_{k-1} ∘ T: child frame pose expressed in parent coords."""
    o = np.asarray(origin, dtype=np.float64) + _rot2(yaw) @ np.array(
        [transfer.dx, transfer.dy])
    return o, yaw + transfer.dyaw


# ---------------------------------------------------------------------------
# Clip structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transfer:
    """Frame-to-frame motion: this keyframe's local frame in the previous
    keyframe's local frame.  dur = seconds from previous keyframe."""

    dx: float
    dy: float
    dyaw: float
    dur: float

    def __post_init__(self) -> None:
        for n in ("dx", "dy", "dyaw", "dur"):
            if not np.isfinite(getattr(self, n)):
                raise TaskValidationError(f"transfer.{n} must be finite")
        if self.dur <= 0.0:
            raise TaskValidationError("transfer.dur must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {"dx": float(self.dx), "dy": float(self.dy),
                "dyaw": float(self.dyaw), "dur": float(self.dur)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Transfer":
        return cls(dx=float(d["dx"]), dy=float(d["dy"]),
                   dyaw=float(d["dyaw"]), dur=float(d["dur"]))


@dataclass(frozen=True)
class Keyframe:
    """One clip keyframe: a local P5 pose + how its frame was reached."""

    pos: np.ndarray                          # local_pos[5,3]
    transfer: Optional[Transfer] = None      # required unless first frame
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _pos5(self.pos, "keyframe.pos"))
        object.__setattr__(self, "meta", dict(self.meta))

    def to_dict(self) -> Dict[str, Any]:
        d = {"p5": self.pos.tolist()}
        if self.transfer is not None:
            d["transfer"] = self.transfer.to_dict()
        if self.label:
            d["label"] = self.label
        if self.meta:
            d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Keyframe":
        return cls(pos=np.asarray(d["p5"], dtype=np.float64),
                   transfer=(Transfer.from_dict(d["transfer"])
                             if "transfer" in d else None),
                   label=str(d.get("label", "")),
                   meta=dict(d.get("meta", {})))


@dataclass(frozen=True)
class P5Clip:
    """A clip: >=1 keyframes forming a transfer chain."""

    name: str
    keyframes: Tuple[Keyframe, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kfs = tuple(self.keyframes)
        if len(kfs) < 1:
            raise TaskValidationError("clip needs at least 1 keyframe")
        if kfs[0].transfer is not None:
            raise TaskValidationError(
                "first keyframe must not carry a transfer")
        for i, k in enumerate(kfs[1:], start=1):
            if k.transfer is None:
                raise TaskValidationError(
                    f"keyframe #{i} must carry a transfer")
        object.__setattr__(self, "keyframes", kfs)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_segments(self) -> int:
        return len(self.keyframes) - 1

    @property
    def duration(self) -> float:
        return float(sum(k.transfer.dur for k in self.keyframes[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": CLIP_FORMAT,
                "name": self.name,
                "keyframes": [k.to_dict() for k in self.keyframes],
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "P5Clip":
        fmt = d.get("type", CLIP_FORMAT)
        if fmt != CLIP_FORMAT:
            raise TaskValidationError(
                f"unsupported clip format {fmt!r}; expected {CLIP_FORMAT!r}")
        return cls(name=str(d.get("name", "")),
                   keyframes=tuple(Keyframe.from_dict(k)
                                   for k in d["keyframes"]),
                   metadata=dict(d.get("metadata", {})))


# ---------------------------------------------------------------------------
# Point5State: a measured/sampled local pose
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point5State:
    """A local P5 pose.  ``vel`` is optional adapter-side data."""

    pos: np.ndarray                          # local_pos[5,3]
    time: float = 0.0
    vel: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _pos5(self.pos))
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

    def twist_deg(self) -> Optional[float]:
        return twist_deg(self.pos)

    def is_degenerate(self) -> bool:
        return is_degenerate(self.pos)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": STATE_FORMAT, "p5": self.pos.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Point5State":
        return cls(pos=np.asarray(d["p5"], dtype=np.float64))
