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
CLIP_FORMAT = "p5clip.v3"       # serialized clip tag
FRAMES_FORMAT = "point5.frames.v2"


class Point5Error(ValueError):
    """Base class for point5 contract violations."""


class TaskValidationError(Point5Error):
    """Clip/keyframe/state structure is invalid."""


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


# ---------------------------------------------------------------------------
# Local-frame geometry derivations (all from stored positions only)
# ---------------------------------------------------------------------------

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


def to_local(pos_world: np.ndarray, origin_xy: np.ndarray,
             yaw_deg: float) -> np.ndarray:
    """world -> local under frame pose (origin_xy, yaw_deg)."""
    p = np.asarray(pos_world, dtype=np.float64).copy()
    p[..., :2] = (p[..., :2] - np.asarray(origin_xy)) @ _rot2(yaw_deg)
    return p


# ---------------------------------------------------------------------------
# Clip structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transfer:
    """One target pose expressed as a delta from Base — in Base's frame.

        pose_i = Base.pos + delta_pos        (point-wise delta, (5,3))
        yaw_i  = delta_yaw                   (torso-heading delta vs base)
    """

    delta_pos: np.ndarray                          # (5,3) in Base frame
    delta_yaw: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_pos",
                           _pos5(self.delta_pos, "transfer.delta_pos"))
        if not np.isfinite(self.delta_yaw):
            raise TaskValidationError("transfer.delta_yaw must be finite")
        object.__setattr__(self, "delta_yaw", float(self.delta_yaw))

    def to_dict(self) -> Dict[str, Any]:
        return {"delta_pos": self.delta_pos.tolist(),
                "delta_yaw": float(self.delta_yaw)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Transfer":
        return cls(delta_pos=np.asarray(d["delta_pos"], dtype=np.float64),
                   delta_yaw=float(d["delta_yaw"]))


@dataclass(frozen=True)
class P5Clip:
    """A clip: a Base pose + ordered pose deltas, all in Base's frame."""

    Base: P5
    Transfers: Tuple[Transfer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "Transfers", tuple(self.Transfers))

    @property
    def p5(self):
        """p5[i] — i-th frame's P5; p5[0] = Base.

        i >= 1: P5(pos = Base.pos + Transfers[i-1].delta_pos),
        all coordinates in the Base frame."""
        return _P5Frames(self)

    @property
    def transfers(self):
        """transfers[i] — the transition from frame i to frame i+1.

        Stored deltas are base->frame(i+1); the i->i+1 transition is
        the difference of consecutive base deltas."""
        return _TransferSeq(self)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": CLIP_FORMAT,
                "base": {"pos": self.Base.pos.tolist()},
                "transfers": [t.to_dict() for t in self.Transfers]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "P5Clip":
        fmt = d.get("type", CLIP_FORMAT)
        if fmt != CLIP_FORMAT:
            raise TaskValidationError(
                f"unsupported clip format {fmt!r}; expected {CLIP_FORMAT!r}")
        return cls(Base=P5(pos=np.asarray(d["base"]["pos"],
                                          dtype=np.float64)),
                   Transfers=tuple(Transfer.from_dict(t)
                                   for t in d["transfers"]))


class _P5Frames:
    """Indexable view: clip.p5[i] -> the i-th frame's P5."""

    def __init__(self, clip: "P5Clip"):
        self._c = clip

    def __getitem__(self, i: int) -> P5:
        c = self._c
        if not 0 <= i < len(self):
            raise IndexError(f"p5 index {i} out of range (len {len(self)})")
        if i == 0:
            return c.Base
        t = c.Transfers[i - 1]
        raw = c.Base.pos + t.delta_pos          # Base-frame coords
        # re-anchor into THIS frame's own local system:
        # own ankle mid -> origin, own yaw (delta vs base) -> +x
        mid = 0.5 * (raw[POINT_INDEX["foot_right"], :2]
                     + raw[POINT_INDEX["foot_left"], :2])
        return P5(pos=to_local(raw, mid,
                               np.degrees(t.delta_yaw)))

    def __len__(self) -> int:
        return 1 + len(self._c.Transfers)


class _TransferSeq:
    """Indexable view: clip.transfers[i] -> frame i -> frame i+1."""

    def __init__(self, clip: "P5Clip"):
        self._c = clip

    def __getitem__(self, i: int) -> "Transfer":
        if not 0 <= i < len(self):
            raise IndexError(
                f"transfers index {i} out of range (len {len(self)})")
        cur = self._c.Transfers[i]                  # base -> frame i+1
        if i == 0:
            return cur                              # base -> frame 1
        prev = self._c.Transfers[i - 1]             # base -> frame i
        return Transfer(delta_pos=cur.delta_pos - prev.delta_pos,
                        delta_yaw=cur.delta_yaw - prev.delta_yaw)

    def __len__(self) -> int:
        return len(self._c.Transfers)


# ---------------------------------------------------------------------------
# P5: a measured/sampled local pose
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P5:
    """A local P5 pose — 5 landmark points in the local frame."""

    pos: np.ndarray                          # local_pos[5,3]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _pos5(self.pos))

    def point_pos(self, name: str) -> np.ndarray:
        return self.pos[_idx(name)]

    def twist_deg(self) -> Optional[float]:
        return twist_deg(self.pos)

    def is_degenerate(self) -> bool:
        return is_degenerate(self.pos)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": STATE_FORMAT, "p5": self.pos.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "P5":
        return cls(pos=np.asarray(d["p5"], dtype=np.float64))
