"""Continuous pose-only target paths for reclip.

A TargetPath represents q = gamma(s).  It deliberately carries no
timeline: objective evaluators decide how s progresses over time.  Pose
interpolation uses the pose fields only; intermediate source velocities
are not targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from .core_clip import CoreClip, _slerp
from .rederive_action import DEFAULT_WEIGHTS


POSE_WEIGHTS = {
    "jp": DEFAULT_WEIGHTS["jp"],
    "rp": DEFAULT_WEIGHTS["rp"],
    "rq": DEFAULT_WEIGHTS["rq"],
}
ROOT_POS_MODES = ("absolute", "free")
VEL_WEIGHTS = {
    "jv": DEFAULT_WEIGHTS["jv"],
    "rv": DEFAULT_WEIGHTS["rv"],
    "ra": DEFAULT_WEIGHTS["ra"],
}


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_log_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion -> rotation vector."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    v = q[1:4]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return 2.0 * v
    ang = 2.0 * np.arctan2(s, np.clip(q[0], -1.0, 1.0))
    return v / s * ang


def _quat_exp_wxyz(rv: np.ndarray) -> np.ndarray:
    """Rotation vector -> wxyz unit quaternion."""
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / th
    return np.array([np.cos(th / 2.0),
                     *(axis * np.sin(th / 2.0))])


def pose_dict(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {f: np.asarray(state[f]) for f in
            ("root_pos", "root_rot", "joint_pos_norm")}


def _merged_pose_weights(
        weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    w = dict(POSE_WEIGHTS)
    if weights is None:
        return w
    unknown = set(weights) - set(w)
    if unknown:
        raise ValueError(
            f"unknown pose weight keys: {sorted(unknown)}")
    w.update({k: float(v) for k, v in weights.items()})
    return w


def resolve_pose_weights(
        weights: Optional[Dict[str, float]] = None,
        root_pos_mode: Optional[str] = None
        ) -> tuple[Dict[str, float], str]:
    """Resolve partial pose weights and the root-position objective mode.

    ``root_pos_mode='free'`` removes absolute root position from every
    pose-distance use in TargetPath (parameterization, projection and
    terminal pose error).  ``rp`` remains reported as a diagnostic
    component, but does not contribute to ``total``.
    """
    w = _merged_pose_weights(weights)
    if root_pos_mode is None:
        root_pos_mode = "free" if w["rp"] == 0.0 else "absolute"
    if root_pos_mode not in ROOT_POS_MODES:
        raise ValueError(
            f"unknown root_pos_mode {root_pos_mode!r}; expected "
            f"one of {ROOT_POS_MODES}")
    if root_pos_mode == "free":
        if weights is not None and "rp" in weights \
                and float(weights["rp"]) != 0.0:
            raise ValueError(
                "root_pos_mode='free' conflicts with nonzero rp weight")
        w["rp"] = 0.0
    elif w["rp"] == 0.0:
        raise ValueError(
            "rp=0 requires root_pos_mode='free'")
    return w, root_pos_mode


def pose_error(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
               weights: Optional[Dict[str, float]] = None
               ) -> Dict[str, Any]:
    """Pose-only weighted error (no velocity terms)."""
    w = _merged_pose_weights(weights)
    q_rel = _quat_mul_wxyz(_quat_inv_wxyz(b["root_rot"]),
                           a["root_rot"])
    comp = {
        "jp": float(np.abs(a["joint_pos_norm"]
                           - b["joint_pos_norm"]).mean()),
        "rp": float(5.0 * np.linalg.norm(a["root_pos"]
                                         - b["root_pos"])),
        "rq": float(np.linalg.norm(_quat_log_wxyz(q_rel)) / np.pi),
    }
    return {"total": float(sum(w.get(k, 0.0) * comp[k]
                               for k in comp)),
            "components": comp}


def velocity_error(a: Dict[str, np.ndarray],
                   b: Dict[str, np.ndarray],
                   weights: Optional[Dict[str, float]] = None
                   ) -> Dict[str, Any]:
    """Velocity-only weighted error for terminal reporting."""
    w = dict(VEL_WEIGHTS if weights is None else weights)
    comp = {
        "jv": float(np.abs(a["joint_vel_norm"]
                           - b["joint_vel_norm"]).mean()),
        "rv": float(np.linalg.norm(a["root_vel_local"]
                                   - b["root_vel_local"]) / 2.0),
        "ra": float(np.linalg.norm(a["root_angular_vel_local"]
                                   - b["root_angular_vel_local"]) / 5.0),
    }
    return {"total": float(sum(w.get(k, 0.0) * comp[k]
                               for k in comp)),
            "components": comp}


@dataclass(frozen=True)
class PoseSample:
    s: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    joint_pos_norm: np.ndarray

    def as_state(self) -> Dict[str, np.ndarray]:
        return {
            "root_pos": self.root_pos.copy(),
            "root_rot": self.root_rot.copy(),
            "joint_pos_norm": self.joint_pos_norm.copy(),
        }


@dataclass(frozen=True)
class PathProjection:
    s: float
    pose: PoseSample
    errors: Dict[str, Any]


class TargetPath:
    """Pose-only continuous path in s ∈ [0, 1+s_ext]."""

    def __init__(self, poses: list[Dict[str, np.ndarray]],
                 mode: str, source_clip_id: str,
                 frame_indices: np.ndarray,
                 s_ext: float = 0.1,
                 pose_weights: Optional[Dict[str, float]] = None,
                 root_pos_mode: Optional[str] = None) -> None:
        if len(poses) < 2:
            raise ValueError("TargetPath needs at least two poses")
        if s_ext < 0:
            raise ValueError("s_ext must be >= 0")
        self.mode = mode
        self.source_clip_id = source_clip_id
        self.frame_indices = np.asarray(frame_indices, dtype=int)
        self.s_end = 1.0
        self.s_ext = float(s_ext)
        self.s_max = 1.0 + self.s_ext
        self.pose_weights, self.root_pos_mode = resolve_pose_weights(
            pose_weights, root_pos_mode)

        self.root_pos = np.stack([p["root_pos"] for p in poses])
        self.root_rot = np.stack([
            np.asarray(p["root_rot"], np.float64)
            / np.linalg.norm(p["root_rot"]) for p in poses])
        self.joint_pos_norm = np.stack([p["joint_pos_norm"]
                                        for p in poses])
        self.knot_s = self._parameterize()

        scalar = np.concatenate(
            [self.root_pos, self.joint_pos_norm], axis=1)
        self._scalar_interp = PchipInterpolator(
            self.knot_s, scalar, axis=0, extrapolate=False)
        self._scalar_end = scalar[-1]
        self._scalar_end_slope = np.asarray(
            self._scalar_interp.derivative()(self.knot_s[-1]))

    @classmethod
    def from_pose_pair(cls, start: Dict[str, np.ndarray],
                       end: Dict[str, np.ndarray],
                       source_clip_id: str = "pose_pair",
                       s_ext: float = 0.1,
                       pose_weights: Optional[Dict[str, float]] = None,
                       root_pos_mode: Optional[str] = None
                       ) -> "TargetPath":
        return cls([pose_dict(start), pose_dict(end)], "endpoints",
                   source_clip_id, np.array([0, 1], dtype=int),
                   s_ext=s_ext, pose_weights=pose_weights,
                   root_pos_mode=root_pos_mode)

    @classmethod
    def from_clip(cls, clip: CoreClip, mode: str = "all",
                  start_frame: int = 0, end_frame: int = -1,
                  s_ext: float = 0.1,
                  pose_weights: Optional[Dict[str, float]] = None,
                  root_pos_mode: Optional[str] = None
                  ) -> "TargetPath":
        end = clip.T - 1 if end_frame < 0 else int(end_frame)
        start = int(start_frame)
        if not (0 <= start < end < clip.T):
            raise ValueError(
                f"invalid frame range [{start},{end}] for T={clip.T}")
        if mode == "endpoints":
            idx = np.array([start, end], dtype=int)
        elif mode == "all":
            idx = np.arange(start, end + 1, dtype=int)
        else:
            raise ValueError(f"unknown path mode {mode!r}")
        poses = [pose_dict(clip.state(int(k))) for k in idx]
        return cls(poses, mode, clip.clip_id, idx,
                   s_ext=s_ext, pose_weights=pose_weights,
                   root_pos_mode=root_pos_mode)

    def _parameterize(self) -> np.ndarray:
        """Arc-length-like knot parameterization by weighted pose error."""
        n = len(self.root_pos)
        if n == 2:
            return np.array([0.0, 1.0])
        ds = np.zeros(n)
        for i in range(1, n):
            a = {"root_pos": self.root_pos[i - 1],
                 "root_rot": self.root_rot[i - 1],
                 "joint_pos_norm": self.joint_pos_norm[i - 1]}
            b = {"root_pos": self.root_pos[i],
                 "root_rot": self.root_rot[i],
                 "joint_pos_norm": self.joint_pos_norm[i]}
            ds[i] = pose_error(a, b, self.pose_weights)["total"]
        if float(ds.sum()) < 1e-9:
            return np.linspace(0.0, 1.0, n)
        # Keep knot_s strictly increasing for PCHIP, even if two poses
        # are numerically identical.
        ds[1:] = np.maximum(ds[1:], 1e-9)
        return np.cumsum(ds) / ds.sum()

    def frame_to_s(self, frame_index: int) -> float:
        """Map a source clip frame index to its path knot parameter."""
        hits = np.nonzero(self.frame_indices == int(frame_index))[0]
        if len(hits) == 0:
            raise ValueError(
                f"frame {frame_index} is not a waypoint on this path")
        return float(self.knot_s[int(hits[0])])

    def _scalar_at(self, s: float) -> np.ndarray:
        if s <= self.s_end:
            return np.asarray(self._scalar_interp(s))
        return self._scalar_end + self._scalar_end_slope * (s - self.s_end)

    def _quat_at(self, s: float) -> np.ndarray:
        knots = self.knot_s
        if s <= self.s_end:
            i = int(np.clip(np.searchsorted(knots, s, side="right") - 1,
                            0, len(knots) - 2))
            a = 0.0 if knots[i + 1] == knots[i] else float(
                (s - knots[i]) / (knots[i + 1] - knots[i]))
            return _slerp(self.root_rot[i], self.root_rot[i + 1], a)
        # Bounded tangent continuation in rotation space.
        i = len(knots) - 2
        q_rel = _quat_mul_wxyz(_quat_inv_wxyz(self.root_rot[i]),
                               self.root_rot[i + 1])
        rv = _quat_log_wxyz(q_rel)
        seg = max(knots[i + 1] - knots[i], 1e-12)
        alpha = (s - knots[i]) / seg
        rv_ext = rv * alpha
        q = _quat_mul_wxyz(self.root_rot[i], _quat_exp_wxyz(rv_ext))
        return (q / np.linalg.norm(q)).astype(np.float32)

    def sample(self, s: float) -> PoseSample:
        s = float(s)
        if not (0.0 <= s <= self.s_max):
            raise ValueError(f"s {s} outside [0,{self.s_max}]")
        scalar = self._scalar_at(s)
        return PoseSample(
            s=s,
            root_pos=scalar[:3].astype(np.float32),
            root_rot=self._quat_at(s),
            joint_pos_norm=scalar[3:].astype(np.float32))

    def project(self, state: Dict[str, np.ndarray],
                s_hint: float, ds_max: float,
                n_eval: int = 33) -> PathProjection:
        """Project state onto the local forward portion of the path.

        ``ds_max`` limits progress search; it is not a path-distance
        tolerance.  The optimizer decides how much path deviation to
        accept through its scoring.
        """
        if ds_max <= 0:
            raise ValueError("ds_max must be positive")
        lo = float(np.clip(s_hint, 0.0, self.s_max))
        hi = min(self.s_max, lo + float(ds_max))
        cand = np.linspace(lo, hi, max(3, int(n_eval)))
        # Include local knots plus milestones so progress can land exactly
        # on source poses and report s=1 / s_max without grid error.
        local_knots = self.knot_s[(self.knot_s >= lo) &
                                  (self.knot_s <= hi)]
        milestones = [v for v in (self.s_end, self.s_max)
                      if lo <= v <= hi]
        cand = np.unique(np.concatenate(
            [cand, local_knots, [lo, hi], milestones]))
        best = None
        for s in cand:
            p = self.sample(float(s))
            err = pose_error(state, p.as_state(), self.pose_weights)
            if best is None or err["total"] < best.errors["total"]:
                best = PathProjection(float(s), p, err)
        assert best is not None
        return best
