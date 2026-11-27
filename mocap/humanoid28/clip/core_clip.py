"""CoreClip28 — Humanoid28-bound motion clip (50 ms control frames).

Same role as mocap/humanoid21/clip/core_clip.py but for the MimicKit
humanoid: joint fields are RAW RADIANS (matching the env's
get_core_state/set_core_state contract — no normalization layer),
28 controlled hinges, pelvis root.

A CoreClip is a sequence of simulator CoreState frames plus optional
per-frame reference action and keypoint tracks. Frame k IS the dict
returned by ``sim.get_core_state()[robot_id]``.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SPEC_VERSION = 1
CONTROL_DT = 0.05            # 20 Hz control frames — fixed, not stored per frame
CONTROL_FPS = 1.0 / CONTROL_DT
PHY_DT = 0.002               # Humanoid28Simulator.DT
PHY_STEPS_PER_ACTION = 25    # CONTROL_DT / PHY_DT
ROBOT_ID = "humanoid28"
N_DOF = 28

# CoreState fields, identical to Humanoid28Simulator.get_core_state().
CORE_FIELDS = (
    "root_pos", "root_rot",
    "root_vel_local", "root_angular_vel_local",
    "joint_pos", "joint_vel",
)
FIELD_DIMS = {"root_pos": 3, "root_rot": 4,
              "root_vel_local": 3, "root_angular_vel_local": 3,
              "joint_pos": N_DOF, "joint_vel": N_DOF}

META_REQUIRED = ("spec_version", "clip_id")


def record_lineage(meta: Dict[str, Any], op: str,
                   parent: Optional[str] = None,
                   **params) -> Dict[str, Any]:
    """Append a provenance hop to meta['lineage'] (append-only list)."""
    m = dict(meta)
    lin = list(m.get("lineage") or [])
    lin.append({"op": op, "parent": parent, "params": params,
                "ts": datetime.now().isoformat(timespec="seconds")})
    m["lineage"] = lin
    return m

# Validation tolerances
_QUAT_TOL = 1e-3


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """wxyz quaternion slerp, shortest arc."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + a * (q1 - q0)
    else:
        th = np.arccos(np.clip(d, -1.0, 1.0))
        out = (np.sin((1.0 - a) * th) * q0
               + np.sin(a * th) * q1) / np.sin(th)
    return (out / np.linalg.norm(out)).astype(np.float32)


class CoreClip:
    """A Humanoid28 clip: T frames of CoreState (+ optional ref_action /
    keypoint_pos sidecars)."""

    def __init__(self, states: Dict[str, np.ndarray],
                 meta: Dict[str, Any],
                 ref_action: Optional[np.ndarray] = None,
                 keypoint_pos: Optional[np.ndarray] = None) -> None:
        self.states = {k: np.asarray(v, dtype=np.float32)
                       for k, v in states.items()}
        self.meta = dict(meta)
        self.ref_action = (None if ref_action is None
                           else np.asarray(ref_action, dtype=np.float32))
        self.keypoint_pos = (None if keypoint_pos is None
                             else np.asarray(keypoint_pos,
                                             dtype=np.float32))
        warnings = self.validate()
        if warnings:
            import warnings as _w
            for w in warnings:
                _w.warn(f"CoreClip[{self.clip_id}]: {w}")

    # --------------------------------------------------------------
    @property
    def T(self) -> int:
        return self.states["root_pos"].shape[0]

    @property
    def duration_s(self) -> float:
        return self.T * CONTROL_DT

    @property
    def clip_id(self) -> str:
        return str(self.meta.get("clip_id", "<unnamed>"))

    # --------------------------------------------------------------
    def state(self, k: int) -> Dict[str, np.ndarray]:
        """Frame k as a CoreState dict — feed directly to set_core_state."""
        k = int(k)
        if not (0 <= k < self.T):
            raise IndexError(f"frame {k} out of range [0,{self.T})")
        return {f: self.states[f][k].copy() for f in CORE_FIELDS}

    def state_at(self, t: float) -> Dict[str, np.ndarray]:
        """Continuous-time state: lerp all fields, slerp root_rot."""
        tau = float(t) / CONTROL_DT
        k0 = int(np.clip(np.floor(tau), 0, self.T - 1))
        k1 = min(k0 + 1, self.T - 1)
        a = float(np.clip(tau - k0, 0.0, 1.0))
        out: Dict[str, np.ndarray] = {}
        for f in CORE_FIELDS:
            if f == "root_rot":
                continue
            out[f] = ((1.0 - a) * self.states[f][k0]
                      + a * self.states[f][k1]).astype(np.float32)
        out["root_rot"] = _slerp(self.states["root_rot"][k0],
                                 self.states["root_rot"][k1], a)
        return out

    def keypoint(self, k: int) -> Dict[str, np.ndarray]:
        """Frame k keypoints as {name: pos} — needs keypoint_pos +
        meta['keypoint_names']."""
        if self.keypoint_pos is None:
            raise KeyError(f"CoreClip[{self.clip_id}] has no keypoint_pos")
        names = self.meta.get("keypoint_names")
        if not names:
            raise KeyError(f"CoreClip[{self.clip_id}] meta lacks "
                           "'keypoint_names'")
        return {n: self.keypoint_pos[int(k), i].copy()
                for i, n in enumerate(names)}

    def slice(self, k0: int, k1: int, clip_id: Optional[str] = None
              ) -> "CoreClip":
        states = {f: self.states[f][k0:k1] for f in CORE_FIELDS}
        meta = dict(self.meta)
        meta["clip_id"] = clip_id or f"{self.clip_id}_s{k0}_{k1}"
        meta["source"] = {"type": "slice", "parent": self.clip_id,
                          "frames": [k0, k1]}
        meta = record_lineage(meta, "slice", parent=self.clip_id,
                              frames=[k0, k1])
        ra = None if self.ref_action is None \
            else self.ref_action[k0:k1]
        kp = None if self.keypoint_pos is None \
            else self.keypoint_pos[k0:k1]
        return CoreClip(states, meta, ra, kp)

    # --------------------------------------------------------------
    @classmethod
    def from_states(cls, states: Sequence[Dict[str, np.ndarray]],
                    meta: Dict[str, Any],
                    ref_action: Optional[Sequence[np.ndarray]] = None,
                    keypoint_pos: Optional[np.ndarray] = None
                    ) -> "CoreClip":
        """Build from a list of per-robot CoreState dicts."""
        if not states:
            raise ValueError("CoreClip.from_states: empty state list")
        stacked = {f: np.stack([np.asarray(s[f]) for s in states])
                   for f in CORE_FIELDS}
        ra = None if ref_action is None else np.stack(
            [np.asarray(a) for a in ref_action])
        m = dict(meta)
        m.setdefault("spec_version", SPEC_VERSION)
        return cls(stacked, m, ra, keypoint_pos)

    # --------------------------------------------------------------
    def validate(self) -> List[str]:
        """Hard errors raise ValueError; returns warning strings."""
        errors: List[str] = []
        warnings: List[str] = []
        for k in META_REQUIRED:
            if k not in self.meta:
                errors.append(f"meta missing '{k}'")
        for f in CORE_FIELDS:
            if f not in self.states:
                errors.append(f"missing field '{f}'")
        if errors:
            raise ValueError(f"CoreClip invalid: {'; '.join(errors)}")
        T = self.T
        for f, d in FIELD_DIMS.items():
            s = self.states[f].shape
            if len(s) != 2 or s[0] != T or s[1] != d:
                errors.append(f"{f} shape {s}, expected ({T},{d})")
        if self.ref_action is not None:
            s = self.ref_action.shape
            if len(s) != 2 or s[0] != T or s[1] != N_DOF:
                errors.append(f"ref_action shape {s}, expected ({T},{N_DOF})")
            elif not np.isfinite(self.ref_action).all():
                errors.append("ref_action contains NaN/inf")
            if "ref_action_semantics" not in self.meta:
                warnings.append("ref_action present but meta lacks "
                                "'ref_action_semantics'")
        if self.keypoint_pos is not None:
            s = self.keypoint_pos.shape
            names = self.meta.get("keypoint_names")
            if len(s) != 3 or s[0] != T or s[2] != 3:
                errors.append(f"keypoint_pos shape {s}, "
                              f"expected ({T},K,3)")
            elif names and s[1] != len(names):
                errors.append(f"keypoint_pos K={s[1]} but "
                              f"keypoint_names has {len(names)}")
            elif not np.isfinite(self.keypoint_pos).all():
                errors.append("keypoint_pos contains NaN/inf")
            if not names:
                warnings.append("keypoint_pos present but meta lacks "
                                "'keypoint_names'")
        q = self.states["root_rot"]
        if not np.isfinite(q).all():
            errors.append("root_rot contains NaN/inf")
        else:
            dev = np.abs(np.linalg.norm(q, axis=1) - 1.0)
            if dev.max() > _QUAT_TOL:
                errors.append(f"root_rot non-unit (max dev {dev.max():.2e})")
        for f in ("joint_pos", "joint_vel",
                  "root_vel_local", "root_angular_vel_local"):
            v = self.states[f]
            if not np.isfinite(v).all():
                errors.append(f"{f} contains NaN/inf")
        if errors:
            raise ValueError(f"CoreClip[{self.clip_id}] invalid: "
                             + "; ".join(errors))
        return warnings

    # --------------------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.setdefault("spec_version", SPEC_VERSION)
        meta["control_dt"] = CONTROL_DT
        meta["phy_dt"] = PHY_DT
        meta.setdefault("robot", ROBOT_ID)
        meta.setdefault("joint_space", ROBOT_ID)
        arrays = dict(self.states)
        if self.ref_action is not None:
            arrays["ref_action"] = self.ref_action
        if self.keypoint_pos is not None:
            arrays["keypoint_pos"] = self.keypoint_pos
        arrays["meta_json"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "CoreClip":
        z = np.load(Path(path), allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        states = {f: z[f] for f in CORE_FIELDS}
        ra = z["ref_action"] if "ref_action" in z.files else None
        kp = z["keypoint_pos"] if "keypoint_pos" in z.files else None
        return cls(states, meta, ra, kp)

    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.T

    def __repr__(self) -> str:
        return (f"CoreClip28({self.clip_id!r}, T={self.T}, "
                f"{self.duration_s:.2f}s, "
                f"ref_action={'yes' if self.ref_action is not None else 'no'})")
