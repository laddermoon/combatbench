"""Three-stage P5 -> full-body pose solver (pure geometry, no dynamics).

Stage order per plan: feet -> head -> hands; earlier point constraints stay
active (within tolerance) in later stages. Each stage minimizes a
high-weight task residual plus a low-weight spring-prior energy toward the
natural pose q0.

Deterministic: fixed init rule, fixed stage order, fixed solver settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as _R

from mocap.point5.contracts import POINT_NAMES, POINT_INDEX

from .fk import Humanoid21FK
from .landmarks import LANDMARKS
from .spring import SpringModel, load_spring_model

_FEET = ("foot_right", "foot_left")
_STAGE_POINTS = (("foot_right", "foot_left"),
                 ("foot_right", "foot_left", "head"),
                 POINT_NAMES)


@dataclass(frozen=True)
class SolveResult:
    robot_id: str
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray          # full controlled-joint vector (rad)
    landmark_pos: np.ndarray       # (5,3) achieved world positions
    point_errors: np.ndarray       # (5,) metres
    satisfied: bool
    failed_stage: Optional[str]
    cost: float
    diagnostics: Dict = field(default_factory=dict)


class PoseSolver:
    """P5 -> Humanoid21 representative pose via staged optimization."""

    def __init__(self, fk: Optional[Humanoid21FK] = None,
                 spring: Optional[SpringModel] = None,
                 tolerance: float = 0.02,
                 task_weight: float = 100.0,
                 energy_weight: float = 1.0) -> None:
        self.fk = fk or Humanoid21FK()
        self.spring = spring or load_spring_model(self.fk)
        self.names = list(self.spring.joint_names)
        self.q0 = self.spring.q0
        self.fixed = {n for n, f in zip(self.names, self.spring.fixed_mask)
                      if f}
        self.ranges = self.spring.ranges
        self.tolerance = float(tolerance)
        self.task_weight = float(task_weight)
        self.energy_weight = float(energy_weight)

    # -- public API ----------------------------------------------------------

    def solve(self, target_pos: np.ndarray,
              robot_id: str = "robot_a") -> SolveResult:
        target = np.asarray(target_pos, dtype=np.float64)
        if target.shape != (5, 3):
            raise ValueError(f"target_pos must be (5,3), got {target.shape}")
        x = self._initial_guess(target)
        failed_stage = None
        for stage_idx, pts in enumerate(_STAGE_POINTS):
            x = self._stage_solve(x, target, robot_id, pts)
            errs = self._point_errors(x, target, robot_id)
            want = [POINT_INDEX[p] for p in pts]
            if errs[want].max() > self.tolerance:
                failed_stage = pts[-1] if len(pts) < 5 else "hands"
                # continue anyway (best-effort), but record first failure
                break
        lm, root_pos, quat, q = self._unpack(x, robot_id)
        errs = np.linalg.norm(lm - target, axis=1)
        return SolveResult(
            robot_id=robot_id, root_pos=root_pos, root_quat_wxyz=quat,
            joint_pos=q, landmark_pos=lm, point_errors=errs,
            satisfied=bool(errs.max() <= self.tolerance),
            failed_stage=failed_stage, cost=float(errs.max()),
            diagnostics={"tolerance": self.tolerance})

    # -- internals -----------------------------------------------------------

    def _initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Feet line defines yaw; COM of feet/head box defines root region."""
        fr, fl = target[POINT_INDEX["foot_right"]], \
            target[POINT_INDEX["foot_left"]]
        mid = (fr + fl) / 2.0
        d = fl - fr
        yaw = float(np.arctan2(-d[0], d[1])) if np.linalg.norm(d[:2]) > 1e-3 \
            else 0.0
        # root roughly above ankle midpoint at natural-pose height offset
        height_off = 1.216 - 0.087
        root_pos = np.array([mid[0], mid[1], mid[2] + height_off])
        rot = _R.from_euler("z", yaw).as_rotvec()
        return self._pack(root_pos, rot, self.q0)

    def _pack(self, root_pos, rot_vec, q) -> np.ndarray:
        return np.concatenate([np.asarray(root_pos, dtype=np.float64),
                               np.asarray(rot_vec, dtype=np.float64), q])

    def _unpack(self, x, robot_id):
        self.fk.set_pose(robot_id, x[0:3], _quat_of(x[3:6]), x[6:])
        return (self.fk.landmark_positions(robot_id),
                x[0:3].copy(), _quat_of(x[3:6]), x[6:].copy())

    def _point_errors(self, x, target, robot_id) -> np.ndarray:
        lm, *_ = self._unpack(x, robot_id)
        return np.linalg.norm(lm - target, axis=1)

    def _residual(self, x, target, robot_id, point_idx):
        self.fk.set_pose(robot_id, x[0:3], _quat_of(x[3:6]), x[6:])
        lm = self.fk.landmark_positions(robot_id)
        task = self.task_weight * (lm[point_idx] - target[point_idx]).ravel()
        energy = self.energy_weight * self.spring.energy_residual(x[6:])
        return np.concatenate([task, energy])

    def _stage_solve(self, x0, target, robot_id, pts) -> np.ndarray:
        idx = [POINT_INDEX[p] for p in pts]
        lo = np.full(len(x0), -np.inf)
        hi = np.full(len(x0), np.inf)
        # root pos loose bounds around targets
        lo[0:3], hi[0:3] = target.min(axis=0) - 1.5, target.max(axis=0) + 1.5
        lo[6:], hi[6:] = self.ranges[:, 0], self.ranges[:, 1]
        # fixed joints clamped to q0 (leastsq needs lo<hi strictly)
        for i, n in enumerate(self.names):
            if n in self.fixed:
                lo[6 + i] = self.q0[i] - 1e-9
                hi[6 + i] = self.q0[i] + 1e-9
        x0 = np.clip(x0, lo, hi)
        sol = least_squares(self._residual, x0, bounds=(lo, hi),
                            args=(target, robot_id, idx),
                            method="trf", xtol=1e-10, ftol=1e-10,
                            gtol=1e-10, max_nfev=400)
        return sol.x


def _quat_of(rot_vec) -> np.ndarray:
    """rotvec -> wxyz quaternion (identity-safe at zero)."""
    rv = np.asarray(rot_vec, dtype=np.float64)
    if np.linalg.norm(rv) < 1e-12:
        return np.array([1.0, 0, 0, 0])
    q = _R.from_rotvec(rv).as_quat()          # xyzw
    return np.array([q[3], q[0], q[1], q[2]])
