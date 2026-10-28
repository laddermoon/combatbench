"""Mechanics model: P5 -> full-body equilibrium pose.

Model: weightless rigid-body chain + joint springs (natural pose q0,
stiffness k). The 5 P5 points are HARD constraints (pinned in space).
Solving for force balance = constrained minimum of elastic energy:

    min_q  E(q) = 1/2 * sum_j k_j ((q_j - q0_j)/s_j)^2
    s.t.   landmark_i(q) = p_i   (equality, not penalty)

The KKT condition grad E - J^T lambda = 0 IS the force balance: spring
torques on constrained DOFs are absorbed by pin reaction forces (lambda).

Variables = root(6) + joints(21) — root is a *technical* unknown: the two
pinned ankles (6 eq) determine it through the leg chains. Stage order
feet -> head -> hands; each stage warm-starts from the previous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as _R

from mocap.point5.contracts import POINT_NAMES, POINT_INDEX

from .fk import Humanoid21FK
from .spring import SpringModel, load_spring_model

_STAGE_POINTS = (("foot_right", "foot_left"),
                 ("foot_right", "foot_left", "head"),
                 POINT_NAMES)
_STAGE_NAMES = ("feet", "head", "hands")


@dataclass(frozen=True)
class SolveResult:
    robot_id: str
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    landmark_pos: np.ndarray
    point_errors: np.ndarray
    satisfied: bool
    failed_stage: Optional[str]
    energy: float
    diagnostics: Dict = field(default_factory=dict)


class PoseSolver:
    """Constrained-equilibrium solver: min E(q) s.t. pinned points."""

    def __init__(self, fk: Optional[Humanoid21FK] = None,
                 spring: Optional[SpringModel] = None,
                 tolerance: float = 0.01,
                 max_nfev: int = 800) -> None:
        self.fk = fk or Humanoid21FK()
        self.spring = spring or load_spring_model(self.fk)
        self.names = list(self.spring.joint_names)
        # Natural pose = spring zero (all-unloaded). The solve is a pure
        # function of the 5 targets: start at zero, pin feet -> head ->
        # hands quasi-statically, take the equilibrium.
        self.q0 = np.zeros(len(self.names))
        self.tolerance = float(tolerance)
        self.max_nfev = int(max_nfev)

    def solve(self, target_pos: np.ndarray,
              robot_id: str = "robot_a") -> SolveResult:
        target = np.asarray(target_pos, dtype=np.float64)
        if target.shape != (5, 3):
            raise ValueError(f"target_pos must be (5,3), got {target.shape}")
        x = self._initial_guess(target)
        failed = None
        for stage, pts in zip(_STAGE_NAMES, _STAGE_POINTS):
            x = self._stage_solve(x, target, robot_id, pts)
            errs = self._errors(x, target, robot_id)
            if errs[[POINT_INDEX[p] for p in pts]].max() > self.tolerance:
                failed = stage
                break
        lm, root_pos, quat, q_norm = self._unpack(x, robot_id)
        errs = np.linalg.norm(lm - target, axis=1)
        return SolveResult(
            robot_id=robot_id, root_pos=root_pos, root_quat_wxyz=quat,
            joint_pos=self.spring.to_rad(q_norm),  # output in radians
            landmark_pos=lm, point_errors=errs,
            satisfied=bool(errs.max() <= self.tolerance),
            failed_stage=failed,
            energy=self.spring.energy(q_norm),
            diagnostics={"tolerance": self.tolerance})

    # -- internals -----------------------------------------------------------

    def _initial_guess(self, target: np.ndarray) -> np.ndarray:
        fr = target[POINT_INDEX["foot_right"]]
        fl = target[POINT_INDEX["foot_left"]]
        mid = (fr + fl) / 2.0
        d = fl - fr
        yaw = float(np.arctan2(-d[0], d[1])) \
            if np.linalg.norm(d[:2]) > 1e-3 else 0.0
        # all-zero pose stands ~0.79 m above the ankle centers (deep crouch)
        root_pos = np.array([mid[0], mid[1], mid[2] + 0.79])
        return np.concatenate([root_pos,
                               _R.from_euler("z", yaw).as_rotvec(), self.q0])

    def _unpack(self, x, robot_id):
        quat = _quat_of(x[3:6])
        self.fk.set_pose(robot_id, x[0:3], quat,
                         self.spring.to_rad(x[6:]))
        return (self.fk.landmark_positions(robot_id),
                x[0:3].copy(), quat, x[6:].copy())

    def _errors(self, x, target, robot_id) -> np.ndarray:
        lm, *_ = self._unpack(x, robot_id)
        return np.linalg.norm(lm - target, axis=1)

    def _stage_solve(self, x0, target, robot_id, pts) -> np.ndarray:
        """Quasi-static pull: interpolate pinned-point targets from the
        current achieved positions to the final targets in small steps,
        solving equilibrium at each step (continuation). Avoids jumping
        to distant local minima (e.g. hip-fold instead of knee-bend)."""
        idx = np.array([POINT_INDEX[p] for p in pts])
        lm_now = self._unpack(x0, robot_id)[0][idx]
        dst = target[idx]
        n = max(1, int(np.ceil(
            np.linalg.norm(dst - lm_now, axis=1).max() / 0.05)))
        x = np.asarray(x0, dtype=np.float64)
        for s in range(1, n + 1):
            sub = lm_now + (dst - lm_now) * (s / n)
            x = self._equilibrium(x, idx, sub, robot_id)
        return x

    def _equilibrium(self, x0, idx, sub_target, robot_id) -> np.ndarray:
        def landmark_res(x):
            self.fk.set_pose(robot_id, x[0:3], _quat_of(x[3:6]),
                             self.spring.to_rad(x[6:]))
            return (self.fk.landmark_positions(robot_id)[idx]
                    - sub_target).ravel()

        bounds = [(-np.inf, np.inf)] * 6 + [
            (-1.0, 1.0)] * len(self.names)
        sol = minimize(
            lambda x: self.spring.energy(x[6:]), np.asarray(x0),
            method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": landmark_res}],
            options={"maxiter": self.max_nfev, "ftol": 1e-12})
        return sol.x


def _quat_of(rot_vec) -> np.ndarray:
    rv = np.asarray(rot_vec, dtype=np.float64)
    if np.linalg.norm(rv) < 1e-12:
        return np.array([1.0, 0, 0, 0])
    q = _R.from_rotvec(rv).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])
