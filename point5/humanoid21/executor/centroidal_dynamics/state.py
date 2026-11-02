"""Centroidal state extraction for Humanoid21.

Two independent paths are provided:

- ``manual_state``: mass-weighted rigid-body summation over a robot subtree
- ``subtree_state``: MuJoCo ``subtree_*`` fields rooted at the torso

``state()`` cross-checks both paths and returns the manual estimate. Its
metadata contains the subtraction metrics used for the consistency check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from .contracts import CentroidalState, ROBOT_IDS, StateNotConsistent
from .model_spec import CentroidalModelSpec, build_model_spec


@dataclass(frozen=True)
class ConsistencyTolerances:
    mass: float = 1e-9
    com: float = 1e-9
    lin_mom: float = 1e-8
    ang_mom: float = 1e-7


class CentroidalStateExtractor:
    """Extract whole-robot ``CentroidalState`` from a live simulator."""

    def __init__(self, sim, model_spec: Optional[CentroidalModelSpec] = None):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.model_spec = model_spec or build_model_spec(sim)

    def refresh(self) -> None:
        """Refresh kinematics/dynamics fields needed by both paths."""
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_subtreeVel(self.model, self.data)

    def manual_state(self, robot_id: str, *, refresh: bool = True
                     ) -> CentroidalState:
        """Mass/inertia-weighted sum over all bodies in a robot subtree."""
        robot = self._robot(robot_id)
        if refresh:
            self.refresh()
        model, data = self.model, self.data
        body_ids = np.asarray(robot.body_ids, dtype=np.int32)
        masses = np.asarray(model.body_mass[body_ids], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            raise ValueError(f"{robot_id} has non-positive subtree mass")

        body_com = np.asarray(data.xipos[body_ids], dtype=np.float64)
        com = (masses @ body_com) / total_mass
        lin_mom = np.zeros(3, dtype=np.float64)
        ang_mom = np.zeros(3, dtype=np.float64)
        jac_p = np.zeros((3, model.nv), dtype=np.float64)
        jac_r = np.zeros((3, model.nv), dtype=np.float64)
        for local_i, body_id in enumerate(body_ids):
            # mj_jacBodyCom returns the world-frame velocity of body COM.
            # data.cvel is deliberately not used here: its components are not
            # the world-frame body-COM point velocity required by this sum.
            jac_p.fill(0.0)
            jac_r.fill(0.0)
            mujoco.mj_jacBodyCom(model, data, jac_p, jac_r,
                                 int(body_id))
            linvel = jac_p @ data.qvel
            angvel = jac_r @ data.qvel
            lin_mom += masses[local_i] * linvel
            ang_mom += np.cross(body_com[local_i] - com,
                                masses[local_i] * linvel)
            R = np.asarray(data.ximat[body_id], dtype=np.float64).reshape(3, 3)
            I_body = np.diag(np.asarray(model.body_inertia[body_id],
                                      dtype=np.float64))
            ang_mom += R @ I_body @ R.T @ angvel

        return CentroidalState(
            robot_id=robot_id,
            time=float(data.time),
            mass=total_mass,
            com_pos=com,
            lin_mom=lin_mom,
            ang_mom=ang_mom,
            source="manual_body_sum",
            metadata={
                "model_fingerprint": self.model_spec.fingerprint,
                "root_body_id": robot.root_body_id,
                "body_count": len(robot.body_ids),
            },
        )

    def subtree_state(self, robot_id: str, *, refresh: bool = True
                      ) -> CentroidalState:
        """MuJoCo subtree COM / velocity / angular-momentum path."""
        robot = self._robot(robot_id)
        if refresh:
            self.refresh()
        data, model = self.data, self.model
        root = int(robot.root_body_id)
        mass = float(model.body_subtreemass[root])
        if mass <= 0.0:
            raise ValueError(f"{robot_id} has non-positive subtree mass")
        return CentroidalState(
            robot_id=robot_id,
            time=float(data.time),
            mass=mass,
            com_pos=np.asarray(data.subtree_com[root], dtype=np.float64),
            lin_mom=mass * np.asarray(data.subtree_linvel[root],
                                      dtype=np.float64),
            ang_mom=np.asarray(data.subtree_angmom[root],
                               dtype=np.float64),
            source="mujoco_subtree",
            metadata={
                "model_fingerprint": self.model_spec.fingerprint,
                "root_body_id": robot.root_body_id,
                "body_count": len(robot.body_ids),
            },
        )

    def state(self, robot_id: str,
              tolerances: ConsistencyTolerances = ConsistencyTolerances()
              ) -> CentroidalState:
        """Return manual state after cross-checking MuJoCo subtree state."""
        self.refresh()
        manual = self.manual_state(robot_id, refresh=False)
        subtree = self.subtree_state(robot_id, refresh=False)
        metrics = self._difference_metrics(manual, subtree)
        violations = {
            name: (value, getattr(tolerances, name))
            for name, value in metrics.items()
            if value > getattr(tolerances, name)
        }
        if violations:
            raise StateNotConsistent(
                f"manual/subtree centroidal mismatch for {robot_id}: "
                f"{violations}")
        metadata = dict(manual.metadata)
        metadata.update({
            "source": "manual_body_sum+mujoco_subtree_checked",
            "subtree_source": "mujoco_subtree",
            "consistency": metrics,
        })
        return CentroidalState(
            robot_id=robot_id,
            time=manual.time,
            mass=manual.mass,
            com_pos=manual.com_pos,
            lin_mom=manual.lin_mom,
            ang_mom=manual.ang_mom,
            source="manual_body_sum+mujoco_subtree_checked",
            metadata=metadata,
        )

    def _robot(self, robot_id: str):
        if robot_id not in ROBOT_IDS:
            raise ValueError(f"unknown robot_id {robot_id!r}")
        return self.model_spec.robots[robot_id]

    @staticmethod
    def _difference_metrics(a: CentroidalState,
                            b: CentroidalState) -> Dict[str, float]:
        return {
            "mass": float(abs(a.mass - b.mass)),
            "com": float(np.linalg.norm(a.com_pos - b.com_pos)),
            "lin_mom": float(np.linalg.norm(a.lin_mom - b.lin_mom)),
            "ang_mom": float(np.linalg.norm(a.ang_mom - b.ang_mom)),
        }


def extract_centroidal_state(sim, robot_id: str,
                             model_spec: Optional[CentroidalModelSpec] = None,
                             tolerances: ConsistencyTolerances =
                             ConsistencyTolerances()) -> CentroidalState:
    """Convenience wrapper around ``CentroidalStateExtractor.state``."""
    return CentroidalStateExtractor(sim, model_spec).state(
        robot_id, tolerances)
