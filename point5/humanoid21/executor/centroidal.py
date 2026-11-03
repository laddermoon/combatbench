"""Centroidal state + trajectory — the only things kept from the old model.

Core state (per the meta-design's single useful line):

    z = (c, l, L)          # 9 continuous quantities, robot local frame
    com_vel = l / MASS_KG  # linear momentum gives COM velocity

Contact mode, footstep positions, phase timing, whole-body config
are *accompanying* information — not part of z.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

CTRL_DT = 0.05             # simulator control period (20 Hz)

# Humanoid21 total body mass (kg) — from
# envs.humanoid21.meta.Humanoid21Meta.build_runtime_tables
# ['robot_a']['total_mass'].  Constant for this robot.
MASS_KG = 40.844


@dataclass
class CentroidalState:
    """z = (c, l, L)."""

    com: np.ndarray                      # c, (3,) local (P5 frame)
    lin_mom: np.ndarray                  # l, (3,) local
    ang_mom: np.ndarray                  # L, (3,) local

    @property
    def com_vel(self) -> np.ndarray:
        return self.lin_mom / MASS_KG

    @property
    def z(self) -> np.ndarray:
        return np.concatenate([self.com, self.lin_mom, self.ang_mom])

    def to_dict(self) -> dict:
        return {"com": self.com.tolist(),
                "lin_mom": self.lin_mom.tolist(),
                "ang_mom": self.ang_mom.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "CentroidalState":
        return cls(com=np.asarray(d["com"], np.float64),
                   lin_mom=np.asarray(d["lin_mom"], np.float64),
                   ang_mom=np.asarray(d["ang_mom"], np.float64))


@dataclass
class CentroidalTrajectory:
    """A planned centroidal rollout: states[k] is the z AFTER step k.

    Two access modes:
      - ``n_steps`` (or len) -> number of steps
      - ``state_after(k)`` / ``traj[k]`` -> CentroidalState after step k
    """

    states: list                       # List[CentroidalState]

    @property
    def n_steps(self) -> int:
        return len(self.states)

    def __len__(self) -> int:
        return self.n_steps

    def state_after(self, k: int) -> CentroidalState:
        return self.states[k]

    def __getitem__(self, k: int) -> CentroidalState:
        return self.states[k]


def save_traj(traj: CentroidalTrajectory, path: str) -> None:
    """Serialize a CentroidalTrajectory to JSON."""
    with open(path, "w") as fh:
        json.dump({"format": "centroidal_trajectory/1",
                   "dt": CTRL_DT,
                   "states": [s.to_dict() for s in traj.states]}, fh)


def load_traj(path: str) -> CentroidalTrajectory:
    """Load a trajectory saved by ``save_traj``."""
    with open(path) as fh:
        d = json.load(fh)
    if d.get("format") != "centroidal_trajectory/1":
        raise ValueError(f"not a centroidal trajectory file: {path}")
    return CentroidalTrajectory(
        states=[CentroidalState.from_dict(s) for s in d["states"]])
