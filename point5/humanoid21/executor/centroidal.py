"""Centroidal state — the only thing kept from the old model.

Core state (per the meta-design's single useful line):

    z = (c, l, L)          # 9 continuous quantities, robot local frame
    com_vel = l / MASS_KG  # linear momentum gives COM velocity

Contact mode, footstep positions, phase timing, whole-body config
are *accompanying* information — not part of z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
