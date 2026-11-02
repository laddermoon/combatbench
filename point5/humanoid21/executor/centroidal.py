"""Centroidal state — the only thing kept from the old model.

Core state (per the meta-design's single useful line):

    z = (c, l, L)          # 9 continuous world-frame quantities
    com_vel = l / mass     # linear momentum gives COM velocity

Contact mode, footstep positions, phase timing, whole-body config
are *accompanying* information — not part of z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CentroidalState:
    """z = (c, l, L) + total mass.  Contact mode, footsteps,
    phase timing etc. are accompanying info — tracked elsewhere."""

    com: np.ndarray                      # c, (3,) world
    lin_mom: np.ndarray                  # l, (3,) world
    ang_mom: np.ndarray                  # L, (3,) world
    mass: float

    @property
    def com_vel(self) -> np.ndarray:
        return self.lin_mom / self.mass

    @property
    def z(self) -> np.ndarray:
        return np.concatenate([self.com, self.lin_mom, self.ang_mom])
