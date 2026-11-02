"""Centroidal state — the only thing kept from the old model.

Core state (per the meta-design's single useful line):

    z = (c, l, L)          # 9 continuous world-frame quantities
    com_vel = l / mass     # linear momentum gives COM velocity

Contact mode, footstep positions, phase timing, whole-body config
are *accompanying* information — not part of z.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class CentroidalState:
    """z = (c, l, L) + total mass + accompanying info."""

    com: np.ndarray                      # c, (3,) world
    lin_mom: np.ndarray                  # l, (3,) world
    ang_mom: np.ndarray                  # L, (3,) world
    mass: float
    contacts: Dict[str, Any] = field(default_factory=dict)  # 伴随信息

    @property
    def com_vel(self) -> np.ndarray:
        return self.lin_mom / self.mass

    @property
    def z(self) -> np.ndarray:
        return np.concatenate([self.com, self.lin_mom, self.ang_mom])
