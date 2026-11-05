"""Tool 1: P5 local pose -> recommended COM position (prior algorithm).

Wraps ``point5.mass_model.mass_proxy`` — a model-free virtual-body COM
estimator (~4 cm RMSE vs real equilibrium-solved poses).  Kept behind
an interface so the estimator can improve independently.
"""
from __future__ import annotations

import numpy as np

from point5.mass_model import MassProxy, mass_proxy


def com_estimate(local_p5: np.ndarray) -> np.ndarray:
    """Recommended centroid position (3,) in the P5 local frame."""
    return np.asarray(mass_proxy(np.asarray(local_p5, np.float64)).com)


def mass_estimate(local_p5: np.ndarray) -> MassProxy:
    """Full proxy: COM + mass-normalized inertia about it."""
    return mass_proxy(np.asarray(local_p5, np.float64))
