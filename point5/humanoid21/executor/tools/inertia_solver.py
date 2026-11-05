"""Tool 3: P5 local pose -> moment of inertia.

Wraps ``point5.mass_model.mass_proxy``'s ``inertia_per_mass`` (3,3)
about the COM proxy, in the P5 local frame; scaled by MASS_KG to
absolute kg·m^2.
"""
from __future__ import annotations

import numpy as np

from point5.mass_model import mass_proxy

from ..centroidal import MASS_KG


def inertia_per_mass(local_p5: np.ndarray) -> np.ndarray:
    """Mass-normalized inertia (3,3) about the COM proxy, local frame.

    Units m^2; tensor J = I/M.
    """
    return np.asarray(
        mass_proxy(np.asarray(local_p5, np.float64)).inertia_per_mass)


def inertia_estimate(local_p5: np.ndarray) -> np.ndarray:
    """Absolute inertia (3,3) kg·m^2 about the COM proxy, local frame."""
    return inertia_per_mass(local_p5) * MASS_KG


def gyration_z(local_p5: np.ndarray) -> float:
    """Radius of gyration about the vertical axis through COM (m)."""
    return float(mass_proxy(
        np.asarray(local_p5, np.float64)).gyration_z)
