"""Joint spring model for Humanoid21 (the robot's pose prior).

Works in the project's NORMALIZED joint coordinates:
``q_norm = (q_rad - mid) / half_range``, every joint spans [-1, 1].
``pose_config.yaml`` values (natural pose q0) are normalized too.

Defines, per controlled joint:
  - ``q0``        natural angle, normalized coords
  - ``k``         stiffness weight resisting deviation from q0
  - ``fixed``     whether the joint is excluded from optimization
  - ``range``     physical rad limits (for converting to FK input)

Elastic energy:  E(q) = 1/2 * sum_j k_j * (q_j - q0_j)^2   (normalized)

This is a *pose preference* model, not a physical PD/stiffness model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .fk import Humanoid21FK

CONFIG_PATH = Path(__file__).with_name("pose_config.yaml")


@dataclass(frozen=True)
class SpringModel:
    """Asymmetric progressive springs on normalized coords.

    Spring reference is the all-zero pose (mechanical midpoint for every
    joint, limits at ±1). Force law is quadratic:

        F(q) = -k_side * q * |q|      (soft mid-range, stiff near limits)
        E(q) =  k_side * |q|^3 / 3

    ``k_low`` applies when q < 0, ``k_high`` when q > 0.
    ``q_init`` is the user-tuned natural pose — used ONLY as the solver's
    initial guess, it is NOT the spring center.
    """
    joint_names: tuple
    k_low: np.ndarray           # (J,) stiffness below 0
    k_high: np.ndarray          # (J,) stiffness above 0
    law: np.ndarray             # (J,) bool: True=linear, False=quadratic
    fixed_mask: np.ndarray      # (J,) bool, True = excluded from optimization
    ranges: np.ndarray          # (J,2) physical rad limits
    reference: np.ndarray       # (J,) rad midpoint
    scale: np.ndarray           # (J,) rad half-range
    q0: np.ndarray = None       # spring center, always zeros

    @property
    def active_idx(self) -> np.ndarray:
        return np.nonzero(~self.fixed_mask)[0]

    @property
    def k(self) -> np.ndarray:
        """Symmetric view (max of the two sides) for introspection."""
        return np.maximum(self.k_low, self.k_high)

    # normalized <-> rad conversions -----------------------------------------
    def to_rad(self, q_norm: np.ndarray) -> np.ndarray:
        return np.asarray(q_norm) * self.scale + self.reference

    def to_norm(self, q_rad: np.ndarray) -> np.ndarray:
        return (np.asarray(q_rad) - self.reference) / self.scale

    # energy in normalized coords ---------------------------------------------
    def _k_side(self, q_norm: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(q_norm) < 0.0,
                        self.k_low, self.k_high)

    def energy(self, q_norm: np.ndarray) -> float:
        q = np.asarray(q_norm, dtype=np.float64)
        a = np.abs(q)
        k = self._k_side(q)
        return float(np.sum(np.where(
            self.law, 0.5 * k * q * q, k * a * a * a / 3.0)))


def load_spring_model(fk: Optional[Humanoid21FK] = None,
                      config_path: Optional[Path] = None,
                      robot_id: str = "robot_a") -> SpringModel:
    """Build the spring model from pose_config.yaml (normalized values)."""
    fk = fk or Humanoid21FK()
    cfg = yaml.safe_load(Path(config_path or CONFIG_PATH).read_text())
    names = tuple(n.rsplit("_", 1)[0]
                  for n in fk.controlled_joints(robot_id))
    fixed = set(cfg.get("fixed_joints", []))
    k_low = np.array([_k_of(cfg, n)[0] for n in names], dtype=np.float64)
    k_high = np.array([_k_of(cfg, n)[1] for n in names], dtype=np.float64)
    law = np.array([_law_of(cfg, n) for n in names], dtype=bool)
    ranges = np.asarray(fk.joint_ranges(robot_id), dtype=np.float64)
    reference = (ranges[:, 0] + ranges[:, 1]) / 2.0
    scale = (ranges[:, 1] - ranges[:, 0]) / 2.0
    return SpringModel(
        joint_names=names, k_low=k_low, k_high=k_high, law=law,
        scale=scale,
        fixed_mask=np.array([n in fixed for n in names]),
        ranges=ranges, reference=reference,
        q0=np.zeros(len(names)))


def _law_of(cfg, name: str) -> bool:
    """True = linear force, False = quadratic (default quadratic)."""
    for group in cfg.get("stiffness_groups", {}).values():
        if name in group["joints"]:
            return group.get("law", "quadratic") == "linear"
    return False


def _k_of(cfg, name: str):
    """Return (k_low, k_high); scalar k expands to symmetric pair."""
    for group in cfg.get("stiffness_groups", {}).values():
        if name in group["joints"]:
            v = group["k"]
            if isinstance(v, dict):
                return float(v.get("low", 1.0)), float(v.get("high", 1.0))
            return float(v), float(v)
    return 1.0, 1.0
