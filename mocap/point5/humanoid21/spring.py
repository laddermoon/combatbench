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
    """Asymmetric piecewise-quadratic springs on normalized coords.

    Per joint, the restoring force pulls toward q0 from both sides but
    stiffness may differ:  F(q) = -k_side * (q - q0)
      k_low  : applies when q < q0
      k_high : applies when q > q0
    """
    joint_names: tuple          # suffix-stripped CONTROLLED_JOINTS order
    q0: np.ndarray              # (J,) natural angles, NORMALIZED [-1,1]
    k_low: np.ndarray           # (J,) stiffness below q0
    k_high: np.ndarray          # (J,) stiffness above q0
    fixed_mask: np.ndarray      # (J,) bool, True = excluded from optimization
    ranges: np.ndarray          # (J,2) physical rad limits
    reference: np.ndarray       # (J,) rad midpoint
    scale: np.ndarray           # (J,) rad half-range

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
        return np.where(np.asarray(q_norm) < self.q0,
                        self.k_low, self.k_high)

    def energy(self, q_norm: np.ndarray) -> float:
        d = np.asarray(q_norm, dtype=np.float64) - self.q0
        return 0.5 * float(np.sum(self._k_side(q_norm) * d * d))

    def energy_residual(self, q_norm: np.ndarray) -> np.ndarray:
        return (np.sqrt(self._k_side(q_norm))
                * (np.asarray(q_norm, dtype=np.float64) - self.q0))


def load_spring_model(fk: Optional[Humanoid21FK] = None,
                      config_path: Optional[Path] = None,
                      robot_id: str = "robot_a") -> SpringModel:
    """Build the spring model from pose_config.yaml (normalized values)."""
    fk = fk or Humanoid21FK()
    cfg = yaml.safe_load(Path(config_path or CONFIG_PATH).read_text())
    names = tuple(n.rsplit("_", 1)[0]
                  for n in fk.controlled_joints(robot_id))
    q0 = np.array([cfg["natural_pose"][n] for n in names],
                  dtype=np.float64)
    fixed = set(cfg.get("fixed_joints", []))
    k_low = np.array([_k_of(cfg, n)[0] for n in names], dtype=np.float64)
    k_high = np.array([_k_of(cfg, n)[1] for n in names], dtype=np.float64)
    ranges = np.asarray(fk.joint_ranges(robot_id), dtype=np.float64)
    reference = (ranges[:, 0] + ranges[:, 1]) / 2.0
    scale = (ranges[:, 1] - ranges[:, 0]) / 2.0
    return SpringModel(
        joint_names=names, q0=q0, k_low=k_low, k_high=k_high, scale=scale,
        fixed_mask=np.array([n in fixed for n in names]),
        ranges=ranges, reference=reference)


def _k_of(cfg, name: str):
    """Return (k_low, k_high); scalar k expands to symmetric pair."""
    for group in cfg.get("stiffness_groups", {}).values():
        if name in group["joints"]:
            v = group["k"]
            if isinstance(v, dict):
                return float(v.get("low", 1.0)), float(v.get("high", 1.0))
            return float(v), float(v)
    return 1.0, 1.0
