"""Joint spring model for Humanoid21 (the robot's pose prior).

Defines, per controlled joint:
  - ``q0``        natural angle (the "unloaded" rest position, rad)
  - ``k``         stiffness weight resisting deviation from q0
  - ``fixed``     whether the joint is excluded from optimization
  - ``scale``     normalization length (half joint range) for the energy term

Elastic energy:  E(q) = 1/2 * sum_j k_j * ((q_j - q0_j) / scale_j)^2

This is a *pose preference* model, not a physical PD/stiffness model.
Config lives in pose_config.yaml; this class turns it into vectors.
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
    joint_names: tuple          # suffix-stripped CONTROLLED_JOINTS order
    q0: np.ndarray              # (J,) natural angles
    k: np.ndarray               # (J,) stiffness weights
    scale: np.ndarray           # (J,) normalization lengths
    fixed_mask: np.ndarray      # (J,) bool, True = excluded from optimization
    ranges: np.ndarray          # (J,2) joint limits

    @property
    def active_idx(self) -> np.ndarray:
        return np.nonzero(~self.fixed_mask)[0]

    def energy(self, q: np.ndarray) -> float:
        d = (np.asarray(q, dtype=np.float64) - self.q0) / self.scale
        return 0.5 * float(np.sum(self.k * d * d))

    def energy_residual(self, q: np.ndarray) -> np.ndarray:
        """Per-joint sqrt-energy residual for least-squares solvers."""
        return np.sqrt(self.k) * (np.asarray(q, dtype=np.float64)
                                  - self.q0) / self.scale


def load_spring_model(fk: Optional[Humanoid21FK] = None,
                      config_path: Optional[Path] = None,
                      robot_id: str = "robot_a") -> SpringModel:
    """Build the spring model from pose_config.yaml + joint ranges."""
    fk = fk or Humanoid21FK()
    cfg = yaml.safe_load(Path(config_path or CONFIG_PATH).read_text())
    names = tuple(n.rsplit("_", 1)[0]
                  for n in fk.controlled_joints(robot_id))
    q0 = np.array([cfg["natural_pose"][n] for n in names])
    fixed = set(cfg.get("fixed_joints", []))
    k = np.array([_k_of(cfg, n) for n in names], dtype=np.float64)
    ranges = np.asarray(fk.joint_ranges(robot_id), dtype=np.float64)
    scale = np.maximum((ranges[:, 1] - ranges[:, 0]) / 2.0, 1e-3)
    return SpringModel(
        joint_names=names, q0=q0, k=k, scale=scale,
        fixed_mask=np.array([n in fixed for n in names]),
        ranges=ranges)


def _k_of(cfg, name: str) -> float:
    for group in cfg.get("stiffness_groups", {}).values():
        if name in group["joints"]:
            return float(group["k"])
    return 1.0
