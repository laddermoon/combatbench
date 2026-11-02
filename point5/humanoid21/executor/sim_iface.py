"""Interface constants derived from the simulator — no magic numbers.

Validates that the executor's timing/normalization assumptions match
``Humanoid21Simulator`` at import/use time (fail loud if they drift).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SimInterface:
    dt_control: float                # seconds per action (20 Hz)
    dt_physics: float
    steps_per_action: int
    action_dim: int
    joint_reference: np.ndarray      # (21,) rad — norm = (q-ref)/scale
    joint_scale: np.ndarray          # (21,)
    standing_action: np.ndarray      # (21,) PD target for spawn pose


def load_interface(sim=None) -> SimInterface:
    """Pull constants from a live simulator (or build one-off sim)."""
    own = sim is None
    if own:
        from envs.humanoid21.simulator import Humanoid21Simulator
        sim = Humanoid21Simulator(initial_distance=6.0)
        sim.reset(seed=0)
    try:
        np_ = sim._norm_params["robot_a"]
        from envs.humanoid21.meta import Humanoid21Meta
        stand = np.asarray(
            Humanoid21Meta.INITIAL_POSES["standing"]["action"],
            dtype=np.float64)
        # consistency: reset computes action=(joint_pos-ref)/scale
        joint0 = np.asarray(
            Humanoid21Meta.INITIAL_POSES["standing"]["joint_pos"],
            dtype=np.float64)
        recomputed = (joint0 - np_["reference"]) / np_["scale"]
        assert np.allclose(stand, recomputed, atol=1e-3), (
            "standing action != norm(standing joint_pos) — "
            "action semantics changed")
        return SimInterface(
            dt_control=0.05, dt_physics=float(sim.DT),
            steps_per_action=int(round(0.05 / sim.DT)),
            action_dim=int(sim.ACTION_DIM),
            joint_reference=np.asarray(np_["reference"], np.float64),
            joint_scale=np.asarray(np_["scale"], np.float64),
            standing_action=stand)
    finally:
        if own:
            sim.close()


def action_to_joint(action: np.ndarray, iface: SimInterface) -> np.ndarray:
    """Normalized PD target -> joint target (rad)."""
    return np.asarray(action) * iface.joint_scale + iface.joint_reference


def joint_to_action(joint_rad: np.ndarray, iface: SimInterface) -> np.ndarray:
    """Joint position (rad) -> normalized PD target."""
    return (np.asarray(joint_rad) - iface.joint_reference) \
        / iface.joint_scale
