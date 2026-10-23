"""Decode the policy-visible Humanoid21 observation into a self-state.

The 96-dimensional observation intentionally omits absolute yaw.  For the
single-robot dynamics used by this baseline that is a symmetry: we choose a
canonical yaw gauge and reconstruct a physically equivalent root pose.  This
module never reads the live simulator's privileged state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.humanoid21.meta import Humanoid21Meta


OBS_DIM = 96
ACTION_DIM = Humanoid21Meta.ACTION_DIM


@dataclass(frozen=True)
class DecodedSelfObservation:
    """Policy-visible quantities needed by the standing controller."""

    core_state: Dict[str, np.ndarray]
    projected_gravity: np.ndarray
    height: float
    linear_vel_local: np.ndarray
    angular_vel_local: np.ndarray
    feet_forces: np.ndarray
    arena_center_local: np.ndarray


def decode_self_observation(observation: np.ndarray) -> DecodedSelfObservation:
    """Decode the ego part of the flat 96-D observation.

    The returned ``root_pos`` / ``root_rot`` live in a canonical yaw gauge.
    They are suitable for reconstructing single-robot dynamics in the circular
    arena, but should not be interpreted as the original absolute world yaw.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (OBS_DIM,):
        raise ValueError(f"expected Humanoid21 observation shape (96,), got {obs.shape}")

    joint_pos_norm = obs[0:21].copy()
    joint_vel_norm = np.sign(obs[21:42]) * (2.0 * obs[21:42]) ** 2
    projected_gravity = obs[42:45].copy()
    height = float(obs[45])
    linear_vel_local = obs[46:49].copy()
    angular_vel_local = np.sign(obs[49:52]) * 2.0 * obs[49:52] ** 2
    feet_forces = obs[52:54].copy()
    arena_center_local = obs[54:57].copy()

    # Choose a canonical yaw gauge: body +Z is fixed by projected gravity, and
    # body +X is chosen as the tilt-minimal continuation of world +Y × body +Z.
    # Any other global yaw gives the same observation and equivalent dynamics.
    body_z_world = -projected_gravity
    body_z_world /= np.linalg.norm(body_z_world)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(reference @ body_z_world)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    body_x_world = np.cross(reference, body_z_world)
    body_x_world /= np.linalg.norm(body_x_world)
    body_y_world = np.cross(body_z_world, body_x_world)
    rot = np.column_stack((body_x_world, body_y_world, body_z_world))

    # arena_center_local = R^T * (-root_pos), so root_pos = -R @ center.
    root_pos = -(rot @ arena_center_local)
    root_pos[2] = height
    quat_xyzw = R.from_matrix(rot).as_quat()
    root_rot = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )

    return DecodedSelfObservation(
        core_state={
            "root_pos": root_pos.astype(np.float32),
            "root_rot": root_rot,
            "root_vel_local": linear_vel_local.astype(np.float32),
            "root_angular_vel_local": angular_vel_local.astype(np.float32),
            "joint_pos_norm": joint_pos_norm.astype(np.float32),
            "joint_vel_norm": joint_vel_norm.astype(np.float32),
        },
        projected_gravity=projected_gravity,
        height=height,
        linear_vel_local=linear_vel_local,
        angular_vel_local=angular_vel_local,
        feet_forces=feet_forces,
        arena_center_local=arena_center_local,
    )


def standing_action() -> np.ndarray:
    """Return the normalized PD target for the canonical standing pose."""
    return Humanoid21Meta.INITIAL_POSES["standing"]["action"].astype(np.float32).copy()
