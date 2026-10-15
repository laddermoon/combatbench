"""Forward kinematics for CMU ASF/AMC skeleton.

Computes world-space positions and orientations for each bone per frame.

CMU ASF/AMC convention:
  - Root: [TX, TY, TZ, RX, RY, RZ] in world frame (degrees)
  - Each bone: DOF values (rx, ry, rz in degrees) applied in bone's local frame
  - Bone local transform = C_axis @ R_dof @ C_axis^T @ T_along_direction
    where C_axis is the axis rotation matrix, R_dof is from DOF values,
    and T_along_direction is translation by direction * length * units_length

Reference: https://mocap.cs.cmu.edu/info.php
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from baseline.humanoid21.mocap.asf_parser import Bone, Skeleton


def _dof_to_rotation(dof: List[str], values: np.ndarray) -> np.ndarray:
    """Convert DOF values to a rotation matrix.

    DOF can be any subset of ['rx', 'ry', 'rz'] in any order.
    We build the rotation as individual axis rotations composed.
    """
    if len(dof) == 0 or np.allclose(values, 0):
        return np.eye(3)

    # Build rotation from Euler angles
    # CMU uses the axis order specified in dof
    euler_angles = []
    euler_axes = []
    for i, d in enumerate(dof):
        if d == "rx":
            euler_angles.append(values[i])
            euler_axes.append("x")
        elif d == "ry":
            euler_angles.append(values[i])
            euler_axes.append("y")
        elif d == "rz":
            euler_angles.append(values[i])
            euler_axes.append("z")

    if not euler_angles:
        return np.eye(3)

    # Compose rotations: first DOF is outermost
    R = np.eye(3)
    for angle, axis in zip(euler_angles, euler_axes):
        r = Rotation.from_euler(axis, float(angle), degrees=True)
        R = r.as_matrix() @ R

    return R


def compute_bone_local_transform(
    bone: Bone, dof_values: np.ndarray, units_length: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local transform (rotation, translation) for a bone.

    Returns:
        (R_local, t_local) where:
        - R_local = C_axis @ R_dof @ C_axis^T
        - t_local = C_axis @ (direction * length * units_length)
    """
    C = bone.axis_mat  # (3,3) axis rotation
    R_dof = _dof_to_rotation(bone.dof, dof_values)

    # Bone rotation: C @ R_dof @ C^T
    R_local = C @ R_dof @ C.T

    # Translation along bone direction (in axis-rotated frame)
    t_local = C @ (bone.direction * bone.length * units_length)

    return R_local, t_local


def compute_fk(
    skel: Skeleton, frame: Dict[str, np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute world transforms for all bones in a frame.

    Args:
        skel: Parsed skeleton
        frame: AMC frame data (bone_name → DOF values)

    Returns:
        Dict mapping bone_name → (R_world, p_world) where:
        - R_world: (3,3) world rotation matrix
        - p_world: (3,) world position
    """
    transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Root transform
    root_vals = frame["root"]  # [TX, TY, TZ, RX, RY, RZ]
    root_pos = root_vals[:3] * skel.units_length  # scale position
    root_rot = Rotation.from_euler("xyz", root_vals[3:6], degrees=True).as_matrix()
    transforms["root"] = (root_rot, root_pos)

    # Traverse hierarchy
    def _traverse(bone_name: str, parent_R: np.ndarray, parent_p: np.ndarray):
        bone = skel.bones[bone_name]
        if bone.has_dof and bone_name in frame:
            dof_values = frame[bone_name]
        else:
            dof_values = np.zeros(len(bone.dof))

        R_local, t_local = compute_bone_local_transform(bone, dof_values, skel.units_length)

        # World transform
        R_world = parent_R @ R_local
        p_world = parent_p + parent_R @ t_local

        transforms[bone_name] = (R_world, p_world)

        for child in bone.children:
            _traverse(child, R_world, p_world)

    for child in skel.bones["root"].children:
        _traverse(child, root_rot, root_pos)

    return transforms


def compute_fk_batch(
    skel: Skeleton, frames: List[Dict[str, np.ndarray]]
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Compute FK for all frames."""
    return [compute_fk(skel, frame) for frame in frames]


def get_joint_positions(
    skel: Skeleton, frames: List[Dict[str, np.ndarray]]
) -> np.ndarray:
    """Get world positions for all bones across all frames.

    Returns:
        (T, N, 3) array where T=frames, N=bones, 3=xyz
    """
    all_transforms = compute_fk_batch(skel, frames)
    bone_names = sorted(skel.bones.keys())
    T = len(all_transforms)
    N = len(bone_names)
    positions = np.zeros((T, N, 3), dtype=np.float32)

    for t, transforms in enumerate(all_transforms):
        for n, name in enumerate(bone_names):
            if name in transforms:
                positions[t, n] = transforms[name][1]

    return positions, bone_names
