"""Retarget CMU MoCap to humanoid21 21-DOF joint angles.

Hybrid approach:
  - Spine + legs: compute relative rotations from FK, decompose onto humanoid21 joint axes
  - Arms: compute hand world positions from FK, use analytical IK for 3-DOF arm

humanoid21 controlled joints (21):
  abdomen_z, abdomen_y, abdomen_x,
  hip_x_right, hip_z_right, hip_y_right, knee_right, ankle_y_right, ankle_x_right,
  hip_x_left, hip_z_left, hip_y_left, knee_left, ankle_y_left, ankle_x_left,
  shoulder1_right, shoulder2_right, elbow_right,
  shoulder1_left, shoulder2_left, elbow_left

humanoid21 joint axes (from battle_v1.xml):
  abdomen_z:  axis [0,0,1],  range [-45, 45]
  abdomen_y:  axis [0,1,0],  range [-75, 30]
  abdomen_x:  axis [1,0,0],  range [-35, 35]
  hip_x_right: axis [1,0,0],  range [-30, 10]
  hip_z_right: axis [0,0,1],  range [-60, 35]
  hip_y_right: axis [0,1,0],  range [-150, 20]
  knee_right:  axis [0,-1,0], range [-160, 2]
  ankle_y_right: axis [0,1,0], range [-50, 50]
  ankle_x_right: axis [1,0,0.5], range [-50, 50]  (approx, normalized)
  (left side mirrors axis signs)
  shoulder1_right: axis [2,1,1], range [-85, 60]
  shoulder2_right: axis [0,-1,1], range [-85, 60]
  elbow_right: axis [0,-1,1], range [-100, 50]
  (left side mirrors)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from baseline.humanoid21.mocap.asf_parser import Skeleton
from baseline.humanoid21.mocap.fk import compute_fk


# humanoid21 joint specs: (name, axis, range_min, range_max)
# axis is in parent body's local frame, normalized
JOINT_SPECS = {
    # Spine (abdomen_z, abdomen_y, abdomen_x)
    "abdomen_z":    (np.array([0, 0, 1], dtype=np.float64), -45, 45),
    "abdomen_y":    (np.array([0, 1, 0], dtype=np.float64), -75, 30),
    "abdomen_x":    (np.array([1, 0, 0], dtype=np.float64), -35, 35),
    # Right leg
    "hip_x_right":  (np.array([1, 0, 0], dtype=np.float64), -30, 10),
    "hip_z_right":  (np.array([0, 0, 1], dtype=np.float64), -60, 35),
    "hip_y_right":  (np.array([0, 1, 0], dtype=np.float64), -150, 20),
    "knee_right":   (np.array([0, -1, 0], dtype=np.float64), -160, 2),
    "ankle_y_right":(np.array([0, 1, 0], dtype=np.float64), -50, 50),
    "ankle_x_right":(np.array([1, 0, 0.5], dtype=np.float64) / np.linalg.norm([1, 0, 0.5]), -50, 50),
    # Left leg (mirrored axes)
    "hip_x_left":   (np.array([-1, 0, 0], dtype=np.float64), -30, 10),
    "hip_z_left":   (np.array([0, 0, -1], dtype=np.float64), -60, 35),
    "hip_y_left":   (np.array([0, 1, 0], dtype=np.float64), -150, 20),
    "knee_left":    (np.array([0, -1, 0], dtype=np.float64), -160, 2),
    "ankle_y_left": (np.array([0, 1, 0], dtype=np.float64), -50, 50),
    "ankle_x_left": (np.array([-1, 0, -0.5], dtype=np.float64) / np.linalg.norm([1, 0, 0.5]), -50, 50),
    # Right arm
    "shoulder1_right": (np.array([2, 1, 1], dtype=np.float64) / np.linalg.norm([2, 1, 1]), -85, 60),
    "shoulder2_right": (np.array([0, -1, 1], dtype=np.float64) / np.linalg.norm([0, -1, 1]), -85, 60),
    "elbow_right":     (np.array([0, -1, 1], dtype=np.float64) / np.linalg.norm([0, -1, 1]), -100, 50),
    # Left arm
    "shoulder1_left":  (np.array([-2, 1, -1], dtype=np.float64) / np.linalg.norm([2, 1, 1]), -85, 60),
    "shoulder2_left":  (np.array([0, -1, -1], dtype=np.float64) / np.linalg.norm([0, -1, 1]), -85, 60),
    "elbow_left":      (np.array([0, -1, -1], dtype=np.float64) / np.linalg.norm([0, -1, 1]), -100, 50),
}

JOINT_ORDER = [
    "abdomen_z", "abdomen_y", "abdomen_x",
    "hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
    "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
    "shoulder1_right", "shoulder2_right", "elbow_right",
    "shoulder1_left", "shoulder2_left", "elbow_left",
]


def _clip_angle(angle: float, joint_name: str) -> float:
    """Clip angle to joint range."""
    _, lo, hi = JOINT_SPECS[joint_name]
    return float(np.clip(angle, lo, hi))


def _rotation_angle_around_axis(R: np.ndarray, axis: np.ndarray) -> float:
    """Extract rotation angle around a specific axis from a rotation matrix.

    Uses: θ = atan2(axis · (R @ axis_perp1), axis · axis_perp2)
    where axis_perp1, axis_perp2 are orthonormal vectors perpendicular to axis.
    """
    axis = axis / np.linalg.norm(axis)
    # Build orthonormal basis
    if abs(axis[2]) < 0.9:
        perp1 = np.cross(axis, [0, 0, 1])
    else:
        perp1 = np.cross(axis, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    # Project R onto the plane perpendicular to axis
    rotated_perp1 = R @ perp1
    # Angle in that plane
    cos_theta = np.dot(rotated_perp1, perp1)
    sin_theta = np.dot(rotated_perp1, perp2)
    return float(np.degrees(np.arctan2(sin_theta, cos_theta)))


def _decompose_3dof(
    R_rel: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    axis3: np.ndarray,
) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into 3 angles around given axes (in order).

    Uses sequential Euler decomposition: R = R1(a1) @ R2(a2) @ R3(a3)
    where R1 is rotation around axis1, etc.

    For simplicity, if axes are close to standard XYZ, use scipy Euler.
    Otherwise, use iterative projection.
    """
    # Try standard Euler if axes are close to X, Y, Z
    ax1 = axis1 / np.linalg.norm(axis1)
    ax2 = axis2 / np.linalg.norm(axis2)
    ax3 = axis3 / np.linalg.norm(axis3)

    # Check if axes are approximately standard basis
    std = np.eye(3)
    if (np.allclose(ax1, std[0]) and np.allclose(ax2, std[1]) and np.allclose(ax3, std[2])):
        angles = Rotation.from_matrix(R_rel).as_euler("xyz", degrees=True)
        return float(angles[0]), float(angles[1]), float(angles[2])

    # General case: extract angles one by one
    # Angle around axis1: project the rotation of axis2 onto the axis2-axis3 plane
    a1 = _rotation_angle_around_axis(R_rel, ax1)
    R1 = Rotation.from_rotvec(np.radians(a1) * ax1).as_matrix()
    R_remaining = R1.T @ R_rel

    a2 = _rotation_angle_around_axis(R_remaining, ax2)
    R2 = Rotation.from_rotvec(np.radians(a2) * ax2).as_matrix()
    R_remaining2 = R2.T @ R_remaining

    a3 = _rotation_angle_around_axis(R_remaining2, ax3)

    return a1, a2, a3


def _retarget_spine(
    cmu_transforms: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float, float]:
    """Retarget CMU spine (lowerback + upperback + thorax) to humanoid21 abdomen joints.

    CMU spine: root → lowerback → upperback → thorax (each 3-DOF)
    humanoid21: torso → waist_lower(abdomen_z, abdomen_y) → pelvis(abdomen_x)

    We compute the total relative rotation from root to thorax, then decompose
    into humanoid21's abdomen_z, abdomen_y, abdomen_x.
    """
    R_root = cmu_transforms["root"][0]
    R_thorax = cmu_transforms["thorax"][0]

    # Total relative rotation
    R_rel = R_root.T @ R_thorax

    # Decompose into abdomen_z (Z), abdomen_y (Y), abdomen_x (X)
    # humanoid21 order: abdomen_z → abdomen_y → abdomen_x (parent to child)
    a_z, a_y, a_x = _decompose_3dof(
        R_rel,
        JOINT_SPECS["abdomen_z"][0],
        JOINT_SPECS["abdomen_y"][0],
        JOINT_SPECS["abdomen_x"][0],
    )

    return _clip_angle(a_z, "abdomen_z"), _clip_angle(a_y, "abdomen_y"), _clip_angle(a_x, "abdomen_x")


def _retarget_leg(
    cmu_transforms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    side: str,  # 'right' or 'left'
) -> Tuple[float, float, float, float, float]:
    """Retarget CMU leg to humanoid21 leg joints.

    CMU: root → hipjoint(no DOF) → femur(rx,ry,rz) → tibia(rx) → foot(rx,rz)
    humanoid21: pelvis → thigh(hip_x, hip_z, hip_y) → shin(knee) → foot(ankle_y, ankle_x)

    Returns: (hip_x, hip_z, hip_y, knee, ankle_y, ankle_x)
    """
    s = side[0].lower()  # 'r' or 'l'
    hip_name = f"{s}femur"
    tibia_name = f"{s}tibia"
    foot_name = f"{s}foot"

    R_root = cmu_transforms["root"][0]
    R_hip = cmu_transforms[hip_name][0]
    R_tibia = cmu_transforms[tibia_name][0]
    R_foot = cmu_transforms[foot_name][0]

    # Hip: relative rotation from root to femur
    R_hip_rel = R_root.T @ R_hip

    hip_x_name = f"hip_x_{side}"
    hip_z_name = f"hip_z_{side}"
    hip_y_name = f"hip_y_{side}"

    a_hx, a_hz, a_hy = _decompose_3dof(
        R_hip_rel,
        JOINT_SPECS[hip_x_name][0],
        JOINT_SPECS[hip_z_name][0],
        JOINT_SPECS[hip_y_name][0],
    )

    # Knee: relative rotation from femur to tibia
    R_knee_rel = R_hip.T @ R_tibia
    knee_name = f"knee_{side}"
    a_knee = _rotation_angle_around_axis(R_knee_rel, JOINT_SPECS[knee_name][0])

    # Ankle: relative rotation from tibia to foot
    R_ankle_rel = R_tibia.T @ R_foot
    ankle_y_name = f"ankle_y_{side}"
    ankle_x_name = f"ankle_x_{side}"
    a_ay = _rotation_angle_around_axis(R_ankle_rel, JOINT_SPECS[ankle_y_name][0])
    # Remove the Y rotation component before extracting X
    R_ay = Rotation.from_rotvec(np.radians(a_ay) * JOINT_SPECS[ankle_y_name][0]).as_matrix()
    R_ankle_remaining = R_ay.T @ R_ankle_rel
    a_ax = _rotation_angle_around_axis(R_ankle_remaining, JOINT_SPECS[ankle_x_name][0])

    return (
        _clip_angle(a_hx, hip_x_name),
        _clip_angle(a_hz, hip_z_name),
        _clip_angle(a_hy, hip_y_name),
        _clip_angle(a_knee, knee_name),
        _clip_angle(a_ay, ankle_y_name),
        _clip_angle(a_ax, ankle_x_name),
    )


def _retarget_arm(
    cmu_transforms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    side: str,  # 'right' or 'left'
) -> Tuple[float, float, float]:
    """Retarget CMU arm to humanoid21 arm joints using rotation decomposition.

    CMU arm: thorax → clavicle(ry,rz) → humerus(rx,ry,rz) → radius(rx)
    humanoid21: torso → upper_arm(shoulder1, shoulder2) → lower_arm(elbow)

    Strategy:
    1. Shoulder: relative rotation from thorax to humerus → decompose onto shoulder1, shoulder2 axes
    2. Elbow: relative rotation from humerus to radius → extract angle around elbow axis

    Returns: (shoulder1, shoulder2, elbow) in degrees
    """
    s = side[0].lower()
    humerus_name = f"{s}humerus"
    radius_name = f"{s}radius"

    R_torso = cmu_transforms["thorax"][0]
    R_humerus = cmu_transforms[humerus_name][0]
    R_radius = cmu_transforms[radius_name][0]

    # Shoulder: relative rotation from thorax to humerus
    R_shoulder_rel = R_torso.T @ R_humerus

    s1_name = f"shoulder1_{side}"
    s2_name = f"shoulder2_{side}"
    elbow_name = f"elbow_{side}"

    # Decompose 2-DOF shoulder: extract angles around shoulder1 and shoulder2 axes
    a_s1 = _rotation_angle_around_axis(R_shoulder_rel, JOINT_SPECS[s1_name][0])
    R_s1 = Rotation.from_rotvec(np.radians(a_s1) * JOINT_SPECS[s1_name][0]).as_matrix()
    R_remaining = R_s1.T @ R_shoulder_rel
    a_s2 = _rotation_angle_around_axis(R_remaining, JOINT_SPECS[s2_name][0])

    # Elbow: relative rotation from humerus to radius
    R_elbow_rel = R_humerus.T @ R_radius
    a_elbow = _rotation_angle_around_axis(R_elbow_rel, JOINT_SPECS[elbow_name][0])

    return _clip_angle(a_s1, s1_name), _clip_angle(a_s2, s2_name), _clip_angle(a_elbow, elbow_name)


def retarget_frame(
    skel: Skeleton,
    frame: Dict[str, np.ndarray],
) -> np.ndarray:
    """Retarget a single CMU frame to humanoid21 21-DOF joint angles.

    Args:
        skel: Parsed CMU skeleton
        frame: AMC frame data

    Returns:
        np.ndarray of shape (21,) — joint angles in degrees, ordered as JOINT_ORDER
    """
    transforms = compute_fk(skel, frame)

    # Spine
    ab_z, ab_y, ab_x = _retarget_spine(transforms)

    # Right leg
    r_hx, r_hz, r_hy, r_knee, r_ay, r_ax = _retarget_leg(transforms, "right")

    # Left leg
    l_hx, l_hz, l_hy, l_knee, l_ay, l_ax = _retarget_leg(transforms, "left")

    # Arms (rotation-based)
    r_s1, r_s2, r_elbow = _retarget_arm(transforms, "right")
    l_s1, l_s2, l_elbow = _retarget_arm(transforms, "left")

    angles = np.array([
        ab_z, ab_y, ab_x,
        r_hx, r_hz, r_hy, r_knee, r_ay, r_ax,
        l_hx, l_hz, l_hy, l_knee, l_ay, l_ax,
        r_s1, r_s2, r_elbow,
        l_s1, l_s2, l_elbow,
    ], dtype=np.float32)

    return angles


def retarget_motion(
    skel: Skeleton,
    frames: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Retarget all frames to humanoid21 joint angles.

    Returns:
        (T, 21) array of joint angles in degrees
    """
    T = len(frames)
    motion = np.zeros((T, 21), dtype=np.float32)

    for t in range(T):
        motion[t] = retarget_frame(skel, frames[t])

    return motion


def angles_to_normalized_action(
    joint_angles_deg: np.ndarray,
    joint_limits: np.ndarray,  # (21, 2) [min, max] in degrees
) -> np.ndarray:
    """Convert joint angles (degrees) to normalized action [-1, 1].

    normalized = (angle - mid) / half_range
    """
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2.0
    half = (joint_limits[:, 1] - joint_limits[:, 0]) / 2.0
    half = np.where(half < 1e-6, 1.0, half)  # avoid div by zero
    return (joint_angles_deg - mid) / half
