"""Retarget CMU MoCap to humanoid21 21-DOF joint angles.

Direct DOF mapping approach:
  The AMC file contains joint angles in CMU's local bone frame.
  We map these directly to humanoid21 joints with sign corrections,
  based on matching physical motion semantics (flexion, abduction, rotation).

CMU DOF semantics:
  - femur: rx=flexion, ry=abduction, rz=rotation
  - tibia: rx=knee flexion
  - foot:  rx=ankle flexion, rz=ankle rotation
  - humerus: rx=flexion, ry=abduction, rz=rotation
  - radius: rx=elbow flexion
  - spine (lowerback/upperback/thorax): rx=flexion, ry=lateral, rz=rotation

humanoid21 joint semantics (from axis + range):
  - hip_y [-150,20]: main flexion (large range) ← CMU femur rx
  - hip_z [-60,35]:  abduction              ← CMU femur ry
  - hip_x [-30,10]:  lateral/rotation       ← CMU femur rz
  - knee [-160,2]:   flexion (negative)     ← -CMU tibia rx
  - ankle_y [-50,50]: flexion               ← -CMU foot rx
  - ankle_x [-50,50]: rotation              ← CMU foot rz
  - shoulder1 [-85,60]: flexion             ← CMU humerus rx
  - shoulder2 [-85,60]: abduction           ← CMU humerus ry
  - elbow [-100,50]: flexion (negative)     ← -CMU radius rx
  - abdomen_z [-45,45]: rotation            ← sum CMU spine rz
  - abdomen_y [-75,30]: flexion             ← sum CMU spine ry
  - abdomen_x [-35,35]: lateral             ← sum CMU spine rx
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


JOINT_ORDER = [
    "abdomen_z", "abdomen_y", "abdomen_x",
    "hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
    "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
    "shoulder1_right", "shoulder2_right", "elbow_right",
    "shoulder1_left", "shoulder2_left", "elbow_left",
]

# humanoid21 joint limits in degrees (from battle_v1.xml)
JOINT_LIMITS = np.array([
    [-45, 45],    # abdomen_z
    [-75, 30],    # abdomen_y
    [-35, 35],    # abdomen_x
    [-30, 10],    # hip_x_right
    [-60, 35],    # hip_z_right
    [-150, 20],   # hip_y_right
    [-160, 2],    # knee_right
    [-50, 50],    # ankle_y_right
    [-50, 50],    # ankle_x_right
    [-30, 10],    # hip_x_left
    [-60, 35],    # hip_z_left
    [-150, 20],   # hip_y_left
    [-160, 2],    # knee_left
    [-50, 50],    # ankle_y_left
    [-50, 50],    # ankle_x_left
    [-85, 60],    # shoulder1_right
    [-85, 60],    # shoulder2_right
    [-100, 50],   # elbow_right
    [-85, 60],    # shoulder1_left
    [-85, 60],    # shoulder2_left
    [-100, 50],   # elbow_left
], dtype=np.float32)


def _clip(angle: float, lo: float, hi: float) -> float:
    return float(np.clip(angle, lo, hi))


def retarget_frame(
    frame: Dict[str, np.ndarray],
) -> np.ndarray:
    """Retarget a single CMU AMC frame to humanoid21 21-DOF joint angles.

    Args:
        frame: AMC frame data (bone_name → DOF values in degrees)

    Returns:
        np.ndarray of shape (21,) — joint angles in degrees, ordered as JOINT_ORDER
    """
    # --- Spine: sum lowerback + upperback + thorax ---
    lb = frame.get("lowerback", np.zeros(3))
    ub = frame.get("upperback", np.zeros(3))
    th = frame.get("thorax", np.zeros(3))
    spine_rx = lb[0] + ub[0] + th[0]  # flexion
    spine_ry = lb[1] + ub[1] + th[1]  # lateral
    spine_rz = lb[2] + ub[2] + th[2]  # rotation

    abdomen_z = _clip(spine_rz, -45, 45)
    abdomen_y = _clip(spine_ry, -75, 30)
    abdomen_x = _clip(spine_rx, -35, 35)

    # --- Right leg ---
    rfem = frame.get("rfemur", np.zeros(3))  # [rx, ry, rz]
    rtib = frame.get("rtibia", np.zeros(1))  # [rx]
    rft = frame.get("rfoot", np.zeros(2))    # [rx, rz]

    hip_y_r = _clip(rfem[0], -150, 20)       # flexion
    hip_z_r = _clip(rfem[1], -60, 35)        # abduction
    hip_x_r = _clip(rfem[2], -30, 10)        # rotation
    knee_r = _clip(-rtib[0], -160, 2)        # flexion (sign flip)
    ankle_y_r = _clip(-rft[0], -50, 50)      # flexion (sign flip)
    ankle_x_r = _clip(rft[1], -50, 50)       # rotation

    # --- Left leg (mirror signs for x and z) ---
    lfem = frame.get("lfemur", np.zeros(3))
    ltib = frame.get("ltibia", np.zeros(1))
    lft = frame.get("lfoot", np.zeros(2))

    hip_y_l = _clip(lfem[0], -150, 20)       # flexion (same sign)
    hip_z_l = _clip(-lfem[1], -60, 35)       # abduction (mirror)
    hip_x_l = _clip(-lfem[2], -30, 10)       # rotation (mirror)
    knee_l = _clip(-ltib[0], -160, 2)        # flexion (sign flip)
    ankle_y_l = _clip(-lft[0], -50, 50)      # flexion (sign flip)
    ankle_x_l = _clip(-lft[1], -50, 50)      # rotation (mirror)

    # --- Right arm ---
    rhum = frame.get("rhumerus", np.zeros(3))  # [rx, ry, rz]
    rrad = frame.get("rradius", np.zeros(1))   # [rx]

    shoulder1_r = _clip(rhum[0], -85, 60)     # flexion
    shoulder2_r = _clip(-rhum[1], -85, 60)    # abduction (sign flip for humanoid21 convention)
    elbow_r = _clip(-rrad[0], -100, 50)       # flexion (sign flip)

    # --- Left arm (mirror) ---
    lhum = frame.get("lhumerus", np.zeros(3))
    lrad = frame.get("lradius", np.zeros(1))

    shoulder1_l = _clip(-lhum[0], -85, 60)    # flexion (mirror)
    shoulder2_l = _clip(lhum[1], -85, 60)     # abduction (mirror)
    elbow_l = _clip(-lrad[0], -100, 50)       # flexion (sign flip)

    angles = np.array([
        abdomen_z, abdomen_y, abdomen_x,
        hip_x_r, hip_z_r, hip_y_r, knee_r, ankle_y_r, ankle_x_r,
        hip_x_l, hip_z_l, hip_y_l, knee_l, ankle_y_l, ankle_x_l,
        shoulder1_r, shoulder2_r, elbow_r,
        shoulder1_l, shoulder2_l, elbow_l,
    ], dtype=np.float32)

    return angles


def retarget_motion(
    frames: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Retarget all frames to humanoid21 joint angles.

    Returns:
        (T, 21) array of joint angles in degrees
    """
    T = len(frames)
    motion = np.zeros((T, 21), dtype=np.float32)
    for t in range(T):
        motion[t] = retarget_frame(frames[t])
    return motion


def angles_to_normalized_action(joint_angles_deg: np.ndarray) -> np.ndarray:
    """Convert joint angles (degrees) to normalized action [-1, 1].

    normalized = (angle - mid) / half_range
    """
    mid = (JOINT_LIMITS[:, 0] + JOINT_LIMITS[:, 1]) / 2.0
    half = (JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]) / 2.0
    half = np.where(half < 1e-6, 1.0, half)
    return (joint_angles_deg - mid) / half
