"""Pure mechanical structure of the Humanoid21 skeleton.

A weightless rigid-body tree + hinge joints + 5 landmark points.
NO MuJoCo dependency at runtime: all quantities below are extracted
constants; FK is implemented directly on the tree with numpy.

Conventions (identical to MuJoCo body kinematics):
  body_xform = parent_xform
               * T(body.pos) * R(body.quat)
               * Π_j  T(anchor_j) * R(axis_j, q_j) * T(-anchor_j)

Joint values are raw radians in JOINT_ORDER. The root is a free joint:
(root_pos, quat_wxyz) defines the torso frame.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------- bodies ---
# name -> (parent_name | None, pos_offset_in_parent_frame)
# body quat is identity for every link in this model.
BODIES: List[Tuple[str, str, Tuple[float, float, float]]] = [
    ("torso",            None,          (0.0, 0.0, 0.0)),
    ("head",             "torso",       (0.0, 0.0, 0.19)),
    ("waist_lower",      "torso",       (-0.01, 0.0, -0.26)),
    ("pelvis",           "waist_lower", (0.0, 0.0, -0.165)),
    ("thigh_right",      "pelvis",      (0.0, -0.1, -0.04)),
    ("shin_right",       "thigh_right", (0.0, 0.01, -0.4)),
    ("foot_right",       "shin_right",  (0.0, 0.0, -0.39)),
    ("thigh_left",       "pelvis",      (0.0, 0.1, -0.04)),
    ("shin_left",        "thigh_left",  (0.0, -0.01, -0.4)),
    ("foot_left",        "shin_left",   (0.0, 0.0, -0.39)),
    ("upper_arm_right",  "torso",       (0.0, -0.17, 0.06)),
    ("lower_arm_right",  "upper_arm_right", (0.18, -0.18, -0.18)),
    ("hand_right",       "lower_arm_right", (0.18, 0.18, 0.18)),
    ("upper_arm_left",   "torso",       (0.0, 0.17, 0.06)),
    ("lower_arm_left",   "upper_arm_left",  (0.18, 0.18, -0.18)),
    ("hand_left",        "lower_arm_left",  (0.18, -0.18, 0.18)),
]

# ---------------------------------------------------------------- joints ---
# joint name -> (body_name, anchor, axis, (range_lo, range_hi))
# hinge rotation applies about the anchor, axes in the body frame.
JOINTS: List[Tuple[str, str, Tuple, Tuple, Tuple[float, float]]] = [
    ("abdomen_z",       "waist_lower", (0, 0, 0.065), (0, 0, 1), (-0.7854, 0.7854)),
    ("abdomen_y",       "waist_lower", (0, 0, 0.065), (0, 1, 0), (-1.3090, 0.5236)),
    ("abdomen_x",       "pelvis",      (0, 0, 0.1),   (1, 0, 0), (-0.6109, 0.6109)),
    ("hip_x_right",     "thigh_right", (0, 0, 0),     (1, 0, 0), (-0.5236, 0.1745)),
    ("hip_z_right",     "thigh_right", (0, 0, 0),     (0, 0, 1), (-1.0472, 0.6109)),
    ("hip_y_right",     "thigh_right", (0, 0, 0),     (0, 1, 0), (-2.6180, 0.3491)),
    ("knee_right",      "shin_right",  (0, 0, 0.02),  (0, -1, 0), (-2.7925, 0.0349)),
    ("ankle_y_right",   "foot_right",  (0, 0, 0.08),  (0, 1, 0), (-0.8727, 0.8727)),
    ("ankle_x_right",   "foot_right",  (0, 0, 0.04),  (0.8944, 0, 0.4472), (-0.8727, 0.8727)),
    ("hip_x_left",      "thigh_left",  (0, 0, 0),     (-1, 0, 0), (-0.5236, 0.1745)),
    ("hip_z_left",      "thigh_left",  (0, 0, 0),     (0, 0, -1), (-1.0472, 0.6109)),
    ("hip_y_left",      "thigh_left",  (0, 0, 0),     (0, 1, 0), (-2.6180, 0.3491)),
    ("knee_left",       "shin_left",   (0, 0, 0.02),  (0, -1, 0), (-2.7925, 0.0349)),
    ("ankle_y_left",    "foot_left",   (0, 0, 0.08),  (0, 1, 0), (-0.8727, 0.8727)),
    ("ankle_x_left",    "foot_left",   (0, 0, 0.04),  (-0.8944, 0, -0.4472), (-0.8727, 0.8727)),
    ("shoulder1_right", "upper_arm_right", (0, 0, 0), (0.8165, 0.4082, 0.4082), (-1.4835, 1.0472)),
    ("shoulder2_right", "upper_arm_right", (0, 0, 0), (0, -0.7071, 0.7071), (-1.4835, 1.0472)),
    ("elbow_right",     "lower_arm_right", (0, 0, 0), (0, -0.7071, 0.7071), (-1.7453, 0.8727)),
    ("shoulder1_left",  "upper_arm_left",  (0, 0, 0), (-0.8165, 0.4082, -0.4082), (-1.4835, 1.0472)),
    ("shoulder2_left",  "upper_arm_left",  (0, 0, 0), (0, -0.7071, -0.7071), (-1.4835, 1.0472)),
    ("elbow_left",      "lower_arm_left",  (0, 0, 0), (0, -0.7071, -0.7071), (-1.7453, 0.8727)),
]

JOINT_NAMES: Tuple[str, ...] = tuple(j[0] for j in JOINTS)
JNT_RANGE: np.ndarray = np.array([j[4] for j in JOINTS], dtype=np.float64)

# -------------------------------------------------------------- landmarks --
# point_name -> (body_name, local_offset). Foot = ankle center (midpoint of
# the ankle_y / ankle_x joint anchors: local z .08 / .04 -> 0.06).
LANDMARKS: Dict[str, Tuple[str, np.ndarray]] = {
    "head":       ("head", np.zeros(3)),
    "hand_right": ("hand_right", np.zeros(3)),
    "hand_left":  ("hand_left", np.zeros(3)),
    "foot_right": ("foot_right", np.array([0.0, 0.0, 0.06])),
    "foot_left":  ("foot_left", np.array([0.0, 0.0, 0.06])),
}
LANDMARK_ORDER: Tuple[str, ...] = (
    "head", "hand_right", "hand_left", "foot_right", "foot_left")


# ------------------------------------------------------------- math bits ---
def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _quat_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)]])


# ------------------------------------------------------------------- FK ---
class Skeleton:
    """The mechanical model: joint values + root pose -> body/landmark poses."""

    def __init__(self) -> None:
        self.joint_index = {n: i for i, n in enumerate(JOINT_NAMES)}
        self.joints_on: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] \
            = {b[0]: [] for b in BODIES}
        for name, body, anchor, axis, rng in JOINTS:
            self.joints_on[body].append(
                (self.joint_index[name],
                 np.asarray(anchor, dtype=np.float64),
                 np.asarray(axis, dtype=np.float64)))
        self.parent = {b[0]: b[1] for b in BODIES}
        self.bpos = {b[0]: np.asarray(b[2], dtype=np.float64) for b in BODIES}
        self.order = [b[0] for b in BODIES]      # already parent-first

    def body_transforms(self, root_pos, root_quat_wxyz,
                        q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return {body: (xpos, xmat)} in world frame."""
        q = np.asarray(q, dtype=np.float64)
        xf = {"__root__": (np.asarray(root_pos, dtype=np.float64),
                           _quat_mat(root_quat_wxyz))}
        out = {}
        for name in self.order:
            parent = self.parent[name] or "__root__"
            ppos, pmat = xf[parent]
            pos = ppos + pmat @ self.bpos[name]
            mat = pmat.copy()          # body quat is identity here
            for jidx, anchor, axis in self.joints_on[name]:
                R = _rot(axis, q[jidx])
                # rotate about anchor: x -> a + R(x - a)
                pos = pos + mat @ (anchor - R @ anchor)
                mat = mat @ R
            xf[name] = (pos, mat)
            out[name] = (pos, mat)
        return out

    def landmarks(self, root_pos, root_quat_wxyz, q) -> np.ndarray:
        """(5,3) landmark world positions in LANDMARK_ORDER."""
        xf = self.body_transforms(root_pos, root_quat_wxyz, q)
        return np.stack([
            xf[b][0] + xf[b][1] @ off
            for b, off in (LANDMARKS[n] for n in LANDMARK_ORDER)])
