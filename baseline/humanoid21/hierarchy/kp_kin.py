"""Keypoint FK + two-phase target construction for strike_kp_2phase.

Single definition of the 15 keypoints shared by live states and
reference clips (STRIKE_KEYPOINT_2PHASE_PLAN.md §2-4):

  body origins : head, torso, pelvis, hand_l/r, foot_l/r,
                 shoulder (= upper_arm origin), elbow (= lower_arm
                 origin), hip (= thigh origin)   — joints defined in
                 those bodies all have jnt_pos = 0, so the joint anchor
                 IS the body origin.
  joint anchors: knee_l/r = shin origin + R_shin @ jnt_pos (0,0,0.02)
  foot sole    : foot origin z - FOOT_SOLE_R (capsule radius 0.027) —
                 used only for clip frame-0 ground alignment.

All positions come from the analytic ``fk_positions`` (verified against
mj_forward at 1e-15), so live states (normalized joint positions from
core state) and clip frames (q_rad) are evaluated by the SAME math —
no live/derived-state keypoint plumbing is needed.

Targets are FROZEN once per phase:

  T1 (ready): clip frame-0 keypoints, z-shifted so the lowest foot
      sole touches ground, then yaw-rotated/translated so the ref
      frame-0 root maps onto the robot's xy/yaw at strike onset.
  T2 (landing): clip landing-frame keypoints under the SAME anchor
      transform, then rigidly 3D-translated so the power-foot keypoint
      sits exactly at the robot's actual power-foot position on the
      phase-1 success frame.  Pinning the target point to the observed
      foot IS the rear-foot constraint — no separate foot channel.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from mocap.humanoid21.robot_kin import (
    BODY_TREE, CONTROLLED_JOINTS, _JNT_POS, fk_positions, joint_limits)

# ---------------------------------------------------------------------
# Keypoint definitions
# ---------------------------------------------------------------------
KP_NAMES: Tuple[str, ...] = (
    "head", "torso", "pelvis",
    "shoulder_left", "shoulder_right",
    "elbow_left", "elbow_right",
    "hand_left", "hand_right",
    "hip_left", "hip_right",
    "knee_left", "knee_right",
    "foot_left", "foot_right",
)
N_KP = len(KP_NAMES)
KP_IDX = {n: i for i, n in enumerate(KP_NAMES)}

# semantic keypoint -> body whose origin is the point
_KP_BODY = {
    "head": "head", "torso": "torso", "pelvis": "pelvis",
    "shoulder_left": "upper_arm_left", "shoulder_right": "upper_arm_right",
    "elbow_left": "lower_arm_left", "elbow_right": "lower_arm_right",
    "hand_left": "hand_left", "hand_right": "hand_right",
    "hip_left": "thigh_left", "hip_right": "thigh_right",
    "foot_left": "foot_left", "foot_right": "foot_right",
}
# semantic keypoint -> joint whose anchor is the point
_KP_JNT = {"knee_left": "knee_left", "knee_right": "knee_right"}
_JNT_BODY = {jn: b for b, (_, joints) in BODY_TREE.items() for jn in joints}

FOOT_SOLE_R = 0.027          # foot capsule radius (m), from the XML

# Per-point success tolerance (plan §5): end-effectors (hands/feet)
# 3 cm, everything else 5 cm.  All 15 points must pass simultaneously.
TOL_M: Dict[str, float] = {
    n: (0.03 if n.startswith(("hand", "foot")) else 0.05)
    for n in KP_NAMES}

# Uniform weights for the dense reward (plan §5, first version).
W_KP = np.ones(N_KP, dtype=np.float64)

# clip_id -> power/rear foot keypoint (rule, not geometry): heavy uses
# the punching-side foot, jab the opposite foot.  Mirrors
# strike_track_observer.strike_power_foot.
POWER_FOOT = {
    "hdm05_3146_88_2895_127": "foot_right",     # jab_l
    "hdm05_3146_88_2895_127_m": "foot_left",    # jab_r
    "cmu_144_20_seg01": "foot_right",           # heavy_r
    "cmu_144_20_m_seg01": "foot_left",          # heavy_l
}

_MID = np.array([0.5 * (joint_limits(n)[1] + joint_limits(n)[0])
                 for n in CONTROLLED_JOINTS])
_HALF = np.array([0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
                  for n in CONTROLLED_JOINTS])


def _rz(dyaw: float) -> np.ndarray:
    c, s = np.cos(dyaw), np.sin(dyaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def yaw_of(quat_wxyz: np.ndarray) -> float:
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    # z-yaw of the wxyz quaternion
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


# ---------------------------------------------------------------------
# FK -> 15 keypoints (world frame)
# ---------------------------------------------------------------------

def fk_keypoints(q: Dict[str, float], root_pos, root_quat_wxyz
                 ) -> np.ndarray:
    """(15,3) world keypoints in KP_NAMES order for one pose."""
    pos, rot = fk_positions(q, np.asarray(root_pos, dtype=np.float64),
                            np.asarray(root_quat_wxyz, dtype=np.float64))
    pts = np.empty((N_KP, 3), dtype=np.float64)
    for i, n in enumerate(KP_NAMES):
        if n in _KP_BODY:
            pts[i] = pos[_KP_BODY[n]]
        else:
            jn = _KP_JNT[n]
            b = _JNT_BODY[jn]
            pts[i] = pos[b] + rot[b] @ _JNT_POS[jn]
    return pts


def live_keypoints(core: Dict[str, np.ndarray]) -> np.ndarray:
    """15 keypoints of a live robot from its core-state dict."""
    q_rad = (np.asarray(core["joint_pos_norm"], dtype=np.float64)
             * _HALF + _MID)
    q = {n: float(v) for n, v in zip(CONTROLLED_JOINTS, q_rad)}
    return fk_keypoints(q, core["root_pos"], core["root_rot"])


def ref_keypoints(ref, k: int) -> np.ndarray:
    """15 keypoints of clip frame k."""
    q = {n: float(v) for n, v in zip(CONTROLLED_JOINTS, ref.q_rad[k])}
    return fk_keypoints(q, ref.root_pos[k], ref.root_quat[k])


# ---------------------------------------------------------------------
# Target construction (frozen per phase)
# ---------------------------------------------------------------------

def ground_align(pts: np.ndarray) -> np.ndarray:
    """z-shift so the lowest foot sole (origin - FOOT_SOLE_R) sits at z=0."""
    out = pts.copy()
    foot_z = out[[KP_IDX["foot_left"], KP_IDX["foot_right"]], 2].min()
    out[:, 2] -= (foot_z - FOOT_SOLE_R)
    return out


def anchored(pts_ref: np.ndarray, ref_root_xy0: np.ndarray,
             ref_yaw0: float, robot_xy: np.ndarray,
             robot_yaw: float) -> np.ndarray:
    """Map ref keypoints into the robot's world frame: the ref frame-0
    root xy/yaw maps onto the robot's xy/yaw (same convention as
    StrikeTrackObserver's ego anchor)."""
    M = _rz(robot_yaw - ref_yaw0)
    out = pts_ref.copy()
    out[:, :2] = (M @ (pts_ref[:, :2]
                       - np.asarray(ref_root_xy0, dtype=np.float64)).T).T \
        + np.asarray(robot_xy, dtype=np.float64)
    return out


def build_p2_target(land_pts_ref: np.ndarray, ref_root_xy0: np.ndarray,
                    ref_yaw0: float, robot_xy: np.ndarray,
                    robot_yaw: float, rear_idx: int,
                    rear_actual: np.ndarray) -> np.ndarray:
    """Landing target anchored like T1, then rigidly 3D-translated so the
    power-foot keypoint equals the robot's actual power-foot position on
    the phase-1 success frame."""
    t = anchored(land_pts_ref, ref_root_xy0, ref_yaw0,
                 robot_xy, robot_yaw)
    return t + (np.asarray(rear_actual, dtype=np.float64) - t[rear_idx])


def kp_errors(live: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-point Euclidean errors (15,)."""
    return np.linalg.norm(target - live, axis=1)


def kp_loss(errs: np.ndarray) -> float:
    """Dense reward potential: weighted mean squared distance."""
    return float(np.sum(W_KP * errs * errs) / np.sum(W_KP))


def kp_pass(errs: np.ndarray) -> bool:
    """Success verdict: every point within its tolerance."""
    return bool(np.all(errs <= np.array(
        [TOL_M[n] for n in KP_NAMES], dtype=np.float64)))
