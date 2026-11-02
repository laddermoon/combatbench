"""Balance reflex — big-logic split: trajectory joints follow the IK
guide, balance joints close the loop on measured COM error.

Open-loop PD hold is unstable on this robot (verified: every solved
keyframe pose falls in ~1.2 s), so the support leg's ankle/hip get a
local COM-error correction added to their PD targets.  The correction
is in NORMALIZED action space; gains are per the config.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from point5.contracts import POINT_INDEX

# action-space joint indices (same order as actuators/JOINT_NAMES)
ANKLE_Y = {"right": 7, "left": 13}
ANKLE_X = {"right": 8, "left": 14}
HIP_X = {"right": 3, "left": 9}
HIP_Y = {"right": 5, "left": 11}
ABDOMEN_X = 2      # side bend


def balance_action(base: np.ndarray, com_pos: np.ndarray,
                   com_vel: np.ndarray, com_ref: np.ndarray,
                   com_vel_ref: np.ndarray, frame_origin: np.ndarray,
                   yaw_deg: float, support: np.ndarray,
                   gains: Optional[dict] = None) -> np.ndarray:
    """Return base action plus COM-error correction on balance joints.

    Correction drives the COM toward ``com_ref`` in the P5 local frame:
      e = R * (com_ref - com_pos), v = R * (com_vel_ref - com_vel).
    Sign convention is empirical (ankle strategy verified on sim).
    """
    g = {"ax": 1.2, "dx": 0.25, "ay": 1.2, "dy": 0.25,
         "hx": 0.6, "abx": 0.3}
    if gains:
        g.update(gains)
    a = np.asarray(base, dtype=np.float64).copy()
    r = np.deg2rad(yaw_deg)
    R = np.array([[np.cos(r), np.sin(r)],
                  [-np.sin(r), np.cos(r)]])
    e = R @ (np.asarray(com_ref)[:2] - np.asarray(com_pos)[:2])
    v = R @ (np.asarray(com_vel_ref)[:2] - np.asarray(com_vel)[:2])
    corr_x = g["ax"] * e[0] + g["dx"] * v[0]   # sagittal (fwd)
    corr_y = g["ay"] * e[1] + g["dy"] * v[1]   # lateral
    legs = []
    if support[0]:
        legs.append("right")
    if support[1]:
        legs.append("left")
    for side in legs:
        # ankle strategy: pitch pushes body along local x; roll along y
        a[ANKLE_Y[side]] += -corr_x if side == "right" else -corr_x
        a[ANKLE_X[side]] += corr_y if side == "right" else -corr_y
        # hip assists (smaller share)
        a[HIP_Y[side]] += g["hx"] * 0.5 * corr_x * (
            1 if side == "right" else -1)
        a[HIP_X[side]] += g["hx"] * 0.5 * corr_y * (
            -1 if side == "right" else 1)
    a[ABDOMEN_X] += g["abx"] * corr_y          # torso counter-lean
    return np.clip(a, -1.0, 1.0)
