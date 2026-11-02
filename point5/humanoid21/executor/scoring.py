"""E4: interpretable per-channel rollout scoring + pose-based init.

Each channel returns a residual with a dead-zone tolerance; the total is
a weighted sum, plus a hard fall penalty.  Weights live only in
``ExecutorConfig``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from point5.contracts import POINT_INDEX
from .contracts import ExecutorConfig, RefFrame
from .state import Current, fall_check

FOOT = (POINT_INDEX["foot_right"], POINT_INDEX["foot_left"])


def score_landed(cur_like: Dict[str, Any], ref: RefFrame,
                 p5_vel: np.ndarray, action: np.ndarray,
                 prev_action: np.ndarray,
                 cfg: ExecutorConfig,
                 pose_target_norm: Optional[np.ndarray] = None,
                 joint_pos_norm: Optional[np.ndarray] = None
                 ) -> Dict[str, Any]:
    """Score one measured landing against a RefFrame (lower better).

    ``cur_like``: dict with p5_world, com_pos, com_vel, yaw (rad),
    torso_tilt, contacts.  ``p5_vel`` is the measured world point
    velocity (finite difference is fine).
    """
    pw = np.asarray(cur_like["p5_world"])
    c: Dict[str, float] = {}
    perr = np.linalg.norm(pw - ref.p5_world, axis=1)
    c["point"] = float(np.maximum(0, perr - ref.tol_point).sum()
                       + 0.25 * perr.sum())
    if p5_vel is not None:
        verr = np.linalg.norm(p5_vel - ref.p5_vel_world, axis=1)
        c["point_vel"] = float(verr.sum() / 5.0)
    c["com"] = float(max(0.0, np.linalg.norm(
        np.asarray(cur_like["com_pos"]) - ref.com_pos) - cfg.tol_com))
    c["com_vel"] = float(np.linalg.norm(
        np.asarray(cur_like["com_vel"]) - ref.com_vel))
    dy = (float(cur_like["yaw"]) - ref.yaw + np.pi) % (2 * np.pi) - np.pi
    c["yaw"] = float(max(0.0, abs(dy) - cfg.tol_yaw))
    slip = 0.0
    for i, planted in ((FOOT[0], ref.support[0]),
                       (FOOT[1], ref.support[1])):
        if planted and p5_vel is not None:
            slip += max(0.0, float(np.linalg.norm(p5_vel[i]))
                        - cfg.slip_v)
    c["slip"] = slip
    c["contact"] = float(cur_like["contacts"].get("extra", 0))
    if not (ref.support[0] and ref.support[1]):
        anchor = FOOT[1] if not ref.support[1] else FOOT[0]
        d = np.linalg.norm(np.asarray(cur_like["com_pos"])[:2]
                           - ref.p5_world[anchor, :2])
        c["com_support"] = float(max(0.0, d - cfg.support_tol))
    else:
        c["com_support"] = 0.0
    c["action"] = float(np.mean(np.square(action)))
    c["delta"] = float(np.mean(np.square(action - prev_action)))
    if pose_target_norm is not None and joint_pos_norm is not None:
        c["pose"] = float(np.abs(np.asarray(joint_pos_norm)
                                 - np.asarray(pose_target_norm)).mean())
    else:
        c["pose"] = 0.0
    total = (cfg.w_point * c["point"]
             + cfg.w_point_vel * c["point_vel"]
             + cfg.w_com * c["com"] + cfg.w_com_vel * c["com_vel"] * 0.1
             + cfg.w_yaw * c["yaw"] + cfg.w_slip * c["slip"]
             + cfg.w_extra_contact * c["contact"]
             + cfg.w_com_support * c["com_support"]
             + cfg.w_action * c["action"]
             + cfg.w_action_delta * c["delta"]
             + cfg.w_pose_pref * c["pose"])
    fallen = (float(cur_like["torso_tilt"]) > cfg.tilt_fail
              or float(pw[POINT_INDEX["head"], 2]) < cfg.head_min_z)
    if fallen:
        total += cfg.fall_penalty
    return {"total": float(total), "components": c, "fallen": fallen}


def pose_init(local_p5: np.ndarray, iface, skeleton=None):
    """Optional action init from solve_equilibrium, or None.

    Returns the normalized PD target whose joint goal equals the solved
    posture; ``None`` when the solve fails (caller falls back to prev
    action).
    """
    from point5.humanoid21.mechanics import Skeleton, solve_equilibrium
    from .sim_iface import joint_to_action
    sk = skeleton or Skeleton()
    try:
        res = solve_equilibrium(np.asarray(local_p5, np.float64),
                                skeleton=sk, tolerance=0.03)
    except Exception:
        return None
    if not res.get("satisfied"):
        return None
    return np.clip(joint_to_action(np.asarray(res["q_rad"]), iface),
                   -1.0, 1.0)
