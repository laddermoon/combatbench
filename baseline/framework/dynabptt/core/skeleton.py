"""Differentiable skeleton + balance metrics, from WMState.

Ports point5/humanoid21/mechanics.py Skeleton FK to batched torch —
hinge joints about anchors, body quats identity, free root:

  body_xform = parent_xform * T(body.pos)
               * prod_j T(anchor_j) R(axis_j,q_j) T(-anchor_j)

Everything is matmul/cross-product → fully differentiable through
(rep, aux).  Balance metric = capture-point margin to a soft
support-polygon approximation (support_state.py constants).
"""
from __future__ import annotations

import torch

from ..hybrid_ppo.gencoord import (R_QJ, R_VB, matrix_from_quat,
                                   quat_apply)

# ------------------------------------------------------- constants (numpy->lazy torch)
# (name, body, anchor, axis, (lo,hi)) — mechanics.py JOINTS
JOINTS = [
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
# (name, parent, pos_offset)
BODIES = [
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
    ("hand_left",        "lower_arm_left",  (0.18, 0.18, 0.18)),
]
BODY_MASS = {
    "torso":           (5.853834, (-0.003864, 0.0, -0.046369)),
    "head":            (3.053628, (0.0, 0.0, 0.0)),
    "waist_lower":     (2.261947, (0.0, 0.0, 0.0)),
    "pelvis":          (6.616194, (-0.02, 0.0, 0.0)),
    "thigh_right":     (4.751751, (0.0, 0.005, -0.17)),
    "shin_right":      (2.755696, (0.0, 0.0, -0.15)),
    "foot_right":      (1.131141, (0.035, 0.0, 0.0)),
    "thigh_left":      (4.751751, (0.0, -0.005, -0.17)),
    "shin_left":       (2.755696, (0.0, 0.0, -0.15)),
    "foot_left":       (1.131141, (0.035, 0.0, 0.0)),
    "upper_arm_right": (1.661080, (0.08, -0.08, -0.08)),
    "lower_arm_right": (0.961458, (0.09, 0.09, 0.09)),
    "hand_right":      (0.268083, (0.0, 0.0, 0.0)),
    "upper_arm_left":  (1.661080, (0.08, 0.08, -0.08)),
    "lower_arm_left":  (0.961458, (0.09, 0.09, 0.09)),
    "hand_left":       (0.268083, (0.0, 0.0, 0.0)),
}
# foot = ankle center = foot body pos + mat @ (0,0,0.06) — landmark convention
FOOT_LOCAL = (0.0, 0.0, 0.06)

# support_state.py constants
ANKLE_Z_PLANTED = 0.107
LIP_MARGIN_TOE = 0.167
LIP_MARGIN_HEEL = 0.097
LIP_MARGIN_SIDE = 0.047
GRAV = 9.81

_RANGE = torch.tensor([j[4] for j in JOINTS], dtype=torch.float32)
_MID = (_RANGE[:, 0] + _RANGE[:, 1]) / 2.0
_HALF = (_RANGE[:, 1] - _RANGE[:, 0]) / 2.0
_JOINT_INDEX = {j[0]: i for i, j in enumerate(JOINTS)}
_JOINTS_ON = {}
for jname, body, anchor, axis, rng in JOINTS:
    _JOINTS_ON.setdefault(body, []).append(
        (_JOINT_INDEX[jname],
         torch.tensor(anchor, dtype=torch.float32),
         torch.tensor(axis, dtype=torch.float32)))
_BPOS = {b[0]: torch.tensor(b[2], dtype=torch.float32) for b in BODIES}
_PARENT = {b[0]: b[1] for b in BODIES}
_ORDER = [b[0] for b in BODIES]
_BMASS = {b: (m, torch.tensor(ip, dtype=torch.float32))
          for b, (m, ip) in BODY_MASS.items()}
_MASS_TOT = sum(m for m, _ in BODY_MASS.values())
_FOOT_LOCAL_T = torch.tensor(FOOT_LOCAL, dtype=torch.float32)


def qn_to_rad(qn: torch.Tensor) -> torch.Tensor:
    return qn * _HALF.to(qn) + _MID.to(qn)


def _rodrigues(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix R(axis, q).  axis (B,3), q (B,) -> (B,3,3)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(q), torch.sin(q)
    C = 1.0 - c
    R = torch.stack([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return R.reshape(*q.shape, 3, 3)


def body_xf(root_pos: torch.Tensor, root_quat: torch.Tensor,
            q_rad: torch.Tensor):
    """(B,3)+(B,4)+(B,21) -> {name: (pos(B,3), mat(B,3,3))} world."""
    dev = q_rad.device
    xf = {"__root__": (root_pos,
                       matrix_from_quat(root_quat).to(dev))}
    out = {}
    for name in _ORDER:
        parent = _PARENT[name] or "__root__"
        ppos, pmat = xf[parent]
        pos = ppos + torch.einsum("bij,j->bi", pmat, _BPOS[name].to(dev))
        mat = pmat
        for jidx, anchor, axis in _JOINTS_ON.get(name, ()):
            R = _rodrigues(axis.to(dev).expand_as(pos), q_rad[:, jidx])
            # pos += mat @ (anchor - R @ anchor)
            a = anchor.to(dev)
            pos = pos + torch.einsum(
                "bij,bj->bi", mat,
                a.unsqueeze(0) - torch.einsum("bij,j->bi", R, a))
            mat = mat @ R
        xf[name] = (pos, mat)
        out[name] = (pos, mat)
    return out


def com_and_feet(aux: torch.Tensor, rep: torch.Tensor):
    """WMState -> (com(B,3), foot_l(B,3), foot_r(B,3))."""
    root_pos, root_quat = aux[..., :3], aux[..., 3:7]
    q_rad = qn_to_rad(rep[..., R_QJ])
    xf = body_xf(root_pos, root_quat, q_rad)
    com = torch.zeros_like(root_pos)
    for name in _ORDER:
        pos, mat = xf[name]
        m, ipos = _BMASS[name]
        xip = pos + torch.einsum("bij,j->bi", mat, ipos.to(rep.device))
        com = com + m * xip
    com = com / _MASS_TOT
    def foot(side):
        pos, mat = xf[f"foot_{side}"]
        return pos + torch.einsum(
            "bij,j->bi", mat, _FOOT_LOCAL_T.to(rep.device))
    return com, foot("left"), foot("right")


def _margin(com, fl, fr, yaw, com_vel_xy, plant_tau: float = 0.02):
    """Core margin from precomputed quantities (single FK shared)."""
    z = com[..., 2].clamp_min(0.05)
    omega = (GRAV / z).sqrt()
    cp = com[..., :2] + com_vel_xy / omega.unsqueeze(-1)

    hl = fl[..., 2] - ANKLE_Z_PLANTED
    hr = fr[..., 2] - ANKLE_Z_PLANTED
    wl = torch.sigmoid(-hl / plant_tau)
    wr = torch.sigmoid(-hr / plant_tau)
    wsum = wl + wr
    center = (wl.unsqueeze(-1) * fl[..., :2]
              + wr.unsqueeze(-1) * fr[..., :2]
              ) / wsum.clamp_min(1e-4).unsqueeze(-1)
    span = (fl[..., :2] - fr[..., :2]).norm(dim=-1)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rel = cp - center
    cp_fwd = rel[..., 0] * cy + rel[..., 1] * sy
    cp_lat = -rel[..., 0] * sy + rel[..., 1] * cy

    ext_lat = LIP_MARGIN_SIDE + (wl * wr) * span / 2.0
    m_lat = ext_lat - cp_lat.abs()
    m_fwd = torch.minimum(LIP_MARGIN_TOE - cp_fwd,
                          LIP_MARGIN_HEEL + cp_fwd)
    margin = torch.minimum(m_lat, m_fwd)
    return margin, cp, {"wl": wl, "wr": wr, "cp_fwd": cp_fwd,
                        "cp_lat": cp_lat, "ext_lat": ext_lat,
                        "omega": omega}


def torso_yaw(aux: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = (aux[..., 3], aux[..., 4],
                    aux[..., 5], aux[..., 6])
    return torch.atan2(2 * (qw * qz + qx * qy),
                       1 - 2 * (qy * qy + qz * qz))


def balance_step(aux: torch.Tensor, rep: torch.Tensor,
                 prev_com=None, dt: float = 0.0):
    """One imagined step's balance quantities (single FK).

    com_vel proxy = world-frame root linear velocity (rep.v_b).
    A finite-difference com velocity amplifies WM rep error into
    phantom velocity (~cm per step / dt), so the root velocity —
    a directly-predicted, smooth rep channel — is used instead.
    Returns (margin(B,), cp(B,2), com(B,3), diag)."""
    com, fl, fr = com_and_feet(aux, rep)
    vel_xy = root_linvel_world(aux, rep)[..., :2]
    margin, cp, diag = _margin(com, fl, fr, torso_yaw(aux), vel_xy)
    return margin, cp, com, diag


def root_linvel_world(aux: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
    """rep.v_b (body frame) -> world-frame linear velocity (B,3)."""
    return quat_apply(aux[..., 3:7], rep[..., R_VB])
