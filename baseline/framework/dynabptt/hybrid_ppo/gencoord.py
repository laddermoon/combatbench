"""GenCoord state representation (ported from HybridRobotics/supermimic).

Splits the world-model state into a frame-invariant learned part (rep)
and an analytically integrated world part (aux):

    cs  (55): [root_pos(3), quat wxyz(4), v_local(3), w_local(3),
               jp_norm(21), jv_norm(21)]
    rep (52): [z(1), g_b(3), v_b(3), w_b(3), qj(21), vj(21)]
    aux (7):  [root_pos(3), root_quat wxyz(4)]
    pred(27): [dv_b(3), dw_b(3), dvj(21)]

The network only predicts velocity-level deltas; every position-level
quantity advances by exact kinematic integration, so model error stays
inside the bounded velocity channels (see SUPERMIMIC_BOUNDARY.md §1.1).

cs velocities are already body-frame (get_core_state stores
root_vel_local / root_angular_vel_local), so represent() only needs the
gravity-direction transform.

Control step: DT_PHYS 0.002s x 25 substeps = dt 0.05s.  Pose updates
weight the delta by substep_coeff = (dec+1)/(2*dec) = 0.52 to match how
the simulator composes its substeps.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

N_JOINTS = 21
CS_DIM = 55
REP_DIM = 1 + 3 + 3 + 3 + N_JOINTS + N_JOINTS   # 52
AUX_DIM = 7
PRED_DIM = 6 + N_JOINTS                        # 27

DT = 0.05
SUBSTEP_COEFF = (25 + 1) / (2 * 25)            # 0.52

# cs slices
CS_POS = slice(0, 3)
CS_QUAT = slice(3, 7)
CS_V = slice(7, 10)
CS_W = slice(10, 13)
CS_QJ = slice(13, 13 + N_JOINTS)
CS_VJ = slice(13 + N_JOINTS, 13 + 2 * N_JOINTS)

# rep slices
R_Z = slice(0, 1)
R_GB = slice(1, 4)
R_VB = slice(4, 7)
R_WB = slice(7, 10)
R_QJ = slice(10, 10 + N_JOINTS)
R_VJ = slice(10 + N_JOINTS, 10 + 2 * N_JOINTS)


# --- quaternion helpers (wxyz) ---------------------------------------------

def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (world result) by quaternion q (wxyz)."""
    qw, qx, qy, qz = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v + qw*t + cross(q_vec, t)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the conjugate of q."""
    qc = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return quat_apply(qc, v)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_integrate(q: torch.Tensor, w_world: torch.Tensor,
                   dt: float) -> torch.Tensor:
    """Integrate orientation by world-frame angular velocity (wxyz)."""
    omega = torch.cat(
        [torch.zeros_like(w_world[..., :1]), w_world], dim=-1)
    q_next = q + 0.5 * dt * quat_mul(omega, q)
    return F.normalize(q_next, p=2, dim=-1)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(wxyz) -> 3x3 rotation matrix (last two dims)."""
    w, x, y, z = q.unbind(-1)
    n = F.normalize(q, p=2, dim=-1)
    w, x, y, z = n.unbind(-1)
    m = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


# --- representation ---------------------------------------------------------

def represent(cs: torch.Tensor) -> torch.Tensor:
    """55-dim cs -> 52-dim frame-invariant rep."""
    quat = cs[..., CS_QUAT]
    down = torch.zeros_like(cs[..., CS_POS])
    down[..., 2] = -1.0
    g_b = quat_apply_inverse(quat, down)
    return torch.cat([
        cs[..., 2:3],            # z
        g_b,                     # gravity dir in body frame
        cs[..., CS_V],           # v_b (already body frame)
        cs[..., CS_W],           # w_b (already body frame)
        cs[..., CS_QJ],
        cs[..., CS_VJ],
    ], dim=-1)


def extract_aux(cs: torch.Tensor) -> torch.Tensor:
    return cs[..., :AUX_DIM]


def apply_prediction(aux: torch.Tensor, rep: torch.Tensor,
                     pred: torch.Tensor, dt: float = DT,
                     c: float = SUBSTEP_COEFF,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit integration of accel-level deltas (official port).

    Deltas are expressed in the CURRENT root frame; joint positions
    integrate joint velocities, the root pose integrates the twist, and
    z/g_b are re-derived from the integrated pose.
    """
    root_pos = aux[..., :3]
    root_quat = aux[..., 3:7]
    v_b = rep[..., R_VB]
    w_b = rep[..., R_WB]
    qj = rep[..., R_QJ]
    vj = rep[..., R_VJ]
    dv_b = pred[..., 0:3]
    dw_b = pred[..., 3:6]
    dvj = pred[..., 6:6 + N_JOINTS]

    next_v_cur = v_b + dv_b          # next twist in current root frame
    next_w_cur = w_b + dw_b
    next_vj = vj + dvj
    next_qj = qj + (vj + c * dvj) * dt

    v_global = quat_apply(root_quat, next_v_cur)
    w_global = quat_apply(root_quat, next_w_cur)
    v_pose = quat_apply(root_quat, v_b + c * dv_b)
    w_pose = quat_apply(root_quat, w_b + c * dw_b)
    next_root_pos = root_pos + v_pose * dt
    next_root_quat = quat_integrate(root_quat, w_pose, dt)

    next_v_b = quat_apply_inverse(next_root_quat, v_global)
    next_w_b = quat_apply_inverse(next_root_quat, w_global)
    world_down = torch.zeros_like(v_global)
    world_down[..., 2] = -1.0
    next_g_b = quat_apply_inverse(next_root_quat, world_down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
        [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b,
         next_qj, next_vj], dim=-1)
    return next_aux, next_rep


def reconstruct_cs(aux: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
    """(aux, rep) -> 55-dim cs (root_pos.z overridden by rep's z)."""
    root_pos = torch.cat(
        [aux[..., :2], rep[..., R_Z]], dim=-1)
    return torch.cat([
        root_pos, aux[..., 3:7],
        rep[..., R_VB], rep[..., R_WB],
        rep[..., R_QJ], rep[..., R_VJ],
    ], dim=-1)


def loss_blocks(rep: torch.Tensor, aux: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
    """Squared-error channel groups (Bryson weighting applied outside).

    Mirrors the official GenCoord loss_blocks: only the integrated root
    pose and joint pos/vel are scored; velocities are supervised
    implicitly through the integration chain.
    """
    rot = matrix_from_quat(aux[..., 3:7]).flatten(-2, -1)
    return {
        "root_pos": aux[..., :3],
        "root_ori": rot * (0.5 ** 0.5),
        "joint_pos": rep[..., R_QJ],
        "joint_vel": rep[..., R_VJ],
    }


# Bryson's rule nominal errors (1/e^2 weighting).  Units are the rep's:
# joint_pos/joint_vel are norm-space (jp_norm, jv_norm), not raw radians.
NOMINAL_ERRORS = {
    "root_pos": 0.01,
    "root_ori": 0.02,
    "joint_pos": 0.1,
    "joint_vel": 1.0,
}
