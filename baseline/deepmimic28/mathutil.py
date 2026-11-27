"""快速 numpy 四元数工具（xyzw 约定，与 scipy 一致）。

全部为 MimicKit torch_util 的 numpy 移植，语义严格对齐：
- quat_to_tan_norm: 旋转的 6D 编码（旋转后 x 轴 + z 轴）
- quat_diff_angle: 两旋转夹角（弧度）
- 其余为组合/插值基础算子
"""
from __future__ import annotations

import numpy as np


def quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """xyzw 四元数乘法 q0⊗q1。支持 (...,4) 广播。"""
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    return np.stack([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用 q 旋转向量 v。q (...,4) xyzw，v (...,3)。"""
    xyz = q[..., :3]
    w = q[..., 3:4]
    # t = 2 * xyz × v
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_diff_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """两四元数夹角（弧度，非负）。MimicKit quat_diff_angle 语义：
    angle(q0 * conj(q1)) 取旋转量。"""
    qd = quat_mul(q0, quat_conj(q1))
    w = np.clip(np.abs(qd[..., 3]), 0.0, 1.0)  # 双覆盖取短弧
    return 2.0 * np.arccos(w)


def quat_to_tan_norm(q: np.ndarray) -> np.ndarray:
    """旋转 → 6D 编码 [rotated_x_axis(3), rotated_z_axis(3)]。
    MimicKit: ref_axes x=(1,0,0), z=(0,0,1) 经 q 旋转。"""
    qexp = q[..., None, :]  # (...,1,4)
    x_ax = np.zeros(q.shape[:-1] + (3,))
    x_ax = x_ax.copy()
    x_ax[..., 0] = 1.0
    z_ax = np.zeros(q.shape[:-1] + (3,))
    z_ax = z_ax.copy()
    z_ax[..., 2] = 1.0
    rx = quat_rotate(q, x_ax)
    rz = quat_rotate(q, z_ax)
    return np.concatenate([rx, rz], axis=-1)  # (...,6)


def nlerp_quat(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """归一化线性插值（上游用的就是 nlerp 而非严格 slerp）。"""
    # 同半球
    if np.sum(q0 * q1) < 0.0:
        q1 = -q1
    out = (1.0 - a) * q0 + a * q1
    return out / np.linalg.norm(out)


def compose_group_quat(angles: np.ndarray, kind: str) -> np.ndarray:
    """关节组 hinge 角度 → 组净旋转 quat (xyzw)。

    angles: spherical 组为 (x,y,z) 三角；hinge 关节为单角。
    kind: 'sph' 或 ('hinge', axis_idx)。约定 q = qx⊗qy⊗qz（已验证）。
    """
    if kind == 'sph':
        ax, ay, az = angles
        qx = np.array([np.sin(ax / 2), 0.0, 0.0, np.cos(ax / 2)])
        qy = np.array([0.0, np.sin(ay / 2), 0.0, np.cos(ay / 2)])
        qz = np.array([0.0, 0.0, np.sin(az / 2), np.cos(az / 2)])
        return quat_mul(quat_mul(qx, qy), qz)
    else:
        axis_idx = kind[1]
        q = np.zeros(4)
        q[axis_idx] = np.sin(angles / 2.0)
        q[3] = np.cos(angles / 2.0)
        return q
