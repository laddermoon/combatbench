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

# ------------------------------------------------------------- body mass ---
# name -> (mass kg, local COM offset). battle_circular_v2.xml 单台机器人
# 导出（*_a 与 *_b 同质量）；整树总质量 40.844 kg。
BODY_MASS: Dict[str, Tuple[float, Tuple[float, float, float]]] = {
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
    "lower_arm_left":  (0.961458, (0.09, -0.09, 0.09)),
    "hand_left":       (0.268083, (0.0, 0.0, 0.0)),
}
_MASS_TOT = float(sum(m for m, _ in BODY_MASS.values()))

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

# ------------------------------------------------- normalized coordinates --
# q_norm = (q_rad - mid)/half_range; every joint spans [-1, 1].
_MID = (JNT_RANGE[:, 0] + JNT_RANGE[:, 1]) / 2.0
_HALF = (JNT_RANGE[:, 1] - JNT_RANGE[:, 0]) / 2.0


def norm_to_rad(q_norm: np.ndarray) -> np.ndarray:
    return np.asarray(q_norm, dtype=np.float64) * _HALF + _MID


def rad_to_norm(q_rad: np.ndarray) -> np.ndarray:
    return (np.asarray(q_rad, dtype=np.float64) - _MID) / _HALF


# ---------------------------------------------------------------- springs --
# Per-joint restoring spring in normalized coords, centered at 0
# (unloaded configuration = the natural pose, by definition).
#   linear:    F = -k*q     E = k*q^2/2
#   quadratic: F = -k*q|q|  E = k*|q|^3/3   (soft mid-range, stiff near limits)
# k = (k_low, k_high) applied for q<0 / q>0. Ratios only (knee = 1.0).
SPRINGS: Dict[str, Tuple[str, float, float]] = {
    # law, k_low, k_high
    "abdomen_z":       ("linear",    20.0, 20.0),
    "abdomen_y":       ("linear",    20.0, 20.0),
    "abdomen_x":       ("linear",    20.0, 20.0),
    "hip_x_right":     ("quadratic",  1.0,  1.0),
    "hip_z_right":     ("quadratic",  1.0,  1.0),
    "hip_y_right":     ("quadratic",  1.0,  1.0),
    "knee_right":      ("quadratic",  1.0,  1.0),
    "ankle_y_right":   ("quadratic",  0.7,  0.7),
    "ankle_x_right":   ("quadratic",  0.7,  0.7),
    "hip_x_left":      ("quadratic",  1.0,  1.0),
    "hip_z_left":      ("quadratic",  1.0,  1.0),
    "hip_y_left":      ("quadratic",  1.0,  1.0),
    "knee_left":       ("quadratic",  1.0,  1.0),
    "ankle_y_left":    ("quadratic",  0.7,  0.7),
    "ankle_x_left":    ("quadratic",  0.7,  0.7),
    "shoulder1_right": ("linear",     0.4,  0.4),
    "shoulder2_right": ("linear",     0.4,  0.4),
    "elbow_right":     ("linear",     1.0,  1.0),
    "shoulder1_left":  ("linear",     0.4,  0.4),
    "shoulder2_left":  ("linear",     0.4,  0.4),
    "elbow_left":      ("linear",     1.0,  1.0),
}

_SPRING_K_LOW = np.array([SPRINGS[n][1] for n in JOINT_NAMES])
_SPRING_K_HIGH = np.array([SPRINGS[n][2] for n in JOINT_NAMES])
_SPRING_LIN = np.array([SPRINGS[n][0] == "linear" for n in JOINT_NAMES])


def spring_energy(q_norm: np.ndarray) -> float:
    """Total elastic energy of the unloaded-centered springs."""
    q = np.asarray(q_norm, dtype=np.float64)
    a = np.abs(q)
    k = np.where(q < 0.0, _SPRING_K_LOW, _SPRING_K_HIGH)
    return float(np.sum(np.where(_SPRING_LIN, 0.5 * k * q * q,
                                 k * a * a * a / 3.0)))


# ------------------------------------------------------------- math bits ---
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


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


def _drot(axis: np.ndarray, angle: float) -> np.ndarray:
    """d/dθ R(axis,θ) = cosθ·K + sinθ·K²，K = skew(单位轴)。"""
    K = _skew(np.asarray(axis, dtype=np.float64))
    return np.cos(angle) * K + np.sin(angle) * (K @ K)


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

    def anchors_jac(self, root_pos, root_quat_wxyz, q, joint_names):
        """关节锚点世界坐标 + (n,3,21) 对 q 的解析雅可比。

        锚点 a_w[j] 在 FK 中于该关节自身旋转之前求值：只随祖先关节
        与同窗内先于它的关节动，自身与下游关节不动它。
        踝心 = ankle_y 锚点——双脚踝旋转下不动，即 P5 踝心语义。
        """
        q = np.asarray(q, dtype=np.float64)
        jis = [self.joint_index[n] for n in joint_names]
        pending = set(jis)
        # 每个 body 的祖先关节集（兄弟链不算）——锚点雅可比的非零列域
        anc_of = {}
        for name in self.order:
            par = self.parent[name]
            anc_of[name] = ([] if par is None else
                            anc_of[par]
                            + [j for j, _, _ in self.joints_on[par]])
        xf = {"__root__": (np.asarray(root_pos, dtype=np.float64),
                           _quat_mat(root_quat_wxyz))}
        a_w, k_w, prec = {}, {}, {}
        for name in self.order:
            parent = self.parent[name] or "__root__"
            ppos, pmat = xf[parent]
            pos = ppos + pmat @ self.bpos[name]
            mat = pmat.copy()
            body_seen = list(anc_of[name])
            for jidx, anchor, axis in self.joints_on[name]:
                a_w[jidx] = pos + mat @ anchor
                k_w[jidx] = mat @ axis
                prec[jidx] = body_seen          # 祖先 + 同 body 先于自己的关节
                body_seen = body_seen + [jidx]
                R = _rot(axis, q[jidx])
                pos = pos + mat @ (anchor - R @ anchor)
                mat = mat @ R
            xf[name] = (pos, mat)
            pending -= set(j for j, _, _ in self.joints_on[name])
            if not pending:
                break          # 请求的锚点全部捕获后不再遍历下游 body
        pos_out = np.stack([a_w[j] for j in jis])
        jac = np.zeros((len(jis), 3, len(JOINT_NAMES)))
        for ai, jidx in enumerate(jis):
            for jp in prec[jidx]:
                jac[ai, :, jp] = np.cross(k_w[jp], a_w[jidx] - a_w[jp])
        return pos_out, jac

    def landmarks(self, root_pos, root_quat_wxyz, q) -> np.ndarray:
        """(5,3) landmark world positions in LANDMARK_ORDER."""
        xf = self.body_transforms(root_pos, root_quat_wxyz, q)
        return np.stack([
            xf[b][0] + xf[b][1] @ off
            for b, off in (LANDMARKS[n] for n in LANDMARK_ORDER)])

    def support_jac(self, root_pos, root_quat_wxyz, q, anchor_names):
        """单次遍历：全身质心 + 头点 + 请求关节锚点的世界坐标
        及对 q(rad) 的解析雅可比。

        com = Σ m_b·xipos_b / M（BODY_MASS 表，单台 40.844kg）。
        J_com[:,j] = (M_sub/M)·axis_w×(com_sub − a_w[j])，
        sub = 关节所在 body 的子树（含自身与全部后代）。
        锚点语义同 anchors_jac；头体无关节，关节雅可比恒 0。
        返回 (com, head, anchors, J_com(3,21), J_a(n,3,21))。
        """
        q = np.asarray(q, dtype=np.float64)
        jis = [self.joint_index[n] for n in anchor_names]
        anc_of = {}
        for name in self.order:
            par = self.parent[name]
            anc_of[name] = ([] if par is None else
                            anc_of[par]
                            + [j for j, _, _ in self.joints_on[par]])
        xf = {"__root__": (np.asarray(root_pos, dtype=np.float64),
                           _quat_mat(root_quat_wxyz))}
        a_w, k_w, prec, xip = {}, {}, {}, {}
        for name in self.order:
            parent = self.parent[name] or "__root__"
            ppos, pmat = xf[parent]
            pos = ppos + pmat @ self.bpos[name]
            mat = pmat.copy()
            body_seen = list(anc_of[name])
            for jidx, anchor, axis in self.joints_on[name]:
                a_w[jidx] = pos + mat @ anchor
                k_w[jidx] = mat @ axis
                prec[jidx] = body_seen          # 祖先 + 同 body 先于自己的关节
                body_seen = body_seen + [jidx]
                R = _rot(axis, q[jidx])
                pos = pos + mat @ (anchor - R @ anchor)
                mat = mat @ R
            xf[name] = (pos, mat)
            m_b, ipos = BODY_MASS[name]
            xip[name] = pos + mat @ np.asarray(ipos, dtype=np.float64)
        head = xf["head"][0].copy()
        # 子树质心：逆序向 parent 回传
        msub = {n: BODY_MASS[n][0] for n in self.order}
        csub = {n: BODY_MASS[n][0] * xip[n] for n in self.order}
        for name in reversed(self.order):
            par = self.parent[name]
            if par is not None:
                msub[par] += msub[name]
                csub[par] += csub[name]
        com = csub["torso"] / msub["torso"]
        J_com = np.zeros((3, len(JOINT_NAMES)))
        for jname, body, *_rest in JOINTS:
            j = self.joint_index[jname]
            J_com[:, j] = (msub[body] / msub["torso"]) * np.cross(
                k_w[j], csub[body] / msub[body] - a_w[j])
        anchors = np.stack([a_w[j] for j in jis])
        J_a = np.zeros((len(jis), 3, len(JOINT_NAMES)))
        for ai, jidx in enumerate(jis):
            for jp in prec[jidx]:
                J_a[ai, :, jp] = np.cross(k_w[jp], a_w[jidx] - a_w[jp])
        return com, head, anchors, J_com, J_a

# ------------------------------------------------------- shared indices ---
_LM_IDX = {n: i for i, n in enumerate(LANDMARK_ORDER)}
_JIDX = {n: i for i, n in enumerate(JOINT_NAMES)}

_BODY_OFF = {b[0]: np.asarray(b[2], dtype=np.float64) for b in BODIES}


def _mk_arm(side: str) -> dict:
    """Closed-form-IK constants for one arm (right/left)."""
    ua, la, ha = f"upper_arm_{side}", f"lower_arm_{side}", f"hand_{side}"
    js = {j[0]: j for j in JOINTS}
    s1, s2, eb = (js[f"{n}_{side}"] for n in
                  ("shoulder1", "shoulder2", "elbow"))
    return {
        "s_off": _BODY_OFF[ua],              # shoulder anchor in torso
        "u": _BODY_OFF[la],                  # elbow anchor in upper arm
        "w": _BODY_OFF[ha],                  # hand offset in lower arm
        "k1": np.asarray(s1[3], dtype=np.float64),
        "k2": np.asarray(s2[3], dtype=np.float64),
        "ke": np.asarray(eb[3], dtype=np.float64),
        "ji": np.array([_JIDX[s1[0]], _JIDX[s2[0]], _JIDX[eb[0]]]),
        "rng": np.array([s1[4], s2[4], eb[4]], dtype=np.float64),
    }


_ARM = {"right": _mk_arm("right"), "left": _mk_arm("left")}


def _wrap(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _arm_ik(d_vec: np.ndarray, torso_mat: np.ndarray,
            sp: dict) -> List[dict]:
    """Closed-form 3-DOF IK: hand = S + T·(u + R_e·w), T = tm·R1·R2.

    Returns up to 4 candidates {rad, qn, e, vio}: joint angles (rad),
    normalized values, spring energy, total range/reach violation.
    """
    d = float(np.linalg.norm(d_vec))
    d_loc = torso_mat.T @ d_vec
    u, w, ke, k1, k2 = sp["u"], sp["w"], sp["ke"], sp["k1"], sp["k2"]
    rng = sp["rng"]

    # elbow angle from |h| = d:  P cos(th) + Q sin(th) = R
    uk, kw = float(u @ ke), float(ke @ w)
    p = float(u @ w) - uk * kw
    q = float(u @ np.cross(ke, w))
    rhs = (d * d - float(u @ u) - float(w @ w)) / 2.0 - uk * kw
    amp = float(np.hypot(p, q))
    if amp < 1e-12:
        return []
    vio_reach = max(0.0, abs(rhs) - amp) / max(amp, 1e-12)
    th0, beta = np.arctan2(q, p), np.arccos(np.clip(rhs / amp, -1.0, 1.0))

    out = []
    for th in (_wrap(th0 + beta), _wrap(th0 - beta)):
        h = u + _rot(ke, th) @ w
        # shoulder2 from axis preservation:  k1·R2(φ2)h = k1·d_loc
        k1k2, k2h = float(k1 @ k2), float(k2 @ h)
        p2 = float(k1 @ h) - k1k2 * k2h
        q2 = float(k1 @ np.cross(k2, h))
        r2 = float(k1 @ d_loc) - k1k2 * k2h
        amp2 = float(np.hypot(p2, q2))
        if amp2 < 1e-12:
            continue
        f20, g2 = np.arctan2(q2, p2), \
            np.arccos(np.clip(r2 / amp2, -1.0, 1.0))
        for f2 in (_wrap(f20 + g2), _wrap(f20 - g2)):
            v = _rot(k2, f2) @ h
            vp = v - (v @ k1) * k1
            dp = d_loc - (d_loc @ k1) * k1
            nv, nd = np.linalg.norm(vp), np.linalg.norm(dp)
            if nv < 1e-9 or nd < 1e-9:
                continue
            f1 = np.arctan2(float(k1 @ np.cross(vp, dp)) / (nv * nd),
                            float(vp @ dp) / (nv * nd))
            rads = np.array([f1, f2, th])
            vio = vio_reach + float(
                np.maximum(rads - rng[:, 1], 0.0).sum()
                + np.maximum(rng[:, 0] - rads, 0.0).sum())
            qn = (rads - _MID[sp["ji"]]) / _HALF[sp["ji"]]
            out.append({"rad": rads, "qn": qn,
                        "e": _energy_of(qn, sp["ji"]), "vio": vio})
    return out


def _energy_of(q_partial, idx) -> float:
    """Spring energy of a subset of normalized joint values."""
    q = np.zeros(len(JOINT_NAMES))
    q[idx] = q_partial
    return spring_energy(q)


# --------------------------------------------- P5 -> Pose solver (v3 spec) --
# infer_pose_from_p5.md：标准 P5 帧，三阶段，min 弹性能 + 硬位置约束。
# 求解一变量 = pitch/roll + 11 内部关节（腰 3 + 髋膝 8）；
# 踝关节对位置约束不可见（踝心=上游锚点）故不参与，留到求解三纯先验；
# 手臂留到求解二。
_S1_IDX = np.array([_JIDX[n] for n in (
    "abdomen_z", "abdomen_y", "abdomen_x",
    "hip_x_right", "hip_z_right", "hip_y_right", "knee_right",
    "hip_x_left", "hip_z_left", "hip_y_left", "knee_left")])
_ANKLE_CENTER = ("ankle_y_right", "ankle_y_left")   # P5 踝心 = 上游锚点

_X_AX = np.array([1.0, 0.0, 0.0])
_Y_AX = np.array([0.0, 1.0, 0.0])


def _egrad_hess(qn_vals: np.ndarray, idx: np.ndarray):
    """归一化坐标弹性能在 idx 关节上的梯度/对角 Hessian。"""
    s = np.asarray(qn_vals, dtype=np.float64)
    k = np.where(s < 0.0, _SPRING_K_LOW[idx], _SPRING_K_HIGH[idx])
    lin = _SPRING_LIN[idx]
    g = np.where(lin, k * s, k * s * np.abs(s))
    h = np.where(lin, k, 2.0 * k * np.abs(s))
    return g, h


def solve_p5_pose(target_pos: np.ndarray,
                  skeleton: "Skeleton | None" = None,
                  tolerance: float = 0.02,
                  max_iter: int = 80,
                  restarts: int = 2,
                  max_evals: int = 150) -> dict:
    """标准 P5 帧 5 点 -> 全身姿态（能量最小 + 硬位置约束，KKT-牛顿）。

    输入 yaw≡0 的标准帧；脚 landmark 换
    踝心（ankle_y 锚点，踝角不动它）；阶段一用踝心−头相对向量（6 eq），
    变量 13（pitch/roll + 11 关节）；阶段二躯干冻结、闭式臂 IK、不可达
    放宽；阶段三踝 = 弹簧中心纯先验；root_pos 由头绝对位置恢复。

    target_pos: (5,3)，LANDMARK_ORDER 序，foot_* 槽位放踝心坐标。
    satisfied=False 表示阶段一几何不可达（多初值重试后仍未达标）。
    """
    sk = skeleton or Skeleton()
    target = np.asarray(target_pos, dtype=np.float64)
    if target.shape != (5, 3):
        raise ValueError(f"target_pos must be (5,3), got {target.shape}")
    if not np.isfinite(target).all():
        raise ValueError("target_pos: non-finite values")

    h_t = target[_LM_IDX["head"]]
    rel_t = np.stack([target[_LM_IDX["foot_right"]] - h_t,
                      target[_LM_IDX["foot_left"]] - h_t])     # (2,3)
    zero3 = np.zeros(3)

    head_off = _BODY_OFF["head"]        # head 无关节：头点 = R·offset
    eval_cnt = [0]                      # 跨初值共享 eval 计数（~0.5ms/次）

    def eval1(v):
        """stage-1 残差/雅可比/能量梯度。v = (pitch, roll, qn[11])。

        头点只随 root 转（head 体无关节、父为 torso），关节雅可比恒 0；
        解析算头、只对踝锚点跑一遍 FK——单次遍历拿全部量。
        """
        eval_cnt[0] += 1
        p, r = v[0], v[1]
        qn = np.zeros(len(JOINT_NAMES))
        qn[_S1_IDX] = v[2:]
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        quat = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])  # Ry·Rx
        qr = norm_to_rad(qn)
        h = (_rot(_Y_AX, p) @ _rot(_X_AX, r)) @ head_off
        a, Jq_a = sk.anchors_jac(zero3, quat, qr, _ANKLE_CENTER)
        rel = np.stack([a[0] - h, a[1] - h])                    # (2,3)
        res = (rel - rel_t).ravel()                           # (6,)
        J = np.zeros((6, 13))
        Ry, Rx = _rot(_Y_AX, p), _rot(_X_AX, r)
        dRyRt = _drot(_Y_AX, p) @ Rx @ (Ry @ Rx).T
        dRxRt = Ry @ _drot(_X_AX, r) @ (Ry @ Rx).T
        for k in range(2):
            J[3 * k:3 * k + 3, 0] = dRyRt @ rel[k]
            J[3 * k:3 * k + 3, 1] = dRxRt @ rel[k]
            # 变量是归一化坐标：d rel/d qn = J_rad · half_range
            J[3 * k:3 * k + 3, 2:] = (Jq_a[k, :, :][:, _S1_IDX]
                                      * _HALF[_S1_IDX][None, :])
        return res, J, rel, h, quat, qr

    def solve1(v0, budget):
        """两段：LM 最小二乘先求可行（残差单调降）→ KKT-QP 抛能量。
        budget = 本初值可用的 eval 上限（与全局 eval_cnt 共享）。
        返回 (v, ok)。"""
        v = np.asarray(v0, dtype=np.float64).copy()
        mu = 1e-4
        # phase A：可行性（残差平方和的 Levenberg-Marquardt）
        for _ in range(max_iter):
            if eval_cnt[0] >= budget:
                return v, False
            res, J, *_ = eval1(v)
            cost = float(res @ res)
            if np.abs(res).max() <= tolerance:
                break
            JtJ = J.T @ J
            Jtr = J.T @ res
            improved = False
            for _ in range(10):
                if eval_cnt[0] >= budget:
                    return v, False
                try:
                    d = np.linalg.solve(
                        JtJ + mu * np.eye(13), -Jtr)
                except np.linalg.LinAlgError:
                    return v, False
                vn = v + d
                vn[2:] = np.clip(vn[2:], -1.0, 1.0)
                res_n = eval1(vn)[0]
                if res_n @ res_n < cost:
                    v, mu, improved = vn, max(mu / 3.0, 1e-9), True
                    break
                mu *= 5.0
            if not improved:
                return v, False                  # LM 局部极小：本初值失败
        else:
            return v, False
        # phase B：KKT-QP 抛能量（保持残差 ≤ tol 且 E 严格下降）
        for _ in range(20):
            if eval_cnt[0] >= budget:
                break
            res, J, *_ = eval1(v)
            g = np.zeros(13)
            H = np.zeros(13)
            g[2:], H[2:] = _egrad_hess(v[2:], _S1_IDX)
            A = np.zeros((19, 19))
            A[:13, :13] = np.diag(H + 1e-4)
            A[:13, 13:] = J.T
            A[13:, :13] = J
            try:
                d = np.linalg.solve(A, np.concatenate([-g, -res]))[:13]
            except np.linalg.LinAlgError:
                break
            e0 = _energy_of(v[2:], _S1_IDX)
            for bt in range(9):
                if eval_cnt[0] >= budget:
                    break
                step = 0.5 ** bt
                vn = v + d * step
                vn[2:] = np.clip(vn[2:], -1.0, 1.0)
                res_n = eval1(vn)[0]
                if np.abs(res_n).max() <= tolerance and \
                        _energy_of(vn[2:], _S1_IDX) < e0:
                    v = vn
                    break
            else:
                break
            if np.linalg.norm(d * step) < 1e-7:
                break
        return v, True

    # ---- stage 1：13 变量 6 硬约束，能量最小 --------------------------
    # 初值序列：几何估计（头−踝倾斜→躯干角、余弦定理→膝角、踝向→髋角，
    # 覆盖绝大多数姿态的吸引域）→ 两级确定性扰动（邻近流域）→
    # rad0（直站姿）→ norm0（弹簧中心）→ 可选随机重启。
    def geo_init():
        PELVIS_DROP = 0.615          # 头→骨盆落差（直立近似，BODIES 量出）
        L1, L2 = 0.42, 0.31          # 髋→膝 / 膝→踝心
        mid = (rel_t[0] + rel_t[1]) / 2.0
        v = np.zeros(13)
        down = max(-mid[2], 1e-6)
        v[0] = np.clip(np.arctan2(-mid[0], down), -1.2, 1.2)
        v[1] = np.clip(np.arctan2(mid[1], down), -1.0, 1.0)

        def to_qn(rad, name):
            i = _JIDX[name]
            return float((rad - _MID[i]) / _HALF[i])
        for k, leg in enumerate(("right", "left")):
            pa = rel_t[k] + np.array([0.0, 0.0, PELVIS_DROP])
            d = np.clip(np.linalg.norm(pa), L1 - L2 + 0.02, L1 + L2 - 0.02)
            interior = np.arccos((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2))
            q_knee = -(np.pi - interior)
            hd = pa / max(np.linalg.norm(pa), 1e-6)
            q_hipy = -np.arctan2(hd[0], -hd[2])
            q_hipz = np.arctan2(-hd[1] * (-1.0 if leg == "right" else 1.0),
                                -hd[2])
            base = 2 + (3 if leg == "right" else 7)
            v[base + 0] = 0.0
            v[base + 1] = np.clip(to_qn(q_hipz, f"hip_z_{leg}"), -1, 1)
            v[base + 2] = np.clip(to_qn(q_hipy, f"hip_y_{leg}"), -1, 1)
            v[base + 3] = np.clip(to_qn(q_knee, f"knee_{leg}"), -1, 1)
        return v

    qn_rad0 = rad_to_norm(np.zeros(len(JOINT_NAMES)))[_S1_IDX]
    g0 = geo_init()
    # 直膝变体：抬腿/踢姿下"骨盆垂距"假设失效、膝角估错流域，
    # 同髋估计＋两膝伸直覆盖该流域。
    g_straight = g0.copy()
    for leg in ("right", "left"):
        i = _JIDX[f"knee_{leg}"]
        g_straight[(2 + (3 if leg == "right" else 7)) + 3] = np.clip(
            -_MID[i] / _HALF[i], -1, 1)
    rng = np.random.default_rng(0)
    inits = [g0,
             g_straight,
             g0 + rng.uniform(-0.12, 0.12, 13),
             g0 + rng.uniform(-0.25, 0.25, 13),
             np.concatenate([[0.0, 0.0], qn_rad0]),
             np.zeros(13)]
    for _ in range(max(0, restarts - 1)):
        inits.append(np.concatenate(
            [rng.uniform(-0.4, 0.4, 2), rng.uniform(-0.5, 0.5, 11)]))
    v, ok1 = None, False
    for v0 in inits:
        if eval_cnt[0] >= max_evals:
            break
        v, ok1 = solve1(v0, max_evals)
        if ok1:
            break

    res, _, _, h_m, quat, qr_body = eval1(v)
    r_max = float(np.abs(res).max())

    # ---- stage 4 提前：root 平移 = 目标头 − 模型头（臂/踝不影响头踝）--
    root_pos = h_t - h_m

    # ---- stage 3：踝关节纯先验 = 弹簧中心（norm 0，该模型即 rad 0）----
    qn = np.zeros(len(JOINT_NAMES))
    qn[_S1_IDX] = v[2:]                                  # 踝位已是 0

    # ---- stage 2：冻结躯干+腿，每臂闭式 3x3 IK，不可达放宽 ------------
    stage2 = {}
    if ok1:
        qr_full = norm_to_rad(qn)
        xf = sk.body_transforms(root_pos, quat, qr_full)
        t_pos, t_mat = xf["torso"]
        for side in ("right", "left"):
            sp = _ARM[side]
            s0 = t_pos + t_mat @ sp["s_off"]
            tgt_h = target[_LM_IDX[f"hand_{side}"]]
            cand = _arm_ik(tgt_h - s0, t_mat, sp)
            if cand:
                pick = min(cand, key=lambda c: (c["vio"], c["e"]))
                qn[np.asarray(sp["ji"])] = pick["qn"]
                stage2[side] = {"vio": pick["vio"], "e": pick["e"],
                                "relaxed": bool(pick["vio"] > 1e-9)}
            else:
                stage2[side] = {"vio": np.inf, "e": 0.0,
                                "relaxed": True}          # 退化：留 norm 0
    else:
        for side in ("right", "left"):
            stage2[side] = {"vio": np.inf, "e": 0.0, "relaxed": True}

    qr = norm_to_rad(qn)
    lms = sk.landmarks(root_pos, quat, qr)
    a_f, _ = sk.anchors_jac(root_pos, quat, qr, _ANKLE_CENTER)
    achieved = lms.copy()
    achieved[_LM_IDX["foot_right"]] = a_f[0]
    achieved[_LM_IDX["foot_left"]] = a_f[1]
    errs = np.linalg.norm(achieved - target, axis=1)
    return {
        "root_pos": root_pos,
        "root_quat_wxyz": quat,
        "pitch": float(v[0]), "roll": float(v[1]),
        "q_norm": qn,
        "q_rad": qr,
        "landmarks": achieved,
        "point_errors": errs,
        "satisfied": bool(ok1),
        "stage1": {"converged": bool(ok1), "evals": eval_cnt[0],
                   "max_resid": r_max},
        "stage2": stage2,
        "energy": spring_energy(qn),
        "joint_names": JOINT_NAMES,
    }


# -------------------------------- SupportGeometry -> Pose solver (v1 spec) -
# reference_pose_for_support_geometry.md：标准几何帧，单阶段，
# 23 变量（pitch/roll + 全 21 关节）对 6 硬约束（踝心−com 相对向量），
# 目标 min −w_head·|头−com| + E_spring + w_arm·Σ_arm(qn−ref)²；
# root_pos 由 com 绝对位置恢复。com 对全部关节有非零雅可比列，
# 故变量全收（不照搬 _S1_IDX）。
_ARM_IDX = np.array([_JIDX[n] for n in (
    "shoulder1_right", "shoulder2_right", "elbow_right",
    "shoulder1_left", "shoulder2_left", "elbow_left")])
_ALL_IDX = np.arange(len(JOINT_NAMES))


def solve_support_pose(com_t: np.ndarray,
                       ankle_right_t: np.ndarray,
                       ankle_left_t: np.ndarray,
                       skeleton: "Skeleton | None" = None,
                       tolerance: float = 0.02,
                       max_iter: int = 80,
                       restarts: int = 2,
                       max_evals: int = 150,
                       w_head: float = 10.0,
                       w_arm: float = 40.0,
                       arm_ref: np.ndarray = None) -> dict:
    """标准几何帧 com+双踝心 -> 全身姿态（max|头−com| + 先验正则）。

    com_t: (3,) 目标质心；ankle_*_t: (3,) 目标踝心（标准帧，yaw=0）。
    arm_ref: (6,) 臂关节归一化标称值，None = 弹簧中心。
    satisfied=False = 几何不可达（多初值后残差仍超 tolerance）。
    """
    sk = skeleton or Skeleton()
    NJ = len(JOINT_NAMES)
    NV = 2 + NJ
    com_t = np.asarray(com_t, dtype=np.float64)
    a_rt = np.asarray(ankle_right_t, dtype=np.float64)
    a_lt = np.asarray(ankle_left_t, dtype=np.float64)
    for n, v in (("com_t", com_t), ("ankle_right_t", a_rt),
                 ("ankle_left_t", a_lt)):
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError(f"{n}: expected finite (3,), got {v.shape}")
    rel_t = np.stack([a_rt - com_t, a_lt - com_t])              # (2,3)
    aref = (np.zeros(len(_ARM_IDX)) if arm_ref is None
            else np.asarray(arm_ref, dtype=np.float64))
    if aref.shape != (len(_ARM_IDX),):
        raise ValueError(f"arm_ref: expected (6,), got {aref.shape}")
    zero3 = np.zeros(3)
    eval_cnt = [0]

    def evalS(v):
        """残差/雅可比/目标项。v = (pitch, roll, qn[21])。

        踝−com 相对向量随 root 旋转：pitch/roll 列 = dR/dθ·Rᵀ·rel；
        关节列 = (J_anchor − J_com)·half_range。目标项
        d = 头 − com：头无关节，J_d = [dR·d, dR·d, −J_com·half]。
        """
        eval_cnt[0] += 1
        p, r = v[0], v[1]
        qn = v[2:].copy()
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        quat = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])
        qr = norm_to_rad(qn)
        com, h, a, J_com, J_a = sk.support_jac(zero3, quat, qr,
                                               _ANKLE_CENTER)
        rel = np.stack([a[0] - com, a[1] - com])
        res = (rel - rel_t).ravel()
        Ry, Rx = _rot(_Y_AX, p), _rot(_X_AX, r)
        dRyRt = _drot(_Y_AX, p) @ Rx @ (Ry @ Rx).T
        dRxRt = Ry @ _drot(_X_AX, r) @ (Ry @ Rx).T
        J = np.zeros((6, NV))
        d = h - com
        J_d = np.zeros((3, NV))
        J_d[:, 0] = dRyRt @ d
        J_d[:, 1] = dRxRt @ d
        J_d[:, 2:] = -J_com * _HALF[None, :]
        for k in range(2):
            J[3 * k:3 * k + 3, 0] = dRyRt @ rel[k]
            J[3 * k:3 * k + 3, 1] = dRxRt @ rel[k]
            J[3 * k:3 * k + 3, 2:] = (J_a[k] - J_com) * _HALF[None, :]
        return res, J, rel, d, J_d, com, quat, qr

    def energy_of(v, d):
        """相 B 目标：弹性能 + 臂强先验 − w_head·|头−com|。"""
        qn = v[2:]
        return (spring_energy(qn)
                + w_arm * float(((qn[_ARM_IDX] - aref) ** 2).sum())
                - w_head * float(np.linalg.norm(d)))

    def solve1(v0, budget):
        """LM 可行 → KKT-QP 抛目标（结构同 solve_p5_pose.solve1）。"""
        v = np.asarray(v0, dtype=np.float64).copy()
        mu = 1e-4
        for _ in range(max_iter):
            if eval_cnt[0] >= budget:
                return v, False
            res, J, *_ = evalS(v)
            cost = float(res @ res)
            if np.abs(res).max() <= tolerance:
                break
            JtJ = J.T @ J
            Jtr = J.T @ res
            improved = False
            for _ in range(10):
                if eval_cnt[0] >= budget:
                    return v, False
                try:
                    d = np.linalg.solve(JtJ + mu * np.eye(NV), -Jtr)
                except np.linalg.LinAlgError:
                    return v, False
                vn = v + d
                vn[2:] = np.clip(vn[2:], -1.0, 1.0)
                res_n = evalS(vn)[0]
                if res_n @ res_n < cost:
                    v, mu, improved = vn, max(mu / 3.0, 1e-9), True
                    break
                mu *= 5.0
            if not improved:
                return v, False
        else:
            return v, False
        # phase B：目标含负曲率项（−w_head·|d|），GN 近似 Hessian
        for _ in range(30):
            if eval_cnt[0] >= budget:
                break
            res, J, _, dvec, J_d, *_ = evalS(v)
            qn = v[2:]
            g = np.zeros(NV)
            hdiag = np.zeros(NV)
            g[2:], hdiag[2:] = _egrad_hess(qn, _ALL_IDX)
            g[2 + _ARM_IDX] += w_arm * (qn[_ARM_IDX] - aref)
            hdiag[2 + _ARM_IDX] += w_arm
            A = np.zeros((NV + 6, NV + 6))
            A[:NV, :NV] = np.diag(hdiag + 1e-4)
            dn = float(np.linalg.norm(dvec))
            if dn > 1e-9:
                u = dvec / dn
                g -= w_head * (J_d.T @ u)
                A[:NV, :NV] -= w_head * (
                    J_d.T @ (np.eye(3) - np.outer(u, u)) @ J_d) / dn
            A[:NV, NV:] = J.T
            A[NV:, :NV] = J
            try:
                d = np.linalg.solve(
                    A, np.concatenate([-g, -res]))[:NV]
            except np.linalg.LinAlgError:
                break
            e0 = energy_of(v, dvec)
            for bt in range(9):
                if eval_cnt[0] >= budget:
                    break
                step = 0.5 ** bt
                vn = v + d * step
                vn[2:] = np.clip(vn[2:], -1.0, 1.0)
                res_n, _, _, d_n, *_ = evalS(vn)
                if np.abs(res_n).max() <= tolerance and \
                        energy_of(vn, d_n) < e0:
                    v = vn
                    break
            else:
                break
            if np.linalg.norm(d * step) < 1e-7:
                break
        return v, True

    # 初值：骨盆≈com−0.08z 的几何估计 → 直膝变体 → 确定性扰动
    # → rad0（直站姿）→ norm0 → 可选随机重启。
    def geo_init():
        COM_TO_PELVIS = 0.08       # 直立时 com 约高骨盆 8cm
        L1, L2 = 0.42, 0.31        # 髋→膝 / 膝→踝心
        mid = (rel_t[0] + rel_t[1]) / 2.0
        v = np.zeros(NV)
        down = max(-mid[2], 1e-6)
        v[0] = np.clip(np.arctan2(-mid[0], down), -1.2, 1.2)
        v[1] = np.clip(np.arctan2(mid[1], down), -1.0, 1.0)

        def to_qn(rad, name):
            i = _JIDX[name]
            return float((rad - _MID[i]) / _HALF[i])
        for k, leg in enumerate(("right", "left")):
            pa = rel_t[k] + np.array([0.0, 0.0, COM_TO_PELVIS])
            d = np.clip(np.linalg.norm(pa), L1 - L2 + 0.02,
                        L1 + L2 - 0.02)
            interior = np.arccos(
                (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2))
            q_knee = -(np.pi - interior)
            hd = pa / max(np.linalg.norm(pa), 1e-6)
            q_hipy = -np.arctan2(hd[0], -hd[2])
            q_hipz = np.arctan2(
                -hd[1] * (-1.0 if leg == "right" else 1.0), -hd[2])
            base = 2 + _JIDX[f"hip_x_{leg}"]
            v[base + 0] = 0.0
            v[base + 1] = np.clip(to_qn(q_hipz, f"hip_z_{leg}"), -1, 1)
            v[base + 2] = np.clip(to_qn(q_hipy, f"hip_y_{leg}"), -1, 1)
            v[2 + _JIDX[f"knee_{leg}"]] = np.clip(
                to_qn(q_knee, f"knee_{leg}"), -1, 1)
        return v

    g0 = geo_init()
    g_straight = g0.copy()
    for leg in ("right", "left"):
        i = _JIDX[f"knee_{leg}"]
        g_straight[2 + i] = np.clip(-_MID[i] / _HALF[i], -1, 1)
    rng = np.random.default_rng(0)
    inits = [g0,
             g_straight,
             g0 + rng.uniform(-0.12, 0.12, NV),
             g0 + rng.uniform(-0.25, 0.25, NV),
             np.concatenate([[0.0, 0.0],
                             rad_to_norm(np.zeros(NJ))]),
             np.zeros(NV)]
    for _ in range(max(0, restarts - 1)):
        inits.append(np.concatenate(
            [rng.uniform(-0.4, 0.4, 2), rng.uniform(-0.5, 0.5, NJ)]))
    v, ok1 = None, False
    for v0 in inits:
        if eval_cnt[0] >= max_evals:
            break
        v, ok1 = solve1(v0, max_evals)
        if ok1:
            break

    res, _, rel_m, dvec, _, com_m, quat, qr = evalS(v)
    r_max = float(np.abs(res).max())
    root_pos = com_t - com_m
    return {
        "root_pos": root_pos,
        "root_quat_wxyz": quat,
        "pitch": float(v[0]), "roll": float(v[1]),
        "q_norm": v[2:].copy(),
        "q_rad": qr,
        "com": com_m + root_pos,
        "ankles": rel_m + com_m[None, :] + root_pos,   # (2,3) 右左踝心
        "head_com": dvec,
        "head_com_dist": float(np.linalg.norm(dvec)),
        "satisfied": bool(ok1),
        "stage1": {"converged": bool(ok1), "evals": eval_cnt[0],
                   "max_resid": r_max},
        "energy": energy_of(v, dvec),
        "joint_names": JOINT_NAMES,
    }
