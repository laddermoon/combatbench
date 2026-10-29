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


# -------------------------------------------------- equilibrium solving ---
_LM_IDX = {n: i for i, n in enumerate(LANDMARK_ORDER)}
_JIDX = {n: i for i, n in enumerate(JOINT_NAMES)}
_ARM_JOINTS = ("shoulder1_right", "shoulder2_right", "elbow_right",
               "shoulder1_left", "shoulder2_left", "elbow_left")
_BODY_IDX = np.array([i for i, n in enumerate(JOINT_NAMES)
                      if n not in _ARM_JOINTS])          # 15 leg+torso
_STAGE2_IDX = np.array([_JIDX[n] for n in
                       ("abdomen_z",) + _ARM_JOINTS])    # AZ + arms = 7

_BODY_OFF = {b[0]: np.asarray(b[2], dtype=np.float64) for b in BODIES}
_ANCHOR = {j[0]: np.asarray(j[2], dtype=np.float64) for j in JOINTS}


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


def solve_equilibrium(target_pos: np.ndarray,
                      torso_yaw: float | None = None,
                      skeleton: "Skeleton | None" = None,
                      tolerance: float = 0.01,
                      max_iter: int = 500,
                      pull_step: float = 0.05) -> dict:
    """Pure function: 5 pinned points (+ optional yaw) -> joint angles.

    Two locked stages, per design:
      stage 1 — pin feet + head (+ torso_yaw if given), solve the BODY
                (root + legs + abdomen + ankles). Arms stay unloaded.
                Result is FROZEN.
      stage 2 — everything frozen except abdomen_z + the 6 arm joints;
                pin the two hands. Shoulder pull may twist the waist (AZ)
                and thereby change the final torso yaw — intentional.

    Each stage independently reports feasibility: stage-1 failure means
    the body geometry is unreachable; stage-2 failure means the hands
    are unreachable via arms + waist twist alone.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as _R

    sk = skeleton or Skeleton()
    target = np.asarray(target_pos, dtype=np.float64)
    if target.shape != (5, 3):
        raise ValueError(f"target_pos must be (5,3), got {target.shape}")

    def quat(rv):
        if np.linalg.norm(rv) < 1e-12:
            return np.array([1.0, 0, 0, 0])
        q4 = _R.from_rotvec(rv).as_quat()
        return np.array([q4[3], q4[0], q4[1], q4[2]])

    yaw_rad = np.deg2rad(torso_yaw) if torso_yaw is not None else None

    # ---- stage 1 ---------------------------------------------------------
    def lm1(v):     # landmarks under a stage-1 variable vector
        q = np.zeros(len(JOINT_NAMES))
        q[_BODY_IDX] = v[6:]
        return sk.landmarks(v[0:3], quat(v[3:6]), norm_to_rad(q))

    def yaw_res(v):
        """Signed angle between torso +x (xy proj) and target direction."""
        xf = sk.body_transforms(v[0:3], quat(v[3:6]),
                                norm_to_rad(_full_q_body(v[6:])))
        t = xf["torso"][1][:, 0][:2]
        lmv = lm1(v)
        b = lmv[_LM_IDX["foot_right"], :2] - lmv[_LM_IDX["foot_left"], :2]
        nb = np.linalg.norm(b)
        if nb < 1e-9 or np.linalg.norm(t) < 1e-9:
            return 0.0
        b /= nb
        n = np.array([-b[1], b[0]])
        dir_t = np.cos(yaw_rad) * n + np.sin(yaw_rad) * b
        return float(np.arctan2(t @ np.array([-dir_t[1], dir_t[0]]),
                                t @ dir_t))

    def _full_q_body(body_q):
        q = np.zeros(len(JOINT_NAMES))
        q[_BODY_IDX] = body_q
        return q

    fr = target[_LM_IDX["foot_right"]]
    fl = target[_LM_IDX["foot_left"]]
    mid = (fr + fl) / 2.0
    d = fl - fr
    yaw0 = float(np.arctan2(-d[0], d[1])) \
        if np.linalg.norm(d[:2]) > 1e-3 else 0.0
    x1 = np.concatenate([
        np.array([mid[0], mid[1], mid[2] + 0.79]),
        _R.from_euler("z", yaw0).as_rotvec(),
        np.zeros(len(_BODY_IDX))])
    b1 = [(-np.inf, np.inf)] * 6 + [(-1.0, 1.0)] * len(_BODY_IDX)
    lm_idx1 = np.array([_LM_IDX[p] for p in
                        ("foot_right", "foot_left", "head")])

    failed = None
    tgt1 = target[lm_idx1]
    cur = lm1(x1)[lm_idx1]
    n1 = max(1, int(np.ceil(
        np.linalg.norm(tgt1 - cur, axis=1).max() / pull_step)))
    for s in range(1, n1 + 1):
        sub = cur + (tgt1 - cur) * (s / n1)
        def res(v, sub=sub):
            r = (lm1(v)[lm_idx1] - sub).ravel()
            if yaw_rad is not None:
                r = np.append(r, yaw_res(v))
            return r
        sol = minimize(lambda v: _energy_of(v[6:], _BODY_IDX), x1,
                       method="SLSQP", bounds=b1,
                       constraints=[{"type": "eq", "fun": res}],
                       options={"maxiter": max_iter, "ftol": 1e-12})
        x1 = sol.x
    e1 = np.linalg.norm(lm1(x1)[lm_idx1] - tgt1, axis=1)
    yaw_err = abs(yaw_res(x1)) if yaw_rad is not None else 0.0
    if e1.max() > tolerance or yaw_err > np.deg2rad(3.0):
        failed = "stage1_body"
        q_full = _full_q_body(x1[6:])
        achieved = sk.landmarks(x1[0:3], quat(x1[3:6]), norm_to_rad(q_full))
        return _result(x1[0:3], quat(x1[3:6]), q_full, achieved,
                       target, tolerance, failed)

    # ---- stage 2 ---------------------------------------------------------
    # Feet stay pinned; abdomen_z twists the TORSO side about the world
    # axis line of the AZ joint, swinging the shoulder anchors on an arc
    # — that is the only channel by which arm pull affects the frozen
    # body.  Each arm is a closed-form 3-DOF IK; the solve is a 1-D
    # spring-energy minimisation over the AZ range (milliseconds).
    root_pos, root_quat = x1[0:3].copy(), quat(x1[3:6])
    q_frozen = _full_q_body(x1[6:])
    xf = sk.body_transforms(root_pos, root_quat, norm_to_rad(q_frozen))
    t_pos, t_mat = xf["torso"]

    az_i = _JIDX["abdomen_z"]
    az0 = float(q_frozen[az_i])
    half_az = float(_HALF[az_i])
    # world AZ axis line: through the joint anchor, along torso z
    a_ax = t_pos + t_mat @ (_BODY_OFF["waist_lower"]
                            + _ANCHOR["abdomen_z"])
    k_ax = t_mat @ np.array((0.0, 0.0, 1.0))

    tgt_h = {"right": target[_LM_IDX["hand_right"]],
             "left": target[_LM_IDX["hand_left"]]}
    az_arr = np.array([az_i])

    def _eval_a(a):
        """(energy, per-arm pick, feasible, deficit) at AZ norm a."""
        rm = _rot(k_ax, -(a - az0) * half_az)   # torso counter-rotates
        tm = rm @ t_mat
        e_tot = _energy_of(np.array([a]), az_arr)
        pick, deficit = {}, 0.0
        for side in ("right", "left"):
            sp = _ARM[side]
            s0 = t_pos + t_mat @ sp["s_off"]
            s = a_ax + rm @ (s0 - a_ax)
            cand = _arm_ik(tgt_h[side] - s, tm, sp)
            c = min(cand, key=lambda c: (c["vio"], c["e"])) \
                if cand else None
            if c is not None:
                deficit += c["vio"]
            if c is not None and c["vio"] <= 1e-9:
                pick[side] = c
                e_tot += c["e"]
            else:
                pick[side] = c      # best-effort, not feasible
        ok = all(pick[s] is not None and pick[s]["vio"] <= 1e-9
                 for s in ("right", "left"))
        return e_tot, pick, ok, deficit

    feas_best = alt_best = None
    for a in np.linspace(-1.0, 1.0, 201):
        e_tot, pick, ok, deficit = _eval_a(float(a))
        if ok and (feas_best is None or e_tot < feas_best[0]):
            feas_best = (e_tot, float(a), pick)
        if alt_best is None or deficit < alt_best[0]:
            alt_best = (deficit, float(a), pick)
    if feas_best is not None:
        lo = max(-1.0, feas_best[1] - 0.02)
        hi = min(1.0, feas_best[1] + 0.02)
        for a in np.linspace(lo, hi, 21):
            e_tot, pick, ok, deficit = _eval_a(float(a))
            if ok and e_tot < feas_best[0]:
                feas_best = (e_tot, float(a), pick)

    if feas_best is not None:
        a_sel, pick = feas_best[1], feas_best[2]
    else:
        failed = "stage2_hands"
        a_sel, pick = alt_best[1], alt_best[2]

    rm = _rot(k_ax, -(a_sel - az0) * half_az)
    root_pos = a_ax + rm @ (t_pos - a_ax)
    root_mat = rm @ t_mat
    q4 = _R.from_matrix(root_mat).as_quat()
    root_quat = np.array([q4[3], q4[0], q4[1], q4[2]])
    q_full = q_frozen.copy()
    q_full[az_i] = a_sel
    for side in ("right", "left"):
        c = pick.get(side)
        if c is not None:
            q_full[np.asarray(_ARM[side]["ji"])] = c["qn"]
    achieved = sk.landmarks(root_pos, root_quat, norm_to_rad(q_full))
    return _result(root_pos, root_quat, q_full, achieved,
                   target, tolerance, failed)


def _energy_of(q_partial, idx) -> float:
    """Spring energy of a subset of normalized joint values."""
    q = np.zeros(len(JOINT_NAMES))
    q[idx] = q_partial
    return spring_energy(q)


def _result(root_pos, root_quat, q_norm, achieved, target,
            tolerance, failed) -> dict:
    errs = np.linalg.norm(achieved - target, axis=1)
    return {
        "root_pos": np.asarray(root_pos).copy(),
        "root_quat_wxyz": np.asarray(root_quat).copy(),
        "q_norm": np.asarray(q_norm).copy(),
        "q_rad": norm_to_rad(q_norm),
        "landmarks": achieved,
        "point_errors": errs,
        "satisfied": failed is None and bool(errs.max() <= tolerance),
        "failed_stage": failed,
        "energy": spring_energy(q_norm),
        "joint_names": JOINT_NAMES,
    }
