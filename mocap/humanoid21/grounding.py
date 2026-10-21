"""Contact-aware grounding for retargeted clips.

Replaces the legacy ``foot_z - 0.027`` heuristic (which ignored foot
ORIENTATION — a plantar-flexed ~60° toe-down foot sinks ~10cm deeper
than the flat-foot estimate).

Method per frame:
  1. true sole bottom = min over the 4 foot-capsule endpoints,
     world frame, minus capsule radius (FK, not MuJoCo).
  2. anchor = lowest sole among the feet labelled in contact
     (``contacts`` arg; falls back to the lower foot).
  3. lift root_pos.z so anchor sole == margin.
  4. any OTHER contact foot still floating above ``margin + ik_tol``
     gets a bounded IK repair on its leg joints
     {hip_x, hip_z, hip_y, knee, ankle_y, ankle_x}, regularized toward
     the retargeted angles — feet that are legitimately airborne
     (not in contact) are left alone.

Everything here is a KINEMATIC repair on reference data; it does not
assert dynamic executability.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .robot_kin import CONTROLLED_JOINTS, RobotFK, joint_limits

# foot capsule local endpoints + radius (battle_circular_v2.xml
# foot1/foot2 fromto/size — shared by both feet and both robots)
FOOT_CAPS = ((np.array([-.07, -.01, 0.0]), np.array([.14, -.03, 0.0])),
             (np.array([-.07, +.01, 0.0]), np.array([.14, +.03, 0.0])))
FOOT_CAPS_R = 0.027

LEG_JOINTS = ("hip_x", "hip_z", "hip_y", "knee", "ankle_y", "ankle_x")


def sole_bottom_z(fk: RobotFK, root_pos, root_quat, q: Dict[str, float],
                  side: str) -> float:
    """True lowest point of the two foot capsules, world z."""
    qv = fk.qvec(np.asarray(root_pos), np.asarray(root_quat), q)
    pos, rot = fk.forward(qv)
    p, R = pos[f"foot_{side}"], rot[f"foot_{side}"]
    lows = [p[2] + (R @ e)[2] for cap in FOOT_CAPS for e in cap]
    return float(min(lows)) - FOOT_CAPS_R


def _ik_plant(fk: RobotFK, root_pos, root_quat, q: Dict[str, float],
              side: str, target: float, reg: float, max_iter: int,
              w_sole: float) -> Tuple[Dict[str, float], float]:
    """Drive ``side`` leg joints so sole z -> target. Returns (q, |resid|)."""
    names = [f"{j}_{side}" for j in LEG_JOINTS]
    x0 = np.array([q[n] for n in names])
    lo = np.array([joint_limits(n)[0] for n in names])
    hi = np.array([joint_limits(n)[1] for n in names])
    x = np.clip(x0, lo + 1e-4, hi - 1e-4)

    def resid(xv):
        qq = dict(q); qq.update(dict(zip(names, xv)))
        s = sole_bottom_z(fk, root_pos, root_quat, qq, side)
        return np.concatenate([[w_sole * (s - target)],
                               reg * (xv - x0)])

    for _ in range(max_iter):
        r = resid(x)
        J = np.zeros((len(r), len(x)))
        for j in range(len(x)):
            xp = x.copy(); xp[j] += 1e-5
            J[:, j] = (resid(xp) - r) / 1e-5
        dx = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(len(x)),
                             -J.T @ r, rcond=None)[0]
        x = np.clip(x + dx, lo, hi)
    qq = dict(q); qq.update(dict(zip(names, x)))
    s = sole_bottom_z(fk, root_pos, root_quat, qq, side)
    return qq, abs(s - target)


def _ik_double(fk: RobotFK, root_pos, root_quat, q: Dict[str, float],
               target: float, dz0: float, reg: float, max_iter: int,
               w_sole: float) -> Tuple[Dict[str, float], float, float]:
    """Double-support solve: dz + both legs -> both soles ~= target.

    Bending the anchored leg lets the pelvis drop while its sole stays
    planted, buying reach for the other leg.  Returns (q, dz, resid).
    """
    names = [f"{j}_{s}" for s in ("left", "right") for j in LEG_JOINTS]
    xq0 = np.array([q[n] for n in names])
    lo = np.array([joint_limits(n)[0] for n in names])
    hi = np.array([joint_limits(n)[1] for n in names])
    xq = np.clip(xq0, lo + 1e-4, hi - 1e-4)
    dz = float(dz0)

    def resid(xv):
        qq = dict(q); qq.update(dict(zip(names, xv[1:])))
        rp = np.asarray(root_pos, dtype=np.float64).copy()
        rp[2] += xv[0]
        sl = sole_bottom_z(fk, rp, root_quat, qq, "left")
        sr = sole_bottom_z(fk, rp, root_quat, qq, "right")
        return np.concatenate([
            [w_sole * (sl - target), w_sole * (sr - target)],
            reg * (xv[1:] - xq0), [0.3 * (xv[0] - dz0)]])

    xv = np.concatenate([[dz], xq])
    for _ in range(max_iter):
        r = resid(xv)
        J = np.zeros((len(r), len(xv)))
        for j in range(len(xv)):
            xp = xv.copy(); xp[j] += 1e-5
            J[:, j] = (resid(xp) - r) / 1e-5
        dx = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(len(xv)),
                             -J.T @ r, rcond=None)[0]
        xv = xv + dx
        xv[1:] = np.clip(xv[1:], lo, hi)
    qq = dict(q); qq.update(dict(zip(names, xv[1:])))
    rp = np.asarray(root_pos, dtype=np.float64).copy(); rp[2] += xv[0]
    sl = sole_bottom_z(fk, rp, root_quat, qq, "left")
    sr = sole_bottom_z(fk, rp, root_quat, qq, "right")
    return qq, xv[0], max(abs(sl - target), abs(sr - target))


def ground_pass(fk: RobotFK, qang: np.ndarray, rpos: np.ndarray,
                rquat: np.ndarray, contacts: Optional[np.ndarray] = None,
                margin: float = 0.005, ik_tol: float = 0.008,
                reg: float = 0.05, max_ik_iter: int = 15,
                w_sole: float = 10.0) -> Dict[str, np.ndarray]:
    """Contact-aware grounding over a whole clip (120Hz arrays).

    ``contacts``: (T,2) 0/1 labels [left,right] in ROBOT frame
    (already mirrored when applicable). ``None`` -> per-frame the lower
    foot is treated as the sole contact.
    Returns dict with corrected qang/rpos + report arrays.
    """
    T = len(qang)
    qang = np.asarray(qang, dtype=np.float64).copy()
    rpos = np.asarray(rpos, dtype=np.float64).copy()
    lift = np.zeros(T)
    ik_resid = np.zeros(T)
    ik_fixed = np.zeros(T, dtype=bool)
    sole_l = np.zeros(T)
    sole_r = np.zeros(T)
    names = list(CONTROLLED_JOINTS)
    for t in range(T):
        q = dict(zip(names, qang[t]))
        sl = sole_bottom_z(fk, rpos[t], rquat[t], q, "left")
        sr = sole_bottom_z(fk, rpos[t], rquat[t], q, "right")
        if contacts is not None:
            cset = [s for s, c in (("left", contacts[t, 0]),
                                   ("right", contacts[t, 1])) if c > 0.5]
        else:
            cset = []
        # safety override: a foot below the floor is physically
        # contacting regardless of the label heuristic
        for side, s in (("left", sl), ("right", sr)):
            if s < -0.5 * ik_tol and side not in cset:
                cset.append(side)
        if len(cset) == 2:
            # double support: solve dz + both legs jointly so the
            # anchored leg can bend deeper, letting the pelvis drop
            # enough for the other leg to reach.
            dz0 = margin - min(sl, sr)
            q2, dz, res = _ik_double(fk, rpos[t], rquat[t], q, margin,
                                     dz0, reg, max_ik_iter, w_sole)
            rpos[t, 2] += dz
            lift[t] = dz
            q = q2
            ik_resid[t] = res
            ik_fixed[t] = res <= ik_tol
        else:
            if not cset:                 # airborne: anchor the lower foot
                cset = ["left" if sl <= sr else "right"]
            anchor = min(sl if s == "left" else sr for s in cset)
            dz = margin - anchor
            rpos[t, 2] += dz
            lift[t] = dz
            sl += dz; sr += dz
            for side, s in (("left", sl), ("right", sr)):
                if side not in cset:
                    continue
                if s - margin > ik_tol:
                    q2, res = _ik_plant(fk, rpos[t], rquat[t], q, side,
                                        margin, reg, max_ik_iter, w_sole)
                    q = q2
                    ik_resid[t] = max(ik_resid[t], res)
                    ik_fixed[t] = ik_fixed[t] or res <= ik_tol
        sl = sole_bottom_z(fk, rpos[t], rquat[t], q, "left")
        sr = sole_bottom_z(fk, rpos[t], rquat[t], q, "right")
        sole_l[t], sole_r[t] = sl, sr
        for i, n in enumerate(names):
            qang[t, i] = q[n]
    return {"joint_angles": qang, "root_pos": rpos,
            "report": {"lift": lift, "ik_resid": ik_resid,
                       "ik_fixed": ik_fixed,
                       "sole_left": sole_l, "sole_right": sole_r}}


def repair_npz(npz: Dict[str, np.ndarray],
               fk: Optional[RobotFK] = None, **kw) -> Dict[str, np.ndarray]:
    """Apply ground_pass to a joint_space=humanoid21_v2 clip npz dict.

    Uses npz['foot_contacts'] when present (already robot-frame).
    Recomputes joint_velocities, root_linvel and keypoints so all
    derived fields stay consistent. Returns a NEW dict.
    """
    from .robot_kin import fk_positions    # keypoint FK used by export
    from .export_library import ROBOT_KEYPOINTS
    if str(npz["joint_names"][0]) not in CONTROLLED_JOINTS:
        raise ValueError("npz is not humanoid21_v2 joint space")
    fk = fk or RobotFK()
    dt = float(np.median(np.diff(npz["timestamps"])))
    contacts = npz["foot_contacts"] if "foot_contacts" in npz else None
    out = ground_pass(fk, np.asarray(npz["joint_angles"], np.float64),
                      np.asarray(npz["root_pos"], np.float64),
                      np.asarray(npz["root_quat"], np.float64),
                      contacts=contacts, **kw)
    qang, rpos = out["joint_angles"], out["root_pos"]
    T = len(qang)
    npz = dict(npz)
    npz["root_pos"] = rpos.astype(np.float32)
    npz["root_linvel"] = np.gradient(rpos, dt, axis=0).astype(np.float32)
    npz["joint_angles"] = qang.astype(np.float32)
    npz["joint_velocities"] = np.gradient(qang, dt, axis=0) \
        .astype(np.float32)
    kp = np.zeros((T, len(ROBOT_KEYPOINTS), 3), np.float32)
    names = list(CONTROLLED_JOINTS)
    for t in range(T):
        pos, _ = fk_positions(dict(zip(names, qang[t])),
                              rpos[t], npz["root_quat"][t])
        for i, b in enumerate(ROBOT_KEYPOINTS):
            kp[t, i] = pos[b]
    npz["keypoints"] = kp
    npz["_ground_report"] = out["report"]
    return npz
