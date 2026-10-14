"""M1-W3: retarget CMU source motion to humanoid21 joint angles.

Method: world-frame matching with POSITION objectives for end-effectors and
orientation objectives for frames — never dof-copy (robot has mixed joint
axes, reduced arm/ankle DOFs, different proportions).

Per-frame solve order:
  1. torso quat   := C @ F[thorax] @ C.T   (human thorax frame -> robot chest)
  2. abdomen z,y,x := euler solve of R_torso.T @ R_pelvis*,  pelvis* = C·F[root]·Cᵀ
  3. legs: hip x,z,y + knee — bounded LM matching
       (a) ankle position rel. hip joint (scaled source, 3 dims, w=1.0)
       (b) shin direction (3 dims, w=0.4 — fixes hip twist / knee hinge)
       (c) soft hamstring-tendon hinge penalty (keeps 0.5*hip_y-0.5*knee >= -0.3)
     then ankle y,x — LM matching foot direction (heel->toe)
  4. arms: shoulder1,2 + elbow — bounded LM matching hand position rel.
     shoulder (scaled, 3 dims, w=1.0). The 3-dof arm vs 3-dim target is a
     square system; a direction prior was tried and pinned the elbow at its
     limit — warm start + temporal prior resolves the hinge plane instead.

Root position: scaled CMU root (pelvis) position; robot torso position is
back-computed via analytic FK so the robot pelvis lands exactly on target.
A single per-clip vertical shift grounds feet (min sole ~ 0).

All solves warm-start from previous frame, bounded by joint limits;
clipped joints count as kinematic violations downstream.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baseline.humanoid21.mocap.asf_parser import parse_asf
from baseline.humanoid21.mocap.amc_parser import parse_amc
from mocap.humanoid21.calibrate import cmu_pos_to_mj, cmu_rot_to_mj
from mocap.humanoid21.ref_fk import RefSkeleton, fk_frame
from mocap.humanoid21.robot_kin import (
    BONE_DIR_LOCAL, CONTROLLED_JOINTS, RobotFK, body_world_rot, fk_positions,
    joint_limits,
)

CAL = json.loads((Path(__file__).parent / "calibration.json").read_text())
SCALE = CAL["meters_per_cmu_unit"]

SRC_THIGH = {"left": "lfemur", "right": "rfemur"}
SRC_SHIN = {"left": "ltibia", "right": "rtibia"}
SRC_ANKLE = {"left": "lfoot", "right": "rfoot"}     # ankle joint position
SRC_UPPERARM = {"left": "lhumerus", "right": "rhumerus"}
SRC_FOREARM = {"left": "lradius", "right": "rradius"}
SRC_HAND = {"left": "lhand", "right": "rhand"}      # fist keypoint (joint origin)

W_POS = 1.0          # end-effector relative-position weight (per axis)
W_DIR_LEG = 0.4      # shin-direction regularizer (resolves hip twist)
W_DIR_ARM = 0.0      # upper-arm-dir prior OFF: 3-dof arm vs 3-dim hand target
                     # is square; direction prior conflicts and pins the elbow
                     # at its limit. Warm start resolves the hinge plane.
W_SMOOTH = 0.15      # temporal prior: penalize per-frame joint deviation (rad)
                     # — pins branch selection, prevents elbow/ankle flips
DQ_MAX = 0.35        # rad/frame slew limit at 120 Hz (~2400 deg/s): above any
                     # real joint speed; prevents IK basin flips from creating
                     # physically impossible discontinuities


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _abdomen_angles(R_torso: np.ndarray, R_pelvis: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R_torso.T @ R_pelvis).as_euler("xyz")


@dataclass
class FrameResult:
    q: Dict[str, float]
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    residuals: Dict[str, float]


class Retargeter:
    def __init__(self, warm_start: bool = True):
        self.fk = RobotFK()
        self.warm = warm_start
        self.prev_q: Dict[str, float] = {}
        self._qvec = self.fk.q0.copy()

    # ---- shared eval: analytic FK (verified vs mj_forward, ~30x faster) ----
    def _eval_positions(self, q, root_quat_wxyz, root_pos=None, needed=None):
        return fk_positions(q, np.zeros(3) if root_pos is None else root_pos,
                            root_quat_wxyz, needed)

    @staticmethod
    def _lm(resid_fn, x0, lo, hi, max_iter=12, eps=1e-6):
        """Bounded damped Gauss-Newton: cheap replacement for scipy
        least_squares (whose per-call overhead dominates at 4 dof)."""
        x = np.clip(np.asarray(x0, float), lo, hi)
        r = resid_fn(x)
        cost = r @ r
        lam = 1e-3
        n = len(x)
        for _ in range(max_iter):
            J = np.empty((r.size, n))
            for k in range(n):
                xk = x.copy(); xk[k] += eps
                J[:, k] = (resid_fn(xk) - r) / eps
            A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9)
            try:
                dx = np.linalg.solve(A, -J.T @ r)
            except np.linalg.LinAlgError:
                break
            xn = np.clip(x + dx, lo, hi)
            rn = resid_fn(xn)
            cn = rn @ rn
            if cn < cost:
                if cost - cn < 1e-10:
                    x, r, cost = xn, rn, cn
                    break
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-7)
            else:
                lam *= 5
                if lam > 1e4:
                    break
        return x, float(np.sqrt(cost))

    def _lsq(self, joints: List[str], resid_fn, q0: Dict[str, float],
             extra_starts: Optional[List[Dict[str, float]]] = None,
             retry_thresh: float = 0.03) -> Tuple[Dict, float]:
        """Bounded LM solve, warm-started from q0. If residual stays above
        retry_thresh and extra_starts are given, retry from each and keep the
        lowest-cost solution (escapes elbow/hip boundary local minima)."""
        lo = np.array([joint_limits(j)[0] for j in joints])
        hi = np.array([joint_limits(j)[1] for j in joints])
        xp = np.array([q0.get(j, 0.0) for j in joints])
        ws = [W_SMOOTH]

        def make_fn(w):
            def smooth_fn(x):
                x = np.asarray(x, float)
                return np.concatenate([resid_fn(x), w * (x - xp)])
            return smooth_fn

        best_x, best_f = self._lm(make_fn(ws[0]), xp, lo, hi)
        # branch-flip guard: a big jump means the solver hopped to an
        # alternate IK basin — re-solve with a much stronger temporal prior
        if np.max(np.abs(best_x - xp)) > 0.5:
            x2, f2 = self._lm(make_fn(ws[0] * 5.0), xp, lo, hi)
            if np.linalg.norm(resid_fn(x2)) < 0.5:
                best_x, best_f = x2, f2
        if best_f > retry_thresh and extra_starts:
            for st in extra_starts:
                x, f = self._lm(make_fn(ws[0]),
                                [st.get(j, q0.get(j, 0.0)) for j in joints],
                                lo, hi)
                if f < best_f:
                    best_x, best_f = x, f
        # report geometric residual only (smoothness penalty excluded)
        rf = resid_fn(best_x)
        return dict(zip(joints, best_x)), float(np.linalg.norm(rf))

    def retarget_frame(self, joint_cmu, distal_cmu, F_cmu) -> FrameResult:
        q = dict(self.prev_q) if self.warm else {}
        res: Dict[str, float] = {}

        # 1) torso quat <- human thorax
        R_torso = cmu_rot_to_mj(F_cmu["thorax"])
        rq = Rotation.from_matrix(R_torso).as_quat()
        root_quat = np.array([rq[3], rq[0], rq[1], rq[2]])

        # 2) abdomen <- human pelvis frame
        R_pelvis_t = cmu_rot_to_mj(F_cmu["root"])
        ex, ey, ez = _abdomen_angles(R_torso, R_pelvis_t)
        for name, val in (("abdomen_x", ex), ("abdomen_y", ey), ("abdomen_z", ez)):
            lo, hi = joint_limits(name)
            q[name] = float(np.clip(val, lo, hi))

        # 3) legs
        for side in ("right", "left"):
            d_shin = _unit(cmu_pos_to_mj(
                distal_cmu[SRC_SHIN[side]] - joint_cmu[SRC_SHIN[side]]))
            # ankle rel hip (scaled human) as position target
            p_rel_t = cmu_pos_to_mj(
                joint_cmu[SRC_ANKLE[side]] - joint_cmu[SRC_THIGH[side]]) * SCALE
            joints = [f"hip_x_{side}", f"hip_z_{side}", f"hip_y_{side}", f"knee_{side}"]

            leg_need = {f"foot_{side}", f"thigh_{side}", f"shin_{side}"}

            def leg_resid(x, sd=side, pr=p_rel_t, ds=d_shin, q0=dict(q),
                          rquat=root_quat, need=leg_need):
                qq = dict(q0); qq.update(dict(zip(joints, x)))
                pos, _ = self._eval_positions(qq, rquat, needed=need)
                ankle_rel = pos[f"foot_{sd}"] - pos[f"thigh_{sd}"]
                shin_dir = _unit(pos[f"foot_{sd}"] - pos[f"shin_{sd}"])
                # hamstring fixed tendon: 0.5*hip_y - 0.5*knee >= -0.3 rad
                # soft hinge penalty (margin 0.02) keeps solution feasible
                tendon = 0.5 * x[2] - 0.5 * x[3]
                t_pen = max(0.0, -(tendon + 0.28)) * 3.0
                return np.concatenate([
                    W_POS * (ankle_rel - pr),
                    W_DIR_LEG * (shin_dir - ds),
                    [t_pen]])

            qq, err = self._lsq(joints, leg_resid, q)
            q.update(qq)
            res[f"leg_{side}"] = err

            # ankle: foot direction (heel->toe)
            toes = "ltoes" if side == "left" else "rtoes"
            foot_dir = _unit(cmu_pos_to_mj(
                distal_cmu[toes] - joint_cmu[SRC_ANKLE[side]]))
            aj = [f"ankle_y_{side}", f"ankle_x_{side}"]

            def ank_resid(x, sd=side, df=foot_dir, q0=dict(q)):
                qq = dict(q0); qq.update(dict(zip(aj, x)))
                return (body_world_rot(f"foot_{sd}", qq, R_torso)
                        @ BONE_DIR_LOCAL[f"foot_{sd}"] - df)

            qq, err = self._lsq(aj, ank_resid, q)
            q.update(qq)
            res[f"ankle_{side}"] = err

        # 4) arms
        for side in ("right", "left"):
            d_uarm = _unit(cmu_pos_to_mj(
                distal_cmu[SRC_UPPERARM[side]] - joint_cmu[SRC_UPPERARM[side]]))
            # hand rel shoulder anchor (scaled)
            p_rel_t = cmu_pos_to_mj(
                joint_cmu[SRC_HAND[side]] - joint_cmu[SRC_UPPERARM[side]]) * SCALE
            joints = [f"shoulder1_{side}", f"shoulder2_{side}", f"elbow_{side}"]

            arm_need = {f"hand_{side}", f"upper_arm_{side}",
                        f"lower_arm_{side}"}

            def arm_resid(x, sd=side, pr=p_rel_t, du=d_uarm, q0=dict(q),
                          rquat=root_quat, need=arm_need):
                qq = dict(q0); qq.update(dict(zip(joints, x)))
                pos, _ = self._eval_positions(qq, rquat, needed=need)
                hand_rel = pos[f"hand_{sd}"] - pos[f"upper_arm_{sd}"]
                return np.concatenate([
                    W_POS * (hand_rel - pr),
                    W_DIR_ARM * (_unit(pos[f"lower_arm_{sd}"] - pos[f"upper_arm_{sd}"]) - du)])

            extra = [{f"elbow_{side}": e} for e in (-90.0 * np.pi / 180,
                                                  -20.0 * np.pi / 180)]
            qq, err = self._lsq(joints, arm_resid, q, extra_starts=extra)
            q.update(qq)
            res[f"arm_{side}"] = err

        # slew-rate limit: never allow >DQ_MAX rad change per joint per frame
        if self.prev_q:
            for n, v in q.items():
                dv = v - self.prev_q.get(n, 0.0)
                q[n] = self.prev_q.get(n, 0.0) + float(np.clip(dv, -DQ_MAX, DQ_MAX))
        self.prev_q = q

        # 5) root pos: robot pelvis lands on scaled human pelvis
        pelvis_target = cmu_pos_to_mj(joint_cmu["root"]) * SCALE
        pos, _ = self._eval_positions(q, root_quat, needed={"pelvis"})
        root_pos = pelvis_target - pos["pelvis"]
        return FrameResult(q=q, root_pos=root_pos, root_quat_wxyz=root_quat,
                           residuals=res)


def retarget_clip(asf_path: str, amc_path: str) -> dict:
    skel = RefSkeleton.from_asf(parse_asf(asf_path))
    frames = parse_amc(amc_path)
    rt = Retargeter()

    T = len(frames)
    qang = np.zeros((T, len(CONTROLLED_JOINTS)))
    rpos = np.zeros((T, 3))
    rquat = np.zeros((T, 4))
    res_keys = ["leg_left", "leg_right", "ankle_left", "ankle_right",
                "arm_left", "arm_right"]
    resid = {k: np.zeros(T) for k in res_keys}
    for t, fr in enumerate(frames):
        j, d, F = fk_frame(skel, fr)
        r = rt.retarget_frame(j, d, F)
        rpos[t] = r.root_pos
        rquat[t] = r.root_quat_wxyz
        for i, n in enumerate(CONTROLLED_JOINTS):
            qang[t, i] = r.q.get(n, 0.0)
        for k in res_keys:
            resid[k][t] = r.residuals[k]

    # Leading-edge repair: frame 0 is solved from the default standing
    # seed and can land on a wrong IK branch; the DQ_MAX slew clamp then
    # smears the correction over the next frames (~50 rad/s fake
    # velocities at the clip start, fatal for RSI init). Detect the
    # transient and re-solve the window BACKWARD, warm-starting from the
    # converged interior solution.
    qmax = np.abs(np.diff(qang, axis=0)).max(axis=1)
    t_end = 0
    for t in range(1, min(240, T - 4)):
        if np.all(qmax[t:t + 4] < 0.08):
            t_end = t
            break
    if t_end >= 1:
        bwd = Retargeter()
        bwd.prev_q = dict(zip(CONTROLLED_JOINTS, qang[t_end]))
        for t in range(t_end - 1, -1, -1):
            j, d, F = fk_frame(skel, frames[t])
            r = bwd.retarget_frame(j, d, F)
            rpos[t] = r.root_pos          # root_pos depends on q
            for i, n in enumerate(CONTROLLED_JOINTS):
                qang[t, i] = r.q.get(n, 0.0)
            for k in res_keys:
                resid[k][t] = r.residuals[k]

    # grounding: uniform z shift so min sole ~ 0
    sole_min = np.inf
    for t in range(0, T, 4):
        qvec = rt.fk.qvec(rpos[t], rquat[t], dict(zip(CONTROLLED_JOINTS, qang[t])))
        pos, _ = rt.fk.forward(qvec)
        sole = min(pos["foot_left"][2], pos["foot_right"][2]) - 0.027
        sole_min = min(sole_min, sole)
    rpos[:, 2] -= sole_min - 0.005
    return {"root_pos": rpos, "root_quat": rquat, "joint_angles": qang,
            "residuals": resid}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("asf"); ap.add_argument("amc")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    skel = RefSkeleton.from_asf(parse_asf(args.asf))
    frames = parse_amc(args.amc)
    rt = Retargeter()
    for t in range(args.frame, args.frame + args.n):
        j, d, F = fk_frame(skel, frames[t])
        r = rt.retarget_frame(j, d, F)
        print(f"frame {t}: root_pos={np.round(r.root_pos,3)} quat={np.round(r.root_quat_wxyz,3)}")
        for n in CONTROLLED_JOINTS:
            print(f"  {n:18s} {r.q.get(n,0)*180/np.pi:8.2f} deg")
        print("  residuals:", {k: round(v, 4) for k, v in r.residuals.items()})
