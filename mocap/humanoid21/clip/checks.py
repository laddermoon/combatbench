"""Unified CoreClip check sections.

Four sections, each returning a SectionResult:
  basic      — format validity, ground/self penetration, contact sanity
  velocity   — recorded velocities vs trajectory-implied velocities
               (centered finite differences + accel-spike seam detection)
  actuator   — inverse-dynamics demand screen vs PD/actuator capacity
  ref_action — forward replay of the recorded action sequence

Only ``ref_action`` (replay) is a true executability test; the others
are screens.  All checks are read-only on the clip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from .check import geom_bottom_z, ground_report, _foot_geom_names, \
    GROUND_GEOM
from .core_clip import CONTROL_DT, CORE_FIELDS, CoreClip

DT = CONTROL_DT


# ------------------------------------------------------------------
@dataclass
class Finding:
    level: str                       # INFO / WARN / FAIL
    msg: str
    frame: Optional[int] = None
    fix: Optional[str] = None        # suggested repair command


@dataclass
class SectionResult:
    name: str
    status: str = "PASS"             # PASS / WARN / FAIL / SKIP
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add(self, level, msg, frame=None, fix=None):
        self.findings.append(Finding(level, msg, frame, fix))
        order = {"PASS": 0, "SKIP": 0, "INFO": 0,
                 "WARN": 1, "FAIL": 2}
        if order[level] > order[self.status]:
            self.status = level


# ------------------------------------------------------------------
def _mk_sim(distance: float = 6.0):
    from envs.humanoid21.simulator import Humanoid21Simulator
    return Humanoid21Simulator(initial_distance=distance,
                               initial_pose_b="standing")


def _rot(q_wxyz: np.ndarray) -> np.ndarray:
    """wxyz -> 3x3."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_log_wxyz(q: np.ndarray) -> np.ndarray:
    """Log map of a unit wxyz quaternion -> rotation vector."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    v = q[1:4]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return 2.0 * v
    ang = 2.0 * np.arctan2(s, np.clip(q[0], -1.0, 1.0))
    return v / s * ang


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ==================================================================
# Section 1: basic state sanity
# ==================================================================
def check_basic(clip: CoreClip, robot: str, sim) -> SectionResult:
    r = SectionResult("basic")
    try:
        warns = clip.validate()
        for w in warns:
            r.add("WARN", f"validate: {w}")
    except ValueError as e:
        r.add("FAIL", f"validate: {e}")
        return r

    model, data = sim.model, sim.data
    names = _foot_geom_names(robot)
    foot_gids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in names}
    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   GROUND_GEOM)
    # all geoms of this robot (body prefix _a / _b)
    suffix = "_a" if robot.endswith("a") else "_b"
    robot_gids = {g for g in range(model.ngeom)
                  if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                  and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                  .endswith(suffix)}

    pen = np.zeros(clip.T)
    other_pen = np.zeros(clip.T)
    self_col = np.zeros(clip.T)
    expect_no_contact = np.zeros(clip.T)
    tilt_deg = np.zeros(clip.T)
    root_z = np.zeros(clip.T)

    for k in range(clip.T):
        sim.set_core_state({robot: clip.state(k)})
        root_z[k] = clip.states["root_pos"][k, 2]
        up = _rot(clip.states["root_rot"][k])[:, 2]
        tilt_deg[k] = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
        lows = {}
        for g in robot_gids:
            z = geom_bottom_z(model, data, g)
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            lows[g] = (z, nm)
        fmin = min(lows[g][0] for g in foot_gids)
        omin = min(v[0] for g, v in lows.items() if g not in foot_gids)
        pen[k] = max(0.0, -fmin)
        other_pen[k] = max(0.0, -omin)
        # solver contacts: self-penetration & non-foot ground contact
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 in robot_gids and g2 in robot_gids and c.dist < -1e-3:
                self_col[k] = max(self_col[k], -float(c.dist))
            if ((g1 == ground_gid and g2 in robot_gids - foot_gids) or
                    (g2 == ground_gid and g1 in robot_gids - foot_gids)):
                expect_no_contact[k] += 1

    r.metrics = {
        "max_foot_penetration_m": float(pen.max()),
        "max_other_geom_penetration_m": float(other_pen.max()),
        "max_self_penetration_m": float(self_col.max()),
        "max_tilt_deg": float(tilt_deg.max()),
        "root_z_range": [float(root_z.min()), float(root_z.max())],
        "nonfoot_ground_contacts": int(expect_no_contact.sum()),
    }
    worst = int(np.argmax(pen))
    if pen.max() > 0.002:
        r.add("FAIL",
              f"foot penetration max {pen.max() * 100:.1f}cm",
              worst,
              fix="clip fix --clip <file> --out <file>_groundfix.npz "
                  "--mode lowest")
    if other_pen.max() > 0.002:
        r.add("FAIL",
              f"non-foot geom penetrates ground "
              f"{other_pen.max() * 100:.1f}cm",
              int(np.argmax(other_pen)),
              fix="no auto-fix — pose edit needed")
    if self_col.max() > 0.002:
        r.add("WARN",
              f"self-collision depth {self_col.max() * 100:.1f}cm",
              int(np.argmax(self_col)))
    if expect_no_contact.sum() > 0:
        r.add("WARN",
              f"non-foot geoms touch ground in "
              f"{int(expect_no_contact.sum())} contacts")
    if tilt_deg.max() > 60:
        r.add("WARN", f"torso tilted {tilt_deg.max():.0f}deg "
                      f"(fallen-looking pose)",
              int(np.argmax(tilt_deg)))
    if not (0.2 < root_z.min() and root_z.max() < 1.9):
        r.add("WARN",
              f"root z range {root_z.min():.2f}..{root_z.max():.2f}m "
              f"outside [0.2,1.9]")
    return r


# ==================================================================
# Section 2: velocity consistency
# ==================================================================
def check_velocity(clip: CoreClip) -> SectionResult:
    r = SectionResult("velocity")
    T = clip.T
    if T < 3:
        r.add("INFO", "clip too short for interior velocity check")
        return r

    jp = clip.states["joint_pos_norm"].astype(np.float64)
    jv = clip.states["joint_vel_norm"].astype(np.float64)
    rp = clip.states["root_pos"].astype(np.float64)
    rq = clip.states["root_rot"].astype(np.float64)
    rv = clip.states["root_vel_local"].astype(np.float64)
    ra = clip.states["root_angular_vel_local"].astype(np.float64)

    # implied velocities, interior centered difference
    jv_impl = (jp[2:] - jp[:-2]) / (2.0 * DT)
    jv_res = np.abs(jv_impl - jv[1:-1])          # (T-2,21)

    v_world_impl = (rp[2:] - rp[:-2]) / (2.0 * DT)
    rv_impl = np.stack([
        _rot(rq[k]).T @ v_world_impl[k - 1] for k in range(1, T - 1)])
    rv_res = np.linalg.norm(rv_impl - rv[1:-1], axis=1)

    # local angular velocity implied by k-1 -> k+1 rotation, expressed
    # in frame-k body coordinates
    ra_impl = np.stack([
        _rot(rq[k]).T @ (_quat_log_wxyz(
            _quat_mul_wxyz(_quat_inv_wxyz(rq[k - 1]), rq[k + 1]))
            / (2.0 * DT))
        for k in range(1, T - 1)])
    ra_res = np.linalg.norm(ra_impl - ra[1:-1], axis=1)

    # acceleration seams (joint velocity jumps)
    # TODO: add a dedicated pose-smoothness metric — jerk outliers on
    # joint_pos itself (independent of recorded velocity) would catch
    # noisy poses before rederive-vel amplifies them.
    ja = (jv[1:] - jv[:-1]) / DT                 # (T-1,21)
    ja_mag = np.abs(ja).max(axis=1)
    p95 = np.percentile(ja_mag, 95)
    seam_frames = [int(k + 1) for k in
                   np.nonzero(ja_mag > max(3 * np.median(ja_mag),
                                          p95 * 1.5, 1.0))[0]]

    jmax_j = int(np.unravel_index(np.argmax(jv_res), jv_res.shape)[1])
    jmax_k = int(np.unravel_index(np.argmax(jv_res), jv_res.shape)[0]) + 1
    r.metrics = {
        "joint_vel_res_max": float(jv_res.max()),
        "joint_vel_res_worst_joint": int(jmax_j),
        "root_linvel_res_max": float(rv_res.max()),
        "root_angvel_res_max": float(ra_res.max()),
        "accel_seam_frames": seam_frames,
        "first_frame_vel": {"root_vel_local": rv[0].tolist(),
                            "root_angvel_local": ra[0].tolist()},
        "last_frame_vel": {"root_vel_local": rv[-1].tolist(),
                           "root_angvel_local": ra[-1].tolist()},
    }
    # sign disagreement: recorded & implied both moving but opposite
    # directions — a hard inconsistency independent of downsampling
    # aliasing (residual same-direction can come from 120Hz->20Hz
    # subsampling, which is informational only)
    rec_dir = np.sign(jv[1:-1])
    imp_dir = np.sign(jv_impl)
    sign_dis = int(((rec_dir != imp_dir) & (np.abs(jv[1:-1]) > 0.3) &
                    (np.abs(jv_impl) > 0.3)).sum())
    r.metrics["vel_sign_disagreements"] = sign_dis

    jn = "joint#%d" % jmax_j
    if sign_dis:
        r.add("FAIL",
              f"{sign_dis} (frame,joint) slots have recorded velocity "
              f"opposing the position trajectory (worst k={jmax_k} "
              f"{jn})", jmax_k,
              fix="clip rederive-vel --clip <file> "
                  "--out <file>_vel.npz")
    elif jv_res.max() > 0.8:
        r.add("WARN",
              f"joint velocity residual {jv_res.max():.2f} norm/s at "
              f"k={jmax_k} ({jn}) — consistent with downsampling "
              f"aliasing or noisy source", jmax_k)
    elif jv_res.max() > 0.3:
        r.add("WARN", f"joint velocity residual {jv_res.max():.2f} "
                      f"norm/s at k={jmax_k} ({jn})", jmax_k)
    if rv_res.max() > 0.5:
        r.add("WARN", f"root linvel residual {rv_res.max():.2f} m/s",
              int(np.argmax(rv_res)) + 1)
    if ra_res.max() > 4.0:
        r.add("WARN", f"root angvel residual {ra_res.max():.2f} rad/s",
              int(np.argmax(ra_res)) + 1)
    if seam_frames:
        r.add("WARN",
              f"acceleration seams at frames {seam_frames} — "
              f"per-frame independent edits suspected")
    return r


# ==================================================================
# Section 3: actuator demand screen (inverse dynamics, no training)
# ==================================================================
def check_actuator(clip: CoreClip, robot: str, sim) -> SectionResult:
    """RNE inverse-dynamics demand screen.

    mj_rne gives the generalized force needed to produce the recorded
    acceleration ignoring contact constraint forces — an upper bound
    for contact-supported joints, exact for free segments.  This is a
    screen, not an executability proof.
    """
    r = SectionResult("actuator")
    T = clip.T
    model, data = sim.model, sim.data
    cache = sim._robot(robot)
    qvel_idx = np.asarray(cache["qvel_indices"])
    root_qvel_adr = int(cache["root_qvel_adr"])
    pd = sim._pd_tables[robot]
    kp = np.asarray(sim.KP, np.float64)
    kd = np.asarray(sim.KD, np.float64)
    ref = np.asarray(sim._norm_params[robot]["reference"], np.float64)
    scale = np.asarray(sim._norm_params[robot]["scale"], np.float64)
    cap = np.maximum(np.abs(pd["ctrl_lo"]), np.abs(pd["ctrl_hi"])) \
        * np.abs(pd["gear"])
    rne = np.zeros(model.nv)

    # collect raw qvel per frame (set_core_state writes rad units)
    qvels = np.zeros((T, model.nv))
    for k in range(T):
        sim.set_core_state({robot: clip.state(k)})
        qvels[k] = data.qvel.copy()

    util = np.zeros((T, 21))          # |required torque| / capacity
    a_req = np.zeros((T, 21))         # implied normalized PD action
    root_wrench = np.zeros((T, 2))    # |force|, |torque| at root dofs
    for k in range(T):
        sim.set_core_state({robot: clip.state(k)})
        if T > 1:
            k0, k1 = max(0, k - 1), min(T - 1, k + 1)
            data.qacc[:] = (qvels[k1] - qvels[k0]) / ((k1 - k0) * DT)
        else:
            data.qacc[:] = 0.0
        mujoco.mj_rne(model, data, 1, rne)  # flg_acc=1: use data.qacc
        tau = rne[qvel_idx]
        util[k] = np.abs(tau) / cap
        # PD target needed to supply tau at current state:
        jv_rad = data.qvel[qvel_idx]
        jp_rad = data.qpos[np.asarray(cache["qpos_indices"])]
        err = (tau + kd * jv_rad) / kp
        a_req[k] = (jp_rad + err - ref) / scale
        root_wrench[k] = [np.linalg.norm(
            rne[root_qvel_adr:root_qvel_adr + 3]),
            np.linalg.norm(
                rne[root_qvel_adr + 3:root_qvel_adr + 6])]

    maxu = util.max(axis=0)
    over80 = sorted(set(np.nonzero(util > 0.8)[0].tolist()))
    over100 = sorted(set(np.nonzero(util > 1.0)[0].tolist()))
    a_over = int((np.abs(a_req) > 1.0).sum())
    r.metrics = {
        "joint_util_max_pct": {int(j): round(float(maxu[j] * 100))
                               for j in np.argsort(-maxu)[:5]},
        "frames_over_80pct": len(over80),
        "frames_over_100pct": len(over100),
        "pd_action_over_range_count": a_over,
        "root_wrench_force_max": float(root_wrench[:, 0].max()),
        "root_wrench_torque_max": float(root_wrench[:, 1].max()),
    }
    if over100:
        r.add("FAIL",
              f"required torque exceeds actuator capacity (>100%) "
              f"in {len(over100)} frames",
              int(np.argmax(util.max(axis=1))))
    elif over80:
        r.add("WARN",
              f"joints above 80% torque capacity in {len(over80)} "
              f"frames",
              int(np.argmax(util.max(axis=1))))
    if a_over:
        r.add("WARN",
              f"PD-feasibility: implied normalized action exceeds "
              f"[-1,1] in {a_over} (frame,joint) slots")
    r.add("INFO",
          "screen only: RNE demand ignores contact support "
          "(upper-bound for stance joints, exact for free segments); "
          "definitive test is ref_action replay")
    return r


# ==================================================================
# Section 4: ref_action forward replay
# ==================================================================
def check_ref_action(clip: CoreClip, robot: str, sim) -> SectionResult:
    r = SectionResult("ref_action")
    if clip.ref_action is None:
        r.status = "SKIP"
        r.add("INFO", "clip has no ref_action")
        return r
    other = "robot_b" if robot == "robot_a" else "robot_a"
    # park the other robot far away, hold its standing state
    stand_state = {f: v.copy() for f, v in
                   sim.get_core_state()[other].items()}
    stand_state["root_pos"][0] += 4.0

    sim.set_core_state({robot: clip.state(0), other: stand_state})
    T = clip.T
    jerr = np.zeros(T)
    jverr = np.zeros(T)
    rerr = np.zeros(T)
    rqerr = np.zeros(T)
    rverr = np.zeros(T)
    raerr = np.zeros(T)
    diverge_k = None
    fall_k = None
    for k in range(T - 1):
        sim.set_core_state({other: stand_state})  # keep B parked
        sim.set_action({robot: clip.ref_action[k],
                        other: np.zeros(21, np.float32)})
        for _ in range(25):
            sim.physical_step()
        st = sim.get_core_state()[robot]
        ref = clip.state(k + 1)
        jerr[k + 1] = float(np.abs(
            st["joint_pos_norm"] - ref["joint_pos_norm"]).mean())
        jverr[k + 1] = float(np.abs(
            st["joint_vel_norm"] - ref["joint_vel_norm"]).mean())
        rerr[k + 1] = float(np.linalg.norm(
            st["root_pos"] - ref["root_pos"]))
        q_rel = _quat_mul_wxyz(
            _quat_inv_wxyz(ref["root_rot"]), st["root_rot"])
        rqerr[k + 1] = float(np.linalg.norm(_quat_log_wxyz(q_rel)))
        rverr[k + 1] = float(np.linalg.norm(
            st["root_vel_local"] - ref["root_vel_local"]))
        raerr[k + 1] = float(np.linalg.norm(
            st["root_angular_vel_local"] -
            ref["root_angular_vel_local"]))
        if diverge_k is None and jerr[k + 1] > 0.3:
            diverge_k = k + 1
        z = st["root_pos"][2]
        if fall_k is None and z < 0.7:
            fall_k = k + 1

    r.metrics = {
        "joint_err_mean_max": float(jerr.max()),
        "joint_err_final": float(jerr[-1]),
        "joint_vel_err_mean_max": float(jverr.max()),
        "root_pos_err_max": float(rerr.max()),
        "root_rot_err_max": float(rqerr.max()),
        "root_vel_err_max": float(rverr.max()),
        "root_angvel_err_max": float(raerr.max()),
        "divergence_frame": diverge_k,
        "fall_frame": fall_k,
    }
    if fall_k is not None:
        r.add("FAIL", f"replay falls at k={fall_k}", fall_k,
              fix="clip rederive-action --clip <file> "
                  "--out <file>_act.npz")
    elif diverge_k is not None:
        r.add("FAIL",
              f"replay diverges at k={diverge_k} "
              f"(joint err {jerr[diverge_k]:.2f})", diverge_k,
              fix="clip rederive-action --clip <file> "
                  "--out <file>_act.npz")
    elif jerr.max() > 0.1:
        r.add("WARN", f"replay tracking err max {jerr.max():.2f} norm",
              int(np.argmax(jerr)),
              fix="clip rederive-action --clip <file> "
                  "--out <file>_act.npz")
    else:
        r.add("INFO", f"replay tracks clip (max joint err "
                      f"{jerr.max():.3f} norm)")
    return r


# ==================================================================
def run_checks(clip: CoreClip, robot: str = "robot_a",
               only: Optional[List[str]] = None,
               distance: float = 6.0) -> Dict[str, SectionResult]:
    sections = only or ["basic", "velocity", "actuator", "ref_action"]
    out: Dict[str, SectionResult] = {}
    sim = None
    need_sim = [s for s in sections if s != "velocity"]
    if need_sim:
        sim = _mk_sim(distance)
    for s in sections:
        if s == "basic":
            out[s] = check_basic(clip, robot, sim)
        elif s == "velocity":
            out[s] = check_velocity(clip)
        elif s == "actuator":
            out[s] = check_actuator(clip, robot, sim)
        elif s == "ref_action":
            out[s] = check_ref_action(clip, robot, sim)
        else:
            raise ValueError(f"unknown check section {s!r}")
    if sim is not None:
        sim.close()
    return out


def report_text(clip: CoreClip,
                results: Dict[str, SectionResult]) -> str:
    lines = [f"=== {clip.clip_id}  T={clip.T} "
             f"({clip.duration_s:.2f}s) ==="]
    for name, r in results.items():
        lines.append(f"[{name:>10s}] {r.status}")
        for f in r.findings:
            loc = f" @k{f.frame}" if f.frame is not None else ""
            lines.append(f"             {f.level:4s} {f.msg}{loc}")
            if f.fix:
                lines.append(f"                  → {f.fix}")
    return "\n".join(lines)


def report_json(clip: CoreClip,
                results: Dict[str, SectionResult]) -> str:
    return json.dumps({
        "clip_id": clip.clip_id, "T": clip.T,
        "sections": {
            n: {"status": r.status, "metrics": r.metrics,
                "findings": [{"level": f.level, "msg": f.msg,
                              "frame": f.frame, "fix": f.fix}
                             for f in r.findings]}
            for n, r in results.items()},
    }, indent=2, default=str)
