"""M1-W2/W3: export the minimal motion set to the library layout.

For each of the 17 minimal-set clips this produces TWO npz files under
<library>/clips/:

  cmu_<subj>_<trial>_src.npz   — joint_space "cmu_asf": source skeleton
      (pelvis root pos/quat, per-bone AMC dof values, all joint keypoints,
      velocities, foot contacts) in MuJoCo world frame (z-up, meters).
  cmu_<subj>_<trial>.npz       — joint_space "humanoid21_v2": retargeted
      reference (21 joint angles, root pos/quat, keypoints, contacts,
      per-chain residuals).

and updates <library>/index.json per DATASPEC §4.

Run:  PYTHONPATH=. python3 -m mocap.humanoid21.export_library [--jobs 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.parsers.asf_parser import parse_asf
from mocap.humanoid21.parsers.amc_parser import parse_amc
from mocap.humanoid21.calibrate import cmu_pos_to_mj, cmu_rot_to_mj
from mocap.humanoid21.ref_fk import RefSkeleton, fk_frame
from mocap.humanoid21.retarget import retarget_clip
from mocap.humanoid21.robot_kin import (
    CONTROLLED_JOINTS, fk_positions, joint_limits,
)

LIBRARY = Path("/data1/dev/combatbench_mocap")
RAW_REPO = Path("/data1/dev/combatbench_mocap/raw")  # repo 侧原目录已被清理
RAW_DEV = LIBRARY / "raw"
SRC_FPS = 120.0
SCALE = json.loads(
    (Path(__file__).parent / "calibration.json").read_text()
)["meters_per_cmu_unit"]

MINIMAL_SET = {
    "13_17": ["boxing"], "13_18": ["boxing"],
    "14_01": ["boxing"], "14_02": ["boxing"], "14_03": ["boxing"],
    "15_13": ["boxing"], "17_10": ["boxing"],
    "07_01": ["walk"], "07_06": ["walk"], "07_12": ["walk"],
    "02_01": ["walk"], "02_02": ["walk"],
    "13_01": ["stand_up", "recovery"], "13_02": ["stand_up", "recovery"],
    "13_03": ["stand_up", "recovery"],
    "17_01": ["walk", "stylized"], "17_03": ["walk", "stylized"],
    # ---- M5 扩充：战斗相关动作类别 ----
    # 防守/闪避/后撤 (subject 76, defense vignettes)
    "76_01": ["retreat", "punch"], "76_02": ["retreat", "feint", "punch"],
    "76_03": ["evade", "dodge"], "76_04": ["guard", "defense"],
    "76_09": ["walk", "retreat"], "76_11": ["retreat", "quick_steps"],
    # 空手 kata/招式 (subject 135)
    "135_01": ["kata", "combo"], "135_04": ["kick", "front_kick"],
    "135_05": ["block", "low_block"], "135_06": ["kata", "combo"],
    "135_07": ["kick", "roundhouse"], "135_09": ["punch", "lunge_punch"],
    "135_10": ["block", "knife_hand"], "135_11": ["kick", "side_kick"],
    # 踢拳组合 (subject 86/144/111/113/141/135/10)
    "86_01": ["kick", "punch", "combo"], "86_06": ["kick", "punch", "combo"],
    "144_05": ["kick", "front_kick"], "144_06": ["kick", "front_kick"],
    "144_09": ["kick", "front_kick"],
    "144_07": ["block"], "144_26": ["block"],
    "144_13": ["punch", "combo"], "144_14": ["punch", "combo"],
    "144_20": ["punch", "combo"], "144_21": ["punch", "combo"],
    "144_17": ["lunge"],
    "111_19": ["kick", "punch"], "113_13": ["kick", "punch"],
    "141_14": ["kick", "punch"],
    "10_01": ["kick", "soccer"], "10_02": ["kick", "soccer"],
    "10_03": ["kick", "soccer"],
    # 打击/跳跃/地面起身
    "02_05": ["punch", "strike"], "02_07": ["strike", "swordplay"],
    "13_11": ["jump", "jump_forward"], "13_19": ["jump", "jump_forward"],
    "141_04": ["jump"], "141_05": ["jump", "jump_sideways"],
    "141_06": ["jump", "jump_twist"],
    "111_06": ["stand_up", "recovery"],
    "113_08": ["stand_up", "recovery"],
}

ROBOT_KEYPOINTS = ["pelvis", "torso", "head",
                   "thigh_left", "shin_left", "foot_left",
                   "thigh_right", "shin_right", "foot_right",
                   "upper_arm_left", "lower_arm_left", "hand_left",
                   "upper_arm_right", "lower_arm_right", "hand_right"]

CONTACT_HEIGHT = 0.06   # m — ankle/toe keypoint near ground
CONTACT_SPEED = 0.35    # m/s — foot stationary

DESCRIPTIONS = json.loads(
    (Path(__file__).parent / "cmu_motion_descriptions.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find(subj: str, trial: str):
    amc = RAW_REPO / f"{subj}_{trial}.amc"
    if not amc.exists():
        amc = RAW_DEV / f"{subj}_{trial}.amc"
    asf = RAW_REPO / f"{subj}.asf"
    if not asf.exists():
        asf = RAW_DEV / f"{subj}.asf"
    assert amc.exists() and asf.exists(), (subj, trial)
    return asf, amc


def _quat_angvel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from a quat sequence (wxyz), central diff."""
    R = Rotation.from_quat(np.roll(quat_wxyz[:, [3, 0, 1, 2]], 0, axis=1))
    w = np.zeros((len(quat_wxyz), 3))
    rel = R[2:] * R[:-2].inv()
    w[1:-1] = rel.as_rotvec() / (2 * dt)
    w[0] = (R[1] * R[0].inv()).as_rotvec() / dt
    w[-1] = (R[-1] * R[-2].inv()).as_rotvec() / dt
    return w


def _contacts(ankle: np.ndarray, toe: np.ndarray, dt: float,
              band: float = 0.05, speed_gate: float = 0.8) -> np.ndarray:
    """0/1 contact for one foot: near ground baseline AND low toe speed.
    Baseline = 5th percentile of min(ankle,toe) z over the clip (adapts to
    ankle-vs-sole keypoint offset)."""
    z = np.minimum(ankle[:, 2], toe[:, 2])
    baseline = np.percentile(z, 5)
    toe_v = np.linalg.norm(np.gradient(toe, dt, axis=0), axis=1)
    return ((z - baseline < band) & (toe_v < speed_gate)).astype(np.float32)


def source_export(skel, frames):
    """joint_space=cmu_asf npz dict: pelvis root + per-bone dofs + keypoints."""
    bone_names = list(skel.bone_names)
    dof_names = ["root.tx", "root.ty", "root.tz",
                 "root.rx", "root.ry", "root.rz"]
    dof_slices = {}
    col = 6
    for b in bone_names:
        if b == "root":
            continue
        nd = len(skel.dof[b])
        dof_slices[b] = slice(col, col + nd)
        dof_names += [f"{b}.{d}" for d in skel.dof[b]]
        col += nd
    D = col
    T = len(frames)
    dt = 1.0 / SRC_FPS

    pos_kp = np.zeros((T, len(bone_names), 3))
    root_p = np.zeros((T, 3))
    root_q = np.zeros((T, 4))
    dof_val = np.zeros((T, D))
    for t, fr in enumerate(frames):
        j, d, F = fk_frame(skel, fr)
        root_p[t] = cmu_pos_to_mj(j["root"]) * SCALE
        Rm = cmu_rot_to_mj(F["root"])
        root_q[t] = Rotation.from_matrix(Rm).as_quat()[[3, 0, 1, 2]]
        for i, b in enumerate(bone_names):
            pos_kp[t, i] = cmu_pos_to_mj(j[b]) * SCALE
            if b == "root":
                continue
            s = dof_slices[b]
            dof_val[t, s] = np.asarray(fr.get(b, [0.0] * (s.stop - s.start)),
                                       float)
        rv = np.asarray(fr.get("root", [0.0] * 6), float)
        dof_val[t, :6] = rv[:6]

    bi = {b: i for i, b in enumerate(bone_names)}
    contacts = np.stack([
        _contacts(pos_kp[:, bi["lfoot"]], pos_kp[:, bi["ltoes"]], dt),
        _contacts(pos_kp[:, bi["rfoot"]], pos_kp[:, bi["rtoes"]], dt)], 1)

    return dict(
        timestamps=np.arange(T, dtype=np.float32) * dt,
        root_pos=root_p.astype(np.float32),
        root_quat=root_q.astype(np.float32),
        root_linvel=np.gradient(root_p, dt, axis=0).astype(np.float32),
        root_angvel=_quat_angvel(root_q, dt).astype(np.float32),
        joint_angles=dof_val.astype(np.float32),
        joint_velocities=np.gradient(dof_val, dt, axis=0).astype(np.float32),
        joint_names=np.array(dof_names),
        keypoints=pos_kp.astype(np.float32),
        keypoint_names=np.array(bone_names),
        foot_contacts=contacts,
        valid_mask=np.ones(T, dtype=np.uint8),
    )


def robot_export(skel, frames, asf_path: Path, amc_path: Path,
                 mirror: bool = False):
    """joint_space=humanoid21_v2 npz dict + retarget diagnostics."""
    out = retarget_clip(str(asf_path), str(amc_path), mirror=mirror)
    qang = out["joint_angles"]
    rpos = out["root_pos"]
    rquat = out["root_quat"]
    T = len(qang)
    dt = 1.0 / SRC_FPS

    kp = np.zeros((T, len(ROBOT_KEYPOINTS), 3))
    for t in range(T):
        pos, _ = fk_positions(dict(zip(CONTROLLED_JOINTS, qang[t])),
                              rpos[t], rquat[t])
        for i, b in enumerate(ROBOT_KEYPOINTS):
            kp[t, i] = pos[b]
    li = ROBOT_KEYPOINTS.index("foot_left")
    ri = ROBOT_KEYPOINTS.index("foot_right")
    contacts = np.stack([
        _contacts(kp[:, li], kp[:, li], dt, band=0.05, speed_gate=1.2),
        _contacts(kp[:, ri], kp[:, ri], dt, band=0.05, speed_gate=1.2)], 1)

    resid = np.stack([out["residuals"][k] for k in
                      ("leg_left", "leg_right", "ankle_left", "ankle_right",
                       "arm_left", "arm_right")], 1)
    npz = dict(
        timestamps=np.arange(T, dtype=np.float32) * dt,
        root_pos=rpos.astype(np.float32),
        root_quat=rquat.astype(np.float32),
        root_linvel=np.gradient(rpos, dt, axis=0).astype(np.float32),
        root_angvel=_quat_angvel(rquat, dt).astype(np.float32),
        joint_angles=qang.astype(np.float32),
        joint_velocities=np.gradient(qang, dt, axis=0).astype(np.float32),
        joint_names=np.array(CONTROLLED_JOINTS),
        keypoints=kp.astype(np.float32),
        keypoint_names=np.array(ROBOT_KEYPOINTS),
        foot_contacts=contacts,
        lead_stable_frame=np.int32(_lead_stable_frame(qang)),
        valid_mask=np.ones(T, dtype=np.uint8),
        retarget_residuals=resid.astype(np.float32),
        residual_names=np.array(["leg_left", "leg_right", "ankle_left",
                                 "ankle_right", "arm_left", "arm_right"]),
    )
    return npz, {k: {"mean": float(v.mean()),
                     "p95": float(np.percentile(v, 95)),
                     "max": float(v.max())}
                 for k, v in out["residuals"].items()}


def _lead_stable_frame(qang: np.ndarray, win: int = 6,
                       thresh: float = 0.08) -> int:
    """首个连续 `win` 帧关节变化 < thresh rad 的帧号。

    片段开头可能有残余 IK 分支瞬态或源数据毛刺（AMC 单帧 >20deg
    跳变）。RSI/确定性回放应从该帧之后初始化。
    """
    dq = np.abs(np.diff(qang, axis=0)).max(axis=1)
    for t in range(len(dq)):
        if np.all(dq[t:t + win] < thresh):
            return t
    return 0


def kinematic_issues(npz, resid_stats):
    """Cheap L1-side checks: limits, finite, velocity spikes, penetration."""
    issues = []
    q = np.asarray(npz["joint_angles"])
    if not np.isfinite(q).all():
        issues.append("nonfinite_joint_angles")
    lim_viol = 0
    max_over_deg = 0.0
    for i, n in enumerate(CONTROLLED_JOINTS):
        lo, hi = joint_limits(n)
        over = np.maximum(lo - q[:, i], q[:, i] - hi)
        viol = np.maximum(over, 0.0)
        lim_viol += int((viol > 1e-4).sum())
        max_over_deg = max(max_over_deg, float(np.rad2deg(viol.max())))
    if lim_viol:
        issues.append("joint_limit_violations")
    jv = np.asarray(npz["joint_velocities"])
    if np.abs(jv).max() > 50.0:
        issues.append("joint_velocity_spike")
    if min(npz["keypoints"][:, ROBOT_KEYPOINTS.index("foot_left"), 2].min(),
           npz["keypoints"][:, ROBOT_KEYPOINTS.index("foot_right"), 2].min()
           ) < -0.01:
        issues.append("foot_penetration")
    for k, st in resid_stats.items():
        if st["p95"] > 0.15:
            issues.append(f"high_residual_{k}")
    return {"limit_violations": lim_viol,
            "max_overshoot_deg": round(max_over_deg, 3)}, issues


def _phase_candidates(npz, labels):
    """Auto boxing phase segments: execute = frames where either hand speed
    > 2.0 m/s, merged with <=0.4s gaps, padded 0.25s. prepare/recover are the
    0.3s windows around each execute block."""
    if "boxing" not in labels:
        return []
    kp = npz["keypoints"]
    kn = list(npz["keypoint_names"])
    hl = kp[:, kn.index("hand_left")]
    hr = kp[:, kn.index("hand_right")]
    dt = 1.0 / SRC_FPS
    sp = np.maximum(np.linalg.norm(np.gradient(hl, dt, axis=0), axis=1),
                    np.linalg.norm(np.gradient(hr, dt, axis=0), axis=1))
    act = sp > 2.0
    segs, i = [], 0
    while i < len(act):
        if act[i]:
            s = i
            while i < len(act) and (act[i] or (i - s < int(0.4 / dt)
                                               and act[i:i + int(0.4 / dt)].any())):
                i += 1
            segs.append((s, i))
        i += 1
    pad = int(0.25 / dt)
    out = []
    for s, e in segs:
        s2, e2 = max(0, s - pad), min(len(act), e + pad)
        out.append({"start": round(s2 * dt, 3), "end": round(e2 * dt, 3),
                    "phase": "execute"})
    return out


def process_clip(tag: str, mirror: bool = False):
    """Worker: source npz + retargeted npz + index entry.

    mirror=True 只导出矢状面镜像的机器人片段（cmu_XX_YY_m），
    不重复写 _src npz（源数据不变，镜像发生在重定向目标层）。
    """
    subj, trial = tag.split("_")
    labels = MINIMAL_SET[tag]
    asf_path, amc_path = _find(subj, trial)
    skel = RefSkeleton.from_asf(parse_asf(str(asf_path)))
    frames = parse_amc(str(amc_path))
    t0 = time.time()

    rob_npz, resid_stats = robot_export(skel, frames, asf_path, amc_path,
                                        mirror=mirror)
    checks, issues = kinematic_issues(rob_npz, resid_stats)
    phases = _phase_candidates(rob_npz, labels)

    clips_dir = LIBRARY / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"cmu_{subj}_{trial}" + ("_m" if mirror else "")
    src_id = f"cmu_{subj}_{trial}_src"
    if not mirror:
        src_npz = source_export(skel, frames)
        np.savez_compressed(clips_dir / f"{src_id}.npz", **src_npz)
    np.savez_compressed(clips_dir / f"{clip_id}.npz", **rob_npz)

    official = DESCRIPTIONS["subjects"].get(subj, {}).get("trials", {}) \
        .get(str(int(trial)), "")
    T = len(frames)
    common_time = {"src_fps": SRC_FPS, "fps": SRC_FPS,
                   "frames": T, "duration_s": round(T / SRC_FPS, 2)}
    entries = {}
    if mirror:
        # 镜像片段：单一条目，溯源记 derivation
        entries[clip_id] = {
            "identity": {"source": "derived", "subject": subj,
                         "trial": int(trial),
                         "parent_clip": f"cmu_{subj}_{trial}",
                         "derivation": "mirror_sagittal_x",
                         "source_file": str(amc_path),
                         "source_sha256": _sha256(amc_path),
                         "npz_sha256": _sha256(clips_dir / f"{clip_id}.npz"),
                         "version": 1},
            "time": common_time,
            "frames_convention": "mujoco_world_z_up",
            "joint_space": "humanoid21_v2",
            "classification": {"labels": labels + ["mirrored"],
                               "phase_segments": phases,
                               "label_source": "cmu_official+derived",
                               "official_description": official,
                               "confidence": "high"},
            "quality": {"level": "L0",
                        "kinematic_checks": checks,
                        "residual_stats": resid_stats,
                        "physics": None,
                        "contact_method": "foot body z<0.06m & speed<0.35m/s",
                        "velocity_method": "np.gradient central diff @120Hz",
                        "issues": issues,
                        "rejected": False, "reject_reason": None},
            "adaptation": {"model": "envs/humanoid21/battle_circular_v2.xml",
                           "model_sha256": _sha256(
                               PROJECT_ROOT / "envs/humanoid21/battle_circular_v2.xml"),
                           "retarget_method": "per-frame LM IK on "
                               "sagittal-mirrored source pose (x_cmu->-x, "
                               "l/r swapped)",
                           "retarget_params": {"scale_m_per_unit": SCALE,
                                               "warm_start": True,
                                               "mirror": True,
                                               "dq_max_rad_frame": 0.35}},
        }
        return clip_id, entries, round(time.time() - t0, 1)

    entries[src_id] = {
        "identity": {"source": "cmu", "subject": subj, "trial": int(trial),
                     "parent_clip": None,
                     "source_file": str(amc_path.relative_to(PROJECT_ROOT)
                                        if str(amc_path).startswith(
                                            str(PROJECT_ROOT)) else amc_path),
                     "asf_file": str(asf_path),
                     "source_sha256": _sha256(amc_path),
                     "asf_sha256": _sha256(asf_path),
                     "npz_sha256": _sha256(clips_dir / f"{src_id}.npz"),
                     "version": 1},
        "time": common_time,
        "frames_convention": "mujoco_world_z_up",
        "joint_space": "cmu_asf",
        "classification": {"labels": labels, "phase_segments": [],
                           "label_source": "cmu_official",
                           "official_description": official,
                           "confidence": "high"},
        "quality": {"level": "L0",
                    "kinematic_checks": None, "physics": None,
                    "contact_method": "min(ankle,toe) z<0.06m & toe speed<0.35m/s",
                    "issues": [], "rejected": False, "reject_reason": None},
        "adaptation": {"model": "envs/humanoid21/battle_circular_v2.xml",
                       "retarget_method": None, "retarget_params": None},
    }
    entries[clip_id] = {
        "identity": {"source": "cmu", "subject": subj, "trial": int(trial),
                     "parent_clip": src_id,
                     "source_file": str(amc_path),
                     "source_sha256": _sha256(amc_path),
                     "npz_sha256": _sha256(clips_dir / f"{clip_id}.npz"),
                     "version": 1},
        "time": common_time,
        "frames_convention": "mujoco_world_z_up",
        "joint_space": "humanoid21_v2",
        "classification": {"labels": labels,
                           "phase_segments": phases,
                           "label_source": "cmu_official",
                           "official_description": official,
                           "confidence": "high"},
        "quality": {"level": "L0",
                    "kinematic_checks": checks,
                    "residual_stats": resid_stats,
                    "physics": None,
                    "contact_method": "foot body z<0.06m & speed<0.35m/s",
                    "velocity_method": "np.gradient central diff @120Hz",
                    "issues": issues,
                    "rejected": False, "reject_reason": None},
        "adaptation": {"model": "envs/humanoid21/battle_circular_v2.xml",
                       "model_sha256": _sha256(
                           PROJECT_ROOT / "envs/humanoid21/battle_circular_v2.xml"),
                       "retarget_method": "per-frame LM IK: ankle/hand "
                           "rel-position (w=1.0) + shin-dir prior (w=0.4, legs) "
                           "+ temporal smoothness w=0.15 + slew limit 0.35rad/f",
                       "retarget_params": {"scale_m_per_unit": SCALE,
                                           "warm_start": True,
                                           "dq_max_rad_frame": 0.35}},
    }
    return clip_id, entries, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--mirror", action="store_true",
                    help="导出矢状面镜像变体 (cmu_XX_YY_m)")
    args = ap.parse_args()
    tags = args.only or list(MINIMAL_SET)
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.starmap(
                process_clip, [(t, args.mirror) for t in tags])
    else:
        results = [process_clip(t, args.mirror) for t in tags]

    index_path = LIBRARY / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else \
        {"spec_version": 1, "clips": {}}
    for clip_id, entries, dt in results:
        index["clips"].update(entries)
        print(f"{clip_id}: {dt}s")
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(f"done in {time.time() - t0:.0f}s -> {index_path}")


if __name__ == "__main__":
    main()
