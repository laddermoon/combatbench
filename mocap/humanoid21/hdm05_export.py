"""HDM05 片段导出到运动库（与 export_library 同契约）。

产物：
  clips/hdm05_<take>_<cls>_<start>_<n>_src.npz — joint_space "hdm05_coords":
      31 标记点位置（米，mj 世界系），无旋转角
  clips/hdm05_<take>_<cls>_<start>_<n>.npz     — joint_space "humanoid21_v2"
  index.json 条目，identity.source="hdm05"

Run:
  PYTHONPATH=. python3 -m mocap.humanoid21.hdm05_export \
      --file .../HDM05-122-objects-annotations-specific-coords_normPOS.data \
      --classes punchLFront1Reps kickLFront1Reps --limit 3 [--jobs 8]
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

from mocap.humanoid21.calibrate import cmu_pos_to_mj  # noqa: E402
from mocap.humanoid21.export_library import (  # noqa: E402
    LIBRARY, SRC_FPS, ROBOT_KEYPOINTS, _contacts, _quat_angvel,
    _lead_stable_frame, kinematic_issues)
from mocap.humanoid21.hdm05_retarget import (  # noqa: E402
    retarget_hdm05_segment, segment_scale)
from mocap.humanoid21.parsers.hdm05_data import (  # noqa: E402
    iter_segments, load_action_names, JOINT_NAMES)
from mocap.humanoid21.robot_kin import (  # noqa: E402
    CONTROLLED_JOINTS, fk_positions)

SCALE = json.loads(
    (Path(__file__).parent / "calibration.json").read_text()
)["meters_per_cmu_unit"]

HDM05_DIR = Path("/data1/dev/hdm05/data")
ACTION_NAMES = HDM05_DIR / "hdm05_action_descriptions.txt"

# hdm05 类名 -> 库标签（原始类名保留在 index.hdm05_label）
LABEL_MAP = {
    "punchLFront": ["punch"], "punchRFront": ["punch"],
    "punchLSide": ["punch", "hook"], "punchRSide": ["punch", "hook"],
    "kickLFront": ["kick", "front_kick"], "kickRFront": ["kick", "front_kick"],
    "kickLSide": ["kick", "side_kick"], "kickRSide": ["kick", "side_kick"],
    "elbowToKnee": ["strike", "elbow"],
    "throwStanding": ["throw", "strike"], "throwFar": ["throw", "strike"],
    "throwBasketball": ["throw"], "throwSitting": ["throw"],
    "hitRHandHead": ["block", "self_hit"],
    "standUpKneelToStand": ["stand_up", "recovery"],
    "standUpLieFloor": ["stand_up", "recovery", "face_up"],
    "standUpSitFloor": ["stand_up", "recovery"],
    "standUpSitChair": ["stand_up", "recovery", "chair"],
    "standUpSitTable": ["stand_up", "recovery", "chair"],
    "standUpSitToStand": ["stand_up", "recovery"],
    "lieDownFloor": ["lie_down", "ground"], "sitDownFloor": ["sit_down", "ground"],
    "sitDownKneelTieShoes": ["sit_down", "recovery"],
    "squat": ["squat"], "jumpingJack": ["jump", "jumping_jack"],
    "jumpDown": ["jump", "jump_down"], "jump": ["jump"],
    "hopLLeg": ["jump", "hop"], "hopRLeg": ["jump", "hop"],
    "hopBothLegs": ["jump", "hop"],
    "turnLeft": ["turn"], "turnRight": ["turn"],
    "walkBackwards": ["walk", "backwards", "footwork"],
    "walkLeft": ["walk", "footwork"], "walkRight": ["walk", "footwork"],
    "walk2Steps": ["walk", "footwork"], "walk3Steps": ["walk", "footwork"],
    "walk4Steps": ["walk", "footwork"],
    "walkOnPlace": ["walk", "in_place"],
    "jogOnPlace": ["jog", "in_place"], "runOnPlace": ["run", "in_place"],
    "walkLeftCircle": ["walk", "circle"], "walkRightCircle": ["walk", "circle"],
    "jogLeftCircle": ["jog", "circle"], "jogRightCircle": ["jog", "circle"],
    "cartwheel": ["acrobatic", "cartwheel"],
    "climbLadder": ["climb"],
    "grabDep": ["grab"], "grabFloor": ["grab"], "grabHigh": ["grab"],
    "grabLow": ["grab"], "grabMiddle": ["grab"],
    "depositFloor": ["deposit"], "depositHigh": ["deposit"],
    "depositLow": ["deposit"], "depositMiddle": ["deposit"],
    "staircaseUp": ["stairs"], "staircaseDown": ["stairs"],
    "balanceBeam": ["balance"], "skier": ["stylized", "skier"],
}


def labels_for(hdm_label: str):
    """hdm05_spec_kickLFront1Reps -> ['kick','front_kick'] + 原名。"""
    base = hdm_label.replace("hdm05_spec_", "").replace("hdm05_", "")
    for k, v in LABEL_MAP.items():
        if base.startswith(k):
            return v
    return [base.split("1Reps")[0].split("2Reps")[0].split("3Reps")[0]
            .rstrip("0123456789")] or [base]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def src_npz(seg, k: float) -> dict:
    """源侧 npz：31 标记点（米，mj 世界系）。joint_angles 无效——
    位置流无旋转角，joint_angles 置 NaN 且 valid 仅供占位。"""
    T = len(seg.points)
    dt = 1.0 / SRC_FPS
    pts = seg.points * k                       # cmu 等效单位
    kp = np.zeros((T, 31, 3))
    for t in range(T):
        for i in range(31):
            kp[t, i] = cmu_pos_to_mj(pts[t, i]) * SCALE
    bi = {n: i for i, n in enumerate(JOINT_NAMES)}
    contacts = np.stack([
        _contacts(kp[:, bi["lfoot"]], kp[:, bi["ltoes"]], dt),
        _contacts(kp[:, bi["rfoot"]], kp[:, bi["rtoes"]], dt)], 1)
    return dict(
        timestamps=np.arange(T, dtype=np.float32) * dt,
        root_pos=np.zeros((T, 3), np.float32),   # in-place：无世界位移
        root_quat=np.tile([1., 0., 0., 0.], (T, 1)).astype(np.float32),
        root_linvel=np.zeros((T, 3), np.float32),
        root_angvel=np.zeros((T, 3), np.float32),
        joint_angles=np.full((T, 1), np.nan, np.float32),
        joint_velocities=np.full((T, 1), np.nan, np.float32),
        joint_names=np.array(["<none>"]),
        keypoints=kp.astype(np.float32),
        keypoint_names=np.array(JOINT_NAMES),
        foot_contacts=contacts,
        valid_mask=np.ones(T, dtype=np.uint8),
    )


def robot_npz(seg, mirror: bool = False):
    out = retarget_hdm05_segment(seg, mirror=mirror)
    qang, rpos, rquat = out["joint_angles"], out["root_pos"], out["root_quat"]
    T = len(qang); dt = 1.0 / SRC_FPS
    kp = np.zeros((T, len(ROBOT_KEYPOINTS), 3))
    for t in range(T):
        pos, _ = fk_positions(dict(zip(CONTROLLED_JOINTS, qang[t])),
                              rpos[t], rquat[t])
        for i, b in enumerate(ROBOT_KEYPOINTS):
            kp[t, i] = pos[b]
    li, ri = (ROBOT_KEYPOINTS.index("foot_left"),
              ROBOT_KEYPOINTS.index("foot_right"))
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
    stats = {k: {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
                 "max": float(v.max())}
             for k, v in out["residuals"].items()}
    return npz, stats, out["hdm05_scale_factor"]


def process_segment(seg, data_file: str, mirror: bool = False):
    t0 = time.time()
    labels = labels_for(seg.label)
    k = segment_scale(seg)
    clips_dir = LIBRARY / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_id = f"hdm05_{seg.key}" + ("_m" if mirror else "")
    src_id = f"hdm05_{seg.key}_src"
    rob, stats, _ = robot_npz(seg, mirror=mirror)
    checks, issues = kinematic_issues(rob, stats)
    # 位置流特有告警
    issues.append("inplace_source_no_root_motion")
    np.savez_compressed(clips_dir / f"{clip_id}.npz", **rob)
    if not mirror:
        np.savez_compressed(clips_dir / f"{src_id}.npz", **src_npz(seg, k))

    T = len(seg.points)
    common_time = {"src_fps": SRC_FPS, "fps": SRC_FPS, "frames": T,
                   "duration_s": round(T / SRC_FPS, 2)}
    entries = {}
    if mirror:
        entries[clip_id] = {
            "identity": {"source": "derived",
                         "parent_clip": f"hdm05_{seg.key}",
                         "derivation": "mirror_sagittal_x",
                         "hdm05_key": seg.key,
                         "npz_sha256": _sha256(clips_dir / f"{clip_id}.npz"),
                         "version": 1},
            "time": common_time, "frames_convention": "mujoco_world_z_up",
            "joint_space": "humanoid21_v2",
            "classification": {"labels": labels + ["mirrored"],
                               "phase_segments": [],
                               "label_source": "hdm05_annotation+derived",
                               "official_description": seg.label,
                               "confidence": "high"},
            "quality": {"level": "L0", "kinematic_checks": checks,
                        "residual_stats": stats, "physics": None,
                        "issues": issues, "rejected": False,
                        "reject_reason": None},
            "adaptation": {"retarget_method": "position-only IK: marker "
                           "positions + estimated torso/pelvis frames; "
                           "in-place source (normPOS)",
                           "retarget_params": {"hdm05_scale_factor": k,
                                               "mirror": True}},
        }
        return clip_id, entries, round(time.time() - t0, 1)

    entries[src_id] = {
        "identity": {"source": "hdm05", "subject": "hdm05",
                     "trial": seg.take, "hdm05_key": seg.key,
                     "hdm05_class_id": seg.class_id,
                     "hdm05_start_frame": seg.start_frame,
                     "parent_clip": None,
                     "source_file": data_file,
                     "npz_sha256": _sha256(clips_dir / f"{src_id}.npz"),
                     "version": 1},
        "time": common_time, "frames_convention": "mujoco_world_z_up",
        "joint_space": "hdm05_coords",
        "classification": {"labels": labels, "phase_segments": [],
                           "label_source": "hdm05_annotation",
                           "official_description": seg.label,
                           "confidence": "high"},
        "quality": {"level": "L0", "kinematic_checks": {},
                    "residual_stats": {}, "physics": None,
                    "issues": ["position_only_no_rotations",
                               "inplace_source_no_root_motion"],
                    "rejected": False, "reject_reason": None},
        "adaptation": {},
    }
    entries[clip_id] = {
        "identity": {"source": "hdm05", "subject": "hdm05",
                     "trial": seg.take, "hdm05_key": seg.key,
                     "hdm05_class_id": seg.class_id,
                     "parent_clip": src_id,
                     "source_file": data_file,
                     "npz_sha256": _sha256(clips_dir / f"{clip_id}.npz"),
                     "version": 1},
        "time": common_time, "frames_convention": "mujoco_world_z_up",
        "joint_space": "humanoid21_v2",
        "classification": {"labels": labels, "phase_segments": [],
                           "label_source": "hdm05_annotation",
                           "official_description": seg.label,
                           "confidence": "high"},
        "quality": {"level": "L0", "kinematic_checks": checks,
                    "residual_stats": stats, "physics": None,
                    "issues": issues, "rejected": False,
                    "reject_reason": None},
        "adaptation": {"retarget_method": "position-only IK: marker "
                       "positions + estimated torso/pelvis frames; "
                       "in-place source (normPOS), per-segment leg-drop "
                       "scale",
                       "retarget_params": {"hdm05_scale_factor": k,
                                           "mirror": False}},
    }
    return clip_id, entries, round(time.time() - t0, 1)


def _worker(args):
    seg, data_file, mirror = args
    try:
        return process_segment(seg, data_file, mirror)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {seg.key}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(
        HDM05_DIR / "HDM05-122" /
        "HDM05-122-objects-annotations-specific-coords_normPOS.data"))
    ap.add_argument("--classes", nargs="*", default=None,
                    help="子串过滤，如 punchLFront kickLFront")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-class", type=int, default=None,
                    help="每个类最多导出的片段数")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="指定 record key")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    names = load_action_names(ACTION_NAMES)
    picked = []
    from collections import Counter
    per_cls = Counter()
    for seg in iter_segments(args.file, names):
        if args.keys and seg.key not in args.keys:
            continue
        if args.classes and not any(c in seg.label for c in args.classes):
            continue
        if args.per_class and per_cls[seg.label] >= args.per_class:
            continue
        per_cls[seg.label] += 1
        picked.append(seg)
        if args.limit and len(picked) >= args.limit:
            break
    print(f"{len(picked)} segments")

    index_path = LIBRARY / "index.json"
    index = json.loads(index_path.read_text())
    index.setdefault("clips", {})          # 与 export_library 同契约
    jobs = [(s, args.file, args.mirror) for s in picked]
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            results = p.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]
    n_ok = 0
    for r in results:
        if isinstance(r, str):
            print(" ", r)
            continue
        clip_id, entries, dt_s = r
        index["clips"].update(entries)
        n_ok += 1
        print(f"  {clip_id} ({dt_s}s)")
    index_path.write_text(json.dumps(index, indent=1))
    print(f"done: {n_ok} exported")


if __name__ == "__main__":
    main()
