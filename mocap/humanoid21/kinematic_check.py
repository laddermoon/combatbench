"""M1-W4: kinematic validation of retargeted clips -> index.json L1 marks.

Checks per clip (joint_space=humanoid21_v2 only):
  - joint range violations & max overshoot
  - hamstring fixed-tendon coupling: 0.5*hip_y - 0.5*knee in [-0.3, 2.0] rad
  - joint velocity spikes / frame-to-frame jumps vs slew cap
  - root rotation jumps and inverted head-to-pelvis frames
  - foot penetration below ground
  - nonfinite / missing frames (valid_mask)
  - retarget residuals p95/max per chain

Results merge into /data1/dev/combatbench_mocap/index.json:
  quality.kinematic_checks = {...}, issues updated (deduped),
  quality.level = "L1" (checks ran; issues explicitly recorded — clips are
  NOT silently clipped/hidden per DATASPEC §5).

Usage: PYTHONPATH=. python3 -m mocap.humanoid21.kinematic_check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

LIBRARY = Path("/data1/dev/combatbench_mocap")
SLEW_CAP = 0.35 + 1e-6          # rad/frame — retarget slew limit
VEL_SPIKE = 60.0                # rad/s — beyond any plausible joint speed
FOOT_Z_MIN = -0.005             # m — penetration tolerance
TENDON_LO, TENDON_HI = -0.3, 2.0  # hamstring: 0.5*hip_y - 0.5*knee range

KINETIC_ISSUES = ("joint_limit_violations", "tendon_violations",
                  "joint_velocity_spike", "frame_jump", "root_orientation_jump",
                  "head_below_pelvis", "foot_penetration", "invalid_frames")


def check_clip(npz_path: Path) -> dict:
    d = np.load(npz_path)
    q = np.asarray(d["joint_angles"], float)
    jv = np.asarray(d["joint_velocities"], float)
    kp = d["keypoints"]
    kn = list(d["keypoint_names"])
    valid = np.asarray(d["valid_mask"], bool)
    dt = float(np.median(np.diff(d["timestamps"])))

    lim_viol = 0
    max_over = 0.0
    for i, n in enumerate(CONTROLLED_JOINTS):
        lo, hi = joint_limits(n)
        over = np.maximum.reduce([lo - q[:, i], q[:, i] - hi,
                                  np.zeros(len(q))])
        lim_viol += int((over > 1e-4).sum())
        max_over = max(max_over, float(np.rad2deg(over.max())))

    ji = {n: i for i, n in enumerate(CONTROLLED_JOINTS)}
    tendon = {}
    tendon_viol = 0
    tendon_minmax = [np.inf, -np.inf]
    for side in ("left", "right"):
        tv = 0.5 * q[:, ji[f"hip_y_{side}"]] - 0.5 * q[:, ji[f"knee_{side}"]]
        tendon_viol += int(((tv < TENDON_LO - 1e-3) | (tv > TENDON_HI + 1e-3)
                            ).sum())
        tendon_minmax[0] = min(tendon_minmax[0], float(tv.min()))
        tendon_minmax[1] = max(tendon_minmax[1], float(tv.max()))
    tendon = {"violations": tendon_viol,
              "range": [round(tendon_minmax[0], 3), round(tendon_minmax[1], 3)],
              "allowed": [TENDON_LO, TENDON_HI]}

    dq = np.abs(np.diff(q, axis=0))
    max_jump = float(np.rad2deg(dq.max()))
    vel_max = float(np.abs(jv).max())

    fl = kp[:, kn.index("foot_left"), 2]
    fr_ = kp[:, kn.index("foot_right"), 2]
    foot_min = float(min(fl.min(), fr_.min()))
    head_below_pelvis = int((kp[:, kn.index("head"), 2] <
                             kp[:, kn.index("pelvis"), 2]).sum())
    root_quat = np.asarray(d["root_quat"], float)
    root_rot = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
    root_jump_deg = float(np.rad2deg(
        (root_rot[1:] * root_rot[:-1].inv()).magnitude().max()))

    nonfinite = int((~np.isfinite(q)).sum() + (~np.isfinite(kp)).sum())
    invalid = int((~valid).sum())

    resid = {}
    if "retarget_residuals" in d.files:
        rn = list(d["residual_names"])
        for i, k in enumerate(rn):
            v = d["retarget_residuals"][:, i]
            resid[k] = {"mean": round(float(v.mean()), 4),
                        "p95": round(float(np.percentile(v, 95)), 4),
                        "max": round(float(v.max()), 4)}

    issues = []
    if lim_viol:
        issues.append("joint_limit_violations")
    if tendon_viol:
        issues.append("tendon_violations")
    if vel_max > VEL_SPIKE:
        issues.append("joint_velocity_spike")
    if max_jump > np.rad2deg(SLEW_CAP) + 0.5:
        issues.append("frame_jump")
    if root_jump_deg > 25.0:
        issues.append("root_orientation_jump")
    if head_below_pelvis:
        issues.append("head_below_pelvis")
    if foot_min < FOOT_Z_MIN:
        issues.append("foot_penetration")
    if invalid or nonfinite:
        issues.append("invalid_frames")

    return {"limit_violations": lim_viol,
            "max_overshoot_deg": round(max_over, 2),
            "tendon": tendon,
            "max_frame_jump_deg": round(max_jump, 2),
            "root_orientation_jump_deg": round(root_jump_deg, 2),
            "head_below_pelvis_frames": head_below_pelvis,
            "max_joint_vel_rad_s": round(vel_max, 1),
            "foot_z_min": round(foot_min, 4),
            "nonfinite": nonfinite, "invalid_frames": invalid,
            "retarget_residuals": resid}, issues


def main():
    index_path = LIBRARY / "index.json"
    index = json.loads(index_path.read_text())
    for cid, e in sorted(index["clips"].items()):
        if cid.endswith("_src"):
            continue
        npz = LIBRARY / "clips" / f"{cid}.npz"
        if not npz.exists():
            print(f"{cid}: npz missing, skipped")
            continue
        checks, issues = check_clip(npz)
        q = e["quality"]
        q["kinematic_checks"] = checks
        keep = [i for i in q["issues"] if i not in KINETIC_ISSUES
                and not i.startswith("high_residual_")]
        for k, st in checks["retarget_residuals"].items():
            if st["p95"] > 0.15:
                issues.append(f"high_residual_{k}")
        q["issues"] = sorted(set(keep + issues))
        q["level"] = "L1"
        print(f"{cid:12s} L1  issues={q['issues']}  "
              f"jump={checks['max_frame_jump_deg']}deg  "
              f"foot_z={checks['foot_z_min']}  tendon={checks['tendon']['range']}")
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print("index.json updated")


if __name__ == "__main__":
    main()
