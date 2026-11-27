"""L0/L1 validation for PHC bundles (DATASPEC v2).

L0: required fields present, shapes consistent, all finite, fps=30,
    pose_aa self-consistent vs root_rot + dof*dof_axis.
L1: joint-limit violations, per-frame jumps, root orientation jumps,
    foot penetration, velocity spikes.  (No tendon check — the h28
    morphology has no hamstring tendon.)
Issues are written back into clip['meta']['issues']/quality_level and
manifest clips[]. No silent repair — issues are recorded, clips kept.

Usage:
  mocap/gmr/.venv/bin/python mocap/humanoid28/gmr/check_phc.py [bundle.pkl]
"""
import json, pathlib, sys

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
PHC_DIR = HERE.parents[1] / 'phc'            # mocap/humanoid28/phc (gitignored)
MANIFEST = PHC_DIR / 'manifest.json'
MOCAP_XML = HERE / 'humanoid28_mocap.xml'

REQUIRED = {"fps", "root_trans_offset", "root_rot", "dof",
            "pose_aa", "smpl_joints", "meta"}
SLEW_CAP_DEG = 20.0          # rad->deg per frame jump cap (30fps)
ROOT_JUMP_DEG = 25.0
VEL_SPIKE = 60.0             # rad/s via finite diff
FOOT_Z_MIN = -0.02
L0_KEYS = ("missing_fields", "shape_mismatch", "nonfinite",
           "bad_fps", "pose_aa_inconsistent")
L1_VETO = {"invalid_frames", "frame_jump", "root_orientation_jump"}


def limits(mm, dof_names):
    lo = np.array([mm.jnt_range[mm.joint(n).id, 0] for n in dof_names])
    hi = np.array([mm.jnt_range[mm.joint(n).id, 1] for n in dof_names])
    return lo, hi


def check_clip(cid, c, mm, mf):
    issues, rep = [], {}
    miss = REQUIRED - set(c)
    if miss:
        return ["missing_fields"], {"missing": sorted(miss)}
    T = len(c["root_trans_offset"])
    fps = int(c["fps"])
    if fps != 30:
        issues.append("bad_fps")

    ok_shape = (c["root_rot"].shape == (T, 4) and c["dof"].shape[0] == T
                and c["pose_aa"].shape[0] == T and c["pose_aa"].shape[1] == 29
                and c["smpl_joints"].shape[:1] == (T,))
    if not ok_shape:
        issues.append("shape_mismatch")
    finite = all(np.isfinite(c[k]).all() for k in
                 ("root_trans_offset", "root_rot", "dof", "pose_aa"))
    if not finite:
        issues.append("nonfinite")
    if not ok_shape or not finite:
        return sorted(set(issues)), rep

    dof = c["dof"].astype(float)
    axis = np.asarray(mf["robot"]["dof_axis"], float)
    pose_aa_ok = np.allclose(c["pose_aa"][:, 1:], dof[:, :, None] * axis[None],
                             atol=1e-5)
    root_aa = R.from_quat(c["root_rot"]).as_rotvec()
    pose_aa_ok &= np.allclose(c["pose_aa"][:, 0], root_aa, atol=1e-5)
    if not pose_aa_ok:
        issues.append("pose_aa_inconsistent")

    # ---- L1 kinematics ----
    lo, hi = limits(mm, list(c["dof_names"]))
    over = np.maximum.reduce([lo - dof, dof - hi, np.zeros_like(dof)])
    lim_v = int((over > 1e-4).sum())
    rep["limit_violations"] = lim_v
    rep["max_overshoot_deg"] = round(float(np.rad2deg(over.max())), 2)
    if lim_v:
        issues.append("joint_limit_violations")

    # NOTE: no tendon constraint on h28 (MimicKit morphology has no
    # hamstring tendon) — the h21 tendon check is intentionally absent.

    jump = np.rad2deg(np.abs(np.diff(dof, axis=0)).max())
    rep["max_frame_jump_deg"] = round(float(jump), 2)
    if jump > SLEW_CAP_DEG:
        issues.append("frame_jump")

    rot = R.from_quat(c["root_rot"])
    rj = np.rad2deg((rot[1:] * rot[:-1].inv()).magnitude().max())
    rep["root_orientation_jump_deg"] = round(float(rj), 2)
    if rj > ROOT_JUMP_DEG:
        issues.append("root_orientation_jump")

    qv = np.gradient(dof, 1.0 / fps, axis=0)
    vm = float(np.abs(qv).max())
    rep["max_joint_vel_rad_s"] = round(vm, 1)
    if vm > VEL_SPIKE:
        issues.append("joint_velocity_spike")

    if "keypoint_pos" in c:
        kn = list(mf["robot"]["keypoint_names"])
        kp = c["keypoint_pos"]
        fz = min(kp[:, kn.index("left_foot"), 2].min(),
                 kp[:, kn.index("right_foot"), 2].min())
        rep["foot_z_min"] = round(float(fz), 4)
        if fz < FOOT_Z_MIN:
            issues.append("foot_penetration")
        hb = int((kp[:, kn.index("head"), 2] < kp[:, kn.index("pelvis"), 2]).sum())
        rep["head_below_pelvis_frames"] = hb
        if hb:
            issues.append("head_below_pelvis")

    return sorted(set(issues)), rep


def main():
    bundle_path = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                   else PHC_DIR / 'combatbench_dev_v1.pkl')
    import pickle
    bundle = pickle.load(open(bundle_path, 'rb'))
    mf = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    import mujoco
    mm = mujoco.MjModel.from_xml_path(str(MOCAP_XML))

    for cid, c in bundle.items():
        issues, rep = check_clip(cid, c, mm, mf)
        l0_bad = any(i in L0_KEYS for i in issues)
        veto = any(i in L1_VETO for i in issues)
        level = "rejected" if l0_bad else ("L1" if not veto else "L1_veto")
        c.setdefault("meta", {})["issues"] = issues
        c["meta"]["quality_level"] = level
        if MANIFEST.exists():
            mf["clips"].setdefault(cid, {}).update(
                {"issues": issues, "quality_level": level,
                 "kinematic_checks": rep})
        print(f"{cid:40s} {level:9s} issues={issues}")
        for k, v in rep.items():
            print(f"    {k}: {v}")

    pickle.dump(bundle, open(bundle_path, 'wb'))
    if MANIFEST.exists():
        MANIFEST.write_text(json.dumps(mf, indent=1, ensure_ascii=False))
    print(f"\nupdated: {bundle_path.name}, manifest.json")


if __name__ == '__main__':
    main()
