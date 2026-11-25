"""AMASS smplx npz -> GMR retarget -> PHC clip_dict -> bundle pkl + manifest.

Per DATASPEC v2: PHC-family fields (fps/root_trans_offset/root_rot xyzw/dof/
pose_aa/smpl_joints) + self-contained extensions (dof_names, keypoint_pos,
meta). One invocation inserts/overwrites one clip into a bundle.

Usage (repo root):
  mocap/gmr/.venv/bin/python mocap/humanoid21/gmr/export_phc.py \
      "ACCAD/Male2MartialArtsPunches_c3d/G1 - jab cross_poses.npz" \
      --clip_id accad_m2ma_punches_g1_jab_cross --labels strike_punch \
      --split train --dataset core --bundle dev
"""
import argparse, hashlib, json, os, pathlib, pickle, sys

os.environ.setdefault('MUJOCO_GL', 'egl')
HERE = pathlib.Path(__file__).resolve().parent
MOCAP = HERE.parents[1]                       # mocap/
GMR_ROOT = MOCAP / 'gmr'
sys.path.insert(0, str(GMR_ROOT))

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast, load_smplx_file)
from smplx.joint_names import JOINT_NAMES

SMPLX_FOLDER = GMR_ROOT / 'assets' / 'body_models'
AMASS_ROOTS = [MOCAP / 'amass' / 'smplx_gender_specific',
               MOCAP / 'amass' / 'smplx_from_smplh']
LIBRARY = pathlib.Path('/data1/dev/combatbench_mocap')
PHC_DIR = LIBRARY / 'phc'
MANIFEST = PHC_DIR / 'manifest.json'
MOCAP_XML = HERE / 'humanoid21_mocap.xml'
TGT_FPS = 30
BURN_IN = 10
SMPL_JOINTS = JOINT_NAMES[:22]                # pelvis..right_wrist


def find_clip(rel):
    p = pathlib.Path(rel)
    if p.is_absolute():
        return p, None
    for root in AMASS_ROOTS:
        if (root / rel).exists():
            return root / rel, root.name
    raise FileNotFoundError(rel)


def sha256(path, nbytes=None):
    h = hashlib.sha256()
    if nbytes is None:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    else:
        h.update(nbytes)
    return h.hexdigest()


def load_model():
    mm = mujoco.MjModel.from_xml_path(str(MOCAP_XML))
    dd = mujoco.MjData(mm)
    # dof order = mj joint order minus freejoint (== XML document order ==
    # CONTROLLED_JOINTS). names suffix-less for the spec.
    jids = [i for i in range(mm.njnt) if mm.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    dof_names = [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix('_a')
                 for j in jids]
    dof_axis = mm.jnt_axis[jids].astype(np.float32)      # hinge axis in parent frame
    # all bodies except worldbody, XML order (torso is index 1 = root body)
    bids = list(range(1, mm.nbody))
    kp_names = [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, b).removesuffix('_a')
                for b in bids]
    return mm, dd, dof_names, dof_axis, np.array(bids), kp_names


def retarget_clip(npz_path):
    smplx_data, body_model, smplx_output, h = load_smplx_file(
        str(npz_path), SMPLX_FOLDER)
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=TGT_FPS)
    rt = GMR(actual_human_height=h, src_human='smplx', tgt_robot='humanoid21')
    for _ in range(BURN_IN):                 # kill branch-snap at clip start
        rt.retarget(frames[0])
    qpos, smpl = [], []
    for f in frames:
        qpos.append(rt.retarget(f))
        # smpl_joints = GMR-emitted human keypoints (Z-up world, native human
        # scale, pre-IK-target) — 22 SMPL-X body joints
        smpl.append([f[n][0] for n in SMPL_JOINTS])
    return np.asarray(qpos, np.float32), np.asarray(smpl, np.float32), fps, smplx_data


def build_clip_dict(qpos, smpl, fps, mm, dd, dof_names, dof_axis, bids):
    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7][:, [1, 2, 3, 0]]            # wxyz -> xyzw
    dof = qpos[:, 7:]
    T = len(qpos)

    pose_aa = np.zeros((T, 1 + len(dof_names), 3), np.float32)
    pose_aa[:, 0] = R.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa[:, 1:] = dof[:, :, None] * dof_axis[None]

    kp = np.zeros((T, len(bids), 3), np.float32)
    for t in range(T):
        dd.qpos[:] = qpos[t]
        mujoco.mj_forward(mm, dd)
        kp[t] = dd.xpos[bids]
    return root_pos, root_rot, dof, pose_aa, kp


def foot_contacts(kp, kp_names, fps):
    fl = kp_names.index('foot_left'); fr = kp_names.index('foot_right')
    z = kp[:, [fl, fr], 2]
    spd = np.linalg.norm(np.gradient(kp[:, [fl, fr]], 1.0 / fps, axis=0), axis=2)
    return ((z < 0.06) & (spd < 0.5)).astype(np.float32)


def update_manifest(mm, dof_names, dof_axis, kp_names, bundle_name, bundle_sha,
                    n_clips, clip_id, clip_rec):
    if MANIFEST.exists():
        mf = json.loads(MANIFEST.read_text())
    else:
        mf = {"spec_version": 2, "quat_order": "xyzw",
              "velocity_method": "finite_difference_np_gradient",
              "bundles": {}, "clips": {}}
    mf["robot"] = {
        "model_xml": str(MOCAP_XML.relative_to(MOCAP.parents[1])),
        "model_sha256": sha256(MOCAP_XML),
        "dof_names": dof_names,
        "dof_axis": np.asarray(dof_axis).tolist(),
        "keypoint_names": kp_names,
        "smpl_joint_names": SMPL_JOINTS,
    }
    mf["bundles"][bundle_name] = {"n_clips": n_clips, "sha256": bundle_sha}
    mf["clips"][clip_id] = clip_rec
    MANIFEST.write_text(json.dumps(mf, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', help='amass rel path (searched in both smplx trees) or abs')
    p.add_argument('--clip_id', required=True)
    p.add_argument('--labels', default='', help='comma list')
    p.add_argument('--split', default='train', choices=['train', 'val'])
    p.add_argument('--dataset', default='none', choices=['core', 'extended', 'none'])
    p.add_argument('--bundle', default='dev', help='bundle tag -> combatbench_<tag>_v1.pkl')
    args = p.parse_args()

    npz_path, src_tree = find_clip(args.npz)
    mm, dd, dof_names, dof_axis, bids, kp_names = load_model()

    qpos, smpl, fps, smplx_data = retarget_clip(npz_path)
    root_pos, root_rot, dof, pose_aa, kp = build_clip_dict(
        qpos, smpl, fps, mm, dd, dof_names, dof_axis, bids)

    src_fps = float(np.asarray(smplx_data.get(
        'mocap_frame_rate', smplx_data.get('mocap_framerate', 120.0))).flat[0])
    rel_npz = args.npz if src_tree else str(npz_path)
    meta = {
        "src_npz": rel_npz,
        "src_tree": src_tree or "absolute",
        "src_sha256": sha256(npz_path),
        "src_fps": src_fps,
        "labels": [x for x in args.labels.split(',') if x],
        "split": args.split,
        "dataset": None if args.dataset == 'none' else args.dataset,
        "parent_clip": None,
        "retarget_method": "gmr_mink_ik",
        "retarget_params": {"ik_config": "smplx_to_humanoid21.json",
                            "tgt_fps": TGT_FPS, "burn_in": BURN_IN,
                            "scale_table_flat": 0.9},
        "quality_level": "L0",
        "issues": [],
        "contact_method": "foot_z<0.06 & foot_speed<0.5",
    }
    clip = {
        "fps": TGT_FPS,
        "root_trans_offset": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof": dof.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "smpl_joints": smpl.astype(np.float32),
        "dof_names": np.array(dof_names),
        "keypoint_pos": kp.astype(np.float32),
        "foot_contacts": foot_contacts(kp, kp_names, fps),
        "meta": meta,
    }

    PHC_DIR.mkdir(parents=True, exist_ok=True)
    bundle_name = f'combatbench_{args.bundle}_v1.pkl'
    bundle_path = PHC_DIR / bundle_name
    bundle = pickle.load(open(bundle_path, 'rb')) if bundle_path.exists() else {}
    bundle[args.clip_id] = clip
    payload = pickle.dumps(bundle)
    bundle_path.write_bytes(payload)
    bsha = sha256(None, payload)

    clip_rec = {"frames": int(len(qpos)),
                "duration_s": round(len(qpos) / fps, 2),
                "split": args.split,
                "labels": meta["labels"], "dataset": meta["dataset"],
                "src_sha256": meta["src_sha256"],
                "quality_level": "L0", "issues": []}
    update_manifest(mm, dof_names, dof_axis, kp_names,
                    bundle_name, bsha, len(bundle), args.clip_id, clip_rec)

    print(f"[OK] {args.clip_id}: T={len(qpos)} @{fps:.0f}fps -> {bundle_name} "
          f"({len(bundle)} clips, sha {bsha[:12]})")
    print(f"     pelvis z range [{kp[:, 0, 2].min():.3f},{kp[:, 0, 2].max():.3f}] "
          f"foot z min {min(kp[:, kp_names.index('foot_left'), 2].min(), kp[:, kp_names.index('foot_right'), 2].min()):.3f}")


if __name__ == '__main__':
    main()
