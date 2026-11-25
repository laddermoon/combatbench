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
import mink
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
# Arm-extension bias (see GMR_RETARGET.md pitfall #10): the zigzag forearm
# reaches the wrist target with the elbow folded (~0deg), so strikes look
# bent.  A low-cost PostureTask on the elbow dofs tracks the SMPL-X elbow
# FLEXION angle (pose_body rotvec, flexion axis = local +Y right / -Y left),
# re-anchored so both "straight" ends coincide: target = flexion - 100deg
# (human flexion 0=straight -> robot -100=straight; flexion ~110 fold ->
# robot +10).  Wrist task (w=30) still owns the endpoint.
ARM_EXT_COST = 10.0
ARM_EXT_SLEW_DEG = 15.0                     # max target swing per frame
ARM_EXT_OFFSET_DEG = 100.0                  # straight-anchor offset
ARM_EXT_MIN_DEG, ARM_EXT_MAX_TGT_DEG = -100.0, 30.0
ARM_FLEX_SMOOTH = np.array([0.06, 0.25, 0.38, 0.25, 0.06])
ELBOW_JOINTS = ('elbow_right_a', 'elbow_left_a')


class GMRArmBias(GMR):
    """GMR + elbow PostureTask tracking the SMPL-X elbow flexion angle."""

    def __init__(self, *a, el_targets=None, **kw):
        super().__init__(*a, **kw)
        m = self.configuration.model
        cost = np.zeros(m.nv)
        self._el_qadr = []
        for jn in ELBOW_JOINTS:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            cost[m.jnt_dofadr[j]] = ARM_EXT_COST
            self._el_qadr.append(m.jnt_qposadr[j])
        self._el_pt = mink.PostureTask(m, cost)
        self.tasks1.append(self._el_pt)
        self._el_pt.set_target(self.configuration.q.copy())
        self._el_tgts = el_targets             # (T,2) deg, from elbow_flexion_targets
        self._pending_tgt = None
        self._el_tgt = np.zeros(len(ELBOW_JOINTS))   # slew-limited target rad

    def set_target_frame(self, i):
        if self._el_tgts is not None:
            self._pending_tgt = self._el_tgts[i]

    def update_targets(self, human_data, offset_to_ground=False):
        super().update_targets(human_data, offset_to_ground)
        if self._pending_tgt is None:
            return
        # slew-rate-limit the elbow target so the posture pull can't snap
        goal = np.deg2rad(np.asarray(self._pending_tgt))
        slew = np.deg2rad(ARM_EXT_SLEW_DEG)
        self._el_tgt += np.clip(goal - self._el_tgt, -slew, slew)
        q = self.configuration.q.copy()
        q[self._el_qadr] = self._el_tgt
        self._el_pt.set_target(q)


def elbow_flexion_targets(frames, smplx_data):
    """(T,2) per-frame robot elbow targets (deg) from SMPL-X pose_body.

    pose_body[i] = JOINT_NAMES[i+1] axis-angle; elbow flexion axis = local
    +Y (right) / -Y (left) — verified corr 0.98 with geometric bend.
    Robot target = human flexion - 100deg (both 'straight' ends aligned).
    """
    pb = np.asarray(smplx_data['pose_body'], np.float64)
    T_src, T = len(pb), len(frames)
    idx = np.clip(np.linspace(0, T_src - 1, T).round().astype(int), 0, T_src - 1)
    out = np.zeros((T, 2), np.float32)
    for k, jn in enumerate(('right_elbow', 'left_elbow')):
        i = (JOINT_NAMES.index(jn) - 1) * 3
        sign = 1.0 if jn.startswith('right') else -1.0
        flex = np.degrees(pb[idx, i + 1] * sign)              # local Y comp
        flex = np.convolve(np.clip(flex, 0, 150), ARM_FLEX_SMOOTH, 'same')
        out[:, k] = np.clip(flex - ARM_EXT_OFFSET_DEG,
                            ARM_EXT_MIN_DEG, ARM_EXT_MAX_TGT_DEG)
    return out


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


def retarget_clip(npz_path, arm_bias=True):
    smplx_data, body_model, smplx_output, h = load_smplx_file(
        str(npz_path), SMPLX_FOLDER)
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=TGT_FPS)
    if arm_bias:
        rt = GMRArmBias(actual_human_height=h, src_human='smplx',
                        tgt_robot='humanoid21',
                        el_targets=elbow_flexion_targets(frames, smplx_data))
        rt.set_target_frame(0)
    else:
        rt = GMR(actual_human_height=h, src_human='smplx',
                 tgt_robot='humanoid21')
    for _ in range(BURN_IN):                 # kill branch-snap at clip start
        rt.retarget(frames[0])
    qpos, smpl = [], []
    for i, f in enumerate(frames):
        if arm_bias:
            rt.set_target_frame(i)
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


def export_clip(npz_arg, clip_id, labels, split='train', dataset=None,
                bundle='dev', arm_bias=True, model_cache=None):
    """Retarget one npz and insert it into a bundle. Returns clip_id."""
    npz_path, src_tree = find_clip(npz_arg)
    if model_cache is None:
        model_cache = load_model()
    mm, dd, dof_names, dof_axis, bids, kp_names = model_cache

    qpos, smpl, fps, smplx_data = retarget_clip(npz_path, arm_bias=arm_bias)
    root_pos, root_rot, dof, pose_aa, kp = build_clip_dict(
        qpos, smpl, fps, mm, dd, dof_names, dof_axis, bids)

    src_fps = float(np.asarray(smplx_data.get(
        'mocap_frame_rate', smplx_data.get('mocap_framerate', 120.0))).flat[0])
    labels = [labels] if isinstance(labels, str) else list(labels)
    meta = {
        "src_npz": npz_arg if src_tree else str(npz_path),
        "src_tree": src_tree or "absolute",
        "src_sha256": sha256(npz_path),
        "src_fps": src_fps,
        "labels": labels,
        "split": split,
        "dataset": dataset,
        "parent_clip": None,
        "retarget_method": "gmr_mink_ik",
        "retarget_params": {"ik_config": "smplx_to_humanoid21.json",
                            "tgt_fps": TGT_FPS, "burn_in": BURN_IN,
                            "scale_table_flat": 0.9,
                            "arm_extension_bias": None if not arm_bias else
                            {"mode": "smplx_flexion_direct",
                             "cost": ARM_EXT_COST,
                             "offset_deg": ARM_EXT_OFFSET_DEG,
                             "slew_deg_per_frame": ARM_EXT_SLEW_DEG,
                             "smooth_kernel": ARM_FLEX_SMOOTH.tolist()}},
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
    bundle_name = f'combatbench_{bundle}_v1.pkl'
    bundle_path = PHC_DIR / bundle_name
    bundle = pickle.load(open(bundle_path, 'rb')) if bundle_path.exists() else {}
    bundle[clip_id] = clip
    payload = pickle.dumps(bundle)
    bundle_path.write_bytes(payload)
    bsha = sha256(None, payload)

    clip_rec = {"frames": int(len(qpos)),
                "duration_s": round(len(qpos) / fps, 2),
                "split": split,
                "labels": meta["labels"], "dataset": meta["dataset"],
                "src_sha256": meta["src_sha256"],
                "quality_level": "L0", "issues": []}
    update_manifest(mm, dof_names, dof_axis, kp_names,
                    bundle_name, bsha, len(bundle), clip_id, clip_rec)

    print(f"[OK] {clip_id}: T={len(qpos)} @{fps:.0f}fps -> {bundle_name} "
          f"({len(bundle)} clips, sha {bsha[:12]})", flush=True)
    print(f"     pelvis z range [{kp[:, 0, 2].min():.3f},{kp[:, 0, 2].max():.3f}] "
          f"foot z min {min(kp[:, kp_names.index('foot_left'), 2].min(), kp[:, kp_names.index('foot_right'), 2].min()):.3f}",
          flush=True)
    return clip_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', help='amass rel path (searched in both smplx trees) or abs')
    p.add_argument('--clip_id', required=True)
    p.add_argument('--labels', default='', help='comma list')
    p.add_argument('--split', default='train', choices=['train', 'val'])
    p.add_argument('--dataset', default='none', choices=['core', 'extended', 'none'])
    p.add_argument('--bundle', default='dev', help='bundle tag -> combatbench_<tag>_v1.pkl')
    p.add_argument('--no-arm-bias', action='store_true',
                   help='disable elbow-extension PostureTask bias')
    args = p.parse_args()

    export_clip(args.npz, args.clip_id,
                [x for x in args.labels.split(',') if x],
                split=args.split,
                dataset=None if args.dataset == 'none' else args.dataset,
                bundle=args.bundle, arm_bias=not args.no_arm_bias)


if __name__ == '__main__':
    main()
