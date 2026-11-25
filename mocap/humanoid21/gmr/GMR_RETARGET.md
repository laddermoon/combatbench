# GMR Retarget for humanoid21 — Status & Pitfalls

## Setup

- GMR clone: `mocap/gmr/` (independent git repo, local branch `humanoid21`, ignored by combatbench git)
- Editable install into `mocap/gmr/.venv` (setuptools needed upgrade for PEP 660); reinstalled `mujoco==3.8.0` after GMR pulled 3.17
- Body models: `gmr/assets/body_models/smplx/` ← copied from `mocap/smpl_models/smplx/` (SMPL-X only; GMR does not accept smplh input)
- Robot MJCF: `mocap/humanoid21/gmr/humanoid21_mocap.xml` — `_a` chain extracted from `battle_circular_v2.xml`, root at origin (arena spawn offset x=-1.5 removed), nq=28 (7 freejoint + 21 hinge), 17 bodies
- IK config: `mocap/humanoid21/gmr/smplx_to_humanoid21.json`
- All humanoid21-specific files live in `mocap/humanoid21/gmr/` (this dir, git-tracked by combatbench). Inside the gmr clone they appear as **symlinks** so `params.py` lookups keep working: `gmr/assets/humanoid21/humanoid21_mocap.xml` and `gmr/.../ik_configs/smplx_to_humanoid21.json`
- Registered in `gmr/.../params.py` (ROBOT_XML_DICT, IK_CONFIG_DICT['smplx'], ROBOT_BASE_DICT='pelvis_a', VIEWER_CAM_DISTANCE)

## Run (headless)

```bash
# from repo root
MUJOCO_GL=egl mocap/gmr/.venv/bin/python mocap/humanoid21/gmr/smplx_to_robot_headless.py \
  --smplx_file <amass_smplx.npz> --robot humanoid21 --save_path out.pkl --render_video out.mp4
```

GLFW viewer does not work on this box; use EGL + offline `mujoco.Renderer`.

## Coordinate conventions (verified)

- smplx model zero-pose: T-pose, **Y-up intrinsic**, left arm +X, faces +Z, feet at y≈-1.23
- GMR emits frames as `Rfix * FK` where `Rfix = Rx(+90)` (their Y-up→Z-up fix), i.e. emitted world is Z-up
- humanoid21 q0: faces **+X** (toe geom +X, left side +Y), arms in combat guard (upper arm dir (±.58,±.58,-.58), forearm up-forward), ALL body frames axis-aligned identity at q0
- `rot_offset` (config [wxyz]) = `emitted_orient_ref^-1 * robot_orient_ref`; pos_offset applied in the UPDATED human joint frame — keep zeros for asymmetric-morphology robots
- `human_scale_table` scales each human joint's offset-from-root (limb ratio); use 0.9 flat; `human_height_assumption` + `actual_human_height` rescale per clip

## Verified results

- Reference-pose check (no IK): all link ang_err = 0.0°
- SSM walking_01: natural upright gait, pelvis_z 0.84, arms swing at sides. `/tmp/h21_walk3.mp4`
- SSM punch_kick: guard + leg kick + strikes visible. `/tmp/h21_pk.mp4`
- Known residual: right foot_z dips to -0.025m (L1 threshold -0.005); fast strikes have arm residual — morphology gap, flag via `high_residual_*` per DATASPEC

## Pitfalls hit (do not rediscover)

1. **`ik_match_table1` must contain EVERY human joint used** — `offset_human_data` applies *table1's* offsets to all 55 emitted joints; joints only in table2 get identity offset => constant ~120° orientation error on all limbs (was the "crouched walk" bug)
2. **Arm orientation tracking is unreachable for humanoid21** — zigzag arm chain + 1-DOF hinge elbows cannot realize human forearm twist deltas; IK diverged into arms-up fold. Fix: arms position-dominant (upper_arm pos10/ori2, lower_arm pos15/ori0, hand pos30/ori0). Wrist position is what matters for strikes anyway
3. rot_offset convention: `target_orient = emitted_orient * rot_offset` (right-multiply, human-local frame). Offsets must be computed against a semantically-matched reference pose — robot q0 guard ↔ a human guard built with matching bone directions, NOT human T-pose (else 120°+ systematic error on arms)
4. GMR `load_smplx_file` needed: `betas[:, :10]` (model has 10 PCs; AMASS gives 16), `expression` arg, float32 casting
5. `mano_v1_2` ready-made SMPLH pkls are unusable (missing hands_* keys, chumpy lazy Select) — see AMASS_DOWNLOAD_DETAIL.md
6. `smplx` pip pkg downgraded cudnn to torch-2.7.1 pin 9.5.1.17 — verified OK
7. AMASS smplx npz fields: `pose_body/root_orient/trans`, `mocap_frame_rate` (no i), `num_betas`; SSM/DFaust/CNRS/WEIZMANN are smplx-only
8. **Per-frame IK branch-snap at clip start**: first ~3 frames settle into a wrong IK branch then snap to another one mid-clip on a nearly-static target -> bogus hand jumps (0.44m, "26.8 m/s"). Fix: 10x burn-in `retarget(frame[0])` before recording (`mocap/humanoid21/gmr/retarget_smplx_batch.py`); afterwards peak hand speeds 1.9-11.5 m/s all mid-clip, matching human ~10.3 m/s
9. Source fps: AMASS smplx = 120fps; `tgt_fps=30` keeps duration (~1% off via aligned_fps); fast clips lose transient peaks at 30fps (peak hand speed halved) — use `TGT_FPS=60` for review, keep 30 for the PHC pipeline

## Next

- Port GMR output qpos to PHC-family pkl (`{clip: {fps, root_pos, root_rot(quat order!), dof_pos}}`) → `kinematic_check.py` L1 → `track_replay`/M2
- Consider `use_actual_human_height` + per-clip beta-aware scaling (currently flat 0.9)
