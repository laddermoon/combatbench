"""Extract standard dynamic stepping clips from policy-capture rollouts.

Route B of the hierarchy plan: harvest clean *cyclic two-foot stepping*
segments from a converged stepping policy (``step_mbs_s42``) captured by
``mocap/humanoid21/policy_capture`` and distill them into the clip library
consumed by ``StandardClipObserver``.

Feature contract (must stay in sync with the online observer):

    f_t = [ q_norm (21)        -- (q - MID) / HALF, same as sim joint_pos_norm
          , qv_norm (21)       -- qv / HALF,      same as sim joint_vel_norm
          , gravity_proj (3)   -- R^T @ [0,0,-1]
          , foot_L rel pelvis in pelvis frame (3)
          , foot_R rel pelvis in pelvis frame (3)
          , foot_L contact (1) -- feet_forces > CONTACT_THRESH
          , foot_R contact (1) ]                        # -> 53 dims

The observer pairs frames into transitions ``z_t = [f_{t-DELTA}, f_t]``
and scores ``max_{clip,frame} exp(-alpha * ||z - z_ref||_W^2)``.

Usage::

    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/extract_standard_clips.py \
        --capture-dir /data1/dev/policy_capture/step_mbs_stepping \
        --out baseline/humanoid21/hierarchy/assets/standard_stepping_clips.npz \
        --num-clips 5 --window 40 --stride 8 \
        --plot baseline/humanoid21/hierarchy/assets/standard_stepping_clips.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits
from envs.humanoid21.batch_binding import quat_to_rot_np

_JLIM = np.array([joint_limits(n) for n in CONTROLLED_JOINTS], dtype=np.float64)
_MID = ((_JLIM[:, 0] + _JLIM[:, 1]) / 2.0).astype(np.float32)
_HALF = ((_JLIM[:, 1] - _JLIM[:, 0]) / 2.0).astype(np.float32)

# body_xpos indices (see policy_capture meta body_names order)
_PELVIS, _FOOT_R, _FOOT_L = 3, 6, 9
_CONTACT_THRESH = 0.15          # feet_forces normalized; >0.15 = loaded
DELTA = 4                       # transition span (action steps) = 200 ms

# ---- per-dim distance weights for ||.||_W (same groups as plan) ----
F_DIM = 53
W_Q, W_QV, W_G, W_FEET, W_CONTACT = 1.0, 0.5, 1.0, 3.0, 1.0

# Feature-group boundaries for the whitening std floor: a per-dim std of
# ~0 (e.g. gravity stays upright throughout the gait) would otherwise get
# infinite weight after whitening and dominate the distance.  Each dim's
# effective std is floored at FLOOR_FRAC x its group's median std.
GROUPS = ((0, 21), (21, 42), (42, 45), (45, 51), (51, 53))
STD_FLOOR_FRAC = 0.25


def whitening_params(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sd_eff) for z-scoring clip/live features identically."""
    flat = feats.reshape(-1, feats.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd_eff = sd.copy()
    for a, b in GROUPS:
        floor = STD_FLOOR_FRAC * float(np.median(sd[a:b]))
        sd_eff[a:b] = np.maximum(sd[a:b], floor)
    return mu.astype(np.float32), sd_eff.astype(np.float32)


def feature_weights() -> np.ndarray:
    w = np.zeros(F_DIM, dtype=np.float32)
    w[0:21] = W_Q
    w[21:42] = W_QV
    w[42:45] = W_G
    w[45:51] = W_FEET
    w[51:53] = W_CONTACT
    return w


def frames_to_features(d: dict) -> np.ndarray:
    """npz trajectory -> (T, 53) per-frame feature matrix."""
    q = d['joint_angles'].astype(np.float64)
    qv = d['joint_velocities'].astype(np.float64)
    quat = d['root_quat'].astype(np.float64)            # (T,4) wxyz
    bx = d['body_xpos'].astype(np.float64)              # (T,16,3)
    ff = d['feet_forces'].astype(np.float64)            # (T,2) L,R
    T = q.shape[0]

    q_norm = (q - _MID) / _HALF
    qv_norm = qv / _HALF

    R = quat_to_rot_np(quat)                                  # (T,3,3)
    grav = np.einsum('tij,j->ti', np.transpose(R, (0, 2, 1)),
                     np.array([0.0, 0.0, -1.0]))              # R^T @ [0,0,-1]
    fl_rel = np.einsum('tij,tj->ti', np.transpose(R, (0, 2, 1)),
                       bx[:, _FOOT_L] - bx[:, _PELVIS])
    fr_rel = np.einsum('tij,tj->ti', np.transpose(R, (0, 2, 1)),
                       bx[:, _FOOT_R] - bx[:, _PELVIS])
    feet_rel = np.concatenate([fl_rel, fr_rel], axis=1)
    contact = (ff > _CONTACT_THRESH).astype(np.float64)
    return np.concatenate(
        [q_norm, qv_norm, grav, feet_rel, contact], axis=1
    ).astype(np.float32)


def window_score(d: dict, feats: np.ndarray, i0: int, L: int) -> float | None:
    """Quality score for window [i0, i0+L); None if it fails hard gates."""
    win = feats[i0:i0 + L]
    root_pos = d['root_pos']
    ff = d['feet_forces']

    # hard gates: upright all frames, in-place, healthy contacts
    if (win[:, 44] > -0.80).any():                       # gravity_proj z
        return None
    if root_pos[i0:i0 + L, 2].min() < 1.05:
        return None
    drift = np.linalg.norm(root_pos[i0 + L - 1, :2] - root_pos[i0, :2])
    if drift > 0.40:
        return None

    loaded_diff = ff[i0:i0 + L, 0] - ff[i0:i0 + L, 1]    # L minus R load
    sig = np.sign(loaded_diff[np.abs(loaded_diff) > 0.1])
    alternations = int(np.sum(sig[1:] * sig[:-1] < 0)) if len(sig) > 1 else 0
    if alternations < 2:                                 # <1 full L-R cycle
        return None

    both = ((ff[i0:i0 + L, 0] > _CONTACT_THRESH)
            & (ff[i0:i0 + L, 1] > _CONTACT_THRESH)).mean()
    z_std = float(root_pos[i0:i0 + L, 2].std())
    # prefer: more alternations, some double-support transitions, stable height
    return float(alternations) + 2.0 * both - 5.0 * z_std


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num-clips', type=int, default=5)
    ap.add_argument('--window', type=int, default=40,
                    help='clip length in action steps (40 = 2s = ~2 cycles)')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--plot', default=None, help='optional diagnostic PNG path')
    args = ap.parse_args()

    files = sorted(Path(args.capture_dir).glob('ep_*_robot_*.npz'))
    if not files:
        sys.exit(f'no npz files in {args.capture_dir}')

    L, M = args.window, args.num_clips
    cands = []          # (score, file, robot, i0, feats_window, ...)
    traj_hash = set()   # deterministic rollouts produce identical episodes;
                        # dedupe by content so picks spread across phases
    for f in files:
        npz = np.load(f)
        d = {k: npz[k] for k in npz.files}
        h = hash(d['joint_angles'].tobytes())
        if h in traj_hash:
            continue
        traj_hash.add(h)
        feats = frames_to_features(d)
        T = feats.shape[0]
        for i0 in range(0, T - L + 1, args.stride):
            s = window_score(d, feats, i0, L)
            if s is not None:
                cands.append((s, str(f), i0, feats[i0:i0 + L],
                              d['root_pos'][i0:i0 + L].copy(),
                              d['joint_angles'][i0:i0 + L].copy(),
                              d['root_quat'][i0:i0 + L].copy(),
                              d['body_xpos'][i0:i0 + L].copy(),
                              d['feet_forces'][i0:i0 + L].copy()))

    cands.sort(key=lambda c: -c[0])
    # dedupe near-identical windows (same file, overlapping ranges)
    picked: list = []
    for c in cands:
        if all(not (c[1] == p[1] and abs(c[2] - p[2]) < L) for p in picked):
            picked.append(c)
        if len(picked) == M:
            break
    if not picked:
        sys.exit('no window passed the quality gates — inspect data first')

    feats = np.stack([p[3] for p in picked])            # (M, L, 53)
    meta = [{
        'source_file': Path(p[1]).name, 'start_step': int(p[2]),
        'score': round(float(p[0]), 3),
        'mean_root_z': round(float(p[4][:, 2].mean()), 3),
        'alternations': int(np.sum(
            np.sign((p[8][:, 0] - p[8][:, 1])[1:])
            * np.sign((p[8][:, 0] - p[8][:, 1])[:-1]) < 0)),
    } for p in picked]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mu, sd_eff = whitening_params(feats)
    np.savez_compressed(
        out,
        clip_features=feats.astype(np.float32),          # (M, L, 53)
        delta=np.int32(DELTA),
        feature_weights=feature_weights(),
        feat_mu=mu,                                      # whitening stats
        feat_sd_eff=sd_eff,
        joint_angles=np.stack([p[5] for p in picked]),   # replay/debug
        root_quat=np.stack([p[6] for p in picked]),
        body_xpos=np.stack([p[7] for p in picked]),
        feet_forces=np.stack([p[8] for p in picked]),
        meta_json=json.dumps(meta, indent=1),
    )
    print(f'[extract] {len(files)} trajectories -> {len(picked)} clips '
          f'(L={L}, delta={DELTA}) -> {out}')
    for i, m in enumerate(meta):
        print(f'  clip{i}: {m}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(picked), 2, figsize=(13, 2.2 * len(picked)),
                                 squeeze=False)
        for i, p in enumerate(picked):
            t = np.arange(L) * 0.05
            ax = axes[i, 0]
            ax.plot(t, p[8][:, 0], label='L', lw=1.2)
            ax.plot(t, p[8][:, 1], label='R', lw=1.2)
            ax.set_ylabel(f'clip{i} force')
            ax.legend(fontsize=7)
            ax.set_title(f'{Path(p[1]).name} +{p[2]}  score={p[0]:.2f}',
                         fontsize=8)
            ax = axes[i, 1]
            ax.plot(t, p[7][:, _FOOT_L, 2], label='foot_L z', lw=1.2)
            ax.plot(t, p[7][:, _FOOT_R, 2], label='foot_R z', lw=1.2)
            ax.plot(t, p[4][:, 2] - p[4][:, 2].mean() + 1.25,
                    label='root z', lw=1.0, ls='--')
            ax.legend(fontsize=7)
            ax.set_ylabel('height (m)')
        for ax in axes[-1]:
            ax.set_xlabel('t (s)')
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print(f'[extract] diagnostic plot -> {args.plot}')


if __name__ == '__main__':
    main()
