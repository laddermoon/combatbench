#!/usr/bin/env python3
"""Batch-convert AMASS smplh npz -> smplx-format npz (field-level, motion-preserving).

SMPLH and SMPLX share the same first-22-joint kinematic tree and local axis
conventions, so poses[:, :66] == root_orient(3) + pose_body(63). The trailing
90 hand-pose dims are dropped (humanoid21 has no fingers; GMR zeroes hands
anyway). betas stay (16,) - loaders slice [:10].

Input : amass/smplh_gender_specific/<DS>/<subj>/*.npz
Output: amass/smplx_from_smplh/<DS>/<subj>/<same name>.npz   (resume-safe)

Usage:
  python3 convert_smplh_to_smplx.py [--src DIR] [--dst DIR] [--dataset CMU KIT]
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_SRC = os.path.join(HERE, 'amass/smplh_gender_specific')
DEF_DST = os.path.join(HERE, 'amass/smplx_from_smplh')


def convert_one(inp: str, outp: str) -> str:
    d = np.load(inp, allow_pickle=True)
    if 'poses' not in d.files:
        return 'skip: no poses key'
    poses = d['poses']
    if poses.ndim != 2 or poses.shape[1] < 66:
        return f'skip: poses shape {poses.shape}'
    n = poses.shape[0]
    fps = float(d['mocap_framerate']) if 'mocap_framerate' in d.files else \
          float(d['mocap_frame_rate']) if 'mocap_frame_rate' in d.files else 120.0
    out = {
        'root_orient': poses[:, :3],
        'pose_body': poses[:, 3:66],
        'trans': d['trans'],
        'betas': d['betas'],
        'num_betas': np.array(d['betas'].shape[0], dtype=np.int64),
        'gender': d['gender'] if 'gender' in d.files else np.array('neutral'),
        'mocap_frame_rate': np.array(fps),
        'mocap_time_length': np.array(n / fps),
        'surface_model_type': np.array('smplx'),
        'converted_from': np.array('smplh'),
    }
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    tmp = outp + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, outp)
    return 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEF_SRC)
    ap.add_argument('--dst', default=DEF_DST)
    ap.add_argument('--dataset', nargs='*', default=None)
    args = ap.parse_args()

    files = []
    for root, _, fns in os.walk(args.src):
        if args.dataset:
            rel = os.path.relpath(root, args.src)
            if rel.split(os.sep)[0] not in args.dataset:
                continue
        for fn in fns:
            if fn.endswith('.npz'):
                files.append(os.path.join(root, fn))
    files.sort()
    print(f'{len(files)} smplh npz found under {args.src}')

    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, inp in enumerate(files):
        rel = os.path.relpath(inp, args.src)
        outp = os.path.join(args.dst, rel)
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            n_skip += 1
            continue
        try:
            res = convert_one(inp, outp)
        except Exception as e:
            res = f'fail: {e}'
        if res == 'ok':
            n_ok += 1
        elif res.startswith('skip'):
            n_skip += 1
        else:
            n_fail += 1
            print(f'[FAIL] {rel}: {res}', file=sys.stderr)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f'[{i + 1}/{len(files)}] ok={n_ok} skip={n_skip} fail={n_fail} '
                  f'({el:.0f}s, ~{el / (i + 1) * len(files) / 60:.0f}min total)')
    print(f'done: ok={n_ok} skip={n_skip} fail={n_fail} -> {args.dst}')


if __name__ == '__main__':
    main()
