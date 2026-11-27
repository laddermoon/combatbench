"""CLI for the humanoid28 clip package.

  python -m mocap.humanoid28.clip phc2core [--bundle core] [--only SUB]
  python -m mocap.humanoid28.clip play-env (--clip npz | --dir DIR) ...
  python -m mocap.humanoid28.clip info --clip npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core_clip import CoreClip


def _default_lib(bundle: str) -> Path:
    return Path(__file__).resolve().parents[1] / 'data' / bundle


def _cmd_phc2core(args) -> int:
    import pickle
    from .convert import phc_to_coreclip
    phc_dir = Path(__file__).resolve().parents[1] / 'phc'
    bundle = pickle.load(open(
        phc_dir / f'combatbench_{args.bundle}_v1.pkl', 'rb'))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in sorted(bundle):
        if args.only and args.only not in cid:
            continue
        cc = phc_to_coreclip(bundle[cid], clip_id=cid, bundle=args.bundle,
                             with_ref_action=not args.no_ref_action)
        out = outdir / f'{cid}.npz'
        cc.save(out)
        n += 1
        iss = bundle[cid]['meta'].get('issues', [])
        print(f'[OK] {cid}.npz T={cc.T} bad_frames={len(cc.meta["rsi_bad_frames"])}'
              f' issues={iss}')
    print(f'done: {n} clips -> {outdir}')
    return 0


def _cmd_playenv(args) -> int:
    import os
    os.environ['MUJOCO_GL'] = 'osmesa'     # set before mujoco/backend init
    from .replay_env import render_clip
    if args.clip:
        clips = [Path(args.clip)]
    else:
        clips = sorted(Path(args.dir).glob('*.npz'))
        if args.only:
            clips = [c for c in clips if args.only in c.stem]
    for p in clips:
        cc = CoreClip.load(p)
        out = Path(args.out or (Path(args.outdir) / f'{p.stem}.mp4'))
        render_clip(cc, out, robot=args.robot)
        print(f'[OK] {p.stem}.mp4')
    return 0


def _cmd_info(args) -> int:
    clip = CoreClip.load(args.clip)
    print(repr(clip))
    import json
    print(json.dumps(clip.meta, indent=1, default=str)[:4000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m mocap.humanoid28.clip')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('phc2core', help='PHC bundle pkl -> CoreClip28 npz dir')
    p.add_argument('--bundle', default='core')
    p.add_argument('--only', default=None, help='substring filter on clip_id')
    p.add_argument('--outdir', default=None)
    p.add_argument('--no-ref-action', action='store_true')
    p.set_defaults(fn=_cmd_phc2core)

    p = sub.add_parser('play-env', help='kinematic replay on battle XML -> mp4')
    p.add_argument('--clip', default=None)
    p.add_argument('--dir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--outdir', default=None)
    p.add_argument('--only', default=None)
    p.add_argument('--robot', default='robot_a')
    p.set_defaults(fn=_cmd_playenv)

    p = sub.add_parser('info', help='dump clip summary')
    p.add_argument('--clip', required=True)
    p.set_defaults(fn=_cmd_info)

    args = ap.parse_args(argv)
    if args.cmd == 'phc2core' and not args.outdir:
        args.outdir = str(_default_lib(args.bundle))
    if args.cmd == 'play-env' and not args.clip and not args.dir:
        ap.error('play-env needs --clip or --dir')
    if args.cmd == 'play-env' and not args.out and not args.outdir:
        ap.error('play-env needs --out or --outdir')
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
