"""Unified CoreClip CLI:

    PYTHONPATH=. python3 -m mocap.humanoid21.clip <command> ...

Commands:
    check    unified checks: basic / velocity / actuator / ref_action
    play     kinematic state-overwrite replay -> MP4
    fix      ground-penetration root-lift repair
    convert  retarget RefClip npz -> CoreClip npz
    info     quick summary of a clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core_clip import CONTROL_FPS, CoreClip


def _cmd_check(args) -> int:
    from .checks import report_json, report_text, run_checks
    clip = CoreClip.load(args.clip)
    only = args.only.split(",") if args.only else None
    results = run_checks(clip, robot=args.robot, only=only)
    print(report_text(clip, results))
    if args.json:
        Path(args.json).write_text(report_json(clip, results))
        print(f"json report -> {args.json}")
    return 0 if all(r.status != "FAIL" for r in results.values()) else 1


def _cmd_play(args) -> int:
    from .play import render
    clip = CoreClip.load(args.clip)
    print(f"[play] {clip} — kinematic replay (NOT physical execution)")
    render(clip, args.out, fps=args.fps, slow=args.slow,
           burn_frame=args.burn_frame, distance=args.distance,
           robot=args.robot)
    return 0


def _cmd_fix(args) -> int:
    from .check import ground_report
    from .fix import fix_ground_penetration
    clip = CoreClip.load(args.clip)
    fixed = fix_ground_penetration(clip, robot=args.robot,
                                   mode=args.mode, margin=args.margin,
                                   smooth=args.smooth)
    rep = ground_report(fixed, robot=args.robot)
    print(rep.summary())
    out = args.out or str(args.clip).replace(".npz", "_groundfix.npz")
    if Path(out).resolve() == Path(args.clip).resolve():
        print("[fix] refused: --out must differ from input "
              "(never repair in place)", file=sys.stderr)
        return 2
    fixed.save(out)
    print(f"[fix] saved {out}")
    return 0


def _cmd_rederive_vel(args) -> int:
    from .rederive import rederive_velocities
    clip = CoreClip.load(args.clip)
    out = args.out or str(args.clip).replace(".npz", "_vel.npz")
    if Path(out).resolve() == Path(args.clip).resolve():
        print("[rederive-vel] refused: --out must differ from input",
              file=sys.stderr)
        return 2
    new = rederive_velocities(clip, zero_endpoints=args.zero_endpoints)
    new.save(out)
    print(f"[rederive-vel] {clip.clip_id} -> {new.clip_id}; "
          f"endpoints {'zeroed' if args.zero_endpoints else 'kept'}")
    return 0


def _cmd_rederive_action(args) -> int:
    from .rederive_action import rederive_actions, DEFAULT_WEIGHTS
    clip = CoreClip.load(args.clip)
    out = args.out or str(args.clip).replace(".npz", "_act.npz")
    if Path(out).resolve() == Path(args.clip).resolve():
        print("[rederive-action] refused: --out must differ from input",
              file=sys.stderr)
        return 2
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v)
    new = rederive_actions(
        clip, robot=args.robot, weights=weights,
        tol_solved=args.tol_solved, tol_approx=args.tol_approx,
        n_samples=args.n_samples, n_iters=args.n_iters,
        seed=args.seed)
    new.save(out)
    sv = new.meta["ref_action_solver"]
    n_sol = sum(g == "solved" for g in sv["per_frame_grade"])
    print(f"[rederive-action] {clip.clip_id} -> {new.clip_id}: "
          f"{n_sol}/{new.T} solved, "
          f"unsolved={sv['unsolved_frames']}")
    return 0


def _cmd_convert(args) -> int:
    from .convert import refclip_to_coreclip
    from mocap.humanoid21.tracking.ref_data import RefClip
    ref = RefClip(args.ref_id, args.dir)
    cc = refclip_to_coreclip(
        ref, clip_id=args.clip_id, start=args.start, end=args.end,
        with_ref_action=not args.no_ref_action)
    cc.save(args.out)
    print(f"[convert] {cc} -> {args.out}")
    return 0


def _cmd_info(args) -> int:
    clip = CoreClip.load(args.clip)
    print(f"{clip}")
    for f in ("root_pos", "joint_pos_norm"):
        v = clip.states[f]
        print(f"  {f}: {v.shape} min={v.min():.3f} max={v.max():.3f}")
    import json as _json
    print("  meta:", _json.dumps(clip.meta, ensure_ascii=False)[:400])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clip",
                                 description="CoreClip workbench CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="run clip checks")
    p.add_argument("--clip", required=True)
    p.add_argument("--robot", default="robot_a")
    p.add_argument("--only", default=None,
                   help="comma list: basic,velocity,actuator,ref_action")
    p.add_argument("--json", default=None)
    p.set_defaults(fn=_cmd_check)

    p = sub.add_parser("play", help="kinematic replay -> mp4")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=CONTROL_FPS)
    p.add_argument("--slow", type=float, default=1.0)
    p.add_argument("--burn-frame", action="store_true")
    p.add_argument("--distance", type=float, default=6.0)
    p.add_argument("--robot", default="robot_a")
    p.set_defaults(fn=_cmd_play)

    p = sub.add_parser("fix", help="ground-penetration root-lift fix")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--robot", default="robot_a")
    p.add_argument("--mode", default="lowest",
                   choices=("lowest", "constant"))
    p.add_argument("--margin", type=float, default=0.002)
    p.add_argument("--smooth", type=int, default=3)
    p.set_defaults(fn=_cmd_fix)

    p = sub.add_parser("rederive-vel",
                       help="recompute velocities from pose trajectory")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--zero-endpoints", action="store_true",
                   help="zero first/last frame velocities "
                        "(default keeps recorded values)")
    p.set_defaults(fn=_cmd_rederive_vel)

    p = sub.add_parser("rederive-action",
                       help="best-effort per-frame action solve (CEM)")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--robot", default="robot_a")
    p.add_argument("--weights", default=None,
                   help="override e.g. jp=1.0,rp=1.0 (csv)")
    p.add_argument("--tol-solved", type=float, default=0.05)
    p.add_argument("--tol-approx", type=float, default=0.2)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_cmd_rederive_action)

    p = sub.add_parser("convert", help="RefClip npz -> CoreClip")
    p.add_argument("--ref-id", required=True)
    p.add_argument("--dir", required=True,
                   help="directory containing <ref_id>.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--clip-id", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--no-ref-action", action="store_true")
    p.set_defaults(fn=_cmd_convert)

    p = sub.add_parser("info", help="quick clip summary")
    p.add_argument("--clip", required=True)
    p.set_defaults(fn=_cmd_info)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
