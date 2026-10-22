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

from .core_clip import CONTROL_DT, CONTROL_FPS, CoreClip


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


def _cmd_reclip(args) -> int:
    from .reclip import (RECLIP_WEIGHTS, guide_from_clip,
                         interp_guide, reclip)
    source_clip = CoreClip.load(args.clip) if args.clip else None
    if args.start:
        start_clip = CoreClip.load(args.start)
    elif source_clip is not None:
        start_clip = source_clip
    else:
        print("[reclip] either --clip or --start is required",
              file=sys.stderr)
        return 2
    start = start_clip.state(args.start_frame)
    via = CoreClip.load(args.via) if args.via else None
    end_clip = None
    if via is None:
        if source_clip is not None:
            end_clip = source_clip
        else:
            if not args.end:
                print("[reclip] --end required unless --via or --clip "
                      "is given", file=sys.stderr)
                return 2
            end_clip = CoreClip.load(args.end)

    out = args.out or f"{start_clip.clip_id}_reclip.npz"
    out_path = Path(out).resolve()
    inputs = [args.clip, args.start, args.via, args.end]
    if any(p and Path(p).resolve() == out_path for p in inputs):
        print("[reclip] refused: --out must differ from every input clip",
              file=sys.stderr)
        return 2

    if args.method != "path-sweep" and args.root_pos_mode is not None:
        print("[reclip] --root-pos-mode requires --method path-sweep",
              file=sys.stderr)
        return 2

    if args.method == "path-sweep":
        from .objective_evaluator import ReclipObjectiveConfig
        from .path_reclip import (reclip_path_sweep,
                                  reclip_time_search)
        from .target_path import TargetPath, resolve_pose_weights

        weight_overrides = {}
        if args.weights:
            try:
                for kv in args.weights.split(","):
                    k, v = kv.split("=")
                    key = k.strip()
                    if key not in ("jp", "rp", "rq"):
                        raise ValueError(key)
                    weight_overrides[key] = float(v)
            except ValueError as exc:
                print("[reclip] --weights must be a CSV of "
                      "jp=...,rp=...,rq=...", file=sys.stderr)
                return 2
        try:
            pose_weights, root_pos_mode = resolve_pose_weights(
                weight_overrides or None, args.root_pos_mode)
        except ValueError as exc:
            print(f"[reclip] {exc}", file=sys.stderr)
            return 2
        if via is not None:
            path_mode = args.path or "all"
            path_src = via
        elif source_clip is not None:
            path_mode = args.path or ("all" if args.objective == "track"
                                      else "endpoints")
            path_src = source_clip
        else:
            path_mode = args.path or "endpoints"
            path_src = None
        kind = args.objective or ("track" if via is not None or
                                  path_mode == "all" else "pass")
        if kind == "track" and path_mode != "all":
            print("[reclip] track requires --path all", file=sys.stderr)
            return 2
        if kind == "track" and root_pos_mode == "free":
            print("[reclip] track requires --root-pos-mode absolute",
                  file=sys.stderr)
            return 2
        if path_mode == "all" and path_src is None:
            print("[reclip] --path all requires --clip or --via",
                  file=sys.stderr)
            return 2

        if path_src is not None:
            if path_src is source_clip:
                path = TargetPath.from_clip(
                    path_src, path_mode,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    s_ext=args.s_ext, pose_weights=pose_weights,
                    root_pos_mode=root_pos_mode)
            else:
                path = TargetPath.from_clip(
                    path_src, path_mode, s_ext=args.s_ext,
                    pose_weights=pose_weights,
                    root_pos_mode=root_pos_mode)
            if path_mode == "all" and path_src.ref_action is not None:
                if path_src is source_clip:
                    end_i = (args.end_frame if args.end_frame >= 0
                             else path_src.T - 1)
                    init_a = path_src.ref_action[
                        args.start_frame:end_i]
                else:
                    init_a = path_src.ref_action
            else:
                init_a = None
        else:
            end_i = (args.end_frame if args.end_frame >= 0
                     else end_clip.T - 1)
            path = TargetPath.from_pose_pair(
                start, end_clip.state(end_i),
                source_clip_id=f"{start_clip.clip_id}->{end_clip.clip_id}",
                s_ext=args.s_ext, pose_weights=pose_weights,
                root_pos_mode=root_pos_mode)
            init_a = None

        search_budgets = None
        if args.budget_search:
            try:
                search_budgets = [int(x) for x in
                                  args.budget_search.split(",") if x]
            except ValueError:
                print("[reclip] --budget-search must be a CSV of ints",
                      file=sys.stderr)
                return 2
            if kind == "track":
                print("[reclip] --budget-search is incompatible with "
                      "track", file=sys.stderr)
                return 2
        if args.budget_frames is not None:
            budget = args.budget_frames
        elif args.time_budget is not None:
            budget = int(round(args.time_budget / CONTROL_DT))
        elif search_budgets:
            budget = min(search_budgets)
        elif kind == "track" or path_mode == "all":
            budget = len(path.knot_s) - 1
        else:
            budget = args.T - 1
        seeds = ([int(x) for x in args.seeds.split(",")]
                 if args.seeds else [args.seed])
        try:
            objective = ReclipObjectiveConfig(
                kind=kind, time_budget_frames=budget,
                ds_max=args.ds_max,
                terminal_pose_weight=(
                    args.terminal_pose_weight if
                    args.terminal_pose_weight is not None else
                    (1.0 if kind == "reach-stop" else 0.0)),
                terminal_vel_weight=(
                    args.terminal_vel_weight if
                    args.terminal_vel_weight is not None else
                    (1.0 if kind == "reach-stop" else 0.0)))
            if search_budgets:
                ts = reclip_time_search(
                    start, path, objective, search_budgets,
                    init_actions=init_a, seeds=seeds,
                    robot=args.robot, sweeps=args.sweeps,
                    n_samples=args.n_samples, n_iters=args.n_iters,
                    sigma0=args.sigma0,
                    post_frames=args.post_frames,
                    clip_id=args.clip_id or
                    f"{start_clip.clip_id}_reclip")
                res = ts.best
                res.clip.meta["reclip"]["time_search"] = ts.attempts
            else:
                res = reclip_path_sweep(
                    start, path, objective, init_actions=init_a,
                    budget_frames=budget, robot=args.robot,
                    sweeps=args.sweeps, n_samples=args.n_samples,
                    n_iters=args.n_iters, sigma0=args.sigma0,
                    seed=seeds[0], post_frames=args.post_frames,
                    clip_id=args.clip_id or
                    f"{start_clip.clip_id}_reclip")
        except NotImplementedError as exc:
            print(f"[reclip] {exc}", file=sys.stderr)
            return 2
        new = res.clip
    else:
        guide_actions = None
        if via is not None:
            guide = guide_from_clip(via)
            guide_actions = via.ref_action
        else:
            k = args.end_frame if args.end_frame >= 0 else end_clip.T - 1
            guide = interp_guide(start, end_clip.state(k), args.T)
        weights = dict(RECLIP_WEIGHTS)
        if args.weights:
            for kv in args.weights.split(","):
                k, v = kv.split("=")
                weights[k.strip()] = float(v)
        if args.method == "sweep":
            from .reclip import reclip_sweep
            if guide_actions is not None:
                init_a = guide_actions
            else:
                import numpy as np
                init_a = np.stack(
                    [g["joint_pos_norm"] for g in guide[1:]])
            new = reclip_sweep(
                start, guide, init_a, robot=args.robot,
                weights=weights if args.weights else None,
                sweeps=args.sweeps, n_samples=args.n_samples,
                n_iters=args.n_iters, sigma0=args.sigma0,
                seed=args.seed,
                clip_id=args.clip_id or f"{start_clip.clip_id}_reclip")
        else:
            horizon = args.horizon
            if args.method == "mpc" and horizon == 1:
                horizon = 4
            new = reclip(start, guide, robot=args.robot,
                         weights=weights if args.weights else None,
                         settle=args.settle, tol_final=args.tol_final,
                         n_samples=args.n_samples, n_iters=args.n_iters,
                         horizon=horizon, disc=args.disc,
                         guide_actions=guide_actions,
                         seed=args.seed,
                         clip_id=args.clip_id or
                         f"{start_clip.clip_id}_reclip")
    new.save(out)
    m = new.meta["reclip"]
    if m.get("solver", {}).get("method") == "path_sweep":
        event = m.get("event")
        event_s = ("none" if not event
                   else f"f{event['frame']} @{event['time']:.2f}s")
        print(f"[reclip:{m['solver']['method']}] {m['objective']} "
              f"{m['path']['mode']} status={m['status']} "
              f"event={event_s} s_final={m['metrics']['s_final']:.3f} "
              f"path_err={m['metrics']['path_error_max']:.3f} -> {out}")
    else:
        extra = (f"settle {m['settle_used']}"
                 if "settle_used" in m else f"sweeps {m['sweeps']}")
        print(f"[reclip:{m.get('method','greedy')}] T={new.T} "
              f"({extra})  final_D={m['final_distance']:.3f}  -> {out}")
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

    p = sub.add_parser("reclip",
                       help="synthesize executable clip start->end")
    p.add_argument("--clip", default=None,
                   help="single source npz for --start-frame/--end-frame "
                        "and --path all")
    p.add_argument("--start", default=None,
                   help="npz whose frame supplies the start pose")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end", default=None,
                   help="npz whose frame supplies the end pose "
                        "(required unless --via)")
    p.add_argument("--end-frame", type=int, default=-1)
    p.add_argument("--via", default=None,
                   help="guide-rail clip npz (its T defines transit)")
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--settle", type=int, default=0)
    p.add_argument("--tol-final", type=float, default=0.1)
    p.add_argument("--weights", default=None)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-iters", type=int, default=4)
    p.add_argument("--method", default="path-sweep",
                   choices=("path-sweep", "greedy", "mpc", "sweep"),
                   help="path-sweep = continuous TargetPath + suffix "
                        "rollouts (new default); legacy methods kept")
    p.add_argument("--horizon", type=int, default=1,
                   help="MPC receding horizon in control frames "
                        "(1 = greedy per-frame)")
    p.add_argument("--disc", type=float, default=0.9)
    p.add_argument("--objective", default=None,
                   choices=("track", "pass", "reach-stop", "reach-state"),
                   help="path-sweep objective (default: track with --via, "
                        "pass without)")
    p.add_argument("--path", default=None,
                   choices=("endpoints", "all"),
                   help="TargetPath construction mode")
    p.add_argument("--root-pos-mode", default=None,
                   choices=("absolute", "free"),
                   help="path-sweep root-position objective mode: "
                        "absolute (default) uses rp; free removes "
                        "root_pos from path/pose errors")
    p.add_argument("--budget-frames", type=int, default=None,
                   help="control-frame action budget for path-sweep")
    p.add_argument("--time-budget", type=float, default=None,
                   help="action budget in seconds (converted to 20Hz frames)")
    p.add_argument("--budget-search", default=None,
                   help="CSV of control-frame budgets, e.g. 8,10,12")
    p.add_argument("--seeds", default=None,
                   help="CSV of CEM seeds (default: --seed)")
    p.add_argument("--ds-max", type=float, default=0.2,
                   help="per-frame local progress search window")
    p.add_argument("--s-ext", type=float, default=0.1,
                   help="bounded path extension beyond s=1")
    p.add_argument("--post-frames", type=int, default=0,
                   help="extra output frames after a pass/reach event")
    p.add_argument("--terminal-pose-weight", type=float, default=None,
                   help="reach-stop terminal pose weight (default 1.0 "
                        "for reach-stop, 0 otherwise)")
    p.add_argument("--terminal-vel-weight", type=float, default=None,
                   help="reach-stop zero-velocity weight (default 1.0 "
                        "for reach-stop, 0 otherwise)")
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--sigma0", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot", default="robot_a")
    p.add_argument("--out", default=None)
    p.add_argument("--clip-id", default=None)
    p.set_defaults(fn=_cmd_reclip)

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
