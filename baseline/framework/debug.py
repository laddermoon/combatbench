"""Debug CLI — behavior probes + metric verification + chain/attribute/frame.

S5: ``probe`` and ``metric --verify`` (§3.7, §3.6).
S2: ``snapshot`` and ``replay`` (§3.11).
S3: ``chain``, ``attribute``, ``frame`` (§3.2, §3.3, §3.4).
S6: ``intervene-check``, ``compare``, ``noise``, ``timeline`` (§3.8–§3.10).

Usage::

    # Run behavior probes at update 250
    PYTHONPATH=. python3 baseline/framework/debug.py probe <run_dir> --at 250

    # Verify the 'steps' metric at update 250
    PYTHONPATH=. python3 baseline/framework/debug.py metric <run_dir> \\
        --verify steps --at 250

    # Request a snapshot from a running training
    PYTHONPATH=. python3 baseline/framework/debug.py snapshot <run_dir> \\
        --hypothesis "test why KL is high"

    # Replay a snapshot with full debug recording
    PYTHONPATH=. python3 baseline/framework/debug.py replay <snapshot_dir>

    # Signal chain profile (S3)
    PYTHONPATH=. python3 baseline/framework/debug.py chain <run_dir> \\
        --channel r_left_foot --at 250

    # Update attribution (S3)
    PYTHONPATH=. python3 baseline/framework/debug.py attribute <run_dir> \\
        --window 20

    # Frame inspector (S3)
    PYTHONPATH=. python3 baseline/framework/debug.py frame <snapshot_dir> \\
        --id ep0003:robot_a:137
    PYTHONPATH=. python3 baseline/framework/debug.py frame <snapshot_dir> \\
        --where "combine.aw_normed.r_left_foot < 0" --limit 20

    # Intervene-check (S6)
    PYTHONPATH=. python3 baseline/framework/debug.py intervene-check <run_dir>
    PYTHONPATH=. python3 baseline/framework/debug.py intervene-check <run_dir> \\
        --knob explore_factor

    # Compare two runs (S6)
    PYTHONPATH=. python3 baseline/framework/debug.py compare <runA> <runB>
    PYTHONPATH=. python3 baseline/framework/debug.py compare <runA> <runB> \\
        --noise-band noise.json

    # Noise baseline (S6)
    PYTHONPATH=. python3 baseline/framework/debug.py noise \\
        --experiment standup_step_v3 --seeds 4 --updates 50

    # Event timeline (S6)
    PYTHONPATH=. python3 baseline/framework/debug.py timeline <run_dir>
    PYTHONPATH=. python3 baseline/framework/debug.py timeline <run_dir> \\
        --overlay uncertainty,max_pot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def cmd_probe(args: argparse.Namespace) -> None:
    """Run behavior probe suites on a specific update's policy."""
    from baseline.framework.ppo.debug.probes import (
        list_available_updates,
        load_experiment_from_run,
        load_policy_blueprint,
        run_probe_suite,
    )

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    experiment = load_experiment_from_run(run_dir)
    suites = experiment.probe_suites()
    if not suites:
        print("This experiment has no probe suites.")
        return

    # Select suite
    if args.suite:
        suites = tuple(s for s in suites if s.name == args.suite)
        if not suites:
            available = [s.name for s in experiment.probe_suites()]
            print(f"Suite {args.suite!r} not found. Available: {available}")
            return

    # Determine updates to probe
    if args.sweep_updates:
        parts = args.sweep_updates.split(":")
        if len(parts) != 3:
            print(f"Error: --sweep-updates expects START:STOP:STEP, "
                  f"got {args.sweep_updates!r}")
            raise SystemExit(1)
        start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        updates = list(range(start, stop + 1, step))
    elif args.at is not None:
        updates = [args.at]
    else:
        available = list_available_updates(run_dir)
        if not available:
            print("No policy exports found in run_dir.")
            return
        updates = [available[-1]]
        print(f"[info] no --at specified, using latest available update: {updates[0]}")

    # Run probes
    first_result = True
    for suite in suites:
        for update in updates:
            try:
                policy_bp = load_policy_blueprint(run_dir, update)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue
            result = run_probe_suite(
                experiment, policy_bp, suite, n_workers=args.workers,
            )
            result.update = update
            if not first_result:
                print()
            first_result = False
            _print_probe_result(result)


def _print_probe_result(result) -> None:
    """Print a probe suite result in human-readable format."""
    print(f"行为探针   {result.suite_name} @ update {result.update}")
    n_episodes = result.n_episodes
    if result.probe_results:
        n_total = result.probe_results[0].total
        n_agents_per_ep = n_total // max(n_episodes, 1)
        print(f"  （{n_episodes} 个固定初始状态 × {n_agents_per_ep} agent"
              f" = {n_total} 评估，确定性 rollout）")
    print()
    print(f"  {'谓词':<35s} {'通过率':<12s} {'最佳样本':<8s}")
    for pr in result.probe_results:
        mark = "✓" if pr.passed > 0 else "✗"
        print(f"  {pr.probe_name:<35s} {pr.passed}/{pr.total:<8d} {mark}")


def cmd_metric(args: argparse.Namespace) -> None:
    """Verify a metric definition against a strict alternative."""
    from baseline.framework.ppo.debug.probes import (
        list_available_updates,
        load_experiment_from_run,
        load_policy_blueprint,
    )
    from baseline.framework.ppo.debug.metrics import verify_metric

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    experiment = load_experiment_from_run(run_dir)
    verifiers = experiment.metric_verifiers()

    if not args.verify:
        if verifiers:
            print(f"Available metric verifiers: {list(verifiers.keys())}")
        else:
            print("This experiment has no metric verifiers.")
        return

    if args.verify not in verifiers:
        available = list(verifiers.keys())
        print(f"Metric {args.verify!r} not found. Available: {available}")
        return

    # Determine update
    if args.at is not None:
        update = args.at
    else:
        available = list_available_updates(run_dir)
        if not available:
            print("No policy exports found in run_dir.")
            return
        update = available[-1]
        print(f"[info] no --at specified, using latest available update: {update}")

    try:
        policy_bp = load_policy_blueprint(run_dir, update)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    verification = verify_metric(
        experiment, policy_bp, args.verify,
        n_episodes=args.n_episodes, n_workers=args.workers,
    )
    verification.update = update
    _print_metric_verification(verification)


def _print_metric_verification(v) -> None:
    """Print a metric verification result in human-readable format."""
    print(f"指标验证   {v.metric_name} @ update {v.update}")
    print()
    print(f"  当前定义   {v.current_mean:.1f}  (共 {v.n_agents} 个 agent)")
    print(f"  严格定义   {v.strict_mean:.1f}")
    print(f"  接触抖动   {v.jitter_mean:.1f}")
    print()
    if v.verdict == "falsified":
        if v.current_mean > 0:
            pct = (1.0 - v.strict_mean / v.current_mean) * 100
        else:
            pct = 0.0
        print(f"  判定：✗ 当前指标 {pct:.0f}% 来自接触抖动，不反映物理迈步。")
        print(f"        → 该指标不可用于判断训练进展。修正定义后重新评估历史曲线。")
    else:
        print(f"  判定：✓ 当前指标与严格定义一致。")


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Write a sentinel file to request a snapshot from a running training."""
    import json
    from pathlib import Path as P

    run_dir = P(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    hypothesis = args.hypothesis
    if not hypothesis or not hypothesis.strip():
        print("Error: --hypothesis is required and must be non-empty.")
        print("  If you can't state a hypothesis, run `health` or `chain` first.")
        raise SystemExit(1)

    episodes_mode = "all" if args.episodes == "all" else "subset"
    episodes_n = 0 if args.episodes == "all" else int(args.episodes)

    request = {
        "hypothesis": hypothesis,
        "episodes_mode": episodes_mode,
        "episodes_n": episodes_n,
        "include_full_grad": args.full_grad,
    }
    sentinel = run_dir / "debug_request.json"
    with open(sentinel, "w") as f:
        json.dump(request, f, indent=2)

    print(f"Snapshot requested for next update boundary.")
    print(f"  run_dir: {run_dir}")
    print(f"  hypothesis: {hypothesis}")
    print(f"  episodes: {args.episodes}")
    print(f"  full_grad: {args.full_grad}")
    print(f"  sentinel: {sentinel}")
    print(f"  The snapshot will be captured at the next update's boundary,")
    print(f"  in <run_dir>/debug/u<NNNNN>/.")


def cmd_replay(args: argparse.Namespace) -> None:
    """Re-run ppo_update on a snapshot with full debug recording."""
    import torch
    from baseline.framework.ppo.debug.replay import (
        replay_snapshot,
        verify_against_log,
    )

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        print(f"Error: snapshot_dir does not exist: {snapshot_dir}")
        raise SystemExit(1)

    device = torch.device(args.device) if args.device else None
    result = replay_snapshot(snapshot_dir, device=device)

    print(f"Replay @ update {result.update}")
    print(f"  snapshot: {result.snapshot_dir}")
    print(f"  episodes: {result.episodes_mode}:{result.n_episodes}")
    print(f"  frames: {result.n_frames}")
    print(f"  replay output: {result.replay_dir}")
    if result.debug_arrays:
        print(f"  debug_arrays: {sorted(result.debug_arrays.keys())}")

    if args.verify:
        if not args.run_dir:
            print("\n  --verify requires --run-dir")
            return
        run_dir = Path(args.run_dir).resolve()
        verification = verify_against_log(snapshot_dir, run_dir)
        print()
        if not verification.comparable:
            print(f"Self-verification: NOT COMPARABLE")
            print(f"  episodes_mode={verification.episodes_mode} (only 'all' is comparable)")
            return
        print(f"Self-verification: {verification.verdict.upper()}")
        print(f"  passed={verification.n_passed} failed={verification.n_failed} skipped={verification.n_skipped}")
        if verification.n_failed > 0:
            print()
            print("  Failed fields:")
            for fc in verification.fields:
                if not fc.passed:
                    print(f"    {fc.name}: log={fc.log_value} replay={fc.replay_value} ({fc.note})")


def cmd_chain(args: argparse.Namespace) -> None:
    """Signal chain profile — locate the breakpoint in the nine-ring chain."""
    from baseline.framework.ppo.debug.chain import chain, render_report

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    report = chain(
        run_dir,
        channel=args.channel,
        update=args.at,
        behavior=args.behavior,
        snapshot_dir=Path(args.snapshot) if args.snapshot else None,
        run_probes=not args.no_probe,
        n_workers=args.workers,
    )
    print(render_report(report))


def cmd_attribute(args: argparse.Namespace) -> None:
    """Update attribution — who is driving the policy?"""
    from baseline.framework.ppo.debug.attribute import attribute

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    report = attribute(
        run_dir,
        update=args.at,
        window=args.window,
        by_action_dim=args.by == "action-dim",
    )
    print(report)


def cmd_frame(args: argparse.Namespace) -> None:
    """Frame-level inspector — inspect a single frame or filter by condition."""
    from baseline.framework.ppo.debug.frame import frame

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        print(f"Error: snapshot_dir does not exist: {snapshot_dir}")
        raise SystemExit(1)

    if not args.id and not args.where:
        print("Error: either --id or --where must be given.")
        raise SystemExit(1)

    if args.render:
        print("[info] --render not yet implemented (S3 TODO); showing data only.")

    result = frame(
        snapshot_dir,
        frame_id=args.id,
        where=args.where,
        limit=args.limit,
        render=args.render,
    )
    print(result)


def cmd_intervene_check(args: argparse.Namespace) -> None:
    """Intervene-check — verify configured knobs entered the data pathway."""
    from baseline.framework.ppo.debug.intervene import intervene_check

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    snapshot_dir = Path(args.snapshot).resolve() if args.snapshot else None
    result = intervene_check(
        run_dir,
        update=args.at,
        knob_name=args.knob,
        snapshot_dir=snapshot_dir,
    )
    print(result)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two training runs metric-by-metric."""
    from baseline.framework.ppo.debug.compare import compare

    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    if not run_a.exists():
        print(f"Error: run_a does not exist: {run_a}")
        raise SystemExit(1)
    if not run_b.exists():
        print(f"Error: run_b does not exist: {run_b}")
        raise SystemExit(1)

    noise_band_path = Path(args.noise_band).resolve() if args.noise_band else None
    result = compare(
        run_a, run_b,
        window=args.window,
        noise_band_path=noise_band_path,
    )
    print(result)


def cmd_noise(args: argparse.Namespace) -> None:
    """Noise baseline — launch multi-seed training and compute noise band."""
    from baseline.framework.ppo.debug.noise import noise

    seeds = [int(s) for s in args.seeds.split(",")]
    output = Path(args.output).resolve() if args.output else None

    result = noise(
        experiment=args.experiment,
        seeds=seeds,
        updates=args.updates,
        algo=args.algo,
        output=output,
        window=args.window,
        wait=not args.no_wait,
    )
    print(result)
    if output:
        print(f"\n噪声带已保存到: {output}")


def cmd_timeline(args: argparse.Namespace) -> None:
    """Event timeline — show events and metric sparklines on one axis."""
    from baseline.framework.ppo.debug.timeline import timeline

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run_dir does not exist: {run_dir}")
        raise SystemExit(1)

    overlay = None
    if args.overlay:
        overlay = [m.strip() for m in args.overlay.split(",")]

    result = timeline(run_dir, overlay=overlay)
    print(result)


def cmd_whatif(args: argparse.Namespace) -> None:
    """Offline counterfactual — re-run ppo_update with parameter overrides."""
    import torch
    from baseline.framework.ppo.debug.replay import _load_experiment, _load_manifest
    from baseline.framework.ppo.debug.whatif import (
        parse_set_args,
        parse_sweep_arg,
        whatif,
        render_report,
        save_report,
    )

    snapshot_dir = Path(args.snapshot_dir).resolve()
    if not snapshot_dir.exists():
        print(f"Error: snapshot_dir does not exist: {snapshot_dir}")
        raise SystemExit(1)

    # Load the experiment to get declared params for parsing.
    manifest = _load_manifest(snapshot_dir)
    experiment = _load_experiment(snapshot_dir, manifest)
    declared = experiment.whatif_params()

    # Parse --set / --sweep (mutually exclusive at the argparse level).
    overrides = None
    sweep_key = None
    sweep_values = None
    if args.set_pairs:
        overrides = parse_set_args(args.set_pairs, declared)
    elif args.sweep:
        sweep_key, sweep_values = parse_sweep_arg(args.sweep, declared)

    device = torch.device(args.device) if args.device else None

    report = whatif(
        snapshot_dir,
        overrides=overrides,
        sweep_key=sweep_key,
        sweep_values=sweep_values,
        noise_band_path=Path(args.noise_band).resolve() if args.noise_band else None,
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        full_grad=args.full_grad,
        device=device,
    )

    # Persist to <snapshot>/whatif/<run_id>/
    run_id = "set" if overrides is not None else f"sweep_{sweep_key}"
    whatif_dir = snapshot_dir / "whatif" / run_id
    save_report(report, whatif_dir)

    print(render_report(report))
    print(f"\n  report saved: {whatif_dir / 'report.json'}")
    print(f"  report text:  {whatif_dir / 'report.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="debug.py",
        description="Debug CLI — probes, metrics, snapshots, replay, "
                    "chain, attribute, frame, intervene-check, compare, "
                    "noise, timeline, whatif (S5 + S2 + S3 + S6 + S4)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- probe ---
    p_probe = subparsers.add_parser(
        "probe", help="Run behavior probes on a specific update's policy",
    )
    p_probe.add_argument("run_dir", type=str, help="Training run directory")
    p_probe.add_argument(
        "--at", type=int, default=None,
        help="Update number to probe (default: latest available)",
    )
    p_probe.add_argument(
        "--suite", type=str, default=None,
        help="Probe suite name (default: all suites)",
    )
    p_probe.add_argument(
        "--sweep-updates", type=str, default=None,
        help="Sweep updates as START:STOP:STEP (e.g. 100:300:20)",
    )
    p_probe.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel rollout workers (default: 2)",
    )
    p_probe.set_defaults(func=cmd_probe)

    # --- metric ---
    p_metric = subparsers.add_parser(
        "metric", help="Verify metric definitions against strict alternatives",
    )
    p_metric.add_argument("run_dir", type=str, help="Training run directory")
    p_metric.add_argument(
        "--verify", type=str, default=None,
        help="Metric name to verify (e.g. 'steps'). "
             "If omitted, lists available verifiers.",
    )
    p_metric.add_argument(
        "--at", type=int, default=None,
        help="Update number to verify (default: latest available)",
    )
    p_metric.add_argument(
        "--n-episodes", type=int, default=16,
        help="Number of eval episodes (default: 16)",
    )
    p_metric.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel rollout workers (default: 2)",
    )
    p_metric.set_defaults(func=cmd_metric)

    # --- snapshot (S2) ---
    p_snapshot = subparsers.add_parser(
        "snapshot", help="Request a snapshot from a running training",
    )
    p_snapshot.add_argument("run_dir", type=str, help="Training run directory")
    p_snapshot.add_argument(
        "--hypothesis", type=str, required=True,
        help="What you're looking for (required — DEBUG_GUIDE.md §6 discipline 6)",
    )
    p_snapshot.add_argument(
        "--episodes", type=str, default="8",
        help="Number of episodes to capture: N (subset, first N) or 'all' (default: 8)",
    )
    p_snapshot.add_argument(
        "--full-grad", action="store_true",
        help="Capture full actor gradient for epoch 0, mb 0 (large)",
    )
    p_snapshot.set_defaults(func=cmd_snapshot)

    # --- replay (S2) ---
    p_replay = subparsers.add_parser(
        "replay", help="Re-run ppo_update on a snapshot with debug recording",
    )
    p_replay.add_argument("snapshot_dir", type=str, help="Snapshot directory")
    p_replay.add_argument(
        "--run-dir", type=str, default=None,
        help="Training run directory (for --verify)",
    )
    p_replay.add_argument(
        "--verify", action="store_true",
        help="Verify replayed stats against training log (requires --run-dir)",
    )
    p_replay.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cpu)",
    )
    p_replay.set_defaults(func=cmd_replay)

    # --- chain (S3) ---
    p_chain = subparsers.add_parser(
        "chain", help="Signal chain profile — locate the breakpoint (§3.2)",
    )
    p_chain.add_argument("run_dir", type=str, help="Training run directory")
    p_chain.add_argument(
        "--channel", type=str, required=True,
        help="Reward channel name (e.g. r_left_foot)",
    )
    p_chain.add_argument(
        "--at", type=int, default=None,
        help="Update number (default: latest from log)",
    )
    p_chain.add_argument(
        "--behavior", type=str, default=None,
        help="Behavior name for display (optional)",
    )
    p_chain.add_argument(
        "--snapshot", type=str, default=None,
        help="Snapshot directory (default: auto-find <run_dir>/debug/u<NNNNN>)",
    )
    p_chain.add_argument(
        "--no-probe", action="store_true",
        help="Skip running probes (①⑨ rings will show '未测')",
    )
    p_chain.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel rollout workers for probes (default: 2)",
    )
    p_chain.set_defaults(func=cmd_chain)

    # --- attribute (S3) ---
    p_attr = subparsers.add_parser(
        "attribute", help="Update attribution — who is driving the policy? (§3.3)",
    )
    p_attr.add_argument("run_dir", type=str, help="Training run directory")
    p_attr.add_argument(
        "--at", type=int, default=None,
        help="Update number (default: latest from log)",
    )
    p_attr.add_argument(
        "--window", type=int, default=1,
        help="Number of recent updates to average (default: 1)",
    )
    p_attr.add_argument(
        "--by", type=str, default=None,
        help="Break down by: 'action-dim' for per-action-dimension gradients",
    )
    p_attr.set_defaults(func=cmd_attribute)

    # --- frame (S3) ---
    p_frame = subparsers.add_parser(
        "frame", help="Frame-level inspector — inspect or filter frames (§3.4)",
    )
    p_frame.add_argument("snapshot_dir", type=str, help="Snapshot directory")
    p_frame.add_argument(
        "--id", type=str, default=None,
        help="Frame ID like ep0003:robot_a:137",
    )
    p_frame.add_argument(
        "--where", type=str, default=None,
        help='Filter expression: "combine.aw_normed.r_left_foot < 0"',
    )
    p_frame.add_argument(
        "--limit", type=int, default=20,
        help="Max frames to return from --where (default: 20)",
    )
    p_frame.add_argument(
        "--render", action="store_true",
        help="Render the frame (not yet implemented)",
    )
    p_frame.set_defaults(func=cmd_frame)

    # --- intervene-check (S6) ---
    p_intervene = subparsers.add_parser(
        "intervene-check",
        help="Verify configured knobs entered the data pathway (§3.8)",
    )
    p_intervene.add_argument("run_dir", type=str, help="Training run directory")
    p_intervene.add_argument(
        "--at", type=int, default=None,
        help="Update number (default: latest from log)",
    )
    p_intervene.add_argument(
        "--knob", type=str, default=None,
        help="Only check this knob (default: all)",
    )
    p_intervene.add_argument(
        "--snapshot", type=str, default=None,
        help="Snapshot directory (default: auto-find <run_dir>/debug/u<NNNNN>)",
    )
    p_intervene.set_defaults(func=cmd_intervene_check)

    # --- compare (S6) ---
    p_compare = subparsers.add_parser(
        "compare", help="Compare two training runs metric-by-metric (§3.9)",
    )
    p_compare.add_argument("run_a", type=str, help="First run directory")
    p_compare.add_argument("run_b", type=str, help="Second run directory")
    p_compare.add_argument(
        "--window", type=int, default=1,
        help="Number of recent updates to average (default: 1)",
    )
    p_compare.add_argument(
        "--noise-band", type=str, default=None,
        help="Path to noise band JSON (from `debug.py noise`) for significance testing",
    )
    p_compare.set_defaults(func=cmd_compare)

    # --- noise (S6) ---
    p_noise = subparsers.add_parser(
        "noise", help="Launch multi-seed training to establish noise baseline (§3.9)",
    )
    p_noise.add_argument(
        "--experiment", type=str, required=True,
        help="Experiment name",
    )
    p_noise.add_argument(
        "--seeds", type=str, default="42,43,44,45",
        help="Comma-separated seeds (default: 42,43,44,45)",
    )
    p_noise.add_argument(
        "--updates", type=int, default=50,
        help="Max updates per run (default: 50)",
    )
    p_noise.add_argument(
        "--algo", type=str, default="ppo",
        help="Algorithm: ppo or sac (default: ppo)",
    )
    p_noise.add_argument(
        "--window", type=int, default=5,
        help="Number of recent updates to average per run (default: 5)",
    )
    p_noise.add_argument(
        "--output", type=str, default=None,
        help="Path to save noise band JSON",
    )
    p_noise.add_argument(
        "--no-wait", action="store_true",
        help="Don't wait for runs to complete (just launch)",
    )
    p_noise.set_defaults(func=cmd_noise)

    # --- timeline (S6) ---
    p_timeline = subparsers.add_parser(
        "timeline", help="Event timeline with metric sparklines (§3.10)",
    )
    p_timeline.add_argument("run_dir", type=str, help="Training run directory")
    p_timeline.add_argument(
        "--overlay", type=str, default=None,
        help="Comma-separated metric names to show as sparklines "
             "(e.g. 'uncertainty,max_pot')",
    )
    p_timeline.set_defaults(func=cmd_timeline)

    # --- whatif (S4) ---
    p_whatif = subparsers.add_parser(
        "whatif", help="Offline counterfactual — re-run ppo_update with "
                       "parameter overrides (§3.5)",
    )
    p_whatif.add_argument("snapshot_dir", type=str, help="Snapshot directory")
    mx = p_whatif.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "--set", dest="set_pairs", action="append", default=None,
        metavar="key=value",
        help="Apply a single override (repeatable for multiple). "
             "E.g. --set foot_height_clip=0.30",
    )
    mx.add_argument(
        "--sweep", type=str, default=None,
        metavar="key=v1,v2,...",
        help="Sweep one parameter over a list of values. "
             "E.g. --sweep foot_actor_weight=1,3,5,10",
    )
    p_whatif.add_argument(
        "--noise-band", type=str, default=None,
        help="Path to an S6 noise-band JSON for significance classification.",
    )
    p_whatif.add_argument(
        "--run-dir", type=str, default=None,
        help="Training run directory (for baseline self-verification).",
    )
    p_whatif.add_argument(
        "--full-grad", action="store_true",
        help="Force full-grad capture for gradient-direction cosine.",
    )
    p_whatif.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cpu).",
    )
    p_whatif.set_defaults(func=cmd_whatif)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
