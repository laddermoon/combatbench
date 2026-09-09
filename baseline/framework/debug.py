"""Debug CLI — behavior probes + metric verification.

S5: This CLI implements the ``probe`` and ``metric --verify``
subcommands from ``DEBUG_GUIDE.md`` §3.7 and §3.6.

Usage::

    # Run behavior probes at update 250
    PYTHONPATH=. python3 baseline/framework/debug.py probe <run_dir> --at 250

    # Run a specific suite across multiple updates
    PYTHONPATH=. python3 baseline/framework/debug.py probe <run_dir> \\
        --suite locomotion --sweep-updates 100:300:20

    # Verify the 'steps' metric at update 250
    PYTHONPATH=. python3 baseline/framework/debug.py metric <run_dir> \\
        --verify steps --at 250

    # List available metric verifiers
    PYTHONPATH=. python3 baseline/framework/debug.py metric <run_dir>
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="debug.py",
        description="Debug CLI — probes, metrics, snapshots, replay (S5 + S2)",
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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
