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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="debug.py",
        description="Debug CLI — behavior probes + metric verification (S5)",
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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
