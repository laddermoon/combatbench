#!/usr/bin/env python3
"""Debug CLI — request a one-shot update dump from a running training.

Usage:
    # Request a dump (writes a sentinel, training loop captures at next update):
    PYTHONPATH=. python3 baseline/framework/ppo/debug.py dump <run_dir> \
        --hypothesis "why is KL high at update 250"

    # Render images for a captured episode and auto-verify:
    PYTHONPATH=. python3 baseline/framework/ppo/debug.py render <run_dir> \
        --episode 0

The ``dump`` subcommand writes a sentinel file ``<run_dir>/dump_request.json``.
The training loop polls for this file at the top of each update; when found,
it captures the complete update data (episodes, trajectories, GAE,
combine, gradients) into ``<run_dir>/dumps/u{N:05d}/`` and writes a
``RECORD_GUIDE.md`` with the exact recorder commands for independent
visual inspection.

The ``render`` subcommand reads a captured dump, runs round_runner with
the stochastic wrapped policy to generate per-frame PNG images, and
auto-verifies the recorded data against the dump data.  An association
record is written linking the images to the dump frames.

``--hypothesis`` is mandatory for ``dump``.  Writing a dump without a
hypothesis is rejected — you must articulate what you're looking for first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from baseline.framework.ppo.dumpkit.dump_request import (
    DumpRequest,
    SENTINEL_FILENAME,
)


def _cmd_dump(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"error: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(
            f"error: {config_path} not found — not a valid training run directory",
            file=sys.stderr,
        )
        return 2

    hypothesis = args.hypothesis
    if not hypothesis or not hypothesis.strip():
        print(
            "error: --hypothesis is required and must be non-empty.\n"
            "If you can't state a hypothesis, you're not ready to dump.",
            file=sys.stderr,
        )
        return 2

    sentinel = run_dir / SENTINEL_FILENAME
    if sentinel.exists():
        print(
            f"error: a dump request already exists at {sentinel}\n"
            f"       It will be consumed at the next update boundary.\n"
            f"       Delete it first if you want to replace it.",
            file=sys.stderr,
        )
        return 3

    payload = {
        "hypothesis": hypothesis,
        "include_full_grad": args.full_grad,
    }
    sentinel.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dump requested for run: {run_dir}")
    print(f"  hypothesis: {hypothesis}")
    print(f"  include_full_grad: {args.full_grad}")
    print()
    print("The next update boundary will capture data into:")
    print(f"  {run_dir}/dumps/u<NNNNN>/")
    print()
    print("After capture, open RECORD_GUIDE.md there for recording instructions.")
    print()
    print("To render images and auto-verify:")
    print(f"  PYTHONPATH=. python3 baseline/framework/ppo/debug.py render "
          f"{run_dir}/dumps/u<NNNNN> --episode 0")
    return 0


def _cmd_delta(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_delta import compute_delta

    dump_dir = Path(args.dump_dir).resolve()
    if not dump_dir.is_dir():
        print(f"error: dump directory does not exist: {dump_dir}", file=sys.stderr)
        return 2
    if not (dump_dir / "episodes.npz").exists():
        print(
            f"error: {dump_dir} is not a valid dump directory "
            f"(missing episodes.npz)",
            file=sys.stderr,
        )
        return 2

    try:
        out_dir = compute_delta(
            dump_dir=dump_dir,
            episode_pos=args.episode,
            gens=args.gens,
        )
    except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print()
    print(f"Delta data: {out_dir}")
    print("Open the episode page in the viewer — the Policy Drift section appears automatically.")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_render import render_episode

    dump_dir = Path(args.dump_dir).resolve()
    if not dump_dir.is_dir():
        print(f"error: dump directory does not exist: {dump_dir}", file=sys.stderr)
        return 2

    if not (dump_dir / "episodes.npz").exists():
        print(
            f"error: {dump_dir} is not a valid dump directory "
            f"(missing episodes.npz)",
            file=sys.stderr,
        )
        return 2

    print(f"[render] using dump: {dump_dir}")

    try:
        record_dir = render_episode(
            dump_dir=dump_dir,
            episode_index=args.episode,
            verbose=True,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print()
    print(f"Recording: {record_dir}")
    print(f"Association: {record_dir / 'association.json'}")
    print()
    print("To view the recording:")
    print(f"  PYTHONPATH=. python3 -m envs.framework.recorder_viewer {record_dir}")
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.viewer.server import serve

    target = Path(args.path).resolve()
    if not target.is_dir():
        print(f"error: directory does not exist: {target}", file=sys.stderr)
        return 2

    try:
        serve(target, port=args.port, open_browser=not args.no_browser)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="debug.py",
        description="Debug CLI — request a one-shot update dump from a running training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dump = sub.add_parser(
        "dump",
        help="Request a one-shot dump of the next update's complete data.",
        description=(
            "Write a sentinel file to <run_dir>/dump_request.json. "
            "The training loop will capture the next update's complete "
            "data (episodes, trajectories, GAE, combine, gradients) "
            "into <run_dir>/dumps/u<NNNNN>/."
        ),
    )
    p_dump.add_argument(
        "run_dir",
        type=str,
        help="Training run directory (must contain config.json).",
    )
    p_dump.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help=(
            "Mandatory. State what you're investigating — e.g. "
            "'why is KL high at update 250'. Empty/whitespace is rejected."
        ),
    )
    p_dump.add_argument(
        "--full-grad",
        action="store_true",
        default=False,
        help="Capture the full flat actor gradient (epoch 0, minibatch 0). Off by default (large).",
    )
    p_dump.set_defaults(func=_cmd_dump)

    # --- render subcommand ---
    p_render = sub.add_parser(
        "render",
        help="Render images for a captured episode and auto-verify.",
        description=(
            "Auto-discover the latest dump under <run_dir>/dumps/, run "
            "round_runner with the stochastic wrapped policy to generate "
            "per-frame PNG images, and auto-verify the recorded data "
            "against the dump data.  An association record is written "
            "linking images to dump frames."
        ),
    )
    p_render.add_argument(
        "dump_dir",
        type=str,
        help="Dump directory (e.g. runs/.../dumps/u00008/).",
    )
    p_render.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to render (0-based, default 0).",
    )
    p_render.set_defaults(func=_cmd_render)

    # --- delta subcommand ---
    p_delta = sub.add_parser(
        "delta",
        help="Compute policy-drift deltas for a dumped episode (offline diagnostic).",
        description=(
            "Replay the episode's stored observations through the "
            "deterministic act() of each policy generation "
            "(policy_exports/u{update-g}) and record the action vectors "
            "into <dump_dir>/delta/episode_NNNNN/.  Episode actions are "
            "not used — both sides of the delta are deterministic.  Only "
            "agents that produced trajectories (per traj_map) are "
            "evaluated, so non-self-play episodes only analyse the "
            "trained side."
        ),
    )
    p_delta.add_argument(
        "dump_dir",
        type=str,
        help="Dump directory (e.g. runs/.../dumps/u00008/).",
    )
    p_delta.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode list position to analyse (0-based).",
    )
    p_delta.add_argument(
        "--gens",
        type=int,
        default=3,
        help="How many generations back to compare (1..10, default 3).",
    )
    p_delta.set_defaults(func=_cmd_delta)

    # --- viewer subcommand ---
    p_viewer = sub.add_parser(
        "viewer",
        help="Launch the debug viewer web app for a run or a captured dump.",
        description=(
            "Start an HTTP server that serves the debug viewer frontend "
            "and API endpoints.  Pass a training run directory "
            "(runs/.../ containing dumps/ and train.log) for the run-level "
            "view, or a dump directory (runs/.../dumps/u00008/) to jump "
            "straight into that dump.  Pass a runs-root directory (or no "
            "argument — defaults to baseline/runs) for the multi-run index.  "
            "Open http://localhost:<port>/ in your browser."
        ),
    )
    p_viewer.add_argument(
        "path",
        type=str,
        nargs="?",
        default="baseline/runs",
        help="Run directory, dump directory, or runs-root directory "
             "(default: baseline/runs).",
    )
    p_viewer.add_argument(
        "--port",
        type=int,
        default=8766,
        help="HTTP port (default: 8766).",
    )
    p_viewer.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )
    p_viewer.set_defaults(func=_cmd_viewer)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
