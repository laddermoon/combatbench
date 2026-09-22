#!/usr/bin/env python3
"""Debug toolset for PPO training runs — organized by the question you have.

All read commands print JSON on stdout (--pretty to indent).  Write
commands (dump) take effect at the next update boundary.  Commands work
fully offline — no viewer server needed.  <run> accepts a directory
path or a run name under --runs-root (default baseline/runs).

Usage by question::

    What runs exist?
        debug.py runs [--status running] [--tail 5]

    How did a run's metrics evolve?  (RunData.summary/metrics — same
    flattening code as the viewer's /api/run/*, identical output.)
        debug.py summary <run> [--keys kl]      # digest FIRST: per-key
                                                # zone+hint+latest+min/max
        debug.py metrics <run> [--keys eval.] [--tail 20] [--docs]
                              [--from-update A --to-update B]

    What does a metric key mean?  (metric_catalog.py — same data the
    viewer renders as chart hints.)
        debug.py catalog [--key stats.post_kl_mean]

    What happened INSIDE update N?  (cross-section capture)
        debug.py dump <run_dir> --hypothesis "why is KL high at u250"
          → <run_dir>/dumps/uNNNNN/  (episodes/trajectories/GAE/grads)
          → inspect in the viewer: debug.py viewer <run_dir>

    Analyze a captured dump (same functions as the viewer API):
        debug.py inspect  <dump>             # overview of one captured update
        debug.py samples  <dump> --sort neg_proj --limit 20
        debug.py trace    <dump> --buffer-index N  (or --frame ep:A:t)
        debug.py timeline <dump> [--step N | --key-steps]
          <dump> = dump dir path, or <run>:u<N> shorthand.

    See a dumped episode frame by frame / did the policy drift?
        debug.py render <dump_dir> --episode 0   # PNGs + auto-verify
        debug.py delta  <dump_dir> --episode 0 --gens 3

    Interactive UI (human-facing):
        debug.py viewer [run_dir|dump_dir|runs_root] --port 8766

The ``dump`` subcommand writes a sentinel file ``<run_dir>/dump_request.json``.
The training loop polls for this file at the top of each update; when found,
it captures the complete update data (episodes, trajectories, GAE,
combine, gradients) into ``<run_dir>/dumps/u{N:05d}/`` and writes a
``RECORD_GUIDE.md`` with the exact recorder commands for independent
visual inspection.  ``--hypothesis`` is optional — the dump is a
general-purpose tool, though stating what you're looking for makes the
artifact self-describing.  Dumps can also be scheduled at launch with
``train.py --dump-at``.

The ``render`` subcommand reads a captured dump, runs round_runner with
the stochastic wrapped policy to generate per-frame PNG images, and
auto-verifies the recorded data against the dump data.

Capability map for AI agents: baseline/framework/ppo/dumpkit/CONTEXT.md
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

_DEFAULT_RUNS_ROOT = "baseline/runs"


def _emit(obj, pretty: bool) -> None:
    print(json.dumps(
        obj, indent=2 if pretty else None,
        ensure_ascii=False, default=str,
    ))


def _resolve_run(arg: str, runs_root: str) -> Path:
    """Resolve <run>: a run-dir path, or a run name under runs_root.

    Name lookup follows the same safety rules as the viewer's
    resolve_run (direct child only, must contain config.json or
    train.log).  Raises FileNotFoundError on failure.
    """
    p = Path(arg).resolve()
    if p.is_dir():
        return p
    root = Path(runs_root).resolve()
    candidate = (root / arg).resolve()
    if (
        "/" not in arg and "\\" not in arg
        and candidate.parent == root
        and candidate.is_dir()
        and (
            (candidate / "config.json").exists()
            or (candidate / "train.log").exists()
        )
    ):
        return candidate
    raise FileNotFoundError(
        f"run not found: {arg} (not a directory, and no run named "
        f"'{arg}' under {root})"
    )


def _run_arg_or_error(args) -> Path:
    return _resolve_run(args.run, args.runs_root)


def _resolve_dump(arg: str, runs_root: str) -> Path:
    """Resolve <dump>: a dump-dir path, or ``<run>:u<N>`` / ``<run>:<N>``
    shorthand resolved under runs_root (``<run>`` may itself be a path
    or a run name).  Raises FileNotFoundError on failure.
    """
    p = Path(arg).resolve()
    if p.is_dir():
        return p
    if ":" in arg:
        run_part, _, upd = arg.rpartition(":")
        upd = upd.lstrip("u")
        try:
            u = int(upd)
        except ValueError:
            u = -1
        if u >= 0:
            run_dir = _resolve_run(run_part, runs_root)
            d = run_dir / "dumps" / f"u{u:05d}"
            if d.is_dir():
                return d
            raise FileNotFoundError(
                f"dump u{u:05d} not found under {run_dir}/dumps/")
    raise FileNotFoundError(
        f"dump not found: {arg} (not a directory; for shorthand use "
        f"<run>:u<N> under {Path(runs_root).resolve()})")


def _open_dump(args):
    from baseline.framework.ppo.dumpkit.viewer.server import DumpData
    try:
        return DumpData(_resolve_dump(args.dump, args.runs_root))
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


def _key_patterns(keys_arg: str) -> list:
    return [k.strip() for k in keys_arg.split(",") if k.strip()]


def _cmd_metrics(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.metric_catalog import metric_doc
    from baseline.framework.ppo.dumpkit.viewer.server import RunData

    try:
        run_dir = _run_arg_or_error(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    metrics = RunData(run_dir).metrics()
    if args.from_update is not None:
        metrics = [m for m in metrics if m["update"] >= args.from_update]
    if args.to_update is not None:
        metrics = [m for m in metrics if m["update"] <= args.to_update]
    if args.tail is not None:
        metrics = metrics[-args.tail:] if args.tail > 0 else []
    if args.keys:
        pats = _key_patterns(args.keys)
        metrics = [
            {"update": m["update"], **{
                k: v for k, v in m.items()
                if k != "update" and any(p in k for p in pats)
            }}
            for m in metrics
        ]

    out = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "n_updates": len(metrics),
        "metrics": metrics,
    }
    if args.docs:
        keys = sorted({k for m in metrics for k in m if k != "update"})
        out["docs"] = {k: metric_doc(k) for k in keys}
    _emit(out, args.pretty)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.viewer.server import RunData

    try:
        run_dir = _run_arg_or_error(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = RunData(run_dir).summary()
    if args.keys:
        pats = _key_patterns(args.keys)
        out["metrics"] = {
            k: v for k, v in out["metrics"].items()
            if any(p in k for p in pats)
        }
    _emit(out, args.pretty)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.viewer.server import list_runs

    root = Path(args.runs_root).resolve()
    if not root.is_dir():
        print(f"error: runs root does not exist: {root}", file=sys.stderr)
        return 2

    runs = list_runs(root)
    if args.status:
        runs = [r for r in runs if r.get("status") == args.status]
    runs.sort(key=lambda r: r.get("activity") or 0, reverse=True)
    if args.tail is not None:
        runs = runs[: args.tail] if args.tail > 0 else []

    _emit({"runs_root": str(root), "runs": runs}, args.pretty)
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.metric_catalog import (
        catalog,
        metric_doc,
    )

    if args.key:
        _emit(metric_doc(args.key), args.pretty)
    else:
        _emit(catalog(), args.pretty)
    return 0


# ---------------------------------------------------------------------------
# Dump analysis commands — same functions as the viewer's /api/... endpoints
# (dumpkit/dump_analysis.py); identical output, fully offline.
# ---------------------------------------------------------------------------

def _cmd_inspect(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_analysis import inspect_dump
    _emit(inspect_dump(_open_dump(args)), args.pretty)
    return 0


def _cmd_samples(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_analysis import gradsig_samples
    status, body = gradsig_samples(
        _open_dump(args),
        sort=args.sort, sign=args.sign, limit=args.limit,
        offset=args.offset, group_by=args.group_by,
        include_invalid=args.include_invalid,
    )
    _emit(body, args.pretty)
    return 0 if status == 200 else 1


def _cmd_trace(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_analysis import trace_frame
    dd = _open_dump(args)
    idx = args.buffer_index
    if idx is None:
        if args.frame is None:
            print("error: --buffer-index or --frame is required",
                  file=sys.stderr)
            return 2
        buf = dd.buffer_npz
        if buf is None or "frame_id" not in buf:
            print("error: buffer.npz/frame_id unavailable", file=sys.stderr)
            return 2
        import numpy as np
        hit = np.where(buf["frame_id"] == args.frame)[0]
        if not hit.size:
            print(f"error: frame_id '{args.frame}' not found in buffer",
                  file=sys.stderr)
            return 2
        idx = int(hit[0])
    status, body = trace_frame(dd, idx)
    _emit(body, args.pretty)
    return 0 if status == 200 else 1


def _cmd_timeline(args: argparse.Namespace) -> int:
    from baseline.framework.ppo.dumpkit.dump_analysis import (
        timeline_overview, timeline_step,
    )
    dd = _open_dump(args)
    if args.step is not None:
        status, body = timeline_step(dd, args.step)
    else:
        body = timeline_overview(dd)
        status = 200 if body.get("available") else 1
        if args.key_steps and body.get("key_steps"):
            body = {"key_steps": body["key_steps"],
                    "early_stop_step": body.get("early_stop_step"),
                    "n_steps": body.get("n_steps")}
    _emit(body, args.pretty)
    return 0 if status == 200 else 1


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

    hypothesis = (args.hypothesis or "").strip()

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
    print(f"  hypothesis: {hypothesis or '(none)'}")
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
        description="Debug toolset for PPO training runs.",
        epilog=(
            "by question:\n"
            "  runs     What runs exist?\n"
            "  summary  How is the run doing overall? (digest — run first)\n"
            "  metrics  How did metrics evolve update-by-update?\n"
            "  catalog  What does a metric key mean?\n"
            "  dump     Capture update N's full cross-section.\n"
            "  viewer   Interactive UI over all of the above.\n"
            "  render   PNG frames for a dumped episode (+ auto-verify).\n"
            "  delta    Policy drift across generations for a dumped episode.\n"
            "\n"
            "read commands print JSON to stdout (--pretty to indent);\n"
            "see dumpkit/CONTEXT.md for the full capability map."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dump = sub.add_parser(
        "dump",
        help="What happened inside the next update? Capture its full cross-section into dumps/uNNNNN/.",
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
        default="",
        help=(
            "Optional. What you're investigating — e.g. "
            "'why is KL high at update 250'. Recorded into the dump "
            "for provenance."
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
        help="See a dumped episode frame by frame: PNG render + auto-verify.",
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
        help="Did the policy drift? Deterministic replay of a dumped episode against older generations.",
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
        help="Interactive web UI over runs, metrics, dumps, episodes, trajectories, timelines.",
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

    # --- runs subcommand (read-only, JSON) ---
    p_runs = sub.add_parser(
        "runs",
        help="What runs exist? List runs under a root as JSON.",
        description=(
            "List training runs under --runs-root (default baseline/runs) "
            "as JSON: name, experiment, status, update/max_updates, "
            "eval_success, n_dumps, n_videos, created/activity timestamps.  "
            "Sorted by activity, newest first.  The entry point for "
            "orienting on an unfamiliar machine or runs root."
        ),
    )
    p_runs.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT,
        help="Runs root directory (default: baseline/runs).")
    p_runs.add_argument("--status", type=str, default="",
        help="Keep only runs with this status (running/finished/stopped/unknown).")
    p_runs.add_argument("--tail", type=int, default=None,
        help="Keep only the N most recently active runs.")
    p_runs.add_argument("--pretty", action="store_true",
        help="Pretty-print JSON output.")
    p_runs.set_defaults(func=_cmd_runs)

    # --- metrics subcommand (read-only, JSON) ---
    p_metrics = sub.add_parser(
        "metrics",
        help="How did metrics evolve? Flattened per-update series as JSON (identical to the viewer API).",
        description=(
            "Parse __RAW_STATS__ from <run>/train.log with the same "
            "RunData._flatten_update the viewer serves at "
            "/api/run/metrics — identical output, no server needed.  "
            "<run> is a run directory or a run name under --runs-root."
        ),
    )
    p_metrics.add_argument("run", type=str,
        help="Run directory, or run name under --runs-root.")
    p_metrics.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT,
        help="Root for run-name resolution (default: baseline/runs).")
    p_metrics.add_argument("--keys", type=str, default="",
        help="Comma-separated substrings — keep only matching metric keys "
             "(e.g. 'kl,lr' or 'eval.').")
    p_metrics.add_argument("--from-update", type=int, default=None,
        dest="from_update", help="First update to include (inclusive).")
    p_metrics.add_argument("--to-update", type=int, default=None,
        dest="to_update", help="Last update to include (inclusive).")
    p_metrics.add_argument("--tail", type=int, default=None,
        help="Keep only the last N updates (after range filtering).")
    p_metrics.add_argument("--docs", action="store_true",
        help="Attach per-key {zone, hint} docs from the metric catalog.")
    p_metrics.add_argument("--pretty", action="store_true",
        help="Pretty-print JSON output.")
    p_metrics.set_defaults(func=_cmd_metrics)

    # --- summary subcommand (read-only, JSON) ---
    p_summary = sub.add_parser(
        "summary",
        help="How is the run doing overall? Per-metric digest with semantics attached — run this first.",
        description=(
            "One-shot orientation digest of a run's metrics.  For every "
            "metric key: namespace zone, semantic hint (metric_catalog), "
            "presence count, latest value, and min/max with the update "
            "indices where they occurred.  Run this first before "
            "drilling into 'metrics' or dumps."
        ),
    )
    p_summary.add_argument("run", type=str,
        help="Run directory, or run name under --runs-root.")
    p_summary.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT,
        help="Root for run-name resolution (default: baseline/runs).")
    p_summary.add_argument("--keys", type=str, default="",
        help="Comma-separated substrings — keep only matching metric keys.")
    p_summary.add_argument("--pretty", action="store_true",
        help="Pretty-print JSON output.")
    p_summary.set_defaults(func=_cmd_summary)

    # --- catalog subcommand (read-only, JSON) ---
    p_catalog = sub.add_parser(
        "catalog",
        help="What does a metric mean? Full semantics catalog, or one key's doc.",
        description=(
            "Dump dumpkit/metric_catalog.catalog() — the same data the "
            "viewer serves at /api/catalog: chart layout, hint tables, "
            "and the exp./eval./policy. zone definitions.  "
            "Use --key to resolve a single flattened metric key to its "
            "{zone, hint} doc."
        ),
    )
    p_catalog.add_argument("--key", type=str, default="",
        help="Resolve one flattened key (e.g. stats.post_kl_mean) to its doc.")
    p_catalog.add_argument("--pretty", action="store_true",
        help="Pretty-print JSON output.")
    p_catalog.set_defaults(func=_cmd_catalog)

    # --- dump analysis commands (shared layer: dumpkit/dump_analysis.py) ---
    _DUMP_HELP = (
        "Dump dir path, or <run>:u<N> shorthand under --runs-root."
    )

    p_inspect = sub.add_parser(
        "inspect",
        help="Dump overview: capabilities, ADV/gradsig/update summaries.",
        description=(
            "Cross-section summary of one captured update: what data "
            "exists, and what the advantage/gradient/update look like."
        ),
    )
    p_inspect.add_argument("dump", type=str, help=_DUMP_HELP)
    p_inspect.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT)
    p_inspect.add_argument("--pretty", action="store_true")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_samples = sub.add_parser(
        "samples",
        help="Ranked sampled-gradient frames with episode/trajectory provenance.",
        description=(
            "Per-frame gradient diagnostics (sampled, theta_old): "
            "proj/cos/grad_norm/w_adv joined to episode/agent/frame. "
            "All statistics are over the sampled subset only."
        ),
    )
    p_samples.add_argument("dump", type=str, help=_DUMP_HELP)
    p_samples.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT)
    p_samples.add_argument(
        "--sort", default="abs_proj",
        choices=["abs_proj", "proj", "neg_proj", "grad_norm", "w_adv"])
    p_samples.add_argument(
        "--sign", default="all", choices=["all", "pos", "neg"],
        help="Filter by projection sign onto the aggregate gradient.")
    p_samples.add_argument("--limit", type=int, default=50)
    p_samples.add_argument("--offset", type=int, default=0)
    p_samples.add_argument(
        "--group-by", default=None, choices=["episode"],
        help="Aggregate sampled frames per episode (pos/neg sums kept "
             "separate).")
    p_samples.add_argument(
        "--include-invalid", action="store_true",
        help="Keep frames whose gradient was non-finite/zero-norm.")
    p_samples.add_argument("--pretty", action="store_true")
    p_samples.set_defaults(func=_cmd_samples)

    p_trace = sub.add_parser(
        "trace",
        help="Everything recorded about one buffer frame, joined across stages.",
        description=(
            "Trace one flat buffer index through buffer → GAE → combine "
            "→ gradsig → epoch snapshots → timeline refs, with "
            "episode/agent/frame provenance and deep links."
        ),
    )
    p_trace.add_argument("dump", type=str, help=_DUMP_HELP)
    p_trace.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT)
    p_trace.add_argument("--buffer-index", type=int, default=None)
    p_trace.add_argument(
        "--frame", type=str, default=None,
        help="frame_id like 'ep0007:robot_a:123' — resolved to its "
             "buffer index.")
    p_trace.add_argument("--pretty", action="store_true")
    p_trace.set_defaults(func=_cmd_trace)

    p_timeline = sub.add_parser(
        "timeline",
        help="Per-minibatch update timeline: KL/dtheta/ratio + key steps.",
        description=(
            "Per-minibatch record of the update (all captured fields "
            "including dtheta_norm and extreme-ratio frame indices), "
            "plus a derived key_steps index."
        ),
    )
    p_timeline.add_argument("dump", type=str, help=_DUMP_HELP)
    p_timeline.add_argument("--runs-root", type=str,
        default=_DEFAULT_RUNS_ROOT)
    p_timeline.add_argument("--step", type=int, default=None,
                            help="Single step detail instead of overview.")
    p_timeline.add_argument("--key-steps", action="store_true",
                            help="Only the key_steps index.")
    p_timeline.add_argument("--pretty", action="store_true")
    p_timeline.set_defaults(func=_cmd_timeline)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
