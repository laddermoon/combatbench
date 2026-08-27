"""Unified training CLI — PPO + SAC.

Usage::

    python3 baseline/framework/train.py --experiment basic_balance
    python3 baseline/framework/train.py --experiment basic_balance --smoke
    python3 baseline/framework/train.py --experiment basic_balance --background
    python3 baseline/framework/train.py --experiment sac_balance --algo sac
    python3 baseline/framework/train.py --list-experiments
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from baseline.experiments_v2 import get_v2_experiment, list_v2_experiments
from baseline.experiments_sac import get_sac_experiment, list_sac_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified trainer — PPO."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (e.g. basic_balance, hybrid_standup_balance).",
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "sac"],
        help="Training algorithm (default: ppo).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=8, eval_episodes=4).",
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--reset-update", action="store_true",
        help="Reset update counter to 0 when resuming (for new generation training).",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--list-experiments", action="store_true",
        help="List available experiments and exit.",
    )
    parser.add_argument(
        "--no-confidence", action="store_true",
        help="Disable EV-based confidence weighting in advantage combination.",
    )
    parser.add_argument(
        "--no-snapshot", action="store_true",
        help="Skip git code snapshot (default: snapshot enabled).",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Explicit run output directory (default: baseline/runs/<run_name>).",
    )
    parser.add_argument(
        "--background", action="store_true",
        help="Run in background (like nohup). Logs go to run_dir/train.log, PID to run_dir/pid.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override experiment seed (default: use experiment's built-in seed).",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Set experiment constructor parameter (can be repeated). "
             "Example: --set policy_blueprint_path=.../policy_blueprint.yaml",
    )
    return parser.parse_args()


def _setup_logging(run_dir: Path, background: bool) -> Path:
    """Set up file logging to run_dir/train.log.

    In foreground mode, output goes to both console and file.
    In background mode, output goes to file only.
    """
    import logging

    log_path = run_dir / "train.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    if not background:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(console_handler)

    # Redirect stdout/stderr to the log file as well (for print() and C-level output)
    sys.stdout = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stderr = open(log_path, "a", buffering=1, encoding="utf-8")

    if not background:
        # In foreground mode, also tee stdout to console via a custom stream
        sys.stdout = _TeeStream(open(log_path, "a", buffering=1, encoding="utf-8"), sys.__stdout__)
        sys.stderr = _TeeStream(open(log_path, "a", buffering=1, encoding="utf-8"), sys.__stderr__)

    return log_path


class _TeeStream:
    """Write to both a file and a console stream."""

    def __init__(self, file_stream, console_stream):
        self._file = file_stream
        self._console = console_stream

    def write(self, data):
        self._file.write(data)
        self._file.flush()
        self._console.write(data)
        self._console.flush()

    def flush(self):
        self._file.flush()
        self._console.flush()

    def close(self):
        self._file.close()


def main() -> None:
    args = _parse_args()

    if args.list_experiments:
        print("Available experiments:")
        print("  [PPO]")
        for name in list_v2_experiments():
            try:
                exp = get_v2_experiment(name)
                channels = exp.reward_channels()
                print(f"    {name}: channels={[ch.name for ch in channels]}")
            except Exception as e:
                print(f"    {name}: (requires constructor args: {e})")
        sac_exps = list_sac_experiments()
        if sac_exps:
            print("  [SAC]")
            for name in sac_exps:
                try:
                    exp = get_sac_experiment(name)
                    channels = exp.reward_channels()
                    print(f"    {name}: channels={[ch.name for ch in channels]}")
                except Exception as e:
                    print(f"    {name}: (requires constructor args: {e})")
        return

    if args.experiment is None:
        print("Error: --experiment is required. Use --list-experiments to see options.")
        raise SystemExit(1)

    # Parse --set key=value pairs into a dict
    set_params = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Error: --set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        set_params[key.strip()] = value.strip()

    algo = args.algo

    # Try the appropriate registry based on algo
    if algo == "sac":
        try:
            experiment = get_sac_experiment(args.experiment, **set_params)
        except KeyError:
            print(f"Error: Unknown SAC experiment {args.experiment!r}.")
            print(f"  Available SAC: {list_sac_experiments()}")
            raise SystemExit(1)
    else:
        try:
            experiment = get_v2_experiment(args.experiment, **set_params)
        except KeyError:
            print(f"Error: Unknown experiment {args.experiment!r}.")
            print(f"  Available: {list_v2_experiments()}")
            raise SystemExit(1)

    # --- Override seed if requested ---
    if args.seed is not None:
        experiment.seed = args.seed
        print(f"[seed] overridden to {args.seed}", flush=True)

    if args.smoke:
        import dataclasses
        if algo == "sac":
            cp = experiment.common_params()
            sp = experiment.sac_params()
            cp = dataclasses.replace(
                cp, max_env_steps=10_000, episodes_per_update=8,
                eval_episodes=4, eval_interval=2_000,
                rollout_workers=2,
            )
            sp = dataclasses.replace(
                sp, warmup_steps=200, batch_size=64, utd_ratio=0.5,
                replay_buffer_size=10_000,
            )
            experiment.common_params = lambda: cp
            experiment.sac_params = lambda: sp
        else:
            cp = experiment.common_params()
            pp = experiment.ppo_params()
            cp = dataclasses.replace(cp, max_updates=2, episodes_per_update=8,
                                     eval_episodes=4, eval_interval=1,
                                     rollout_workers=2)
            pp = dataclasses.replace(pp, minibatch_size=64)
            experiment.common_params = lambda: cp
            experiment.ppo_params = lambda: pp

    run_name = args.run_name or f"train_{experiment.name}_{algo}_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    if run_dir.exists():
        raise SystemExit(f"Error: run_dir already exists: {run_dir}")

    # --- Background fork ---
    if args.background:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        pid = os.fork()
        if pid > 0:
            # Parent: print info and exit
            with open(run_dir / "pid", "w") as f:
                f.write(str(pid) + "\n")
            print(f"[run] started in background")
            print(f"[run] dir: {run_dir}")
            print(f"[run] log: {log_path}")
            print(f"[run] pid: {pid}")
            print(f"[run] monitor: tail -f {log_path}")
            print(f"[run] stop: kill {pid}")
            return
        # Child: detach from terminal
        os.setsid()
        # Redirect stdin/stdout/stderr to log file
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    # --- Logging setup ---
    log_path = _setup_logging(run_dir, background=args.background)

    resume_from = Path(args.resume_from).resolve() if args.resume_from else None

    if algo == "sac":
        from baseline.framework.sac.loop import save_run_config_sac
        save_run_config_sac(experiment, run_dir, smoke=args.smoke)
    else:
        from baseline.framework.ppo_loop_v2 import save_run_config_v2
        save_run_config_v2(experiment, run_dir, smoke=args.smoke, algo=algo)
    print(f"[config] saved to {run_dir / 'config.json'}", flush=True)
    print(f"[algo] {algo.upper()}", flush=True)
    print(f"[log] {log_path}", flush=True)

    # --- Code snapshot for reproducibility ---
    if not args.no_snapshot:
        from baseline.framework.code_snapshot import create_code_snapshot, format_repro_command
        snapshot_info = create_code_snapshot(run_name=run_name, run_dir=run_dir)
        if snapshot_info is not None:
            print(f"[snapshot] branch {snapshot_info['branch']} created (commit {snapshot_info['commit'][:8]})", flush=True)
            repro = format_repro_command(
                snapshot_info, args=args,
                original_run_dir=run_dir,
                original_repo_root=Path(snapshot_info["repo_root"]),
            )
            repro_path = run_dir / "REPRODUCE.md"
            with open(repro_path, "w") as f:
                f.write(repro + "\n")
            print(f"[snapshot] reproduction guide saved to {repro_path}", flush=True)
            print(repro, flush=True)

    use_confidence = not args.no_confidence
    if algo == "ppo":
        print(f"[confidence] {'on' if use_confidence else 'off'}", flush=True)

    if algo == "sac":
        from baseline.framework.sac.loop import train_sac
        train_sac(experiment, run_dir=run_dir, resume_from=resume_from, reset_update=args.reset_update)
    else:
        from baseline.framework.ppo_loop_v2 import train_ppo_v2
        train_ppo_v2(experiment, run_dir=run_dir, resume_from=resume_from, use_confidence=use_confidence, reset_update=args.reset_update)


if __name__ == "__main__":
    main()
