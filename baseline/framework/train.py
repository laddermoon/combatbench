"""Unified training CLI — supports PPO and SAC.

Usage::

    python3 baseline/framework/train.py --experiment basic_balance --algo ppo
    python3 baseline/framework/train.py --experiment basic_balance --algo sac
    python3 baseline/framework/train.py --experiment basic_balance --algo ppo --smoke
    python3 baseline/framework/train.py --experiment basic_balance --algo ppo --background
    python3 baseline/framework/train.py --list-experiments
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment, list_experiments
from baseline.experiments_v2 import get_v2_experiment, list_v2_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified trainer — PPO or SAC."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (e.g. basic_balance, hybrid_standup_balance).",
    )
    parser.add_argument(
        "--algo", type=str, default="ppo", choices=["ppo", "sac"],
        help="Training algorithm: ppo or sac (default: ppo).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Short smoke run (max_updates=2, episodes_per_update=8, eval_episodes=4).",
    )
    parser.add_argument("--resume-from", type=str, default=None)
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
        for name in list_experiments():
            exp = get_experiment(name)
            print(f"  {name} (v1): reward_keys={exp.reward_keys}")
        for name in list_v2_experiments():
            exp = get_v2_experiment(name)
            channels = exp.reward_channels()
            print(f"  {name} (v2): channels={[ch.name for ch in channels]}")
        return

    if args.experiment is None:
        print("Error: --experiment is required. Use --list-experiments to see options.")
        raise SystemExit(1)

    # Try v1 registry first, then v2
    try:
        experiment = get_experiment(args.experiment)
        is_v2 = False
    except KeyError:
        try:
            experiment = get_v2_experiment(args.experiment)
            is_v2 = True
        except KeyError:
            print(f"Error: Unknown experiment {args.experiment!r}.")
            print(f"  V1: {list_experiments()}")
            print(f"  V2: {list_v2_experiments()}")
            raise SystemExit(1)

    if args.smoke:
        if is_v2:
            cp = experiment.common_params()
            pp = experiment.ppo_params()
            import dataclasses
            cp = dataclasses.replace(cp, max_updates=2, episodes_per_update=8,
                                     eval_episodes=4, eval_interval=1,
                                     rollout_workers=2)
            pp = dataclasses.replace(pp, minibatch_size=64)
            experiment.common_params = lambda: cp
            experiment.ppo_params = lambda: pp
        else:
            experiment.max_updates = 2
            experiment.episodes_per_update = 8
            experiment.eval_episodes = 4
            experiment.eval_interval = 1
            experiment.rollout_workers = 2
            experiment.minibatch_size = 64

    algo = args.algo
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

    if is_v2:
        from baseline.framework.ppo_loop_v2 import save_run_config_v2
        save_run_config_v2(experiment, run_dir, smoke=args.smoke, algo=algo)
    else:
        experiment.save_run_config(run_dir, smoke=args.smoke, algo=algo)
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
    print(f"[confidence] {'on' if use_confidence else 'off'}", flush=True)

    if is_v2:
        if algo != "ppo":
            raise ValueError(f"ExperimentV2 only supports PPO, got algo={algo}")
        from baseline.framework.ppo_loop_v2 import train_ppo_v2
        train_ppo_v2(experiment, run_dir=run_dir, resume_from=resume_from, use_confidence=use_confidence)
    elif algo == "ppo":
        from baseline.framework.ppo_loop import train_ppo
        train_ppo(experiment, run_dir=run_dir, resume_from=resume_from, use_confidence=use_confidence)
    elif algo == "sac":
        from baseline.framework.sac_loop import train_sac
        train_sac(experiment, run_dir=run_dir, resume_from=resume_from)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    main()
