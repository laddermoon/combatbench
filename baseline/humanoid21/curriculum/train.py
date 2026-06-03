"""Unified curriculum training CLI.

Usage::

    python3 baseline/humanoid21/curriculum/train.py --experiment v2_follow
    python3 baseline/humanoid21/curriculum/train.py --experiment v1_relation --smoke
    python3 baseline/humanoid21/curriculum/train.py --list-experiments
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment, list_experiments
from baseline.humanoid21.curriculum.framework.training_loop import (
    TrainConfig,
    train,
)


def _save_run_config(
    run_dir: Path,
    cfg: TrainConfig,
    experiment,
    smoke: bool,
) -> None:
    """Save the full run configuration to run_dir/config.json."""
    payload = {
        "config": dataclasses.asdict(cfg),
        "experiment": experiment.to_dict(),
        "smoke": smoke,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified curriculum PPO trainer."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (e.g. v1_relation, v2_follow).",
    )
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--terminal-fall-penalty", type=float, default=None)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for name in list_experiments():
            exp = get_experiment(name)
            print(f"  {name}: reward_keys={exp.reward_keys} blueprint={exp.env_blueprint}")
        return

    if args.experiment is None:
        print("Error: --experiment is required. Use --list-experiments to see options.")
        raise SystemExit(1)

    experiment = get_experiment(args.experiment)

    cfg = TrainConfig()

    # Apply experiment-specific PPO overrides
    for k, v in experiment.ppo_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if args.smoke:
        cfg.max_updates = 2
        cfg.episodes_per_update = 8
        cfg.eval_episodes = 4
        cfg.eval_interval = 1
        cfg.rollout_workers = 2
        cfg.minibatch_size = 64
    if args.max_updates is not None:
        cfg.max_updates = int(args.max_updates)
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = int(args.episodes_per_update)
    if args.rollout_workers is not None:
        cfg.rollout_workers = int(args.rollout_workers)
    if args.terminal_fall_penalty is not None:
        cfg.terminal_fall_penalty = float(args.terminal_fall_penalty)

    run_name = args.run_name or f"curriculum_{experiment.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    resume_from = Path(args.resume_from) if args.resume_from else None

    # Save config snapshot before training starts
    _save_run_config(run_dir, cfg, experiment, smoke=args.smoke)
    print(f"[config] saved to {run_dir / 'config.json'}", flush=True)

    train(cfg, experiment, run_dir=run_dir, resume_from=resume_from)


if __name__ == "__main__":
    main()
