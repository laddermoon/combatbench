"""Unified curriculum training CLI.

Usage::

    python3 baseline/humanoid21/curriculum/train.py --experiment v2_follow
    python3 baseline/humanoid21/curriculum/train.py --experiment v1_relation --smoke
    python3 baseline/humanoid21/curriculum/train.py --list-experiments
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment, list_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified curriculum PPO trainer."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (e.g. v1_relation, v2_follow).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_experiments:
        print("Available experiments:")
        for name in list_experiments():
            exp = get_experiment(name)
            print(f"  {name}: reward_keys={exp.reward_keys}")
        return

    if args.experiment is None:
        print("Error: --experiment is required. Use --list-experiments to see options.")
        raise SystemExit(1)

    experiment = get_experiment(args.experiment)

    if args.smoke:
        experiment.max_updates = 2
        experiment.episodes_per_update = 8
        experiment.eval_episodes = 4
        experiment.eval_interval = 1
        experiment.rollout_workers = 2
        experiment.minibatch_size = 64
    run_name = args.run_name or f"curriculum_{experiment.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    resume_from = Path(args.resume_from) if args.resume_from else None

    # Save config snapshot before training starts
    experiment.save_run_config(run_dir, smoke=args.smoke)
    print(f"[config] saved to {run_dir / 'config.json'}", flush=True)

    from baseline.humanoid21.curriculum.framework.training_loop import train
    train(experiment, run_dir=run_dir, resume_from=resume_from)


if __name__ == "__main__":
    main()
