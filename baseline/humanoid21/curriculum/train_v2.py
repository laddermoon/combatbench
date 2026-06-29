"""Unified curriculum training CLI (v2) — supports PPO and SAC.

Usage::

    python3 baseline/humanoid21/curriculum/train_v2.py --experiment basic_balance_v2 --algo ppo
    python3 baseline/humanoid21/curriculum/train_v2.py --experiment basic_balance_v2 --algo sac
    python3 baseline/humanoid21/curriculum/train_v2.py --experiment basic_balance_v2 --algo ppo --smoke
    python3 baseline/humanoid21/curriculum/train_v2.py --list-experiments
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments_v2 import get_experiment, list_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified curriculum trainer (v2) — PPO or SAC."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name (e.g. basic_balance_v2).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_experiments:
        print("Available experiments (v2):")
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

    algo = args.algo
    run_name = args.run_name or f"curriculum_{experiment.name}_{algo}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    resume_from = Path(args.resume_from) if args.resume_from else None

    # Save config snapshot before training starts
    experiment.save_run_config(run_dir, smoke=args.smoke, algo=algo)
    print(f"[config] saved to {run_dir / 'config.json'}", flush=True)
    print(f"[algo] {algo.upper()}", flush=True)

    if algo == "ppo":
        from baseline.humanoid21.curriculum.framework_v2.ppo_loop import train_ppo
        train_ppo(experiment, run_dir=run_dir, resume_from=resume_from)
    elif algo == "sac":
        from baseline.humanoid21.curriculum.framework_v2.sac_loop import train_sac
        train_sac(experiment, run_dir=run_dir, resume_from=resume_from)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    main()
