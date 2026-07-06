"""Unified training CLI — supports PPO and SAC.

Usage::

    python3 baseline/framework/train.py --experiment basic_balance --algo ppo
    python3 baseline/framework/train.py --experiment basic_balance --algo sac
    python3 baseline/framework/train.py --experiment basic_balance --algo ppo --smoke
    python3 baseline/framework/train.py --list-experiments
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from baseline.humanoid21.curriculum.experiments import get_experiment, list_experiments


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

    algo = args.algo
    run_name = args.run_name or f"train_{experiment.name}_{algo}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(__file__).resolve().parent.parent / "runs" / run_name

    resume_from = Path(args.resume_from) if args.resume_from else None

    experiment.save_run_config(run_dir, smoke=args.smoke, algo=algo)
    print(f"[config] saved to {run_dir / 'config.json'}", flush=True)
    print(f"[algo] {algo.upper()}", flush=True)

    use_confidence = not args.no_confidence
    print(f"[confidence] {'on' if use_confidence else 'off'}", flush=True)

    if algo == "ppo":
        from baseline.framework.ppo_loop import train_ppo
        train_ppo(experiment, run_dir=run_dir, resume_from=resume_from, use_confidence=use_confidence)
    elif algo == "sac":
        from baseline.framework.sac_loop import train_sac
        train_sac(experiment, run_dir=run_dir, resume_from=resume_from)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


if __name__ == "__main__":
    main()
