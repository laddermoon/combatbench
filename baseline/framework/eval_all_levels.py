"""Eval a checkpoint across all curriculum levels.

Usage::

    python3 baseline/framework/eval_all_levels.py \
        --experiment balance_recover_v2 \
        --policy-export baseline/runs/recover_v2_14lv_wall05/policy_exports/u16415 \
        --episodes 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from baseline.humanoid21.curriculum.experiments import get_experiment
from baseline.common.rollout import Episode, ParallelRollouter
from envs.framework.policy import PolicyBlueprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval checkpoint across all levels")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--policy-export", type=str, required=True,
                        help="Path to policy export dir (containing policy_blueprint.yaml)")
    parser.add_argument("--episodes", type=int, default=256,
                        help="Episodes per level")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reground", action="store_true",
                        help="Re-seat the robot on the ground after pose perturbation.")
    args = parser.parse_args()

    experiment = get_experiment(args.experiment)
    n_levels = len(experiment.LEVEL_SCALES)

    policy_bp = PolicyBlueprint.load(
        Path(args.policy_export) / "policy_blueprint.yaml"
    )

    rollouter = ParallelRollouter(num_workers=args.workers)

    env_pb = experiment._env_pb()

    print(f"reground={args.reground}  episodes={args.episodes}")
    print(f"{'level':>4} {'scale':>6} {'survived':>9} {'mean_len':>9} {'n_fell':>7} {'n_total':>8}")
    print("-" * 50)

    for level in range(n_levels):
        experiment._level = level
        scale = experiment.current_scale

        jobs = experiment.build_eval_jobs(policy_bp, args.seed + level * 10000)
        # Trim to requested number of episodes
        jobs = jobs[:args.episodes]

        if args.reground:
            perturb = experiment._current_perturb_params()
            env_bps = {
                aid: env_pb.materialize(
                    max_steps=experiment.max_steps, agent_id=aid,
                    tolerance=6, reground=True, **perturb,
                )
                for aid in ("robot_a", "robot_b")
            }
            jobs = [
                (pa, pb, env_bps[meta["agent_id"]], seed, meta)
                for (pa, pb, _env, seed, meta) in jobs
            ]

        episodes: list[Episode] = rollouter.collect(jobs)

        n_total = len(episodes)
        n_survived = sum(
            1 for ep in episodes
            if not all(r.startswith("imbalance") for r in ep.agent_termination_reason.values())
        )
        mean_len = np.mean([ep.num_frames for ep in episodes]) if episodes else 0.0
        survived_ratio = n_survived / n_total if n_total > 0 else 0.0

        print(f"{level:4d} {scale:6.2f} {survived_ratio:9.3f} {mean_len:9.1f} {n_total - n_survived:7d} {n_total:8d}")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
