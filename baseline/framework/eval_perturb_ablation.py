"""Ablate initial-state perturbation dimensions to find what actually causes falls.

Usage::

    python3 baseline/framework/eval_perturb_ablation.py \
        --experiment balance_recover_v2 \
        --policy-export baseline/runs/recover_v2_14lv_wall05/policy_exports/u16415 \
        --scale 0.90 --episodes 256
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.humanoid21.curriculum.experiments import get_experiment
from baseline.common.rollout import Episode, ParallelRollouter
from envs.framework.policy import PolicyBlueprint

DIMS = (
    "joint_pos_delta_max",
    "joint_vel_delta_max",
    "root_tilt_deg_max",
    "root_linear_velocity_delta_max",
    "root_angular_velocity_delta_max",
)


def build_jobs(experiment, policy_bp, base_seed, n_episodes, perturb):
    """Build eval jobs with an explicit perturbation dict."""
    env_pb = experiment._env_pb()
    rng = np.random.default_rng(base_seed)
    env_bps = {
        aid: env_pb.materialize(
            max_steps=experiment.max_steps, agent_id=aid, tolerance=6, **perturb
        )
        for aid in ("robot_a", "robot_b")
    }
    jobs = []
    for i in range(n_episodes):
        seed = int(base_seed + i)
        agent_id = experiment._agent_from_rollout_seed(seed)
        initial_distance = float(
            rng.uniform(
                experiment.custom_config["rollout_distance_min"],
                experiment.custom_config["rollout_distance_max"],
            )
        )
        jobs.append((
            policy_bp, policy_bp, env_bps[agent_id], seed,
            {"agent_id": agent_id, "initial_distance": initial_distance},
        ))
    return jobs


def run(rollouter, experiment, policy_bp, seed, n_episodes, perturb, label):
    jobs = build_jobs(experiment, policy_bp, seed, n_episodes, perturb)
    episodes: List[Episode] = rollouter.collect(jobs)
    n = len(episodes)
    n_surv = sum(1 for ep in episodes if not all(r.startswith("imbalance") for r in ep.agent_termination_reason.values()))
    mean_len = float(np.mean([ep.num_frames for ep in episodes])) if episodes else 0.0
    ratio = n_surv / n if n else 0.0
    print(f"{label:<44} {ratio:8.3f} {mean_len:9.1f} {n - n_surv:7d}")
    return ratio


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--policy-export", required=True)
    p.add_argument("--scale", type=float, default=0.90)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7777)
    args = p.parse_args()

    experiment = get_experiment(args.experiment)
    full = {k: float(v) * args.scale for k, v in experiment.PERTURB_FULL.items()}
    zero = {k: 0.0 for k in DIMS}

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    rollouter = ParallelRollouter(num_workers=args.workers)

    print(f"scale={args.scale}  episodes={args.episodes}")
    print(f"full perturbation: {full}")
    print()
    print(f"{'condition':<44} {'survived':>8} {'mean_len':>9} {'n_fell':>7}")
    print("-" * 72)

    run(rollouter, experiment, policy_bp, args.seed, args.episodes, zero, "none (no perturbation)")
    run(rollouter, experiment, policy_bp, args.seed, args.episodes, full, "ALL dims")
    print("-" * 72)

    # Only one dim active
    for d in DIMS:
        cfg = dict(zero)
        cfg[d] = full[d]
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg, f"ONLY {d}")
    print("-" * 72)

    # Leave one out
    for d in DIMS:
        cfg = dict(full)
        cfg[d] = 0.0
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg, f"ALL except {d}")
    print("-" * 72)

    # Split root linear velocity into horizontal vs vertical
    v = full["root_linear_velocity_delta_max"]
    cfg = dict(zero)
    cfg["root_linear_velocity_delta_max"] = [v, v, 0.0]
    run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
        "ONLY root_lin_vel (horizontal xy only)")
    cfg = dict(zero)
    cfg["root_linear_velocity_delta_max"] = [0.0, 0.0, v]
    run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
        "ONLY root_lin_vel (vertical z only)")

    # Split root angular velocity into roll/pitch vs yaw
    w = full["root_angular_velocity_delta_max"]
    cfg = dict(zero)
    cfg["root_angular_velocity_delta_max"] = [w, w, 0.0]
    run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
        "ONLY root_ang_vel (roll/pitch only)")
    cfg = dict(zero)
    cfg["root_angular_velocity_delta_max"] = [0.0, 0.0, w]
    run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
        "ONLY root_ang_vel (yaw only)")
    print("-" * 72)


if __name__ == "__main__":
    main()
