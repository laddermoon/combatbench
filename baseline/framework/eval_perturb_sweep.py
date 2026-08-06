"""Sweep physically-motivated perturbation designs to see which produce a
usable difficulty gradient.

An instantaneous impact changes *velocities*, not positions. This script sweeps
velocity-only perturbations (and pure horizontal root impulses) over a wide
magnitude range to check whether they can span the full difficulty range that
the curriculum needs.

Usage::

    python3 baseline/framework/eval_perturb_sweep.py \
        --experiment balance_recover_v2 \
        --policy-export baseline/runs/recover_v2_14lv_wall05/policy_exports/u16415 \
        --episodes 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baseline.humanoid21.curriculum.experiments import get_experiment
from baseline.common.rollout import ParallelRollouter
from envs.framework.policy import PolicyBlueprint

from baseline.framework.eval_perturb_ablation import DIMS, build_jobs

ZERO = {k: 0.0 for k in DIMS}


def run(rollouter, experiment, policy_bp, seed, n, perturb, label):
    episodes = rollouter.collect(build_jobs(experiment, policy_bp, seed, n, perturb))
    total = len(episodes)
    surv = sum(1 for ep in episodes if not all(r.startswith("imbalance") for r in ep.agent_termination_reason.values()))
    mean_len = float(np.mean([ep.num_frames for ep in episodes])) if episodes else 0.0
    print(f"{label:<52} {surv / total if total else 0:8.3f} {mean_len:9.1f} {total - surv:7d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--policy-export", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=31337)
    args = p.parse_args()

    experiment = get_experiment(args.experiment)
    base = dict(experiment.PERTURB_FULL)

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    rollouter = ParallelRollouter(num_workers=args.workers)

    print(f"episodes={args.episodes}   reference (scale=1.0): {base}")
    print()
    print(f"{'condition':<52} {'survived':>8} {'mean_len':>9} {'n_fell':>7}")
    print("-" * 80)

    # A. Velocity-only perturbation (physically correct post-impact state),
    #    swept well beyond the current scale=1.0 magnitudes.
    for mult in (1, 2, 4, 8, 16):
        cfg = dict(ZERO)
        cfg["joint_vel_delta_max"] = base["joint_vel_delta_max"] * mult
        cfg["root_linear_velocity_delta_max"] = base["root_linear_velocity_delta_max"] * mult
        cfg["root_angular_velocity_delta_max"] = base["root_angular_velocity_delta_max"] * mult
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
            f"velocities only, {mult}x scale-1.0")
    print("-" * 80)

    # B. Pure horizontal root impulse — the cleanest analogue of a punch.
    for v in (2.0, 4.0, 6.0, 8.0, 10.0):
        cfg = dict(ZERO)
        cfg["root_linear_velocity_delta_max"] = [v, v, 0.0]
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
            f"horizontal root impulse only, +-{v:.0f} m/s")
    print("-" * 80)

    # C. Root angular velocity (roll/pitch) only — tipping impulse.
    for w in (1.0, 2.0, 4.0, 6.0):
        cfg = dict(ZERO)
        cfg["root_angular_velocity_delta_max"] = [w, w, 0.0]
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
            f"root roll/pitch rate only, +-{w:.0f} rad/s")
    print("-" * 80)

    # D. Joint-velocity only — internal disturbance.
    for jv in (2.0, 4.0, 8.0, 16.0):
        cfg = dict(ZERO)
        cfg["joint_vel_delta_max"] = jv
        run(rollouter, experiment, policy_bp, args.seed, args.episodes, cfg,
            f"joint velocities only, +-{jv:.0f} (norm)")
    print("-" * 80)


if __name__ == "__main__":
    main()
