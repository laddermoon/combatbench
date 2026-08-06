"""Measure the temporal profile of posture quantities to tell apart
"recovers by stepping" from "recovers by stiffening", and "alternating
bipedal balance" from "statue".

Reports, averaged over surviving episodes, the mean/max of the posture
observer fields in the recovery transient vs the steady state, plus how often
the shaping thresholds are actually exceeded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baseline.humanoid21.curriculum.experiments import get_experiment
from baseline.common.rollout import ParallelRollouter
from baseline.framework.ppo_trainer import _extract_per_step_field
from envs.framework.policy import PolicyBlueprint

FIELDS = ("foot_height", "joint_deviation", "joint_vel", "torso_tilt")
# Shaping thresholds from extract_rewards() in the balance experiments.
THRESHOLDS = {
    "foot_height": 0.10,
    "joint_deviation": 0.1,
    "joint_vel": 0.1,
    "torso_tilt": 0.26,
}


def profile(episodes, label: str, transient: int) -> None:
    survivors = [ep for ep in episodes if not all(r.startswith("imbalance") for r in ep.agent_termination_reason.values())]
    print(f"\n=== {label} ===")
    print(f"episodes={len(episodes)}  survivors={len(survivors)}")
    if not survivors:
        return

    print(f"{'field':>16} {'transient mean':>15} {'transient max':>14} "
          f"{'steady mean':>12} {'steady max':>11} {'%steps>thr':>11}")
    for f in FIELDS:
        tr, st, over, total = [], [], 0, 0
        for ep in survivors:
            arr = _extract_per_step_field(ep.observer_outputs, "posture", f, ep.num_frames)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            tr.append(arr[:transient])
            if arr.size > transient:
                st.append(arr[transient:])
            over += int(np.sum(arr > THRESHOLDS[f]))
            total += arr.size
        if not tr:
            continue
        trc = np.concatenate(tr)
        stc = np.concatenate(st) if st else np.array([np.nan])
        print(f"{f:>16} {trc.mean():15.4f} {trc.max():14.4f} "
              f"{np.nanmean(stc):12.4f} {np.nanmax(stc):11.4f} "
              f"{100.0 * over / max(total, 1):10.1f}%")

    # Steady-state foot-height oscillation: distinguishes alternating gait
    # (foot repeatedly leaves the ground) from a frozen stance.
    amps, lifts = [], []
    for ep in survivors:
        arr = _extract_per_step_field(ep.observer_outputs, "posture", "foot_height", ep.num_frames)
        if arr is None or ep.num_frames <= transient:
            continue
        seg = np.asarray(arr, dtype=np.float64)[transient:]
        amps.append(seg.max() - seg.min())
        # Nominal grounded foot height is 0.027; count clear lift-offs.
        lifts.append(float(np.mean(seg > 0.027 + 0.01)))
    if amps:
        print(f"  steady-state foot_height peak-to-peak: mean={np.mean(amps):.4f} m  "
              f"median={np.median(amps):.4f} m")
        print(f"  steady-state fraction of steps with a foot lifted >1cm: "
              f"{100.0 * np.mean(lifts):.1f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--policy-export", required=True)
    p.add_argument("--level", type=int, default=None,
                   help="Curriculum level (perturbation experiments only).")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--transient", type=int, default=20,
                   help="Number of leading action steps treated as the recovery transient.")
    args = p.parse_args()

    experiment = get_experiment(args.experiment)
    if args.level is not None:
        experiment._level = args.level

    policy_bp = PolicyBlueprint.load(Path(args.policy_export) / "policy_blueprint.yaml")
    rollouter = ParallelRollouter(num_workers=args.workers)

    jobs = experiment.build_eval_jobs(policy_bp, args.seed)[:args.episodes]
    episodes = rollouter.collect(jobs)

    lvl = "" if args.level is None else f" level={args.level}"
    profile(episodes, f"{args.experiment}{lvl}", args.transient)


if __name__ == "__main__":
    main()
