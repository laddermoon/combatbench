#!/usr/bin/env python3
"""Collect gating classifier data with **even multi-level perturbation coverage**.

Unlike the original ``collect_gating_data.py`` which collects at a single
perturbation level, this script distributes episodes **evenly** across all
levels ``{0, 1, ..., --perturb-level}``.  This ensures the gating classifier
sees data spanning the full perturbation spectrum, preventing the classifier
from becoming miscalibrated on low-magnitude perturbations (the same
"forgetting" problem that motivated the mixed-level training curriculum).

Data labeling uses 'Temporal Back-Tracking Noiseless Labeling':
- If an episode falls ("imbalance" is in termination_proposals), all steps are labeled 0 (unsafe).
- If an episode survives (completes the full episode without falling), all steps are labeled 1 (safe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure combatbench root is in sys.path
COMBATBENCH_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(COMBATBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COMBATBENCH_ROOT))

# Ensure child processes inherit the same PYTHONPATH so they can load 'weakened_policy'
os.environ["PYTHONPATH"] = str(COMBATBENCH_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from envs.framework.policy import PolicyBlueprint
from baseline.common.rollout import ParallelRollouter, Episode
from baseline.humanoid21.curriculum.experiments import get_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect gating classifier dataset (multi-level).")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_004703/policy_exports/u03275",
        help="Path to the policy directory containing model.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data_refine",
        help="Output directory to save collected dataset"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2000,
        help="Total number of episodes to collect (evenly split across levels)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.08,
        help="Standard deviation of Gaussian action noise to weaken the policy"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="balance_recover_plus",
        help="Name of the experiment config to use for perturbations"
    )
    parser.add_argument(
        "--perturb-level",
        type=int,
        default=6,
        help="Maximum perturbation level; data is collected evenly from levels 0..this"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of episodes to run in each parallel batch (prevents memory bloat)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel rollout workers"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=50000,
        help="Base random seed for rollouts"
    )
    return parser.parse_args()


def main() -> None:
    t_start = time.perf_counter()
    args = parse_args()

    policy_dir = Path(args.policy_path)
    model_path = policy_dir / "model.pt"
    if not model_path.exists():
        print(f"Error: model.pt not found in {policy_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load experiment config
    experiment = get_experiment(args.experiment_name)
    experiment.rollout_workers = args.workers

    max_level = int(args.perturb_level)
    levels = list(range(max_level + 1))
    n_levels = len(levels)

    print("=" * 70)
    print(f"🌟 Starting Gating Data Collection (Multi-Level Even Split)")
    print(f"   - Base Policy:      {policy_dir}")
    print(f"   - Weakening Noise:  {args.noise_std:.4f} (Gaussian standard deviation)")
    print(f"   - Perturb Config:   {args.experiment_name}")
    print(f"   - Level Range:      0..{max_level} ({n_levels} levels, ~{100.0 / n_levels:.1f}% each)")
    for lvl in levels:
        experiment._level = lvl
        print(f"     - Level {lvl}: scale={experiment.current_scale:.2f}")
    print(f"   - Total Episodes:   {args.num_episodes}")
    print(f"   - Rollout Workers:  {args.workers}")
    print(f"   - Chunk Size:       {args.chunk_size}")
    print("=" * 70, flush=True)

    # 2. Build PolicyBlueprint pointing to our custom weakened policy
    weakened_policy_py = Path(__file__).resolve().parent / "weakened_policy.py"
    policy_bp = PolicyBlueprint(
        cls=f"file:{weakened_policy_py}:WeakenedExportedMLPPolicy",
        config={
            "model_path": str(model_path),
            "stochastic": False,
            "noise_std": args.noise_std,
        }
    )

    # 3. Accumulators
    all_observations: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    total_episodes_done = 0
    total_falls = 0
    total_survivals = 0
    total_frames = 0
    episode_lengths: List[int] = []

    # Per-level statistics
    level_stats: Dict[int, Dict[str, int]] = {
        lvl: {"episodes": 0, "falls": 0, "survivals": 0, "frames": 0}
        for lvl in levels
    }

    # 4. Run Parallel Collector in Chunks
    with ParallelRollouter(num_workers=args.workers) as rollouter:
        while total_episodes_done < args.num_episodes:
            chunk_episodes = min(args.chunk_size, args.num_episodes - total_episodes_done)
            chunk_seed = args.base_seed + total_episodes_done * 97

            print(
                f"🚀 Collecting chunk [{total_episodes_done + 1} - {total_episodes_done + chunk_episodes}] "
                f"({chunk_episodes} eps across {n_levels} levels) ...",
                end="", flush=True,
            )
            t_chunk_start = time.perf_counter()

            # --- Distribute chunk episodes evenly across all levels ---
            base_per_level = chunk_episodes // n_levels
            remainder = chunk_episodes % n_levels

            chunk_jobs: List[Any] = []
            for li, level in enumerate(levels):
                n_for_level = base_per_level + (1 if li < remainder else 0)
                if n_for_level == 0:
                    continue
                experiment._level = level
                # Large per-level seed offset to avoid collisions across levels/chunks
                level_seed = chunk_seed + level * 1_000_000
                level_jobs = experiment._build_perturbed_jobs(
                    policy_bp=policy_bp,
                    base_seed=level_seed,
                    n_episodes=n_for_level,
                )
                # Tag each job with its perturbation level for per-level tracking
                for job in level_jobs:
                    info = {**job[4], "perturb_level": float(level)}
                    chunk_jobs.append((job[0], job[1], job[2], job[3], info))

            # --- Single parallel collect for the entire chunk ---
            episodes: List[Episode] = rollouter.collect(chunk_jobs)
            t_chunk_elapsed = time.perf_counter() - t_chunk_start

            # --- Process episodes ---
            chunk_falls = 0
            chunk_survivals = 0
            chunk_frames = 0

            for ep in episodes:
                level = int(ep.episode_options.get("perturb_level", 0))
                agent_id = ep.episode_options["agent_id"]
                obs = ep.observations[agent_id]  # Shape: (T, obs_dim)
                T = ep.num_frames

                fell = "imbalance" in ep.termination_proposals

                if fell:
                    chunk_falls += 1
                    labels = np.zeros((T,), dtype=np.float32)
                    level_stats[level]["falls"] += 1
                else:
                    chunk_survivals += 1
                    labels = np.ones((T,), dtype=np.float32)
                    level_stats[level]["survivals"] += 1

                all_observations.append(obs)
                all_labels.append(labels)

                chunk_frames += T
                episode_lengths.append(T)
                level_stats[level]["episodes"] += 1
                level_stats[level]["frames"] += T

            # Update overall stats
            total_episodes_done += chunk_episodes
            total_falls += chunk_falls
            total_survivals += chunk_survivals
            total_frames += chunk_frames

            chunk_success_rate = (chunk_survivals / chunk_episodes) * 100.0
            print(
                f" Done in {t_chunk_elapsed:.2f}s | Success: {chunk_success_rate:.1f}% "
                f"| Avg Len: {np.mean(episode_lengths[-chunk_episodes:]):.1f} steps",
                flush=True,
            )

    # 5. Concatenate and save data
    print("\n💾 Formatting and saving collected dataset...", end="", flush=True)
    X = np.concatenate(all_observations, axis=0)  # Shape: (Total_Frames, obs_dim)
    Y = np.concatenate(all_labels, axis=0)        # Shape: (Total_Frames,)

    npz_path = output_dir / "gating_data.npz"
    np.savez_compressed(npz_path, observations=X, labels=Y)

    # Create summary metadata
    overall_success_rate = (total_survivals / args.num_episodes) * 100.0
    num_positive = int(np.sum(Y == 1.0))
    num_negative = int(np.sum(Y == 0.0))

    # Build per-level metadata (JSON keys must be strings)
    level_stats_jsonable = {}
    for lvl in levels:
        s = level_stats[lvl]
        eps = s["episodes"]
        rate = (s["survivals"] / eps * 100.0) if eps > 0 else 0.0
        level_stats_jsonable[str(lvl)] = {
            "episodes": eps,
            "falls": s["falls"],
            "survivals": s["survivals"],
            "frames": s["frames"],
            "survival_rate_pct": round(rate, 1),
        }

    metadata = {
        "policy_path": args.policy_path,
        "noise_std": args.noise_std,
        "experiment_name": args.experiment_name,
        "max_perturb_level": max_level,
        "level_distribution": "even",
        "total_episodes": args.num_episodes,
        "overall_success_rate": overall_success_rate,
        "total_falls": total_falls,
        "total_survivals": total_survivals,
        "total_frames": total_frames,
        "num_positive_frames": num_positive,
        "num_negative_frames": num_negative,
        "obs_dim": X.shape[1],
        "level_stats": level_stats_jsonable,
        "date_collected": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(" Done!")

    # 6. Beautiful summary output
    t_total = time.perf_counter() - t_start
    print("=" * 70)
    print("🎉 Dataset Collection Successfully Completed!")
    print(f"   - Saved .npz Path:  {npz_path}")
    print(f"   - Saved JSON Path: {json_path}")
    print(f"   - Total Frames:     {total_frames:,}")
    print(f"     - Safe (Label 1):  {num_positive:,} ({num_positive / total_frames * 100.0:.1f}%)")
    print(f"     - Unsafe (Label 0): {num_negative:,} ({num_negative / total_frames * 100.0:.1f}%)")
    print(f"   - Episode stats:")
    print(f"     - Total:          {args.num_episodes}")
    print(f"     - Safe Stands:    {total_survivals} ({overall_success_rate:.1f}% survival rate)")
    print(f"     - Fallen Runs:    {total_falls} ({100.0 - overall_success_rate:.1f}% fall rate)")
    print(f"     - Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"   - Per-level breakdown:")
    print(f"     {'Level':>5} | {'Scale':>6} | {'Eps':>6} | {'Falls':>6} | {'Surv':>6} | {'Frames':>8} | {'Surv%':>6}")
    print(f"     {'-----':>5} | {'------':>6} | {'-----':>6} | {'-----':>6} | {'-----':>6} | {'------':>8} | {'-----':>6}")
    for lvl in levels:
        s = level_stats[lvl]
        experiment._level = lvl
        eps = s["episodes"]
        rate = (s["survivals"] / eps * 100.0) if eps > 0 else 0.0
        print(
            f"     {lvl:>5} | {experiment.current_scale:>6.2f} | {eps:>6} | {s['falls']:>6} | {s['survivals']:>6} "
            f"| {s['frames']:>8,} | {rate:>5.1f}%"
        )
    print(f"   - Total Execution Time: {t_total:.1f} seconds ({t_total / 60.0:.1f} minutes)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
