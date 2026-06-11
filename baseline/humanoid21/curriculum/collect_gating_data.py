#!/usr/bin/env python3
"""Script to collect gating classifier data using a weakened version of u03275.

We use the u03275 recovery policy under the ultra-violent 'balance_recover_plus' 
perturbations, slightly weakened by injecting action-space Gaussian noise (e.g. standard deviation 0.08).

Data labeling uses 'Temporal Back-Tracking Noiseless Labeling':
- If an episode falls ("imbalance" is in termination_proposals), all steps in that episode are labeled 0 (unsafe).
- If an episode survives (completes the full 200 steps without falling), all steps in that episode are labeled 1 (safe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Collect gating classifier dataset.")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_balance_recover_20260611_004703/policy_exports/u03275",
        help="Path to the u03275 policy directory containing model.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_data",
        help="Output directory to save collected dataset"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2000,
        help="Total number of episodes to collect"
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
        help="Perturbation level scale index (level 6 = scale 1.0 = maximum)"
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

    # 1. Load experiment config & force maximum perturbation scale
    experiment = get_experiment(args.experiment_name)
    experiment._level = args.perturb_level
    # Configure parallel workers in the experiment config
    experiment.rollout_workers = args.workers

    scale = experiment.current_scale
    print("=" * 70)
    print(f"🌟 Starting Gating Data Collection Pipeline")
    print(f"   - Base Policy:      {policy_dir}")
    print(f"   - Weakening Noise:  {args.noise_std:.4f} (Gaussian standard deviation)")
    print(f"   - Perturb Config:   {args.experiment_name} (Level {args.perturb_level}, Scale {scale:.3f})")
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

    # 4. Run Parallel Collector in Chunks
    with ParallelRollouter(num_workers=args.workers) as rollouter:
        while total_episodes_done < args.num_episodes:
            chunk_episodes = min(args.chunk_size, args.num_episodes - total_episodes_done)
            chunk_seed = args.base_seed + total_episodes_done * 97

            print(f"🚀 Collecting chunk [{total_episodes_done + 1} - {total_episodes_done + chunk_episodes}] ...", end="", flush=True)
            t_chunk_start = time.perf_counter()

            # Build jobs
            jobs = experiment._build_perturbed_jobs(
                policy_bp=policy_bp,
                base_seed=chunk_seed,
                n_episodes=chunk_episodes,
            )

            # Parallel rollout
            episodes: List[Episode] = rollouter.collect(jobs)
            t_chunk_elapsed = time.perf_counter() - t_chunk_start

            # Process episodes
            chunk_falls = 0
            chunk_survivals = 0
            chunk_frames = 0

            for ep in episodes:
                agent_id = ep.episode_options["agent_id"]
                obs = ep.observations[agent_id]  # Shape: (T, obs_dim)
                T = ep.num_frames
                
                # Determine if the robot fell
                fell = "imbalance" in ep.termination_proposals
                
                if fell:
                    chunk_falls += 1
                    labels = np.zeros((T,), dtype=np.float32)
                else:
                    chunk_survivals += 1
                    labels = np.ones((T,), dtype=np.float32)

                all_observations.append(obs)
                all_labels.append(labels)

                chunk_frames += T
                episode_lengths.append(T)

            # Update overall stats
            total_episodes_done += chunk_episodes
            total_falls += chunk_falls
            total_survivals += chunk_survivals
            total_frames += chunk_frames

            chunk_success_rate = (chunk_survivals / chunk_episodes) * 100.0
            print(f" Done in {t_chunk_elapsed:.2f}s | Success Rate: {chunk_success_rate:.1f}% | Avg Len: {np.mean(episode_lengths[-chunk_episodes:]):.1f} steps", flush=True)

    # 5. Concatenate and save data
    print("\n💾 Formatting and saving collected dataset...", end="", flush=True)
    X = np.concatenate(all_observations, axis=0)  # Shape: (Total_Frames, obs_dim)
    Y = np.concatenate(all_labels, axis=0)        # Shape: (Total_Frames,)
    
    # Save as compressed .npz file
    npz_path = output_dir / "gating_data.npz"
    np.savez_compressed(npz_path, observations=X, labels=Y)

    # Create summary metadata
    overall_success_rate = (total_survivals / args.num_episodes) * 100.0
    num_positive = int(np.sum(Y == 1.0))
    num_negative = int(np.sum(Y == 0.0))
    
    metadata = {
        "policy_path": args.policy_path,
        "noise_std": args.noise_std,
        "experiment_name": args.experiment_name,
        "perturb_level": args.perturb_level,
        "perturb_scale": scale,
        "total_episodes": args.num_episodes,
        "overall_success_rate": overall_success_rate,
        "total_falls": total_falls,
        "total_survivals": total_survivals,
        "total_frames": total_frames,
        "num_positive_frames": num_positive,
        "num_negative_frames": num_negative,
        "obs_dim": X.shape[1],
        "date_collected": time.strftime("%Y-%m-%d %H:%M:%S")
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
    print(f"     - Safe (Label 1):  {num_positive:,} ({num_positive/total_frames*100.0:.1f}%)")
    print(f"     - Unsafe (Label 0): {num_negative:,} ({num_negative/total_frames*100.0:.1f}%)")
    print(f"   - Episode stats:")
    print(f"     - Total:          {args.num_episodes}")
    print(f"     - Safe Stands:    {total_survivals} ({overall_success_rate:.1f}% survival rate)")
    print(f"     - Fallen Runs:    {total_falls} ({100.0 - overall_success_rate:.1f}% fall rate)")
    print(f"     - Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"   - Total Execution Time: {t_total:.1f} seconds ({t_total/60.0:.1f} minutes)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
