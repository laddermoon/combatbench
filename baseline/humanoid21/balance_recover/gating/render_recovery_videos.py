"""Render recovery videos for specific states from a labeled state pool.

Useful for inspecting "surprising" recoveries — states where the robot is
already low (torso height below a threshold) but still recovers.

Usage::

    python3 baseline/humanoid21/balance_recover/gating/render_recovery_videos.py \\
        --input baseline/humanoid21/balance_recover/gating/labeled_state_pool_a.npz \\
        --policy baseline/runs/recovery_v5_gen9/policy_exports/u00635/policy_blueprint.yaml \\
        --output-dir baseline/humanoid21/balance_recover/gating/recovery_videos \\
        --max-height 0.5 \\
        --label 1 \\
        --max-videos 20
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.episode_runner import EpisodeRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Render recovery videos for specific states")
    parser.add_argument("--input", type=str, required=True, help="Labeled state_pool .npz path")
    parser.add_argument("--policy", type=str, required=True, help="Path to π_recover policy_blueprint.yaml")
    parser.add_argument("--output-dir", type=str, required=True, help="Output video directory")
    parser.add_argument("--env-yaml", type=str,
                        default="baseline/humanoid21/balance_recover/gating/label_state_pool_env.yaml")
    parser.add_argument("--max-height", type=float, default=0.5,
                        help="Only render states with root_pos[2] below this value")
    parser.add_argument("--label", type=int, default=1,
                        help="Only render states with this label (1=safe, 0=unsafe)")
    parser.add_argument("--max-videos", type=int, default=20, help="Max videos to render")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    states = data["states"]
    labels = data["labels"]
    n_total = len(states)
    print(f"Loaded {n_total} states from {args.input}")

    heights = states[:, 2]
    mask = (heights < args.max_height) & (labels == args.label)
    candidate_indices = np.where(mask)[0]
    print(f"States with height < {args.max_height} and label={args.label}: {len(candidate_indices)}")

    if len(candidate_indices) == 0:
        print("No matching states found.")
        return

    n = min(args.max_videos, len(candidate_indices))
    selected = np.random.RandomState(42).choice(candidate_indices, size=n, replace=False)
    selected.sort()
    print(f"Rendering {n} videos...")

    # Load policy
    policy_bp = PolicyBlueprint.load(Path(args.policy))
    policy = policy_bp.build()
    print(f"Loaded policy from {args.policy}")

    # Load env blueprint template
    env_pb = ParameterizedEnvBlueprint.load(args.env_yaml)
    state_bank_path = str(Path(args.input).resolve())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(selected):
        height = float(heights[idx])
        label = int(labels[idx])
        video_path = out_dir / f"state_{idx:06d}_h{height:.2f}_l{label}.mp4"

        env_bp = env_pb.materialize(
            state_bank_path=state_bank_path,
            max_steps=100,
        )
        video_plugin = VideoRecorderPlugin(fps=args.fps, output_path=str(video_path))
        runtime = env_bp.build(debug_plugins=[video_plugin])

        runner = EpisodeRunner(
            runtime=runtime,
            policy_a=policy,
            policy_b=policy,
        )
        runner.run_episode(seed=42, options={"state_bank_index": int(idx)})

        reasons = runtime.get_agent_termination()
        reason = reasons.get("robot_a", "unknown")
        print(f"  [{i+1}/{n}] idx={idx} h={height:.3f} label={label} reason={reason} -> {video_path.name}")
        runtime.close()

    print(f"\nDone. {n} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
