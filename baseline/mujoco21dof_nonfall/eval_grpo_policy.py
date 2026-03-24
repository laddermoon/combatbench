import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.baseline.mujoco21dof_nonfall.eval_policy import (
    build_summary,
    load_opponent,
    maybe_write_json,
    round_result_to_dict,
)
from combatbench.baseline.mujoco21dof_nonfall.grpo_policy import GRPOCombatPolicy
from combatbench.envs import RoundRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GRPO attacker policy and optionally export a video.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="standing")
    parser.add_argument("--opponent-policy", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--match-duration", type=float, default=10.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--non-fall-mode", action="store_true")
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=5.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=5.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = GRPOCombatPolicy(
        model_path=args.model_path,
        device=args.device,
        deterministic=not args.stochastic,
    )

    results: List[dict] = []
    for episode_index in range(args.episodes):
        opponent = load_opponent(args.opponent, args.opponent_policy, args.device)
        render_mode = "rgb_array" if args.video and episode_index == 0 else None
        video_path = args.video if episode_index == 0 else None
        if video_path is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        runner = RoundRunner(
            policy_a=policy,
            policy_b=opponent,
            render_mode=render_mode,
            match_duration=args.match_duration,
            control_frequency=args.control_frequency,
            initial_distance=args.initial_distance,
            non_fall_mode=args.non_fall_mode,
            non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
            non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
            damage_scale=args.damage_scale,
            verbose=not args.quiet,
        )
        episode_seed = args.seed + episode_index
        result = runner.run(save_video_path=video_path, seed=episode_seed)
        results.append(round_result_to_dict(result, episode_index, episode_seed))

    summary = build_summary(results, args)
    maybe_write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
