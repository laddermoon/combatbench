import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.baseline.mujoco21dof_nonfall.opponents import make_opponent_policy
from combatbench.baseline.mujoco21dof_nonfall.policy_adapter import SB3PPOCombatPolicy
from combatbench.envs import RoundRunner
from combatbench.tools.run_round import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO attacker policy and optionally export a video.")
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
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=15.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=10.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_opponent(opponent_spec: str, opponent_policy_spec: str | None, device: str):
    if opponent_policy_spec:
        return load_policy(opponent_policy_spec, device=device)
    return make_opponent_policy(opponent_spec)


def round_result_to_dict(result, episode_index: int, seed: int) -> Dict[str, Any]:
    return {
        "episode_index": episode_index,
        "seed": seed,
        "steps": int(result.steps),
        "end_reason": result.end_reason,
        "winner": result.winner,
        "scores": dict(result.scores),
        "initial_scores": dict(result.initial_scores),
        "damage_dealt": dict(result.damage_dealt),
        "total_reward": dict(result.total_reward),
        "video_frames": int(result.video_frames),
    }


def build_summary(results: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_path": args.model_path,
        "episodes": len(results),
        "opponent": args.opponent_policy or args.opponent,
        "deterministic": not args.stochastic,
        "mean_steps": mean(item["steps"] for item in results),
        "mean_robot_a_damage_dealt": mean(item["damage_dealt"]["robot_a"] for item in results),
        "mean_robot_b_damage_dealt": mean(item["damage_dealt"]["robot_b"] for item in results),
        "mean_robot_a_score": mean(item["scores"]["robot_a"] for item in results),
        "mean_robot_b_score": mean(item["scores"]["robot_b"] for item in results),
        "wins_robot_a": sum(1 for item in results if item["winner"] == "robot_a"),
        "wins_robot_b": sum(1 for item in results if item["winner"] == "robot_b"),
        "draws": sum(1 for item in results if item["winner"] in (None, "draw")),
        "results": results,
    }


def maybe_write_json(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    policy = SB3PPOCombatPolicy(
        model_path=args.model_path,
        device=args.device,
        deterministic=not args.stochastic,
    )

    results: List[Dict[str, Any]] = []
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
