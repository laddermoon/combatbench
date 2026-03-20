from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ...tools.run_match import MatchRunner
from .policies import SB3CombatPolicy
from .rewards import resolve_reward_config
from .selfplay_env import make_attacker_standing_env, make_symmetric_selfplay_env



def configure_runtime() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")



def evaluate_shared_env(model_path: str, phase: str, episodes: int, match_duration: float, control_frequency: int, initial_distance: float, opponent_model: str | None = None) -> None:
    reward_config = resolve_reward_config(phase)
    if phase == "fight_attacker":
        opponent_model_path = opponent_model or model_path
        env = make_attacker_standing_env(
            opponent_model_path=opponent_model_path,
            render_mode=None,
            match_duration=match_duration,
            control_frequency=control_frequency,
            initial_distance=initial_distance,
            reward_config=reward_config,
            opponent_device="auto",
        )
    else:
        env = make_symmetric_selfplay_env(
            render_mode=None,
            match_duration=match_duration,
            control_frequency=control_frequency,
            initial_distance=initial_distance,
            reward_config=reward_config,
        )
    model = PPO.load(model_path, env=env, device="auto")

    rewards = []
    lengths = []
    winners = {}
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            if terminated or truncated:
                winners[info.get("winner", "unknown")] = winners.get(info.get("winner", "unknown"), 0) + 1
                break
        rewards.append(episode_reward)
        lengths.append(episode_length)
        print(
            f"Episode {episode + 1:02d} | reward={episode_reward:.3f} | steps={episode_length} | winner={info.get('winner')} | reason={info.get('end_reason')}"
        )

    print("-" * 72)
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Average steps: {np.mean(lengths):.1f}")
    print(f"Winner counts: {winners}")
    env.close()



def evaluate_match(model_a: str, model_b: str | None, video: str | None, duration: float, control_frequency: int, initial_distance: float, phase: str) -> None:
    policy_a = SB3CombatPolicy(model_a)
    policy_b = SB3CombatPolicy(model_b or model_a)
    runner = MatchRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        render_mode="rgb_array" if video else None,
        match_duration=duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
        phase=phase,
    )
    result = runner.run(save_video_path=video)
    print(result)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the SB3 shared-policy combat baseline")
    parser.add_argument("--mode", choices=["selfplay", "match"], default="selfplay")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument("--phase", choices=["stand", "fight", "fight_attacker"], default="fight")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--video", type=str, default=None)
    return parser



def main() -> None:
    configure_runtime()
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.phase == "fight_attacker" and float(args.initial_distance) != 2.0:
        raise ValueError("fight_attacker phase requires initial_distance=2.0")

    if args.mode == "selfplay":
        evaluate_shared_env(
            model_path=args.model,
            phase=args.phase,
            episodes=args.episodes,
            match_duration=args.duration,
            control_frequency=args.control_frequency,
            initial_distance=args.initial_distance,
            opponent_model=args.model_b,
        )
        return

    evaluate_match(
        model_a=args.model,
        model_b=args.model_b,
        video=args.video,
        duration=args.duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
