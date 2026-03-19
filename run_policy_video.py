from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from combatbench.baseline.sb3.policies import SB3CombatPolicy
from combatbench.baseline.sb3.rewards import resolve_reward_config
from combatbench.baseline.sb3.selfplay_env import make_symmetric_selfplay_env
from combatbench.tools.run_match import MatchRunner


def default_video_path(model_path: str, phase: str, mode: str) -> Path:
    model_file = Path(model_path)
    if model_file.suffix == ".zip":
        return model_file.with_name(f"{model_file.stem}_{phase}_{mode}.mp4")
    return model_file.parent / f"{model_file.name}_{phase}_{mode}.mp4"


def export_shared_env_video(
    model_path: str,
    phase: str,
    video_path: str,
    duration: float,
    control_frequency: int,
    initial_distance: float,
    device: str,
) -> None:
    reward_config = resolve_reward_config(phase)
    env = make_symmetric_selfplay_env(
        render_mode="rgb_array",
        match_duration=duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
        reward_config=reward_config,
    )
    model = PPO.load(model_path, env=env, device=device)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        episode_length += 1
        if terminated or truncated:
            break

    output_path = Path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env.base_env.save_video(str(output_path), fps=env.base_env.video_sample_frequency)
    print(f"video_path={output_path}")
    print(f"frame_count={len(env.base_env.get_video_buffer())}")
    print(f"steps={episode_length}")
    print(f"reward={episode_reward:.3f}")
    print(f"winner={info.get('winner')}")
    print(f"end_reason={info.get('end_reason')}")
    env.close()


def export_match_video(
    model_a: str,
    model_b: str | None,
    phase: str,
    video_path: str,
    duration: float,
    control_frequency: int,
    initial_distance: float,
    device: str,
) -> None:
    policy_a = SB3CombatPolicy(model_a, device=device)
    policy_b = SB3CombatPolicy(model_b or model_a, device=device)
    runner = MatchRunner(
        policy_a=policy_a,
        policy_b=policy_b,
        render_mode="rgb_array",
        match_duration=duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
        phase=phase,
    )
    result = runner.run(save_video_path=video_path)
    print(f"video_path={video_path}")
    print(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a rollout video for an SB3 CombatBench policy")
    parser.add_argument("--mode", choices=["shared_env", "match"], default="shared_env")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-b", type=str, default=None)
    parser.add_argument("--phase", choices=["stand", "fight"], default="stand")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    video_path = args.video or str(default_video_path(args.model, args.phase, args.mode))

    if args.mode == "shared_env":
        export_shared_env_video(
            model_path=args.model,
            phase=args.phase,
            video_path=video_path,
            duration=args.duration,
            control_frequency=args.control_frequency,
            initial_distance=args.initial_distance,
            device=args.device,
        )
        return

    export_match_video(
        model_a=args.model,
        model_b=args.model_b,
        phase=args.phase,
        video_path=video_path,
        duration=args.duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
        device=args.device,
    )


if __name__ == "__main__":
    main()
