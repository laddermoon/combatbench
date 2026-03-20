from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions.normal import Normal

from .env import SharedPolicySelfPlayHPEnv, SelfPlayHPRewardConfig
from .train_shared_policy import ActorCritic, obs_dict_to_tensor, select_device, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a shared-policy HP self-play checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--control-frequency", type=int, default=None)
    parser.add_argument("--initial-distance", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    return parser


def resolve_run_dir(checkpoint_path: Path) -> Path:
    candidates = [checkpoint_path.parent, checkpoint_path.parent.parent]
    for candidate in candidates:
        if (candidate / "run_config.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find run_config.json near checkpoint: {checkpoint_path}")


def load_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_config.json").read_text())


def default_video_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_rollout.mp4")


def build_env(run_config: dict, args: argparse.Namespace) -> SharedPolicySelfPlayHPEnv:
    reward_cfg = run_config.get("reward_config", {})
    reward_config = SelfPlayHPRewardConfig(
        damage_reward_scale=float(reward_cfg.get("damage_reward_scale", 1.0)),
        damage_penalty_scale=float(reward_cfg.get("damage_penalty_scale", 1.0)),
        win_bonus=float(reward_cfg.get("win_bonus", 0.0)),
        lose_penalty=float(reward_cfg.get("lose_penalty", 0.0)),
    )
    return SharedPolicySelfPlayHPEnv(
        render_mode="rgb_array" if args.video else None,
        match_duration=float(args.duration if args.duration is not None else run_config.get("match_duration", 15.0)),
        control_frequency=int(args.control_frequency if args.control_frequency is not None else run_config.get("control_frequency", 20)),
        initial_distance=float(args.initial_distance if args.initial_distance is not None else run_config.get("initial_distance", 1.0)),
        non_fall_mode=bool(run_config.get("non_fall_mode", False)),
        non_fall_pitch_limit_deg=float(run_config.get("non_fall_pitch_limit_deg", 15.0)),
        non_fall_roll_limit_deg=float(run_config.get("non_fall_roll_limit_deg", 10.0)),
        damage_scale=float(run_config.get("damage_scale", 100.0)),
        reward_config=reward_config,
    )


def sample_actions(model: ActorCritic, obs_tensor: torch.Tensor, stochastic: bool) -> torch.Tensor:
    with torch.no_grad():
        action_mean = model.actor_mean(obs_tensor)
        if not stochastic:
            return torch.clamp(action_mean, -1.0, 1.0)
        action_std = torch.exp(model.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        return torch.clamp(dist.sample(), -1.0, 1.0)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = resolve_run_dir(checkpoint_path)
    run_config = load_run_config(run_dir)
    set_seed(args.seed)
    device = select_device(args.device)

    env = build_env(run_config, args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    obs_dim = int(checkpoint.get("obs_dim", env.observation_space["robot_a"].shape[0]))
    action_dim = int(checkpoint.get("action_dim", env.action_space.shape[0]))
    model = ActorCritic(obs_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    obs, info = env.reset(seed=args.seed)
    obs_tensor = obs_dict_to_tensor(obs, device)
    episode_returns = {"robot_a": 0.0, "robot_b": 0.0}
    step_count = 0

    while True:
        actions = sample_actions(model, obs_tensor, args.stochastic)
        obs, rewards, terminated, truncated, info = env.step(
            actions[0].cpu().numpy(),
            actions[1].cpu().numpy(),
        )
        episode_returns["robot_a"] += float(rewards["robot_a"])
        episode_returns["robot_b"] += float(rewards["robot_b"])
        step_count += 1
        if terminated or truncated:
            break
        obs_tensor = obs_dict_to_tensor(obs, device)

    if args.video:
        output_path = Path(args.video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env.base_env.save_video(str(output_path), fps=env.base_env.video_sample_frequency)
        print(f"video_path={output_path}")
        print(f"frame_count={len(env.base_env.get_video_buffer())}")

    print(f"checkpoint={checkpoint_path}")
    print(f"steps={step_count}")
    print(f"returns={episode_returns}")
    print(f"scores={info.get('scores')}")
    print(f"winner={info.get('winner')}")
    print(f"end_reason={info.get('end_reason')}")
    env.close()


if __name__ == "__main__":
    main()
