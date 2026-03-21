from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .env import SharedPolicySelfPlayHPEnv, SelfPlayHPRewardConfig


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    rollout_steps: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    minibatch_size: int = 512
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden_sizes = [256, 256, 128]
        actor_layers = []
        critic_layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_sizes:
            actor_layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            critic_layers.extend([nn.Linear(last_dim, hidden_dim), nn.Tanh()])
            last_dim = hidden_dim
        self.actor_mean = nn.Sequential(*actor_layers, nn.Linear(last_dim, action_dim))
        self.critic = nn.Sequential(*critic_layers, nn.Linear(last_dim, 1))
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs)
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, rollout_steps: int, agent_count: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.rollout_steps = rollout_steps
        self.agent_count = agent_count
        self.obs = torch.zeros((rollout_steps, agent_count, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, agent_count, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, agent_count), dtype=torch.float32, device=device)

    def flatten(self) -> tuple[torch.Tensor, ...]:
        return (
            self.obs.reshape(-1, self.obs.shape[-1]),
            self.actions.reshape(-1, self.actions.shape[-1]),
            self.logprobs.reshape(-1),
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
            self.values.reshape(-1),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean shared-policy PPO baseline on HP-only self-play")
    parser.add_argument("--run-name", type=str, default="selfplay_hp_ppo")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--match-duration", type=float, default=15.0)
    parser.add_argument("--control-frequency", type=int, default=20)
    parser.add_argument("--initial-distance", type=float, default=1.0)
    parser.add_argument("--non-fall-mode", action="store_true")
    parser.add_argument("--non-fall-pitch-limit-deg", type=float, default=15.0)
    parser.add_argument("--non-fall-roll-limit-deg", type=float, default=10.0)
    parser.add_argument("--damage-scale", type=float, default=100.0)
    parser.add_argument("--damage-reward-scale", type=float, default=1.0)
    parser.add_argument("--damage-penalty-scale", type=float, default=0.0)
    parser.add_argument("--win-bonus", type=float, default=0.0)
    parser.add_argument("--lose-penalty", type=float, default=0.0)
    parser.add_argument("--approach-reward-weight", type=float, default=0.0, help="Reward weight for approaching opponent")
    parser.add_argument("--action-diversity-weight", type=float, default=0.0, help="Reward weight for action diversity (arm movement)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-updates", type=int, default=10)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return Path(__file__).resolve().parent / "runs" / args.run_name


def save_run_config(args: argparse.Namespace, run_dir: Path) -> None:
    reward_config = SelfPlayHPRewardConfig(
        damage_reward_scale=args.damage_reward_scale,
        damage_penalty_scale=args.damage_penalty_scale,
        win_bonus=args.win_bonus,
        lose_penalty=args.lose_penalty,
        approach_reward_weight=args.approach_reward_weight,
        action_diversity_weight=args.action_diversity_weight,
    )
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )
    config = {
        "experiment": "shared_policy_selfplay_hp",
        "seed": args.seed,
        "device": args.device,
        "match_duration": args.match_duration,
        "control_frequency": args.control_frequency,
        "initial_distance": args.initial_distance,
        "non_fall_mode": args.non_fall_mode,
        "non_fall_pitch_limit_deg": args.non_fall_pitch_limit_deg,
        "non_fall_roll_limit_deg": args.non_fall_roll_limit_deg,
        "damage_scale": args.damage_scale,
        "reward_config": reward_config.to_dict(),
        "ppo_config": asdict(ppo_config),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_env(args: argparse.Namespace) -> SharedPolicySelfPlayHPEnv:
    reward_config = SelfPlayHPRewardConfig(
        damage_reward_scale=args.damage_reward_scale,
        damage_penalty_scale=args.damage_penalty_scale,
        win_bonus=args.win_bonus,
        lose_penalty=args.lose_penalty,
        approach_reward_weight=args.approach_reward_weight,
        action_diversity_weight=args.action_diversity_weight,
    )
    return SharedPolicySelfPlayHPEnv(
        render_mode=None,
        match_duration=args.match_duration,
        control_frequency=args.control_frequency,
        initial_distance=args.initial_distance,
        non_fall_mode=args.non_fall_mode,
        non_fall_pitch_limit_deg=args.non_fall_pitch_limit_deg,
        non_fall_roll_limit_deg=args.non_fall_roll_limit_deg,
        damage_scale=args.damage_scale,
        reward_config=reward_config,
    )


def obs_dict_to_tensor(obs_dict: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([obs_dict["robot_a"], obs_dict["robot_b"]], axis=0),
        dtype=torch.float32,
        device=device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, run_dir)

    env = build_env(args)
    obs, info = env.reset(seed=args.seed)
    obs_tensor = obs_dict_to_tensor(obs, device)
    obs_dim = obs_tensor.shape[-1]
    action_dim = int(env.action_space.shape[0])
    agent_count = 2

    agent = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    transitions_per_env_step = agent_count
    num_updates = max(1, args.total_timesteps // (args.rollout_steps * transitions_per_env_step))

    print("=" * 72)
    print("Shared-policy HP self-play PPO")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Non-fall mode: {args.non_fall_mode}")
    print(f"Pitch/Roll limits: +/-{args.non_fall_pitch_limit_deg} / +/-{args.non_fall_roll_limit_deg}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Num updates: {num_updates}")
    print("=" * 72)

    global_step = 0
    completed_episodes = 0
    episode_returns = np.zeros(agent_count, dtype=np.float32)
    episode_lengths = np.zeros(agent_count, dtype=np.int32)

    for update in range(1, num_updates + 1):
        buffer = RolloutBuffer(args.rollout_steps, agent_count, obs_dim, action_dim, device)

        for step in range(args.rollout_steps):
            global_step += transitions_per_env_step
            buffer.obs[step] = obs_tensor
            with torch.no_grad():
                sampled_action, logprob, _, value = agent.get_action_and_value(obs_tensor)
            clipped_action = torch.clamp(sampled_action, -1.0, 1.0)
            buffer.actions[step] = clipped_action
            buffer.logprobs[step] = logprob
            buffer.values[step] = value

            next_obs, reward_dict, terminated, truncated, info = env.step(
                clipped_action[0].cpu().numpy(),
                clipped_action[1].cpu().numpy(),
            )
            done = bool(terminated or truncated)
            reward_array = np.array([reward_dict["robot_a"], reward_dict["robot_b"]], dtype=np.float32)
            buffer.rewards[step] = torch.as_tensor(reward_array, dtype=torch.float32, device=device)
            buffer.dones[step] = torch.full((agent_count,), float(done), dtype=torch.float32, device=device)

            episode_returns += reward_array
            episode_lengths += 1

            if done:
                completed_episodes += 1
                print(
                    f"Update {update:04d} | Episode {completed_episodes:04d} | "
                    f"steps={int(episode_lengths.max())} | returns={episode_returns.tolist()} | "
                    f"scores={info['scores']} | winner={info.get('winner')} | reason={info.get('end_reason')}"
                )
                next_obs, info = env.reset()
                episode_returns[:] = 0.0
                episode_lengths[:] = 0

            obs_tensor = obs_dict_to_tensor(next_obs, device)

        with torch.no_grad():
            next_value = agent.get_value(obs_tensor)
        last_gae = torch.zeros(agent_count, dtype=torch.float32, device=device)
        next_non_terminal = 1.0 - buffer.dones[-1]
        for step in reversed(range(args.rollout_steps)):
            if step == args.rollout_steps - 1:
                next_values = next_value
                non_terminal = next_non_terminal
            else:
                next_values = buffer.values[step + 1]
                non_terminal = 1.0 - buffer.dones[step + 1]
            delta = buffer.rewards[step] + args.gamma * next_values * non_terminal - buffer.values[step]
            last_gae = delta + args.gamma * args.gae_lambda * non_terminal * last_gae
            buffer.advantages[step] = last_gae
        buffer.returns = buffer.advantages + buffer.values

        b_obs, b_actions, b_logprobs, b_advantages, b_returns, _ = buffer.flatten()
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)
        batch_size = b_obs.shape[0]
        minibatch_size = min(args.minibatch_size, batch_size)
        batch_indices = np.arange(batch_size)

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_idx = batch_indices[start:end]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[minibatch_idx],
                    b_actions[minibatch_idx],
                )
                logratio = new_logprob - b_logprobs[minibatch_idx]
                ratio = logratio.exp()

                mb_advantages = b_advantages[minibatch_idx]
                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_loss = 0.5 * torch.mean((new_value - b_returns[minibatch_idx]) ** 2)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        checkpoint = {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "global_step": global_step,
            "update": update,
        }
        torch.save(checkpoint, run_dir / "latest.pt")
        if update % max(1, args.save_every_updates) == 0:
            torch.save(checkpoint, run_dir / f"checkpoint_{update:04d}.pt")

    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "global_step": global_step,
            "update": num_updates,
        },
        run_dir / "final.pt",
    )
    env.close()
    print(f"Saved final checkpoint to: {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
