from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.distributions import Normal

from combatbench.baseline.mujoco21dof_nonfall.reward import (
    DistanceStageRewardConfig,
    compute_distance_stage_curriculum_returns,
)


ROBOT_KEYS = ("robot_a", "robot_b")


@dataclass(frozen=True)
class GRPOModelConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256)
    log_std_init: float = -0.5


@dataclass(frozen=True)
class GRPOActionPenaltyConfig:
    action_magnitude_coef: float = 1.0
    action_delta_coef: float = 1.0


class GRPOActor(nn.Module):
    def __init__(self, config: GRPOModelConfig):
        super().__init__()
        self.config = config
        layers: List[nn.Module] = []
        input_dim = config.obs_dim
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, config.action_dim))
        self.policy_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), float(config.log_std_init), dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        pre_tanh_action = dist.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = self.log_prob(obs, pre_tanh_action)
        return action, pre_tanh_action, log_prob

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        dist = self.distribution(obs)
        pre_tanh_action = dist.mean if deterministic else dist.sample()
        return torch.tanh(pre_tanh_action)

    def log_prob(self, obs: torch.Tensor, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        squashed_action = torch.tanh(pre_tanh_action)
        correction = torch.log(torch.clamp(1.0 - squashed_action.pow(2), min=1e-6))
        return torch.sum(dist.log_prob(pre_tanh_action) - correction, dim=-1)

    def evaluate_actions(self, obs: torch.Tensor, pre_tanh_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = self.log_prob(obs, pre_tanh_action)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return log_prob, entropy


class GRPORolloutCollector:
    def __init__(
        self,
        vec_env: VecEnv,
        *,
        curriculum_stage: str = "attack",
        distance_stage_reward_config: Optional[DistanceStageRewardConfig] = None,
        self_play_mode: bool = False,
    ):
        self.vec_env = vec_env
        self.curriculum_stage = curriculum_stage
        self.distance_stage_reward_config = distance_stage_reward_config
        self.self_play_mode = bool(self_play_mode)
        self.current_obs: Optional[np.ndarray] = None
        self.partial_episodes: List[Dict[str, List[Any]]] = []
        self.reset()

    def reset(self) -> None:
        self.current_obs = np.asarray(self.vec_env.reset(), dtype=np.float32)
        if self.self_play_mode:
            self.partial_episodes = [
                {robot_key: self._new_partial_episode() for robot_key in ROBOT_KEYS}
                for _ in range(self.vec_env.num_envs)
            ]
        else:
            self.partial_episodes = [self._new_partial_episode() for _ in range(self.vec_env.num_envs)]

    def _new_partial_episode(self) -> Dict[str, List[Any]]:
        return {
            "obs": [],
            "actions": [],
            "pre_tanh_actions": [],
            "log_probs": [],
            "step_rewards": [],
        }

    def collect(self, actor: GRPOActor, device: torch.device, target_episodes: int, group_size: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        if target_episodes <= 0:
            raise ValueError("target_episodes must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.current_obs is None:
            self.reset()

        completed_episodes: List[Dict[str, Any]] = []
        total_env_steps = 0

        while len(completed_episodes) < target_episodes:
            obs_batch = np.asarray(self.current_obs, dtype=np.float32)
            if self.self_play_mode:
                flat_obs_batch = obs_batch.reshape(self.vec_env.num_envs * len(ROBOT_KEYS), -1)
                obs_tensor = torch.as_tensor(flat_obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_tensor, pre_tanh_tensor, log_prob_tensor = actor.sample_action(obs_tensor)
                action_dim = int(action_tensor.shape[-1])
                actions = action_tensor.detach().cpu().numpy().astype(np.float32).reshape(
                    self.vec_env.num_envs,
                    len(ROBOT_KEYS),
                    action_dim,
                )
                pre_tanh_actions = pre_tanh_tensor.detach().cpu().numpy().astype(np.float32).reshape(
                    self.vec_env.num_envs,
                    len(ROBOT_KEYS),
                    action_dim,
                )
                log_probs = log_prob_tensor.detach().cpu().numpy().astype(np.float32).reshape(
                    self.vec_env.num_envs,
                    len(ROBOT_KEYS),
                )
            else:
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_tensor, pre_tanh_tensor, log_prob_tensor = actor.sample_action(obs_tensor)
                actions = action_tensor.detach().cpu().numpy().astype(np.float32)
                pre_tanh_actions = pre_tanh_tensor.detach().cpu().numpy().astype(np.float32)
                log_probs = log_prob_tensor.detach().cpu().numpy().astype(np.float32)

            next_obs, rewards, dones, infos = self.vec_env.step(actions)
            self.current_obs = np.asarray(next_obs, dtype=np.float32)
            total_env_steps += int(self.vec_env.num_envs)

            for env_idx in range(self.vec_env.num_envs):
                info = infos[env_idx]
                if self.self_play_mode:
                    view_infos = info.get("self_play_views", {})
                    for robot_idx, robot_key in enumerate(ROBOT_KEYS):
                        partial_episode = self.partial_episodes[env_idx][robot_key]
                        partial_episode["obs"].append(obs_batch[env_idx, robot_idx].copy())
                        partial_episode["actions"].append(actions[env_idx, robot_idx].copy())
                        partial_episode["pre_tanh_actions"].append(pre_tanh_actions[env_idx, robot_idx].copy())
                        partial_episode["log_probs"].append(float(log_probs[env_idx, robot_idx]))
                        partial_episode["step_rewards"].append(extract_step_reward(view_infos.get(robot_key, {}), 0.0))

                    if not dones[env_idx]:
                        continue

                    for robot_key in ROBOT_KEYS:
                        partial_episode = self.partial_episodes[env_idx][robot_key]
                        robot_info = view_infos.get(robot_key, {})
                        step_rewards = np.asarray(partial_episode["step_rewards"], dtype=np.float32)
                        completed_episodes.append(
                            {
                                "robot_key": robot_key,
                                "obs": np.asarray(partial_episode["obs"], dtype=np.float32),
                                "actions": np.asarray(partial_episode["actions"], dtype=np.float32),
                                "pre_tanh_actions": np.asarray(partial_episode["pre_tanh_actions"], dtype=np.float32),
                                "log_probs": np.asarray(partial_episode["log_probs"], dtype=np.float32),
                                "step_rewards": step_rewards,
                                "return": float(np.sum(step_rewards)),
                                "length": int(step_rewards.shape[0]),
                                "episode_clamp_count": float(robot_info.get("episode_stats", {}).get("clamp_count", 0.0)),
                                "episode_damage_dealt": float(robot_info.get("episode_stats", {}).get("damage_dealt", 0.0)),
                                "episode_min_horizontal_distance": float(
                                    robot_info.get("episode_stats", {}).get(
                                        "min_horizontal_distance",
                                        robot_info.get("attacker_metrics", {}).get("horizontal_distance", 0.0),
                                    )
                                ),
                                "final_distance": float(robot_info.get("attacker_metrics", {}).get("horizontal_distance", 0.0)),
                            }
                        )
                    self.partial_episodes[env_idx] = {
                        robot_key: self._new_partial_episode()
                        for robot_key in ROBOT_KEYS
                    }
                    continue

                partial_episode = self.partial_episodes[env_idx]
                partial_episode["obs"].append(obs_batch[env_idx].copy())
                partial_episode["actions"].append(actions[env_idx].copy())
                partial_episode["pre_tanh_actions"].append(pre_tanh_actions[env_idx].copy())
                partial_episode["log_probs"].append(float(log_probs[env_idx]))
                partial_episode["step_rewards"].append(extract_step_reward(info, rewards[env_idx]))

                if not dones[env_idx]:
                    continue

                step_rewards = np.asarray(partial_episode["step_rewards"], dtype=np.float32)
                completed_episodes.append(
                    {
                        "robot_key": "robot_a",
                        "obs": np.asarray(partial_episode["obs"], dtype=np.float32),
                        "actions": np.asarray(partial_episode["actions"], dtype=np.float32),
                        "pre_tanh_actions": np.asarray(partial_episode["pre_tanh_actions"], dtype=np.float32),
                        "log_probs": np.asarray(partial_episode["log_probs"], dtype=np.float32),
                        "step_rewards": step_rewards,
                        "return": float(np.sum(step_rewards)),
                        "length": int(step_rewards.shape[0]),
                        "episode_clamp_count": float(info.get("episode_stats", {}).get("clamp_count", 0.0)),
                        "episode_damage_dealt": float(info.get("episode_stats", {}).get("damage_dealt", 0.0)),
                        "episode_min_horizontal_distance": float(
                            info.get("episode_stats", {}).get(
                                "min_horizontal_distance",
                                info.get("attacker_metrics", {}).get("horizontal_distance", 0.0),
                            )
                        ),
                        "final_distance": float(info.get("attacker_metrics", {}).get("horizontal_distance", 0.0)),
                    }
                )
                self.partial_episodes[env_idx] = self._new_partial_episode()

        usable_episode_count = len(completed_episodes) if group_size == 1 else (len(completed_episodes) // group_size) * group_size
        if usable_episode_count <= 0:
            raise RuntimeError("No complete GRPO groups collected. Increase target_episodes or reduce group_size.")
        completed_episodes = completed_episodes[:usable_episode_count]

        if (
            self.curriculum_stage == "distance_stage1"
            and self.distance_stage_reward_config is not None
            and self.distance_stage_reward_config.reward_mode == "episode_curriculum"
        ):
            curriculum_metrics = [
                {
                    "episode_clamp_count": episode["episode_clamp_count"],
                    "episode_damage_dealt": episode["episode_damage_dealt"],
                    "episode_min_horizontal_distance": episode["episode_min_horizontal_distance"],
                }
                for episode in completed_episodes
            ]
            curriculum_rewards, _, attack_enabled = compute_distance_stage_curriculum_returns(
                curriculum_metrics,
                self.distance_stage_reward_config,
            )
            for episode, curriculum_reward in zip(completed_episodes, curriculum_rewards):
                episode["return"] = float(curriculum_reward)
            curriculum_attack_enabled = float(1.0 if attack_enabled else 0.0)
        else:
            curriculum_attack_enabled = 0.0

        episode_returns = np.asarray([episode["return"] for episode in completed_episodes], dtype=np.float32)
        episode_lengths = np.asarray([episode["length"] for episode in completed_episodes], dtype=np.float32)
        episode_clamp_counts = np.asarray([episode["episode_clamp_count"] for episode in completed_episodes], dtype=np.float32)
        episode_damage_dealt = np.asarray([episode["episode_damage_dealt"] for episode in completed_episodes], dtype=np.float32)
        episode_min_horizontal_distances = np.asarray(
            [episode["episode_min_horizontal_distance"] for episode in completed_episodes],
            dtype=np.float32,
        )
        final_distances = np.asarray([episode["final_distance"] for episode in completed_episodes], dtype=np.float32)
        episode_advantages = compute_group_advantages(episode_returns, group_size)

        obs_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []
        prev_action_list: List[np.ndarray] = []
        pre_tanh_action_list: List[np.ndarray] = []
        old_log_prob_list: List[np.ndarray] = []
        step_advantage_list: List[np.ndarray] = []

        for episode, episode_advantage in zip(completed_episodes, episode_advantages):
            episode_length = int(episode["length"])
            episode_actions = episode["actions"]
            prev_episode_actions = np.zeros_like(episode_actions, dtype=np.float32)
            if episode_length > 1:
                prev_episode_actions[1:] = episode_actions[:-1]
            obs_list.append(episode["obs"])
            action_list.append(episode_actions)
            prev_action_list.append(prev_episode_actions)
            pre_tanh_action_list.append(episode["pre_tanh_actions"])
            old_log_prob_list.append(episode["log_probs"])
            step_advantage_list.append(np.full((episode_length,), float(episode_advantage), dtype=np.float32))

        batch = {
            "obs": np.concatenate(obs_list, axis=0).astype(np.float32),
            "actions": np.concatenate(action_list, axis=0).astype(np.float32),
            "prev_actions": np.concatenate(prev_action_list, axis=0).astype(np.float32),
            "pre_tanh_actions": np.concatenate(pre_tanh_action_list, axis=0).astype(np.float32),
            "old_log_probs": np.concatenate(old_log_prob_list, axis=0).astype(np.float32),
            "advantages": np.concatenate(step_advantage_list, axis=0).astype(np.float32),
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_clamp_counts": episode_clamp_counts,
            "episode_damage_dealt": episode_damage_dealt,
            "episode_min_horizontal_distances": episode_min_horizontal_distances,
            "final_distances": final_distances,
        }
        mean_action_magnitude = float(np.mean(np.square(batch["actions"])))
        mean_action_delta = float(np.mean(np.square(batch["actions"] - batch["prev_actions"])))
        stats = {
            "env_steps": float(total_env_steps),
            "episodes_collected": float(len(completed_episodes)),
            "samples_collected": float(batch["obs"].shape[0]),
            "mean_episode_return": float(np.mean(episode_returns)),
            "std_episode_return": float(np.std(episode_returns)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_episode_clamp_count": float(np.mean(episode_clamp_counts)),
            "mean_episode_damage_dealt": float(np.mean(episode_damage_dealt)),
            "mean_episode_min_horizontal_distance": float(np.mean(episode_min_horizontal_distances)),
            "mean_final_distance": float(np.mean(final_distances)),
            "mean_action_magnitude": mean_action_magnitude,
            "mean_action_delta": mean_action_delta,
            "mean_group_advantage": float(np.mean(episode_advantages)),
            "std_group_advantage": float(np.std(episode_advantages)),
            "curriculum_attack_enabled": curriculum_attack_enabled,
        }
        return batch, stats


def compute_group_advantages(episode_returns: np.ndarray, group_size: int) -> np.ndarray:
    if episode_returns.ndim != 1:
        raise ValueError("episode_returns must be a 1D array")
    if group_size <= 1:
        std = float(np.std(episode_returns))
        return (episode_returns - float(np.mean(episode_returns))) / (std + 1e-8)
    if episode_returns.shape[0] % group_size != 0:
        raise ValueError("episode_returns length must be divisible by group_size")

    advantages = np.zeros_like(episode_returns, dtype=np.float32)
    for start_idx in range(0, episode_returns.shape[0], group_size):
        group_returns = episode_returns[start_idx:start_idx + group_size]
        group_mean = float(np.mean(group_returns))
        group_std = float(np.std(group_returns))
        advantages[start_idx:start_idx + group_size] = (group_returns - group_mean) / (group_std + 1e-8)
    return advantages


def extract_step_reward(info: Dict[str, Any], fallback_reward: float) -> float:
    reward_terms = info.get("reward_terms")
    if isinstance(reward_terms, dict) and reward_terms:
        return float(sum(float(value) for value in reward_terms.values()))
    return float(fallback_reward)


def optimize_grpo(
    actor: GRPOActor,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    device: torch.device,
    minibatch_size: int,
    update_epochs: int,
    clip_range: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: Optional[float] = None,
    action_penalty_config: Optional[GRPOActionPenaltyConfig] = None,
) -> Dict[str, float]:
    penalty_config = GRPOActionPenaltyConfig() if action_penalty_config is None else action_penalty_config
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    prev_actions = torch.as_tensor(batch["prev_actions"], dtype=torch.float32, device=device)
    pre_tanh_actions = torch.as_tensor(batch["pre_tanh_actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)

    num_samples = int(obs.shape[0])
    if num_samples <= 0:
        raise ValueError("GRPO batch is empty")

    effective_minibatch_size = max(1, min(int(minibatch_size), num_samples))
    clip_fraction_values: List[float] = []
    entropy_values: List[float] = []
    policy_loss_values: List[float] = []
    base_policy_loss_values: List[float] = []
    approx_kl_values: List[float] = []
    grad_norm_values: List[float] = []
    loss_multiplier_values: List[float] = []
    action_magnitude_values: List[float] = []
    action_delta_values: List[float] = []
    updates = 0
    early_stop = False

    for _ in range(int(update_epochs)):
        permutation = np.random.permutation(num_samples)
        for start_idx in range(0, num_samples, effective_minibatch_size):
            batch_indices = permutation[start_idx:start_idx + effective_minibatch_size]
            obs_mb = obs[batch_indices]
            actions_mb = actions[batch_indices]
            prev_actions_mb = prev_actions[batch_indices]
            pre_tanh_mb = pre_tanh_actions[batch_indices]
            old_log_prob_mb = old_log_probs[batch_indices]
            advantage_mb = advantages[batch_indices]

            new_log_prob_mb, entropy_mb = actor.evaluate_actions(obs_mb, pre_tanh_mb)
            log_ratio = new_log_prob_mb - old_log_prob_mb
            ratio = torch.exp(log_ratio)
            unclipped_objective = ratio * advantage_mb
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            clipped_objective = clipped_ratio * advantage_mb
            policy_loss_per_sample = -torch.minimum(unclipped_objective, clipped_objective)
            action_magnitude_mb = torch.mean(actions_mb.pow(2), dim=-1)
            action_delta_mb = torch.mean((actions_mb - prev_actions_mb).pow(2), dim=-1)
            action_magnitude_multiplier_mb = (
                1.0
                + max(0.0, float(penalty_config.action_magnitude_coef) - 1.0) * action_magnitude_mb
            )
            action_delta_multiplier_mb = (
                1.0
                + max(0.0, float(penalty_config.action_delta_coef) - 1.0) * action_delta_mb
            )
            loss_multiplier_mb = action_magnitude_multiplier_mb * action_delta_multiplier_mb
            policy_loss = torch.mean(policy_loss_per_sample)
            regularized_policy_loss = torch.mean(
                policy_loss_per_sample
                + torch.abs(policy_loss_per_sample.detach()) * (loss_multiplier_mb - 1.0)
            )
            entropy = torch.mean(entropy_mb)
            loss = regularized_policy_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            optimizer.step()

            approx_kl = torch.mean(old_log_prob_mb - new_log_prob_mb).item()
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item()

            clip_fraction_values.append(float(clip_fraction))
            entropy_values.append(float(entropy.item()))
            policy_loss_values.append(float(regularized_policy_loss.item()))
            base_policy_loss_values.append(float(policy_loss.item()))
            approx_kl_values.append(float(approx_kl))
            grad_norm_values.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))
            loss_multiplier_values.append(float(torch.mean(loss_multiplier_mb).item()))
            action_magnitude_values.append(float(torch.mean(action_magnitude_mb).item()))
            action_delta_values.append(float(torch.mean(action_delta_mb).item()))
            updates += 1

            if target_kl is not None and approx_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break

    return {
        "updates": float(updates),
        "policy_loss": float(np.mean(policy_loss_values)) if policy_loss_values else 0.0,
        "base_policy_loss": float(np.mean(base_policy_loss_values)) if base_policy_loss_values else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "approx_kl": float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
        "clip_fraction": float(np.mean(clip_fraction_values)) if clip_fraction_values else 0.0,
        "grad_norm": float(np.mean(grad_norm_values)) if grad_norm_values else 0.0,
        "mean_loss_multiplier": float(np.mean(loss_multiplier_values)) if loss_multiplier_values else 1.0,
        "mean_action_magnitude": float(np.mean(action_magnitude_values)) if action_magnitude_values else 0.0,
        "mean_action_delta": float(np.mean(action_delta_values)) if action_delta_values else 0.0,
        "early_stop": float(1.0 if early_stop else 0.0),
    }


def evaluate_grpo_actor(
    actor: GRPOActor,
    env: Any,
    device: torch.device,
    episodes: int,
    deterministic: bool = True,
    seed: int = 0,
    curriculum_stage: str = "attack",
    distance_stage_reward_config: Optional[DistanceStageRewardConfig] = None,
) -> Dict[str, float]:
    episode_returns: List[float] = []
    episode_lengths: List[float] = []
    clamp_counts: List[float] = []
    damage_dealt_values: List[float] = []
    min_horizontal_distances: List[float] = []
    final_distances: List[float] = []

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        final_distance = 0.0
        clamp_count = 0.0
        episode_damage_dealt = 0.0
        episode_min_horizontal_distance = 0.0

        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor = actor.act(obs_tensor, deterministic=deterministic)
            action = action_tensor.detach().cpu().numpy()[0].astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += extract_step_reward(info, reward)
            episode_length += 1
            final_distance = float(info.get("attacker_metrics", {}).get("horizontal_distance", final_distance))
            clamp_count = float(info.get("episode_stats", {}).get("clamp_count", clamp_count))
            episode_damage_dealt = float(info.get("episode_stats", {}).get("damage_dealt", episode_damage_dealt))
            episode_min_horizontal_distance = float(
                info.get("episode_stats", {}).get("min_horizontal_distance", final_distance)
            )

        episode_returns.append(float(episode_return))
        episode_lengths.append(float(episode_length))
        clamp_counts.append(float(clamp_count))
        damage_dealt_values.append(float(episode_damage_dealt))
        min_horizontal_distances.append(float(episode_min_horizontal_distance))
        final_distances.append(float(final_distance))

    if (
        curriculum_stage == "distance_stage1"
        and distance_stage_reward_config is not None
        and distance_stage_reward_config.reward_mode == "episode_curriculum"
    ):
        curriculum_metrics = [
            {
                "episode_clamp_count": clamp_count,
                "episode_damage_dealt": damage_dealt,
                "episode_min_horizontal_distance": min_distance,
            }
            for clamp_count, damage_dealt, min_distance in zip(
                clamp_counts,
                damage_dealt_values,
                min_horizontal_distances,
            )
        ]
        episode_returns, _, attack_enabled = compute_distance_stage_curriculum_returns(
            curriculum_metrics,
            distance_stage_reward_config,
        )
    else:
        attack_enabled = False

    returns_array = np.asarray(episode_returns, dtype=np.float32)
    lengths_array = np.asarray(episode_lengths, dtype=np.float32)
    clamp_array = np.asarray(clamp_counts, dtype=np.float32)
    damage_dealt_array = np.asarray(damage_dealt_values, dtype=np.float32)
    min_distance_array = np.asarray(min_horizontal_distances, dtype=np.float32)
    distance_array = np.asarray(final_distances, dtype=np.float32)
    return {
        "episode_returns": returns_array,
        "episode_lengths": lengths_array,
        "episode_clamp_counts": clamp_array,
        "episode_damage_dealt": damage_dealt_array,
        "episode_min_horizontal_distances": min_distance_array,
        "final_distances": distance_array,
        "mean_reward": float(np.mean(returns_array)),
        "std_reward": float(np.std(returns_array)),
        "mean_episode_length": float(np.mean(lengths_array)),
        "mean_episode_clamp_count": float(np.mean(clamp_array)),
        "mean_episode_damage_dealt": float(np.mean(damage_dealt_array)),
        "mean_episode_min_horizontal_distance": float(np.mean(min_distance_array)),
        "mean_final_distance": float(np.mean(distance_array)),
        "curriculum_attack_enabled": float(1.0 if attack_enabled else 0.0),
    }


def save_grpo_checkpoint(
    path: str | Path,
    actor: GRPOActor,
    optimizer: Optional[torch.optim.Optimizer],
    metadata: Dict[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "grpo",
        "model_config": {
            "obs_dim": int(actor.config.obs_dim),
            "action_dim": int(actor.config.action_dim),
            "hidden_sizes": list(actor.config.hidden_sizes),
            "log_std_init": float(actor.config.log_std_init),
        },
        "model_state_dict": actor.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def load_grpo_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Tuple[GRPOActor, Dict[str, Any]]:
    map_location = torch.device(device) if not isinstance(device, torch.device) else device
    checkpoint = torch.load(Path(path), map_location=map_location)
    model_config_dict = checkpoint["model_config"]
    model_config = GRPOModelConfig(
        obs_dim=int(model_config_dict["obs_dim"]),
        action_dim=int(model_config_dict["action_dim"]),
        hidden_sizes=tuple(int(hidden_size) for hidden_size in model_config_dict["hidden_sizes"]),
        log_std_init=float(model_config_dict["log_std_init"]),
    )
    actor = GRPOActor(model_config)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.to(map_location)
    actor.eval()
    return actor, checkpoint


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
