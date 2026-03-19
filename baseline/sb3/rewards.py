from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    name: str
    target_height: float = 1.282
    height_sigma: float = 0.18
    min_height: float = 0.9
    min_uprightness: float = 0.25
    target_distance: float = 1.35
    distance_sigma: float = 0.45
    survival_reward: float = 0.1
    height_weight: float = 0.35
    upright_weight: float = 0.3
    feet_contact_weight: float = 0.1
    facing_weight: float = 0.1
    distance_weight: float = 0.1
    damage_reward_weight: float = 0.75
    damage_penalty_weight: float = 0.75
    arena_penalty_weight: float = 0.08
    linear_velocity_penalty_weight: float = 0.02
    angular_velocity_penalty_weight: float = 0.01
    action_penalty_weight: float = 0.005
    action_change_penalty_weight: float = 0.01
    fall_penalty: float = 6.0
    win_bonus: float = 2.0
    lose_penalty: float = 2.0
    terminate_on_fall: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


STANDING_REWARD_CONFIG = RewardConfig(
    name="stand",
    target_height=1.282,
    height_sigma=0.14,
    min_height=0.98,
    min_uprightness=0.45,
    target_distance=1.5,
    distance_sigma=0.55,
    survival_reward=0.2,
    height_weight=0.45,
    upright_weight=0.28,
    feet_contact_weight=0.15,
    facing_weight=0.04,
    distance_weight=0.03,
    damage_reward_weight=0.1,
    damage_penalty_weight=0.2,
    arena_penalty_weight=0.04,
    linear_velocity_penalty_weight=0.01,
    angular_velocity_penalty_weight=0.01,
    action_penalty_weight=0.004,
    action_change_penalty_weight=0.008,
    fall_penalty=8.0,
    win_bonus=0.5,
    lose_penalty=0.5,
)

FIGHT_REWARD_CONFIG = RewardConfig(
    name="fight",
    target_height=1.24,
    height_sigma=0.2,
    min_height=0.85,
    min_uprightness=0.2,
    target_distance=1.25,
    distance_sigma=0.35,
    survival_reward=0.1,
    height_weight=0.2,
    upright_weight=0.18,
    feet_contact_weight=0.05,
    facing_weight=0.12,
    distance_weight=0.15,
    damage_reward_weight=1.2,
    damage_penalty_weight=1.0,
    arena_penalty_weight=0.1,
    linear_velocity_penalty_weight=0.015,
    angular_velocity_penalty_weight=0.008,
    action_penalty_weight=0.003,
    action_change_penalty_weight=0.008,
    fall_penalty=10.0,
    win_bonus=2.5,
    lose_penalty=2.5,
)


def resolve_reward_config(phase: str) -> RewardConfig:
    phase_name = phase.lower().strip()
    if phase_name == "stand":
        return STANDING_REWARD_CONFIG
    if phase_name == "fight":
        return FIGHT_REWARD_CONFIG
    raise ValueError(f"Unsupported phase: {phase}")


def _gaussian_score(value: float, target: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-0.5 * ((value - target) / sigma) ** 2))


def _feet_contact_score(feet_contact: dict) -> float:
    count = int(bool(feet_contact.get("left_foot"))) + int(bool(feet_contact.get("right_foot")))
    if count == 2:
        return 1.0
    if count == 1:
        return 0.5
    return 0.0


def compute_shaped_rewards(
    info: dict,
    previous_scores: dict,
    current_scores: dict,
    action_dict: dict,
    previous_action_dict: dict,
    config: RewardConfig,
) -> dict:
    rewards = {}

    for robot_id, opponent_id in (("robot_a", "robot_b"), ("robot_b", "robot_a")):
        state = info["robot_states"][robot_id]
        relative_metrics = info["relative_metrics"][robot_id]
        torso_height = float(state["torso_position"][2])
        uprightness = float(np.clip(state["uprightness"], -1.0, 1.0))
        upright_reward = float(np.clip((uprightness + 1.0) * 0.5, 0.0, 1.0))
        height_reward = _gaussian_score(torso_height, config.target_height, config.height_sigma)
        feet_contact_reward = _feet_contact_score(state["feet_contact"])
        facing_reward = float(max(0.0, relative_metrics["facing_opponent"]))
        distance_reward = _gaussian_score(
            float(relative_metrics["horizontal_distance"]),
            config.target_distance,
            config.distance_sigma,
        )
        arena_penalty = float(max(0.0, np.linalg.norm(state["torso_position"][:2]) - 2.25))
        linear_speed = float(np.linalg.norm(state["linear_velocity"]))
        angular_speed = float(np.linalg.norm(state["angular_velocity"]))
        action_penalty = float(np.mean(np.square(action_dict[robot_id])))
        previous_action = previous_action_dict.get(robot_id)
        action_change_penalty = 0.0
        if previous_action is not None:
            action_change_penalty = float(np.mean(np.square(action_dict[robot_id] - previous_action)))

        damage_inflicted = float(max(0.0, previous_scores[opponent_id] - current_scores[opponent_id]))
        damage_taken = float(max(0.0, previous_scores[robot_id] - current_scores[robot_id]))

        reward = config.survival_reward
        reward += config.height_weight * height_reward
        reward += config.upright_weight * upright_reward
        reward += config.feet_contact_weight * feet_contact_reward
        reward += config.facing_weight * facing_reward
        reward += config.distance_weight * distance_reward
        reward += config.damage_reward_weight * damage_inflicted
        reward -= config.damage_penalty_weight * damage_taken
        reward -= config.arena_penalty_weight * arena_penalty
        reward -= config.linear_velocity_penalty_weight * linear_speed
        reward -= config.angular_velocity_penalty_weight * angular_speed
        reward -= config.action_penalty_weight * action_penalty
        reward -= config.action_change_penalty_weight * action_change_penalty

        if torso_height < config.min_height or uprightness < config.min_uprightness:
            reward -= config.fall_penalty

        winner = info.get("winner")
        if winner == robot_id:
            reward += config.win_bonus
        elif winner == opponent_id:
            reward -= config.lose_penalty

        rewards[robot_id] = float(reward)

    return rewards
