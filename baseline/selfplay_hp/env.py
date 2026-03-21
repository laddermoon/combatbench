from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from gymnasium import spaces

from ...envs.combat_gym import CombatGymEnv
from ...core.humanoid_robot import HumanoidRobot
from ..sb3.normalization import ObservationNormalizer
from ..sb3.rewards import FIGHT_REWARD_CONFIG
from ..sb3.selfplay_env import (
    STAND_PD_KD,
    STAND_PD_KP,
    build_stand_reset_pose,
)


COMBAT_ACTION_SCALE_MULTIPLIER = 5.0


def build_combat_action_scale() -> np.ndarray:
    """Build action scale for combat with larger multipliers for visible movement."""
    action_scale = np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)
    joint_scales = {
        "abdomen_z": 0.15,
        "abdomen_y": 0.20,
        "abdomen_x": 0.15,
        "hip_x_right": 0.15,
        "hip_z_right": 0.12,
        "hip_y_right": 0.25,
        "knee_right": 0.30,
        "ankle_y_right": 0.20,
        "ankle_x_right": 0.15,
        "hip_x_left": 0.15,
        "hip_z_left": 0.12,
        "hip_y_left": 0.25,
        "knee_left": 0.30,
        "ankle_y_left": 0.20,
        "ankle_x_left": 0.15,
        "shoulder1_right": 0.30,
        "shoulder2_right": 0.30,
        "elbow_right": 0.30,
        "shoulder1_left": 0.30,
        "shoulder2_left": 0.30,
        "elbow_left": 0.30,
    }
    joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(HumanoidRobot.CONTROLLED_JOINTS)}
    for joint_name, scale in joint_scales.items():
        action_scale[joint_name_to_index[joint_name]] = float(scale)
    return action_scale


def configure_combat_env(env: CombatGymEnv) -> None:
    """Configure environment for combat with larger action scales."""
    action_scale = build_combat_action_scale()
    stand_reset_pose = build_stand_reset_pose()
    env.set_robot_joint_positions(stand_reset_pose)
    env.set_controller_reference_positions(stand_reset_pose)
    env.set_controller_gains(kp=STAND_PD_KP, kd=STAND_PD_KD)
    env.set_controller_action_scale(
        {
            "robot_a": action_scale,
            "robot_b": action_scale,
        }
    )


@dataclass
class SelfPlayHPRewardConfig:
    damage_reward_scale: float = 1.0
    damage_penalty_scale: float = 0.0
    win_bonus: float = 0.0
    lose_penalty: float = 0.0
    approach_reward_weight: float = 0.0
    action_diversity_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SharedPolicySelfPlayHPEnv:
    def __init__(
        self,
        render_mode: str | None = None,
        match_duration: float = 15.0,
        control_frequency: int = 20,
        initial_distance: float = 1.0,
        non_fall_mode: bool = True,
        non_fall_pitch_limit_deg: float = 15.0,
        non_fall_roll_limit_deg: float = 10.0,
        damage_scale: float = 100.0,
        reward_config: SelfPlayHPRewardConfig | None = None,
        target_height: float = FIGHT_REWARD_CONFIG.target_height,
    ) -> None:
        self.reward_config = reward_config or SelfPlayHPRewardConfig()
        self.base_env = CombatGymEnv(
            render_mode=render_mode,
            match_duration=match_duration,
            control_frequency=control_frequency,
            initial_distance=initial_distance,
            non_fall_mode=non_fall_mode,
            non_fall_pitch_limit_deg=non_fall_pitch_limit_deg,
            non_fall_roll_limit_deg=non_fall_roll_limit_deg,
            damage_scale=damage_scale,
        )
        self.normalizer = ObservationNormalizer(target_height=target_height)
        self.observation_space = spaces.Dict(
            {
                "robot_a": self.normalizer.observation_space(),
                "robot_b": self.normalizer.observation_space(),
            }
        )
        self.action_space = self.base_env.action_space["robot_a"]
        self._previous_scores = {"robot_a": 100.0, "robot_b": 100.0}
        self._previous_horizontal_distance = None

    def _normalize_obs(self, obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized = {
            "robot_a": self.normalizer.normalize(obs_dict["robot_a_obs"]),
            "robot_b": self.normalizer.normalize(obs_dict["robot_b_obs"]),
        }
        for robot_id, obs in normalized.items():
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                normalized[robot_id] = np.zeros_like(obs, dtype=np.float32)
            else:
                normalized[robot_id] = obs.astype(np.float32)
        return normalized

    def _compute_rewards(
        self,
        previous_scores: dict[str, float],
        current_scores: dict[str, float],
        done: bool,
        info: dict,
        action_dict: dict[str, np.ndarray],
        previous_horizontal_distance: float | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        reward_breakdown = {}
        rewards = {}
        winner = info.get("winner")
        
        current_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        approach_progress = 0.0
        if previous_horizontal_distance is not None:
            approach_progress = float(previous_horizontal_distance - current_distance)
        
        for robot_id, opponent_id in (("robot_a", "robot_b"), ("robot_b", "robot_a")):
            damage_inflicted = float(max(0.0, previous_scores[opponent_id] - current_scores[opponent_id]))
            damage_taken = float(max(0.0, previous_scores[robot_id] - current_scores[robot_id]))
            
            hp_reward = (
                self.reward_config.damage_reward_scale * damage_inflicted
                - self.reward_config.damage_penalty_scale * damage_taken
            )
            
            approach_reward = float(self.reward_config.approach_reward_weight * approach_progress)
            
            action = action_dict[robot_id]
            action_magnitude = float(np.abs(action).mean())
            diversity_reward = float(self.reward_config.action_diversity_weight * action_magnitude)
            
            reward = hp_reward + approach_reward + diversity_reward
            
            terminal_bonus = 0.0
            terminal_penalty = 0.0
            if done:
                if winner == robot_id:
                    terminal_bonus = float(self.reward_config.win_bonus)
                    reward += terminal_bonus
                elif winner == opponent_id:
                    terminal_penalty = float(self.reward_config.lose_penalty)
                    reward -= terminal_penalty
            
            rewards[robot_id] = float(reward)
            reward_breakdown[robot_id] = {
                "damage_inflicted": damage_inflicted,
                "damage_taken": damage_taken,
                "approach_reward": approach_reward,
                "diversity_reward": diversity_reward,
                "terminal_bonus": terminal_bonus,
                "terminal_penalty": terminal_penalty,
            }
        return rewards, reward_breakdown

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        configure_combat_env(self.base_env)
        obs = self.base_env._get_obs()
        info = self.base_env._build_info()
        self._previous_scores = dict(info["scores"])
        self._previous_horizontal_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        return self._normalize_obs(obs), info

    def step(
        self,
        action_a: np.ndarray,
        action_b: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict]:
        action_dict = {
            "robot_a": np.clip(np.asarray(action_a, dtype=np.float32), -1.0, 1.0),
            "robot_b": np.clip(np.asarray(action_b, dtype=np.float32), -1.0, 1.0),
        }
        obs, _, terminated, truncated, info = self.base_env.step(action_dict)
        done = bool(terminated or truncated)
        rewards, reward_breakdown = self._compute_rewards(
            self._previous_scores,
            info["scores"],
            done,
            info,
            action_dict,
            self._previous_horizontal_distance,
        )
        self._previous_scores = dict(info["scores"])
        self._previous_horizontal_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        updated_info = dict(info)
        updated_info["hp_rewards"] = reward_breakdown
        updated_info["applied_actions"] = {
            "robot_a": action_dict["robot_a"].copy(),
            "robot_b": action_dict["robot_b"].copy(),
        }
        return self._normalize_obs(obs), rewards, terminated, truncated, updated_info

    def close(self) -> None:
        self.base_env.close()
