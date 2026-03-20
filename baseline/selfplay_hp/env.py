from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from gymnasium import spaces

from ...envs.combat_gym import CombatGymEnv
from ..sb3.normalization import ObservationNormalizer
from ..sb3.rewards import FIGHT_REWARD_CONFIG
from ..sb3.selfplay_env import configure_base_env_for_fight


@dataclass
class SelfPlayHPRewardConfig:
    damage_reward_scale: float = 1.0
    damage_penalty_scale: float = 1.0
    win_bonus: float = 0.0
    lose_penalty: float = 0.0

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
    ) -> tuple[dict[str, float], dict[str, float]]:
        reward_breakdown = {}
        rewards = {}
        winner = info.get("winner")
        for robot_id, opponent_id in (("robot_a", "robot_b"), ("robot_b", "robot_a")):
            damage_inflicted = float(max(0.0, previous_scores[opponent_id] - current_scores[opponent_id]))
            damage_taken = float(max(0.0, previous_scores[robot_id] - current_scores[robot_id]))
            reward = (
                self.reward_config.damage_reward_scale * damage_inflicted
                - self.reward_config.damage_penalty_scale * damage_taken
            )
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
                "terminal_bonus": terminal_bonus,
                "terminal_penalty": terminal_penalty,
            }
        return rewards, reward_breakdown

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        configure_base_env_for_fight(self.base_env)
        obs = self.base_env._get_obs()
        info = self.base_env._build_info()
        self._previous_scores = dict(info["scores"])
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
        rewards, reward_breakdown = self._compute_rewards(self._previous_scores, info["scores"], done, info)
        self._previous_scores = dict(info["scores"])
        updated_info = dict(info)
        updated_info["hp_rewards"] = reward_breakdown
        updated_info["applied_actions"] = {
            "robot_a": action_dict["robot_a"].copy(),
            "robot_b": action_dict["robot_b"].copy(),
        }
        return self._normalize_obs(obs), rewards, terminated, truncated, updated_info

    def close(self) -> None:
        self.base_env.close()
