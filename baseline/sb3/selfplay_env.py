from __future__ import annotations

import numpy as np
import gymnasium as gym

from ...envs.combat_gym import CombatGymEnv
from .normalization import ObservationNormalizer
from .rewards import FIGHT_REWARD_CONFIG, RewardConfig, compute_shaped_rewards


class SymmetricSelfPlayEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        env: CombatGymEnv,
        reward_config: RewardConfig = FIGHT_REWARD_CONFIG,
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.reward_config = reward_config
        self.normalizer = normalizer or ObservationNormalizer(target_height=reward_config.target_height)
        self.observation_space = self.normalizer.observation_space()
        self.action_space = env.action_space["robot_a"]
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        self._previous_actions = {"robot_a": None, "robot_b": None}
        self._previous_scores = {"robot_a": 100.0, "robot_b": 100.0}

    @property
    def base_env(self) -> CombatGymEnv:
        return self.env

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        normalized_obs = self.normalizer.normalize(obs)
        if np.any(np.isnan(normalized_obs)) or np.any(np.isinf(normalized_obs)):
            return np.zeros_like(normalized_obs, dtype=np.float32)
        return normalized_obs.astype(np.float32)

    def _apply_fall_termination(self, terminated: bool, truncated: bool, info: dict) -> tuple[bool, bool, dict]:
        if terminated or truncated or not self.reward_config.terminate_on_fall:
            return terminated, truncated, info

        fallen_robots = []
        for robot_id, state in info["robot_states"].items():
            torso_height = float(state["torso_position"][2])
            uprightness = float(state["uprightness"])
            if torso_height < self.reward_config.min_height or uprightness < self.reward_config.min_uprightness:
                fallen_robots.append(robot_id)

        if not fallen_robots:
            return terminated, truncated, info

        updated_info = dict(info)
        updated_info["fallen_robots"] = fallen_robots
        if len(fallen_robots) == 1:
            loser = fallen_robots[0]
            winner = "robot_b" if loser == "robot_a" else "robot_a"
            updated_info["winner"] = winner
            updated_info["end_reason"] = f"{loser} fell below the stability threshold"
        else:
            updated_info["winner"] = "draw"
            updated_info["end_reason"] = "both robots fell below the stability threshold"
        return True, False, updated_info

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._previous_actions = {
            "robot_a": np.zeros(self.action_space.shape, dtype=np.float32),
            "robot_b": np.zeros(self.action_space.shape, dtype=np.float32),
        }
        self._previous_scores = dict(info["scores"])
        return self._normalize_obs(obs["robot_a_obs"]), info

    def step(self, action: np.ndarray):
        action_array = np.asarray(action, dtype=np.float32)
        action_array = np.clip(action_array, -1.0, 1.0)
        action_dict = {
            "robot_a": action_array,
            "robot_b": action_array.copy(),
        }
        previous_scores = dict(self._previous_scores)
        obs, _, terminated, truncated, info = self.env.step(action_dict)
        terminated, truncated, info = self._apply_fall_termination(terminated, truncated, info)
        shaped_rewards = compute_shaped_rewards(
            info=info,
            previous_scores=previous_scores,
            current_scores=info["scores"],
            action_dict=action_dict,
            previous_action_dict=self._previous_actions,
            config=self.reward_config,
        )
        reward = float(0.5 * (shaped_rewards["robot_a"] + shaped_rewards["robot_b"]))
        self._previous_actions = {
            "robot_a": action_dict["robot_a"].copy(),
            "robot_b": action_dict["robot_b"].copy(),
        }
        self._previous_scores = dict(info["scores"])
        updated_info = dict(info)
        updated_info["shaped_rewards"] = shaped_rewards
        updated_info["shared_policy_action"] = action_array.copy()
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        return self._normalize_obs(obs["robot_a_obs"]), reward, terminated, truncated, updated_info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()



def make_symmetric_selfplay_env(
    render_mode: str | None = None,
    match_duration: float = 15.0,
    control_frequency: int = 20,
    initial_distance: float = 2.0,
    reward_config: RewardConfig = FIGHT_REWARD_CONFIG,
) -> SymmetricSelfPlayEnv:
    base_env = CombatGymEnv(
        render_mode=render_mode,
        match_duration=match_duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
    )
    normalizer = ObservationNormalizer(target_height=reward_config.target_height)
    return SymmetricSelfPlayEnv(base_env, reward_config=reward_config, normalizer=normalizer)
