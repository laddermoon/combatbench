from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from combatbench.envs import CombatGymEnv

from .opponents import make_opponent_policy
from .reward import (
    AttackerRewardConfig,
    DistanceStageRewardConfig,
    compute_attacker_reward,
    compute_distance_stage_reward,
    zero_reward_terms,
)


RewardFn = Callable[[Dict[str, Any]], Tuple[float, Dict[str, float]]]


class SingleAgentAttackerEnv(gym.Env):
    metadata = CombatGymEnv.metadata

    def __init__(
        self,
        render_mode: Optional[str] = None,
        initial_distance: float = 2.0,
        control_frequency: int = 20,
        match_duration: float = 30.0,
        non_fall_mode: bool = True,
        non_fall_pitch_limit_deg: float = 5.0,
        non_fall_roll_limit_deg: float = 5.0,
        damage_scale: float = 100.0,
        opponent: Any = "standing",
        opponent_seed: Optional[int] = None,
        opponent_random_scale: float = 0.1,
        curriculum_stage: str = "attack",
        reward_fn: Optional[RewardFn] = None,
        reward_config: Optional[AttackerRewardConfig] = None,
        distance_stage_reward_config: Optional[DistanceStageRewardConfig] = None,
    ):
        super().__init__()
        self.base_env = CombatGymEnv(
            render_mode=render_mode,
            initial_distance=initial_distance,
            control_frequency=control_frequency,
            match_duration=match_duration,
            non_fall_mode=non_fall_mode,
            non_fall_pitch_limit_deg=non_fall_pitch_limit_deg,
            non_fall_roll_limit_deg=non_fall_roll_limit_deg,
            damage_scale=damage_scale,
        )
        self.action_space = self.base_env.action_space["robot_a"]
        self.observation_space = self.base_env.observation_space["robot_a_obs"]
        self.reward_fn = reward_fn
        self.curriculum_stage = str(curriculum_stage)
        self.reward_config = AttackerRewardConfig() if reward_config is None else reward_config
        self.distance_stage_reward_config = (
            DistanceStageRewardConfig() if distance_stage_reward_config is None else distance_stage_reward_config
        )
        self._opponent_spec = opponent
        self._opponent_seed = opponent_seed
        self._opponent_random_scale = float(opponent_random_scale)
        self.opponent_policy = make_opponent_policy(
            opponent,
            seed=opponent_seed,
            random_scale=opponent_random_scale,
        )
        self._last_full_obs: Optional[Dict[str, np.ndarray]] = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._last_agent_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._episode_reward = 0.0
        self._episode_damage_dealt = 0.0
        self._episode_damage_received = 0.0
        self._episode_hits_dealt = 0
        self._episode_hits_received = 0

    def _distance_target_error(self, horizontal_distance: float) -> float:
        return abs(horizontal_distance - self.distance_stage_reward_config.target_distance)

    def set_opponent(self, opponent: Any, *, seed: Optional[int] = None) -> None:
        self._opponent_spec = opponent
        self._opponent_seed = seed
        self.opponent_policy = make_opponent_policy(
            opponent,
            seed=seed,
            random_scale=self._opponent_random_scale,
        )

    def get_base_env(self) -> CombatGymEnv:
        return self.base_env

    def _coerce_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.action_space.shape),
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

    def _get_opponent_action(self) -> np.ndarray:
        if self._last_full_obs is None or self._last_info is None:
            return np.zeros(self.action_space.shape, dtype=np.float32)
        action = self.opponent_policy.act(self._last_full_obs["robot_b_obs"], self._last_info)
        return self._coerce_action(action)

    def _extract_metrics(
        self,
        info: Dict[str, Any],
        prev_info: Dict[str, Any],
        action: np.ndarray,
    ) -> Dict[str, float]:
        prev_scores = prev_info.get("scores", {})
        current_scores = info.get("scores", {})
        prev_relative_metrics = prev_info.get("relative_metrics", {}).get("robot_a", {})
        relative_metrics = info.get("relative_metrics", {}).get("robot_a", {})
        prev_robot_state = prev_info.get("robot_states", {}).get("robot_a", {})
        robot_state = info.get("robot_states", {}).get("robot_a", {})
        damage_dealt = max(0.0, float(prev_scores.get("robot_b", 0.0) - current_scores.get("robot_b", 0.0)))
        damage_received = max(0.0, float(prev_scores.get("robot_a", 0.0) - current_scores.get("robot_a", 0.0)))
        hit_records = info.get("hit_records", {})
        hits_dealt = float(len(hit_records.get("robot_b", [])))
        hits_received = float(len(hit_records.get("robot_a", [])))
        horizontal_distance = float(relative_metrics.get("horizontal_distance", 0.0))
        prev_horizontal_distance = float(prev_relative_metrics.get("horizontal_distance", horizontal_distance))
        facing_opponent = float(relative_metrics.get("facing_opponent", 0.0))
        prev_facing_opponent = float(prev_relative_metrics.get("facing_opponent", facing_opponent))
        uprightness = float(robot_state.get("uprightness", 1.0))
        prev_uprightness = float(prev_robot_state.get("uprightness", uprightness))
        winner = info.get("winner")
        metrics = {
            "damage_dealt": damage_dealt,
            "damage_received": damage_received,
            "distance": float(relative_metrics.get("distance", 0.0)),
            "horizontal_distance": horizontal_distance,
            "horizontal_distance_delta": prev_horizontal_distance - horizontal_distance,
            "distance_error": self._distance_target_error(horizontal_distance),
            "distance_error_delta": self._distance_target_error(prev_horizontal_distance) - self._distance_target_error(horizontal_distance),
            "facing_opponent": facing_opponent,
            "facing_delta": facing_opponent - prev_facing_opponent,
            "uprightness": uprightness,
            "uprightness_delta": uprightness - prev_uprightness,
            "hits_dealt": hits_dealt,
            "hits_received": hits_received,
            "action_magnitude": float(np.mean(np.abs(action))),
            "action_delta": float(np.mean(np.abs(action - self._last_agent_action))),
            "win": 1.0 if winner == "robot_a" else 0.0,
            "loss": 1.0 if winner == "robot_b" else 0.0,
        }
        return metrics

    def _compute_reward(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        if self.reward_fn is not None:
            reward, reward_terms = self.reward_fn(metrics)
            return float(reward), {key: float(value) for key, value in reward_terms.items()}
        if self.curriculum_stage == "distance_stage1":
            return compute_distance_stage_reward(metrics, self.distance_stage_reward_config)
        return compute_attacker_reward(metrics, self.reward_config)

    def _build_agent_info(
        self,
        info: Dict[str, Any],
        metrics: Dict[str, float],
        reward_terms: Dict[str, float],
    ) -> Dict[str, Any]:
        agent_info = dict(info)
        agent_info["attacker_metrics"] = metrics
        agent_info["reward_terms"] = reward_terms
        agent_info["episode_stats"] = {
            "reward": self._episode_reward,
            "damage_dealt": self._episode_damage_dealt,
            "damage_received": self._episode_damage_received,
            "hits_dealt": self._episode_hits_dealt,
            "hits_received": self._episode_hits_received,
        }
        return agent_info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        full_obs, info = self.base_env.reset(seed=seed, options=options)
        if hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset()
        self._last_full_obs = full_obs
        self._last_info = info
        self._last_agent_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._episode_reward = 0.0
        self._episode_damage_dealt = 0.0
        self._episode_damage_received = 0.0
        self._episode_hits_dealt = 0
        self._episode_hits_received = 0
        reset_info = self._build_agent_info(
            info,
            metrics={
                "damage_dealt": 0.0,
                "damage_received": 0.0,
                "distance": float(info.get("relative_metrics", {}).get("robot_a", {}).get("distance", 0.0)),
                "horizontal_distance": float(info.get("relative_metrics", {}).get("robot_a", {}).get("horizontal_distance", 0.0)),
                "horizontal_distance_delta": 0.0,
                "distance_error": self._distance_target_error(
                    float(info.get("relative_metrics", {}).get("robot_a", {}).get("horizontal_distance", 0.0))
                ),
                "distance_error_delta": 0.0,
                "facing_opponent": float(info.get("relative_metrics", {}).get("robot_a", {}).get("facing_opponent", 0.0)),
                "facing_delta": 0.0,
                "uprightness": float(info.get("robot_states", {}).get("robot_a", {}).get("uprightness", 1.0)),
                "uprightness_delta": 0.0,
                "hits_dealt": 0.0,
                "hits_received": 0.0,
                "action_magnitude": 0.0,
                "action_delta": 0.0,
                "win": 0.0,
                "loss": 0.0,
            },
            reward_terms=zero_reward_terms(),
        )
        return full_obs["robot_a_obs"], reset_info

    def step(self, action: np.ndarray):
        if self._last_full_obs is None or self._last_info is None:
            raise RuntimeError("Environment must be reset before calling step()")

        agent_action = self._coerce_action(action)
        opponent_action = self._get_opponent_action()
        prev_info = self._last_info
        full_obs, _, terminated, truncated, info = self.base_env.step(
            {
                "robot_a": agent_action,
                "robot_b": opponent_action,
            }
        )
        metrics = self._extract_metrics(info, prev_info, agent_action)
        reward, reward_terms = self._compute_reward(metrics)
        self._episode_reward += reward
        self._episode_damage_dealt += metrics["damage_dealt"]
        self._episode_damage_received += metrics["damage_received"]
        self._episode_hits_dealt += int(metrics["hits_dealt"])
        self._episode_hits_received += int(metrics["hits_received"])
        agent_info = self._build_agent_info(info, metrics, reward_terms)
        self._last_full_obs = full_obs
        self._last_info = info
        self._last_agent_action = agent_action
        return full_obs["robot_a_obs"], reward, terminated, truncated, agent_info

    def render(self):
        return self.base_env.render()

    def get_video_buffer(self):
        return self.base_env.get_video_buffer()

    def save_video(self, filepath: str, fps: Optional[int] = None) -> None:
        self.base_env.save_video(
            filepath,
            fps=self.base_env.video_sample_frequency if fps is None else fps,
        )

    def close(self) -> None:
        self.base_env.close()
