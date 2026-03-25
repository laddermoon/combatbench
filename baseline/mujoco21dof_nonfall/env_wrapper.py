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
        self._episode_clamp_count = 0
        self._episode_min_horizontal_distance = 0.0

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

    def _opponent_key(self, agent_key: str) -> str:
        return "robot_b" if agent_key == "robot_a" else "robot_a"

    def _get_opponent_action(self) -> np.ndarray:
        if self._last_full_obs is None or self._last_info is None:
            return np.zeros(self.action_space.shape, dtype=np.float32)
        action = self.opponent_policy.act(self._last_full_obs["robot_b_obs"], self._last_info)
        return self._coerce_action(action)

    def _extract_metrics_for_agent(
        self,
        info: Dict[str, Any],
        prev_info: Dict[str, Any],
        action: np.ndarray,
        *,
        agent_key: str,
        last_action: np.ndarray,
        episode_damage_dealt: float,
        episode_damage_received: float,
        episode_min_horizontal_distance: float,
    ) -> Dict[str, float]:
        opponent_key = self._opponent_key(agent_key)
        prev_scores = prev_info.get("scores", {})
        current_scores = info.get("scores", {})
        prev_relative_metrics = prev_info.get("relative_metrics", {}).get(agent_key, {})
        relative_metrics = info.get("relative_metrics", {}).get(agent_key, {})
        prev_robot_state = prev_info.get("robot_states", {}).get(agent_key, {})
        robot_state = info.get("robot_states", {}).get(agent_key, {})
        damage_dealt = max(0.0, float(prev_scores.get(opponent_key, 0.0) - current_scores.get(opponent_key, 0.0)))
        damage_received = max(0.0, float(prev_scores.get(agent_key, 0.0) - current_scores.get(agent_key, 0.0)))
        hit_records = info.get("hit_records", {})
        hits_dealt = float(len(hit_records.get(opponent_key, [])))
        hits_received = float(len(hit_records.get(agent_key, [])))
        horizontal_distance = float(relative_metrics.get("horizontal_distance", 0.0))
        prev_horizontal_distance = float(prev_relative_metrics.get("horizontal_distance", horizontal_distance))
        facing_opponent = float(relative_metrics.get("facing_opponent", 0.0))
        prev_facing_opponent = float(prev_relative_metrics.get("facing_opponent", facing_opponent))
        uprightness = float(robot_state.get("uprightness", 1.0))
        prev_uprightness = float(prev_robot_state.get("uprightness", uprightness))
        winner = info.get("winner")
        clamp_counts = info.get("non_fall_mode", {}).get("clamp_counts", {})
        current_step_clamp_counts = clamp_counts.get("current_step", {})
        episode_clamp_counts = clamp_counts.get("episode", {})
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
            "action_delta": float(np.mean(np.abs(action - last_action))),
            "clamp_count": float(current_step_clamp_counts.get(agent_key, 0.0)),
            "episode_clamp_count": float(episode_clamp_counts.get(agent_key, 0.0)),
            "episode_damage_dealt": episode_damage_dealt + damage_dealt,
            "episode_damage_received": episode_damage_received + damage_received,
            "episode_min_horizontal_distance": min(episode_min_horizontal_distance, horizontal_distance),
            "win": 1.0 if winner == agent_key else 0.0,
            "loss": 1.0 if winner == opponent_key else 0.0,
        }
        return metrics

    def _extract_metrics(
        self,
        info: Dict[str, Any],
        prev_info: Dict[str, Any],
        action: np.ndarray,
    ) -> Dict[str, float]:
        return self._extract_metrics_for_agent(
            info,
            prev_info,
            action,
            agent_key="robot_a",
            last_action=self._last_agent_action,
            episode_damage_dealt=self._episode_damage_dealt,
            episode_damage_received=self._episode_damage_received,
            episode_min_horizontal_distance=self._episode_min_horizontal_distance,
        )

    def _build_reset_metrics_for_agent(self, info: Dict[str, Any], *, agent_key: str) -> Dict[str, float]:
        relative_metrics = info.get("relative_metrics", {}).get(agent_key, {})
        robot_state = info.get("robot_states", {}).get(agent_key, {})
        horizontal_distance = float(relative_metrics.get("horizontal_distance", 0.0))
        return {
            "damage_dealt": 0.0,
            "damage_received": 0.0,
            "distance": float(relative_metrics.get("distance", 0.0)),
            "horizontal_distance": horizontal_distance,
            "horizontal_distance_delta": 0.0,
            "distance_error": self._distance_target_error(horizontal_distance),
            "distance_error_delta": 0.0,
            "facing_opponent": float(relative_metrics.get("facing_opponent", 0.0)),
            "facing_delta": 0.0,
            "uprightness": float(robot_state.get("uprightness", 1.0)),
            "uprightness_delta": 0.0,
            "hits_dealt": 0.0,
            "hits_received": 0.0,
            "action_magnitude": 0.0,
            "action_delta": 0.0,
            "clamp_count": 0.0,
            "episode_clamp_count": 0.0,
            "episode_damage_dealt": 0.0,
            "episode_damage_received": 0.0,
            "episode_min_horizontal_distance": horizontal_distance,
            "win": 0.0,
            "loss": 0.0,
        }

    def _build_combined_reset_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_agent_info(
            info,
            metrics=self._build_reset_metrics_for_agent(info, agent_key="robot_a"),
            reward_terms=zero_reward_terms(),
        )

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
            "clamp_count": self._episode_clamp_count,
            "min_horizontal_distance": self._episode_min_horizontal_distance,
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
        self._episode_clamp_count = 0
        self._episode_min_horizontal_distance = float(info.get("relative_metrics", {}).get("robot_a", {}).get("horizontal_distance", 0.0))
        reset_info = self._build_combined_reset_info(info)
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
        if (
            self.curriculum_stage == "distance_stage1"
            and self.distance_stage_reward_config.reward_mode in {"episode_uniform", "episode_curriculum"}
            and not (terminated or truncated)
        ):
            reward = 0.0
            reward_terms = zero_reward_terms()
        self._episode_reward += reward
        self._episode_damage_dealt += metrics["damage_dealt"]
        self._episode_damage_received += metrics["damage_received"]
        self._episode_hits_dealt += int(metrics["hits_dealt"])
        self._episode_hits_received += int(metrics["hits_received"])
        self._episode_clamp_count += int(metrics["clamp_count"])
        self._episode_min_horizontal_distance = min(self._episode_min_horizontal_distance, float(metrics["horizontal_distance"]))
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


class SelfPlaySymmetricEnv(SingleAgentAttackerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        single_action_space = self.base_env.action_space["robot_a"]
        single_observation_space = self.base_env.observation_space["robot_a_obs"]
        self.action_space = gym.spaces.Box(
            low=np.stack([single_action_space.low, single_action_space.low], axis=0),
            high=np.stack([single_action_space.high, single_action_space.high], axis=0),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.stack([single_observation_space.low, single_observation_space.low], axis=0),
            high=np.stack([single_observation_space.high, single_observation_space.high], axis=0),
            dtype=np.float32,
        )
        self._last_actions = {
            "robot_a": np.zeros(single_action_space.shape, dtype=np.float32),
            "robot_b": np.zeros(single_action_space.shape, dtype=np.float32),
        }
        self._episode_rewards = {"robot_a": 0.0, "robot_b": 0.0}
        self._episode_damage_dealt_by_robot = {"robot_a": 0.0, "robot_b": 0.0}
        self._episode_damage_received_by_robot = {"robot_a": 0.0, "robot_b": 0.0}
        self._episode_hits_dealt_by_robot = {"robot_a": 0, "robot_b": 0}
        self._episode_hits_received_by_robot = {"robot_a": 0, "robot_b": 0}
        self._episode_clamp_count_by_robot = {"robot_a": 0, "robot_b": 0}
        self._episode_min_horizontal_distance_by_robot = {"robot_a": 0.0, "robot_b": 0.0}

    def _stack_full_obs(self, full_obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([full_obs["robot_a_obs"], full_obs["robot_b_obs"]], axis=0).astype(np.float32)

    def _build_agent_episode_info(
        self,
        info: Dict[str, Any],
        *,
        agent_key: str,
        metrics: Dict[str, float],
        reward_terms: Dict[str, float],
    ) -> Dict[str, Any]:
        agent_info = dict(info)
        agent_info["attacker_metrics"] = metrics
        agent_info["reward_terms"] = reward_terms
        agent_info["episode_stats"] = {
            "reward": self._episode_rewards[agent_key],
            "damage_dealt": self._episode_damage_dealt_by_robot[agent_key],
            "damage_received": self._episode_damage_received_by_robot[agent_key],
            "hits_dealt": self._episode_hits_dealt_by_robot[agent_key],
            "hits_received": self._episode_hits_received_by_robot[agent_key],
            "clamp_count": self._episode_clamp_count_by_robot[agent_key],
            "min_horizontal_distance": self._episode_min_horizontal_distance_by_robot[agent_key],
        }
        return agent_info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        full_obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_full_obs = full_obs
        self._last_info = info
        for agent_key in ("robot_a", "robot_b"):
            self._last_actions[agent_key] = np.zeros(self.base_env.action_space[agent_key].shape, dtype=np.float32)
            self._episode_rewards[agent_key] = 0.0
            self._episode_damage_dealt_by_robot[agent_key] = 0.0
            self._episode_damage_received_by_robot[agent_key] = 0.0
            self._episode_hits_dealt_by_robot[agent_key] = 0
            self._episode_hits_received_by_robot[agent_key] = 0
            self._episode_clamp_count_by_robot[agent_key] = 0
            self._episode_min_horizontal_distance_by_robot[agent_key] = float(
                info.get("relative_metrics", {}).get(agent_key, {}).get("horizontal_distance", 0.0)
            )
        reset_info = {
            "self_play_views": {
                agent_key: self._build_agent_episode_info(
                    info,
                    agent_key=agent_key,
                    metrics=self._build_reset_metrics_for_agent(info, agent_key=agent_key),
                    reward_terms=zero_reward_terms(),
                )
                for agent_key in ("robot_a", "robot_b")
            }
        }
        return self._stack_full_obs(full_obs), reset_info

    def step(self, action: np.ndarray):
        if self._last_full_obs is None or self._last_info is None:
            raise RuntimeError("Environment must be reset before calling step()")

        action_array = np.asarray(action, dtype=np.float32).reshape(2, *self.base_env.action_space["robot_a"].shape)
        action_dict = {
            "robot_a": np.clip(
                action_array[0],
                self.base_env.action_space["robot_a"].low,
                self.base_env.action_space["robot_a"].high,
            ).astype(np.float32),
            "robot_b": np.clip(
                action_array[1],
                self.base_env.action_space["robot_b"].low,
                self.base_env.action_space["robot_b"].high,
            ).astype(np.float32),
        }
        prev_info = self._last_info
        full_obs, _, terminated, truncated, info = self.base_env.step(action_dict)

        view_infos: Dict[str, Any] = {}
        for agent_key in ("robot_a", "robot_b"):
            metrics = self._extract_metrics_for_agent(
                info,
                prev_info,
                action_dict[agent_key],
                agent_key=agent_key,
                last_action=self._last_actions[agent_key],
                episode_damage_dealt=self._episode_damage_dealt_by_robot[agent_key],
                episode_damage_received=self._episode_damage_received_by_robot[agent_key],
                episode_min_horizontal_distance=self._episode_min_horizontal_distance_by_robot[agent_key],
            )
            reward, reward_terms = self._compute_reward(metrics)
            if (
                self.curriculum_stage == "distance_stage1"
                and self.distance_stage_reward_config.reward_mode in {"episode_uniform", "episode_curriculum"}
                and not (terminated or truncated)
            ):
                reward = 0.0
                reward_terms = zero_reward_terms()
            self._episode_rewards[agent_key] += reward
            self._episode_damage_dealt_by_robot[agent_key] += metrics["damage_dealt"]
            self._episode_damage_received_by_robot[agent_key] += metrics["damage_received"]
            self._episode_hits_dealt_by_robot[agent_key] += int(metrics["hits_dealt"])
            self._episode_hits_received_by_robot[agent_key] += int(metrics["hits_received"])
            self._episode_clamp_count_by_robot[agent_key] += int(metrics["clamp_count"])
            self._episode_min_horizontal_distance_by_robot[agent_key] = min(
                self._episode_min_horizontal_distance_by_robot[agent_key],
                float(metrics["horizontal_distance"]),
            )
            view_infos[agent_key] = self._build_agent_episode_info(
                info,
                agent_key=agent_key,
                metrics=metrics,
                reward_terms=reward_terms,
            )
            self._last_actions[agent_key] = action_dict[agent_key]

        self._last_full_obs = full_obs
        self._last_info = info
        combined_info = dict(info)
        combined_info["self_play_views"] = view_infos
        return self._stack_full_obs(full_obs), 0.0, terminated, truncated, combined_info
