from __future__ import annotations

import numpy as np
import gymnasium as gym

from ...core.humanoid_robot import HumanoidRobot
from ...envs.combat_gym import CombatGymEnv
from .normalization import ObservationNormalizer
from .rewards import FIGHT_REWARD_CONFIG, RewardConfig, compute_shaped_rewards


STAND_REFERENCE_POSE = {
    "abdomen_y": -0.01085978047062336,
    "abdomen_x": 0.008895262945626065,
    "hip_y_right": -0.14603185113208156,
    "knee_right": -0.1246737508641517,
    "ankle_y_right": 0.0005306184929703764,
    "hip_y_left": -0.15436512545227296,
    "knee_left": -0.13332025613936188,
    "ankle_y_left": -0.05391723289203508,
    "hip_x_right": -0.059727273869097465,
    "hip_x_left": 0.04935656269190952,
    "ankle_x_right": 0.05406650212513886,
    "ankle_x_left": 0.002856759542333928,
}
STAND_POLICY_ACTION_SCALE = {
    "abdomen_z": 0.0224,
    "abdomen_y": 0.0336,
    "abdomen_x": 0.0224,
    "hip_x_right": 0.0224,
    "hip_z_right": 0.0168,
    "hip_y_right": 0.0504,
    "knee_right": 0.056,
    "ankle_y_right": 0.0336,
    "ankle_x_right": 0.0224,
    "hip_x_left": 0.0224,
    "hip_z_left": 0.0168,
    "hip_y_left": 0.0504,
    "knee_left": 0.056,
    "ankle_y_left": 0.0336,
    "ankle_x_left": 0.0224,
    "shoulder1_right": 0.0084,
    "shoulder2_right": 0.0084,
    "elbow_right": 0.0084,
    "shoulder1_left": 0.0084,
    "shoulder2_left": 0.0084,
    "elbow_left": 0.0084,
}


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
        self._stand_mode = reward_config.name == "stand"
        self.observation_space = self.normalizer.observation_space()
        self.action_space = env.action_space["robot_a"]
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None
        self._previous_actions = {"robot_a": None, "robot_b": None}
        self._previous_scores = {"robot_a": 100.0, "robot_b": 100.0}
        self._joint_names = tuple(HumanoidRobot.CONTROLLED_JOINTS)
        self._joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(self._joint_names)}
        self._stand_reference_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._stand_action_scale = np.zeros(self.action_space.shape, dtype=np.float32)
        for joint_name, joint_value in STAND_REFERENCE_POSE.items():
            self._stand_reference_action[self._joint_name_to_index[joint_name]] = float(joint_value)
        for joint_name, scale in STAND_POLICY_ACTION_SCALE.items():
            self._stand_action_scale[self._joint_name_to_index[joint_name]] = float(scale)
        self._opponent_slice = HumanoidRobot.OBSERVATION_SLICES["opponent_relative_position"]
        self._stand_reset_pose = {
            "robot_a": dict(STAND_REFERENCE_POSE),
            "robot_b": dict(STAND_REFERENCE_POSE),
        }

    @property
    def base_env(self) -> CombatGymEnv:
        return self.env

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        normalized_obs = self.normalizer.normalize(obs)
        if self._stand_mode:
            normalized_obs[self._opponent_slice.start :] = 0.0
        if np.any(np.isnan(normalized_obs)) or np.any(np.isinf(normalized_obs)):
            return np.zeros_like(normalized_obs, dtype=np.float32)
        return normalized_obs.astype(np.float32)

    def _build_stand_targets(self, action_array: np.ndarray) -> dict[str, np.ndarray]:
        robot_a_target = self._stand_reference_action + self._stand_action_scale * action_array
        robot_b_target = self._stand_reference_action.copy()
        return {
            "robot_a": robot_a_target.astype(np.float32),
            "robot_b": robot_b_target.astype(np.float32),
        }

    def _apply_fall_termination(self, terminated: bool, truncated: bool, info: dict) -> tuple[bool, bool, dict]:
        if terminated or truncated or not self.reward_config.terminate_on_fall:
            return terminated, truncated, info

        fallen_robots = []
        robot_ids = ("robot_a",) if self._stand_mode else tuple(info["robot_states"].keys())
        for robot_id in robot_ids:
            state = info["robot_states"][robot_id]
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
        if self._stand_mode:
            self.env.set_robot_joint_positions(self._stand_reset_pose)
            self.env.set_controller_reference_positions(self._stand_reset_pose)
            self.env.set_controller_action_scale(
                {
                    "robot_a": self._stand_action_scale,
                    "robot_b": np.zeros_like(self._stand_action_scale),
                }
            )
        else:
            self.env.reset_controller_config()
        obs = self.env._get_obs()
        info = self.env._build_info()
        self._previous_actions = {
            "robot_a": np.zeros(self.action_space.shape, dtype=np.float32),
            "robot_b": np.zeros(self.action_space.shape, dtype=np.float32),
        }
        self._previous_scores = dict(info["scores"])
        return self._normalize_obs(obs["robot_a_obs"]), info

    def step(self, action: np.ndarray):
        action_array = np.asarray(action, dtype=np.float32)
        action_array = np.clip(action_array, -1.0, 1.0)
        if self._stand_mode:
            policy_action_dict = {
                "robot_a": action_array,
                "robot_b": np.zeros_like(action_array),
            }
            target_positions = self._build_stand_targets(action_array)
            previous_scores = dict(self._previous_scores)
            obs, _, terminated, truncated, info = self.env.step(action_dict=policy_action_dict)
            terminated, truncated, info = self._apply_fall_termination(terminated, truncated, info)
            shaped_rewards = compute_shaped_rewards(
                info=info,
                previous_scores=previous_scores,
                current_scores=info["scores"],
                action_dict=policy_action_dict,
                previous_action_dict=self._previous_actions,
                config=self.reward_config,
            )
            reward = float(shaped_rewards["robot_a"])
            self._previous_actions = {
                "robot_a": policy_action_dict["robot_a"].copy(),
                "robot_b": policy_action_dict["robot_b"].copy(),
            }
            self._previous_scores = dict(info["scores"])
            updated_info = dict(info)
            updated_info["shaped_rewards"] = shaped_rewards
            updated_info["shared_policy_action"] = action_array.copy()
            updated_info["stand_target_positions"] = {
                robot_id: target.copy() for robot_id, target in target_positions.items()
            }
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            return self._normalize_obs(obs["robot_a_obs"]), reward, terminated, truncated, updated_info

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
