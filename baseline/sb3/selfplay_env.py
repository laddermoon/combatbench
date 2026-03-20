from __future__ import annotations

import numpy as np
import gymnasium as gym

from ...core.humanoid_robot import HumanoidRobot
from ...envs.combat_gym import CombatGymEnv
from .normalization import ObservationNormalizer
from .policies import SB3CombatPolicy
from .rewards import ATTACKER_APPROACH_REWARD_CONFIG, ATTACKER_REWARD_CONFIG, FIGHT_REWARD_CONFIG, RewardConfig, compute_shaped_rewards


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
STAND_PD_KP = 12.0
STAND_PD_KD = 0.2
STAND_ACTION_SCALE_MULTIPLIER = 0.8
ATTACKER_APPROACH_PROGRESS_WEIGHT = 4.0
ATTACKER_DISTANCE_PENALTY_WEIGHT = 0.35
ATTACKER_CLOSING_SPEED_WEIGHT = 0.5
ATTACKER_FACING_REWARD_WEIGHT = 0.6
ATTACKER_LATERAL_OFFSET_PENALTY_WEIGHT = 0.45
ATTACKER_APPROACH_FACING_GATE_THRESHOLD = 0.55
ATTACKER_POOR_FACING_PROGRESS_PENALTY_WEIGHT = 2.5
ATTACKER_RESIDUAL_ACTION_SCALE = 0.35
ATTACKER_ENGAGE_DISTANCE = 0.9
ATTACKER_ENGAGE_BONUS = 0.15
ATTACKER_STAGNATION_PROGRESS_THRESHOLD = 0.002
ATTACKER_STAGNATION_SPEED_THRESHOLD = 0.03
ATTACKER_STAGNATION_PENALTY = 0.05
ATTACKER_STANDING_WIN_BONUS = 10.0
ATTACKER_SELF_FALL_PENALTY = 10.0
ATTACKER_DOUBLE_FALL_PENALTY = 14.0
ATTACKER_TIMEOUT_PENALTY = 20.0
ATTACKER_SELF_HEAD_DAMAGE_PENALTY = 0.35
ATTACKER_SELF_TORSO_DAMAGE_PENALTY = 0.15
ATTACKER_OPPONENT_HEAD_DAMAGE_BONUS = 0.25
ATTACKER_OPPONENT_TORSO_DAMAGE_BONUS = 0.12
ATTACKER_UNSTABLE_APPROACH_PENALTY_WEIGHT = 6.0
ATTACKER_UNSTABLE_CLOSING_PENALTY_WEIGHT = 0.8
ATTACKER_ENGAGE_STABILITY_THRESHOLD = 0.65
ATTACKER_CONTACT_REWARD_STABILITY_THRESHOLD = 0.75
ATTACKER_UNSTABLE_HEAD_CONTACT_PENALTY = 0.35
ATTACKER_UNSTABLE_TORSO_CONTACT_PENALTY = 0.18
ATTACKER_APPROACH_SUCCESS_STABILITY_THRESHOLD = 0.72
ATTACKER_APPROACH_SUCCESS_FACING_THRESHOLD = 0.45
ATTACKER_APPROACH_SUCCESS_BONUS = 6.0
ATTACKER_APPROACH_OPPONENT_FALL_PENALTY = 4.0
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
ATTACKER_ACTION_SCALE_MULTIPLIERS = {
    "abdomen_z": 2.5,
    "abdomen_y": 3.0,
    "abdomen_x": 2.0,
    "hip_x_right": 2.0,
    "hip_z_right": 2.0,
    "hip_y_right": 3.5,
    "knee_right": 4.0,
    "ankle_y_right": 3.0,
    "ankle_x_right": 2.0,
    "hip_x_left": 2.0,
    "hip_z_left": 2.0,
    "hip_y_left": 3.5,
    "knee_left": 4.0,
    "ankle_y_left": 3.0,
    "ankle_x_left": 2.0,
    "shoulder1_right": 2.0,
    "shoulder2_right": 2.5,
    "elbow_right": 2.5,
    "shoulder1_left": 2.0,
    "shoulder2_left": 2.5,
    "elbow_left": 2.5,
}


def build_stand_action_scale() -> np.ndarray:
    action_scale = np.zeros(HumanoidRobot.ACTION_DIM, dtype=np.float32)
    joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(HumanoidRobot.CONTROLLED_JOINTS)}
    for joint_name, scale in STAND_POLICY_ACTION_SCALE.items():
        action_scale[joint_name_to_index[joint_name]] = float(scale) * STAND_ACTION_SCALE_MULTIPLIER
    return action_scale


def build_attacker_action_scale() -> np.ndarray:
    action_scale = build_stand_action_scale().copy()
    joint_name_to_index = {joint_name: idx for idx, joint_name in enumerate(HumanoidRobot.CONTROLLED_JOINTS)}
    for joint_name, multiplier in ATTACKER_ACTION_SCALE_MULTIPLIERS.items():
        joint_index = joint_name_to_index[joint_name]
        action_scale[joint_index] = action_scale[joint_index] * float(multiplier)
    return action_scale


def build_attacker_base_action_compensation() -> np.ndarray:
    stand_action_scale = build_stand_action_scale()
    attacker_action_scale = build_attacker_action_scale()
    compensation = np.ones(HumanoidRobot.ACTION_DIM, dtype=np.float32)
    nonzero_mask = attacker_action_scale > 1e-6
    compensation[nonzero_mask] = stand_action_scale[nonzero_mask] / attacker_action_scale[nonzero_mask]
    return compensation.astype(np.float32)


def build_stand_reset_pose() -> dict[str, dict[str, float]]:
    return {
        "robot_a": dict(STAND_REFERENCE_POSE),
        "robot_b": dict(STAND_REFERENCE_POSE),
    }


def configure_base_env_for_stand(env: CombatGymEnv) -> np.ndarray:
    action_scale = build_stand_action_scale()
    stand_reset_pose = build_stand_reset_pose()
    env.set_robot_joint_positions(stand_reset_pose)
    env.set_controller_reference_positions(stand_reset_pose)
    env.set_controller_gains(kp=STAND_PD_KP, kd=STAND_PD_KD)
    env.set_controller_action_scale(
        {
            "robot_a": action_scale,
            "robot_b": np.zeros_like(action_scale),
        }
    )
    return action_scale


def configure_base_env_for_fight_attacker(env: CombatGymEnv) -> np.ndarray:
    attacker_action_scale = build_attacker_action_scale()
    stand_action_scale = build_stand_action_scale()
    stand_reset_pose = build_stand_reset_pose()
    env.set_robot_joint_positions(stand_reset_pose)
    env.set_controller_reference_positions(stand_reset_pose)
    env.set_controller_gains(kp=STAND_PD_KP, kd=STAND_PD_KD)
    env.set_controller_action_scale(
        {
            "robot_a": attacker_action_scale,
            "robot_b": stand_action_scale,
        }
    )
    return attacker_action_scale


def configure_base_env_for_fight(env: CombatGymEnv) -> np.ndarray:
    action_scale = build_stand_action_scale()
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
    return action_scale


def get_fallen_robots(info: dict, reward_config: RewardConfig, robot_ids: tuple[str, ...] | None = None) -> list[str]:
    selected_robot_ids = robot_ids or tuple(info["robot_states"].keys())
    fallen_robots = []
    for robot_id in selected_robot_ids:
        state = info["robot_states"][robot_id]
        torso_height = float(state["torso_position"][2])
        uprightness = float(state["uprightness"])
        if torso_height < reward_config.min_height or uprightness < reward_config.min_uprightness:
            fallen_robots.append(robot_id)
    return fallen_robots


def is_robot_standing(info: dict, reward_config: RewardConfig, robot_id: str) -> bool:
    return robot_id not in get_fallen_robots(info, reward_config, robot_ids=(robot_id,))


def compute_attacker_stability_score(info: dict, reward_config: RewardConfig, robot_id: str = "robot_a") -> float:
    state = info["robot_states"][robot_id]
    torso_height = float(state["torso_position"][2])
    uprightness = float(state["uprightness"])
    height_denom = max(1e-6, reward_config.target_height - reward_config.min_height)
    upright_denom = max(1e-6, 1.0 - reward_config.min_uprightness)
    height_score = float(np.clip((torso_height - reward_config.min_height) / height_denom, 0.0, 1.0))
    upright_score = float(np.clip((uprightness - reward_config.min_uprightness) / upright_denom, 0.0, 1.0))
    return float(min(height_score, upright_score))


def compute_attacker_contact_reward(info: dict, reward_config: RewardConfig, attacker_stability: float) -> tuple[float, dict[str, float]]:
    reward = 0.0
    reward_breakdown = {
        "self_head_damage": 0.0,
        "self_torso_damage": 0.0,
        "opponent_head_damage": 0.0,
        "opponent_torso_damage": 0.0,
        "unstable_contact_penalty": 0.0,
    }
    self_hit_records = info.get("hit_records", {}).get("robot_a", [])
    opponent_hit_records = info.get("hit_records", {}).get("robot_b", [])

    for record in self_hit_records:
        damage_amount = float(max(0.0, -float(record.get("damage", 0.0))))
        damage_part = record.get("damage_part")
        if damage_part == "head":
            penalty = ATTACKER_SELF_HEAD_DAMAGE_PENALTY * damage_amount
            reward -= penalty
            reward_breakdown["self_head_damage"] += penalty
        elif damage_part == "torso":
            penalty = ATTACKER_SELF_TORSO_DAMAGE_PENALTY * damage_amount
            reward -= penalty
            reward_breakdown["self_torso_damage"] += penalty

    if reward_config.name == ATTACKER_APPROACH_REWARD_CONFIG.name:
        return float(reward), reward_breakdown

    stability_margin = max(0.0, attacker_stability - ATTACKER_CONTACT_REWARD_STABILITY_THRESHOLD)
    stability_scale = float(np.clip(stability_margin / max(1e-6, 1.0 - ATTACKER_CONTACT_REWARD_STABILITY_THRESHOLD), 0.0, 1.0))
    instability_scale = float(np.clip((ATTACKER_CONTACT_REWARD_STABILITY_THRESHOLD - attacker_stability) / ATTACKER_CONTACT_REWARD_STABILITY_THRESHOLD, 0.0, 1.0))

    for record in opponent_hit_records:
        damage_amount = float(max(0.0, -float(record.get("damage", 0.0))))
        damage_part = record.get("damage_part")
        if damage_part == "head":
            bonus = ATTACKER_OPPONENT_HEAD_DAMAGE_BONUS * damage_amount * stability_scale
            reward += bonus
            reward_breakdown["opponent_head_damage"] += bonus
            if instability_scale > 0.0:
                penalty = ATTACKER_UNSTABLE_HEAD_CONTACT_PENALTY * damage_amount * instability_scale
                reward -= penalty
                reward_breakdown["unstable_contact_penalty"] += penalty
        elif damage_part == "torso":
            bonus = ATTACKER_OPPONENT_TORSO_DAMAGE_BONUS * damage_amount * stability_scale
            reward += bonus
            reward_breakdown["opponent_torso_damage"] += bonus
            if instability_scale > 0.0:
                penalty = ATTACKER_UNSTABLE_TORSO_CONTACT_PENALTY * damage_amount * instability_scale
                reward -= penalty
                reward_breakdown["unstable_contact_penalty"] += penalty

    return float(reward), reward_breakdown


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
        self._stand_action_scale = build_stand_action_scale()
        for joint_name, joint_value in STAND_REFERENCE_POSE.items():
            self._stand_reference_action[self._joint_name_to_index[joint_name]] = float(joint_value)
        self._opponent_slice = HumanoidRobot.OBSERVATION_SLICES["opponent_relative_position"]
        self._stand_reset_pose = build_stand_reset_pose()

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

        robot_ids = ("robot_a",) if self._stand_mode else tuple(info["robot_states"].keys())
        fallen_robots = get_fallen_robots(info, self.reward_config, robot_ids=robot_ids)

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

    def _apply_approach_completion(self, terminated: bool, truncated: bool, info: dict) -> tuple[bool, bool, dict]:
        if terminated or truncated or not self._approach_mode:
            return terminated, truncated, info

        current_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        attacker_stability = compute_attacker_stability_score(info, self.reward_config, "robot_a")
        facing_score = float(max(0.0, info["relative_metrics"]["robot_a"]["facing_opponent"]))
        if current_distance > self.reward_config.target_distance:
            return terminated, truncated, info
        if attacker_stability < ATTACKER_APPROACH_SUCCESS_STABILITY_THRESHOLD:
            return terminated, truncated, info
        if facing_score < ATTACKER_APPROACH_SUCCESS_FACING_THRESHOLD:
            return terminated, truncated, info

        updated_info = dict(info)
        updated_info["winner"] = "robot_a"
        updated_info["end_reason"] = "robot_a reached the target approach distance while remaining stable and facing the opponent"
        return True, False, updated_info

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self._stand_mode:
            self._stand_action_scale = configure_base_env_for_stand(self.env)
        else:
            self._stand_action_scale = configure_base_env_for_fight(self.env)
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


class AttackerStandingOpponentEnv(SymmetricSelfPlayEnv):
    def __init__(
        self,
        env: CombatGymEnv,
        opponent_policy: SB3CombatPolicy,
        attacker_base_policy: SB3CombatPolicy,
        reward_config: RewardConfig = ATTACKER_REWARD_CONFIG,
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        super().__init__(env, reward_config=reward_config, normalizer=normalizer)
        self.opponent_policy = opponent_policy
        self.attacker_base_policy = attacker_base_policy
        self.attacker_base_action_compensation = build_attacker_base_action_compensation()
        self._approach_mode = reward_config.name == ATTACKER_APPROACH_REWARD_CONFIG.name
        self._previous_horizontal_distance = 0.0
        self._last_obs = None
        self._last_info = None

    def _apply_attacker_termination(self, terminated: bool, truncated: bool, info: dict) -> tuple[bool, bool, dict]:
        if terminated or truncated or not self.reward_config.terminate_on_fall:
            return terminated, truncated, info

        fallen_robots = get_fallen_robots(info, self.reward_config)
        if not fallen_robots:
            return terminated, truncated, info

        updated_info = dict(info)
        updated_info["fallen_robots"] = fallen_robots
        attacker_standing = is_robot_standing(info, self.reward_config, "robot_a")
        defender_standing = is_robot_standing(info, self.reward_config, "robot_b")
        if self._approach_mode:
            if not attacker_standing and defender_standing:
                updated_info["winner"] = "robot_b"
                updated_info["end_reason"] = "robot_a fell below the stability threshold during approach curriculum"
            elif attacker_standing and not defender_standing:
                updated_info["winner"] = "draw"
                updated_info["end_reason"] = "robot_b fell during approach curriculum; stable close-range arrival is required instead"
            else:
                updated_info["winner"] = "draw"
                updated_info["end_reason"] = "approach curriculum ended because both robots fell below the stability threshold"
            return True, False, updated_info
        winner = info.get("winner")
        if winner == "robot_a" and attacker_standing and not defender_standing:
            updated_info["winner"] = "robot_a"
            updated_info["end_reason"] = "robot_b fell below the stability threshold while robot_a remained standing"
        elif winner == "robot_b":
            updated_info["winner"] = "robot_b"
            updated_info["end_reason"] = "robot_a fell below the stability threshold before defeating robot_b"
        else:
            updated_info["winner"] = "draw"
            updated_info["end_reason"] = "both robots fell below the stability threshold"
        return True, False, updated_info

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._stand_action_scale = configure_base_env_for_fight_attacker(self.env)
        obs = self.env._get_obs()
        info = self.env._build_info()
        self._previous_actions = {
            "robot_a": np.zeros(self.action_space.shape, dtype=np.float32),
            "robot_b": np.zeros(self.action_space.shape, dtype=np.float32),
        }
        self._previous_scores = dict(info["scores"])
        self._previous_horizontal_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        self._last_obs = obs
        self._last_info = info
        self.opponent_policy.reset()
        self.attacker_base_policy.reset()
        return self._normalize_obs(obs["robot_a_obs"]), info

    def step(self, action: np.ndarray):
        residual_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        attacker_base_action = self.attacker_base_policy.act(self._last_obs["robot_a_obs"], self._last_info)
        attacker_base_action = np.asarray(attacker_base_action, dtype=np.float32) * self.attacker_base_action_compensation
        opponent_action = self.opponent_policy.act(self._last_obs["robot_b_obs"], self._last_info)
        action_dict = {
            "robot_a": np.clip(attacker_base_action + ATTACKER_RESIDUAL_ACTION_SCALE * residual_action, -1.0, 1.0),
            "robot_b": np.clip(np.asarray(opponent_action, dtype=np.float32), -1.0, 1.0),
        }
        previous_scores = dict(self._previous_scores)
        previous_distance = float(self._previous_horizontal_distance)
        obs, _, terminated, truncated, info = self.env.step(action_dict)
        terminated, truncated, info = self._apply_attacker_termination(terminated, truncated, info)
        terminated, truncated, info = self._apply_approach_completion(terminated, truncated, info)
        shaped_rewards = compute_shaped_rewards(
            info=info,
            previous_scores=previous_scores,
            current_scores=info["scores"],
            action_dict=action_dict,
            previous_action_dict=self._previous_actions,
            config=self.reward_config,
        )
        current_distance = float(info["relative_metrics"]["robot_a"]["horizontal_distance"])
        approach_progress = previous_distance - current_distance
        direction_to_opponent = np.asarray(info["relative_metrics"]["robot_a"]["direction_to_opponent"], dtype=np.float32)
        relative_position = np.asarray(info["relative_metrics"]["robot_a"]["relative_position"], dtype=np.float32)
        attacker_rotation = np.asarray(info["robot_states"]["robot_a"]["rotation_matrix"], dtype=np.float32)
        local_relative_position = attacker_rotation.T @ relative_position
        lateral_offset = float(abs(local_relative_position[1]))
        facing_score = float(max(0.0, info["relative_metrics"]["robot_a"]["facing_opponent"]))
        facing_gate = float(np.clip((facing_score - ATTACKER_APPROACH_FACING_GATE_THRESHOLD) / max(1e-6, 1.0 - ATTACKER_APPROACH_FACING_GATE_THRESHOLD), 0.0, 1.0))
        attacker_velocity = np.asarray(info["robot_states"]["robot_a"]["linear_velocity"], dtype=np.float32)
        closing_speed = float(np.dot(attacker_velocity[:2], direction_to_opponent[:2]))
        attacker_stability = compute_attacker_stability_score(info, self.reward_config, "robot_a")
        stable_approach_progress = max(0.0, approach_progress) * attacker_stability
        aligned_approach_progress = stable_approach_progress * facing_gate
        misaligned_approach_progress = stable_approach_progress * (1.0 - facing_gate)
        unstable_approach_progress = max(0.0, approach_progress) * (1.0 - attacker_stability)
        stable_closing_speed = max(0.0, closing_speed) * attacker_stability
        unstable_closing_speed = max(0.0, closing_speed) * (1.0 - attacker_stability)
        contact_reward, contact_reward_breakdown = compute_attacker_contact_reward(info, self.reward_config, attacker_stability)
        reward = float(
            shaped_rewards["robot_a"]
            + ATTACKER_APPROACH_PROGRESS_WEIGHT * aligned_approach_progress
            + ATTACKER_CLOSING_SPEED_WEIGHT * stable_closing_speed
            + ATTACKER_FACING_REWARD_WEIGHT * facing_score * attacker_stability
            - ATTACKER_DISTANCE_PENALTY_WEIGHT * current_distance
            - ATTACKER_LATERAL_OFFSET_PENALTY_WEIGHT * lateral_offset
            + contact_reward
            - ATTACKER_POOR_FACING_PROGRESS_PENALTY_WEIGHT * misaligned_approach_progress
            - ATTACKER_UNSTABLE_APPROACH_PENALTY_WEIGHT * unstable_approach_progress
            - ATTACKER_UNSTABLE_CLOSING_PENALTY_WEIGHT * unstable_closing_speed
        )
        if current_distance <= ATTACKER_ENGAGE_DISTANCE and attacker_stability >= ATTACKER_ENGAGE_STABILITY_THRESHOLD:
            reward += ATTACKER_ENGAGE_BONUS
        elif approach_progress < ATTACKER_STAGNATION_PROGRESS_THRESHOLD and closing_speed < ATTACKER_STAGNATION_SPEED_THRESHOLD:
            reward -= ATTACKER_STAGNATION_PENALTY

        attacker_standing = is_robot_standing(info, self.reward_config, "robot_a")
        defender_standing = is_robot_standing(info, self.reward_config, "robot_b")
        winner = info.get("winner")
        if self._approach_mode and winner == "robot_a":
            reward += ATTACKER_APPROACH_SUCCESS_BONUS
        elif winner == "robot_a" and attacker_standing and not defender_standing:
            reward += ATTACKER_STANDING_WIN_BONUS
        elif winner == "robot_b":
            reward -= ATTACKER_SELF_FALL_PENALTY
        elif self._approach_mode and winner == "draw" and attacker_standing and not defender_standing:
            reward -= ATTACKER_APPROACH_OPPONENT_FALL_PENALTY
        elif winner == "draw" and not attacker_standing and not defender_standing:
            reward -= ATTACKER_DOUBLE_FALL_PENALTY
        elif truncated:
            reward -= ATTACKER_TIMEOUT_PENALTY

        self._previous_actions = {
            "robot_a": action_dict["robot_a"].copy(),
            "robot_b": action_dict["robot_b"].copy(),
        }
        self._previous_scores = dict(info["scores"])
        self._previous_horizontal_distance = current_distance
        self._last_obs = obs
        self._last_info = info
        updated_info = dict(info)
        updated_info["shaped_rewards"] = shaped_rewards
        updated_info["attacker_reward"] = reward
        updated_info["approach_progress"] = approach_progress
        updated_info["attacker_stability"] = attacker_stability
        updated_info["attacker_facing_score"] = facing_score
        updated_info["attacker_facing_gate"] = facing_gate
        updated_info["attacker_lateral_offset"] = lateral_offset
        updated_info["attacker_contact_reward"] = contact_reward
        updated_info["attacker_contact_reward_breakdown"] = contact_reward_breakdown
        updated_info["attacker_residual_action"] = residual_action.copy()
        updated_info["attacker_base_action"] = np.asarray(attacker_base_action, dtype=np.float32).copy()
        updated_info["attacker_applied_action"] = action_dict["robot_a"].copy()
        updated_info["opponent_policy_action"] = action_dict["robot_b"].copy()
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


def make_attacker_standing_env(
    opponent_model_path: str,
    attacker_base_model_path: str | None = None,
    render_mode: str | None = None,
    match_duration: float = 15.0,
    control_frequency: int = 20,
    initial_distance: float = 2.0,
    reward_config: RewardConfig = ATTACKER_REWARD_CONFIG,
    opponent_device: str = "cpu",
) -> AttackerStandingOpponentEnv:
    base_env = CombatGymEnv(
        render_mode=render_mode,
        match_duration=match_duration,
        control_frequency=control_frequency,
        initial_distance=initial_distance,
    )
    normalizer = ObservationNormalizer(target_height=reward_config.target_height)
    opponent_policy = SB3CombatPolicy(opponent_model_path, device=opponent_device)
    attacker_base_policy = SB3CombatPolicy(
        attacker_base_model_path or opponent_model_path,
        device=opponent_device,
        approach_base_mode="stepping_forward",
        approach_distance_threshold=float(reward_config.target_distance),
        approach_abdomen_y_action=0.6,
    )
    return AttackerStandingOpponentEnv(base_env, opponent_policy=opponent_policy, attacker_base_policy=attacker_base_policy, reward_config=reward_config, normalizer=normalizer)
