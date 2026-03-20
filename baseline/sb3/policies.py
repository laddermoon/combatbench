from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ...core.humanoid_robot import HumanoidRobot
from .normalization import ObservationNormalizer


DEFAULT_TARGET_HEIGHT = 1.282
DEFAULT_APPROACH_DISTANCE_THRESHOLD = 1.0
DEFAULT_APPROACH_ABDOMEN_Y_ACTION = 0.6
DEFAULT_APPROACH_MIN_HEIGHT = 1.0
DEFAULT_APPROACH_MIN_UPRIGHTNESS = 0.55
DEFAULT_APPROACH_TURN_ACTION = 0.32
ABDOMEN_Y_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("abdomen_y")
ABDOMEN_Z_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("abdomen_z")
HIP_X_RIGHT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("hip_x_right")
HIP_Y_RIGHT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("hip_y_right")
KNEE_RIGHT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("knee_right")
ANKLE_Y_RIGHT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("ankle_y_right")
HIP_X_LEFT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("hip_x_left")
HIP_Y_LEFT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("hip_y_left")
KNEE_LEFT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("knee_left")
ANKLE_Y_LEFT_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("ankle_y_left")
DEFAULT_APPROACH_STEP_PERIOD = 24
DEFAULT_APPROACH_HIP_SWING_ACTION = 0.32
DEFAULT_APPROACH_KNEE_LIFT_ACTION = 0.34
DEFAULT_APPROACH_ANKLE_SWING_ACTION = 0.18



def load_training_metadata(model_path: str | Path) -> dict:
    model_file = Path(model_path)
    candidate_paths = [
        model_file.parent / "run_config.json",
        model_file.parent.parent / "run_config.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


class SB3CombatPolicy:
    def __init__(
        self,
        model_path: str | Path,
        deterministic: bool = True,
        device: str = "auto",
        target_height: float | None = None,
        mask_opponent_features: bool | None = None,
        approach_base_mode: str | None = None,
        approach_distance_threshold: float | None = None,
        approach_abdomen_y_action: float | None = None,
        approach_min_height: float | None = None,
        approach_min_uprightness: float | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.deterministic = deterministic
        self.metadata = load_training_metadata(model_path)
        effective_target_height = target_height
        if effective_target_height is None:
            effective_target_height = self.metadata.get("target_height", DEFAULT_TARGET_HEIGHT)
        self.normalizer = ObservationNormalizer(target_height=float(effective_target_height))
        if mask_opponent_features is None:
            mask_opponent_features = self.metadata.get("phase") == "stand"
        self.mask_opponent_features = bool(mask_opponent_features)
        self.action_interpretation = self.metadata.get("action_interpretation", "direct")
        self.attacker_residual_action_scale = float(self.metadata.get("attacker_residual_action_scale", 0.35))
        reward_config_metadata = self.metadata.get("reward_config") or {}
        reward_config_target_distance = reward_config_metadata.get("target_distance")
        if approach_base_mode is None:
            approach_base_mode = self.metadata.get("approach_base_mode")
        if approach_distance_threshold is None:
            if self.metadata.get("phase") in {"fight_attacker_approach", "fight_attacker"} and reward_config_target_distance is not None:
                approach_distance_threshold = reward_config_target_distance
            else:
                approach_distance_threshold = self.metadata.get("approach_distance_threshold", DEFAULT_APPROACH_DISTANCE_THRESHOLD)
        if approach_abdomen_y_action is None:
            approach_abdomen_y_action = self.metadata.get("approach_abdomen_y_action", DEFAULT_APPROACH_ABDOMEN_Y_ACTION)
        if approach_min_height is None:
            approach_min_height = self.metadata.get("approach_min_height", DEFAULT_APPROACH_MIN_HEIGHT)
        if approach_min_uprightness is None:
            approach_min_uprightness = self.metadata.get("approach_min_uprightness", DEFAULT_APPROACH_MIN_UPRIGHTNESS)
        self.approach_base_mode = approach_base_mode
        self.approach_distance_threshold = float(approach_distance_threshold)
        self.approach_abdomen_y_action = float(approach_abdomen_y_action)
        self.approach_min_height = float(approach_min_height)
        self.approach_min_uprightness = float(approach_min_uprightness)
        compensation_values = self.metadata.get("attacker_base_action_compensation")
        if compensation_values is None:
            self.attacker_base_action_compensation = None
        else:
            self.attacker_base_action_compensation = np.asarray(compensation_values, dtype=np.float32)
        self.base_policy = None
        if self.action_interpretation == "attacker_base_residual":
            attacker_base_model = self.metadata.get("attacker_base_model") or self.metadata.get("opponent_model")
            if attacker_base_model:
                self.base_policy = SB3CombatPolicy(
                    attacker_base_model,
                    deterministic=deterministic,
                    device=device,
                    approach_base_mode=self.approach_base_mode,
                    approach_distance_threshold=self.approach_distance_threshold,
                    approach_abdomen_y_action=self.approach_abdomen_y_action,
                    approach_min_height=self.approach_min_height,
                    approach_min_uprightness=self.approach_min_uprightness,
                )
        self.model = PPO.load(self.model_path, device=device)

    def _apply_stepping_forward_base(self, updated_action: np.ndarray, step_count: int, distance_scale: float) -> np.ndarray:
        half_period = max(1, DEFAULT_APPROACH_STEP_PERIOD // 2)
        swing_right = ((step_count // half_period) % 2) == 0
        hip_swing = DEFAULT_APPROACH_HIP_SWING_ACTION * distance_scale
        knee_lift = DEFAULT_APPROACH_KNEE_LIFT_ACTION * distance_scale
        ankle_swing = DEFAULT_APPROACH_ANKLE_SWING_ACTION * distance_scale

        if swing_right:
            updated_action[HIP_Y_RIGHT_INDEX] = np.clip(updated_action[HIP_Y_RIGHT_INDEX] + hip_swing, -1.0, 1.0)
            updated_action[KNEE_RIGHT_INDEX] = np.clip(updated_action[KNEE_RIGHT_INDEX] + knee_lift, -1.0, 1.0)
            updated_action[ANKLE_Y_RIGHT_INDEX] = np.clip(updated_action[ANKLE_Y_RIGHT_INDEX] - ankle_swing, -1.0, 1.0)
            updated_action[HIP_Y_LEFT_INDEX] = np.clip(updated_action[HIP_Y_LEFT_INDEX] - 0.45 * hip_swing, -1.0, 1.0)
            updated_action[KNEE_LEFT_INDEX] = np.clip(updated_action[KNEE_LEFT_INDEX] - 0.2 * knee_lift, -1.0, 1.0)
            updated_action[ANKLE_Y_LEFT_INDEX] = np.clip(updated_action[ANKLE_Y_LEFT_INDEX] + 0.5 * ankle_swing, -1.0, 1.0)
        else:
            updated_action[HIP_Y_LEFT_INDEX] = np.clip(updated_action[HIP_Y_LEFT_INDEX] + hip_swing, -1.0, 1.0)
            updated_action[KNEE_LEFT_INDEX] = np.clip(updated_action[KNEE_LEFT_INDEX] + knee_lift, -1.0, 1.0)
            updated_action[ANKLE_Y_LEFT_INDEX] = np.clip(updated_action[ANKLE_Y_LEFT_INDEX] - ankle_swing, -1.0, 1.0)
            updated_action[HIP_Y_RIGHT_INDEX] = np.clip(updated_action[HIP_Y_RIGHT_INDEX] - 0.45 * hip_swing, -1.0, 1.0)
            updated_action[KNEE_RIGHT_INDEX] = np.clip(updated_action[KNEE_RIGHT_INDEX] - 0.2 * knee_lift, -1.0, 1.0)
            updated_action[ANKLE_Y_RIGHT_INDEX] = np.clip(updated_action[ANKLE_Y_RIGHT_INDEX] + 0.5 * ankle_swing, -1.0, 1.0)
        return updated_action

    def _apply_turn_toward_opponent_base(
        self,
        updated_action: np.ndarray,
        relative_position: np.ndarray,
        info: dict | None,
        distance_scale: float,
    ) -> np.ndarray:
        local_relative_position = np.asarray(relative_position, dtype=np.float32)
        if info is not None:
            robot_state = info.get("robot_states", {}).get("robot_a")
            if robot_state is not None and robot_state.get("rotation_matrix") is not None:
                rotation_matrix = np.asarray(robot_state["rotation_matrix"], dtype=np.float32)
                local_relative_position = rotation_matrix.T @ local_relative_position
        heading_error = float(np.arctan2(local_relative_position[1], max(1e-6, local_relative_position[0])))
        turn_scale = float(np.clip(heading_error / 0.75, -1.0, 1.0)) * distance_scale
        abdomen_turn = DEFAULT_APPROACH_TURN_ACTION * turn_scale
        updated_action[ABDOMEN_Z_INDEX] = np.clip(updated_action[ABDOMEN_Z_INDEX] + abdomen_turn, -1.0, 1.0)
        return updated_action

    def _apply_approach_base(self, obs: np.ndarray, action_array: np.ndarray, info: dict | None = None) -> np.ndarray:
        if self.approach_base_mode not in {"lean_forward", "stepping_forward"}:
            return action_array
        relative_position_slice = self.normalizer.slices["opponent_relative_position"]
        relative_position = np.asarray(obs, dtype=np.float32)[relative_position_slice]
        horizontal_distance = float(np.linalg.norm(relative_position[:2]))
        updated_action = np.asarray(action_array, dtype=np.float32).copy()
        if info is not None:
            robot_state = info.get("robot_states", {}).get("robot_a")
            if robot_state is not None:
                torso_height = float(robot_state["torso_position"][2])
                uprightness = float(robot_state["uprightness"])
                if torso_height < self.approach_min_height or uprightness < self.approach_min_uprightness:
                    return updated_action
        if horizontal_distance > self.approach_distance_threshold:
            abdomen_y_action = self.approach_abdomen_y_action
            if self.approach_base_mode == "stepping_forward":
                abdomen_y_action *= 0.35
            updated_action[ABDOMEN_Y_INDEX] = np.clip(updated_action[ABDOMEN_Y_INDEX] + abdomen_y_action, -1.0, 1.0)
            if self.approach_base_mode == "stepping_forward":
                distance_scale = float(np.clip((horizontal_distance - self.approach_distance_threshold) / max(1e-6, self.approach_distance_threshold), 0.35, 1.0))
                step_count = 0
                if info is not None:
                    step_count = int(info.get("current_step", info.get("physics_step_count", 0)))
                updated_action = self._apply_turn_toward_opponent_base(updated_action, relative_position, info, distance_scale)
                updated_action = self._apply_stepping_forward_base(updated_action, step_count, distance_scale)
        return updated_action

    def act(self, obs, info=None):
        raw_obs = np.asarray(obs, dtype=np.float32)
        normalized_obs = self.normalizer.normalize(raw_obs)
        if self.mask_opponent_features:
            opponent_slice = self.normalizer.slices["opponent_relative_position"]
            normalized_obs[opponent_slice.start :] = 0.0
        action, _ = self.model.predict(normalized_obs, deterministic=self.deterministic)
        action_array = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.base_policy is not None:
            base_action = self.base_policy.act(obs, info)
            if self.attacker_base_action_compensation is not None:
                base_action = np.asarray(base_action, dtype=np.float32) * self.attacker_base_action_compensation

            base_policy_mode = getattr(self.base_policy, "approach_base_mode", None)
            if self.approach_base_mode == "lean_forward" and base_policy_mode != "lean_forward":
                base_action = self._apply_approach_base(raw_obs, base_action, info)

            action_array = np.clip(base_action + self.attacker_residual_action_scale * action_array, -1.0, 1.0)
        else:
            action_array = self._apply_approach_base(raw_obs, action_array, info)
        return action_array

    def reset(self):
        return None
