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
ABDOMEN_Y_INDEX = HumanoidRobot.CONTROLLED_JOINTS.index("abdomen_y")



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
        if approach_base_mode is None:
            approach_base_mode = self.metadata.get("approach_base_mode")
        if approach_distance_threshold is None:
            approach_distance_threshold = self.metadata.get("approach_distance_threshold", DEFAULT_APPROACH_DISTANCE_THRESHOLD)
        if approach_abdomen_y_action is None:
            approach_abdomen_y_action = self.metadata.get("approach_abdomen_y_action", DEFAULT_APPROACH_ABDOMEN_Y_ACTION)
        self.approach_base_mode = approach_base_mode
        self.approach_distance_threshold = float(approach_distance_threshold)
        self.approach_abdomen_y_action = float(approach_abdomen_y_action)
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
                )
        self.model = PPO.load(self.model_path, device=device)

    def _apply_approach_base(self, obs: np.ndarray, action_array: np.ndarray) -> np.ndarray:
        if self.approach_base_mode != "lean_forward":
            return action_array
        relative_position_slice = self.normalizer.slices["opponent_relative_position"]
        relative_position = np.asarray(obs, dtype=np.float32)[relative_position_slice]
        horizontal_distance = float(np.linalg.norm(relative_position[:2]))
        updated_action = np.asarray(action_array, dtype=np.float32).copy()
        if horizontal_distance > self.approach_distance_threshold:
            updated_action[ABDOMEN_Y_INDEX] = np.clip(updated_action[ABDOMEN_Y_INDEX] + self.approach_abdomen_y_action, -1.0, 1.0)
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
            base_action = self._apply_approach_base(raw_obs, base_action)
            action_array = np.clip(base_action + self.attacker_residual_action_scale * action_array, -1.0, 1.0)
        else:
            action_array = self._apply_approach_base(raw_obs, action_array)
        return action_array

    def reset(self):
        return None
