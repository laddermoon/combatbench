from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ...core.humanoid_robot import HumanoidRobot


class ObservationNormalizer:
    def __init__(
        self,
        target_height: float = 1.282,
        height_scale: float = 0.5,
        joint_velocity_scale: float = 10.0,
        linear_velocity_scale: float = 5.0,
        angular_velocity_scale: float = 10.0,
        force_scale: float = 150.0,
        relative_position_scale: float = 4.0,
        keypoint_position_scale: float = 4.0,
        keypoint_velocity_scale: float = 10.0,
    ) -> None:
        self.target_height = float(target_height)
        self.height_scale = float(height_scale)
        self.joint_velocity_scale = float(joint_velocity_scale)
        self.linear_velocity_scale = float(linear_velocity_scale)
        self.angular_velocity_scale = float(angular_velocity_scale)
        self.force_scale = float(force_scale)
        self.relative_position_scale = float(relative_position_scale)
        self.keypoint_position_scale = float(keypoint_position_scale)
        self.keypoint_velocity_scale = float(keypoint_velocity_scale)
        self.slices = HumanoidRobot.OBSERVATION_SLICES

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        normalized = obs_array.copy()
        slices = self.slices

        normalized[slices['joint_positions']] = np.clip(
            normalized[slices['joint_positions']] / np.pi,
            -1.0,
            1.0,
        )
        normalized[slices['joint_velocities']] = np.clip(
            normalized[slices['joint_velocities']] / self.joint_velocity_scale,
            -1.0,
            1.0,
        )
        normalized[slices['height']] = np.clip(
            (obs_array[slices['height']] - self.target_height) / self.height_scale,
            -1.0,
            1.0,
        )
        normalized[slices['linear_velocity']] = np.clip(
            normalized[slices['linear_velocity']] / self.linear_velocity_scale,
            -1.0,
            1.0,
        )
        normalized[slices['angular_velocity']] = np.clip(
            normalized[slices['angular_velocity']] / self.angular_velocity_scale,
            -1.0,
            1.0,
        )
        normalized[slices['external_forces']] = np.tanh(
            normalized[slices['external_forces']] / self.force_scale
        )
        normalized[slices['opponent_relative_position']] = np.clip(
            normalized[slices['opponent_relative_position']] / self.relative_position_scale,
            -1.0,
            1.0,
        )
        normalized[slices['opponent_relative_velocity']] = np.clip(
            normalized[slices['opponent_relative_velocity']] / self.linear_velocity_scale,
            -1.0,
            1.0,
        )
        normalized[slices['opponent_keypoint_positions']] = np.clip(
            normalized[slices['opponent_keypoint_positions']] / self.keypoint_position_scale,
            -1.0,
            1.0,
        )
        normalized[slices['opponent_keypoint_velocities']] = np.clip(
            normalized[slices['opponent_keypoint_velocities']] / self.keypoint_velocity_scale,
            -1.0,
            1.0,
        )

        return normalized.astype(np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(HumanoidRobot.OBSERVATION_DIM,),
            dtype=np.float32,
        )
