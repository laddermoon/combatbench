from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RootPose:
    joint_name: str
    position: np.ndarray
    orientation: np.ndarray


class BaseResetter:
    name = "base"

    def reset(
        self,
        env: Any,
        rng: np.random.Generator,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"type": self.name}


class SymmetricStandResetter(BaseResetter):
    name = "symmetric_stand"

    def __init__(
        self,
        initial_distance: Optional[float] = None,
        root_height: Optional[float] = None,
        lateral_offset: float = 0.0,
        yaw_jitter_deg: float = 0.0,
    ):
        self.initial_distance = initial_distance
        self.root_height = root_height
        self.lateral_offset = float(lateral_offset)
        self.yaw_jitter_deg = float(yaw_jitter_deg)

    def reset(
        self,
        env: Any,
        rng: np.random.Generator,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        runtime_options = {} if options is None else dict(options)
        initial_distance = float(
            runtime_options.get(
                "initial_distance",
                env.initial_distance if self.initial_distance is None else self.initial_distance,
            )
        )
        root_height = float(
            runtime_options.get(
                "root_height",
                env.DEFAULT_ROOT_HEIGHT if self.root_height is None else self.root_height,
            )
        )
        lateral_offset = float(runtime_options.get("lateral_offset", self.lateral_offset))
        yaw_jitter_deg = float(runtime_options.get("yaw_jitter_deg", self.yaw_jitter_deg))
        env.initial_distance = initial_distance
        root_poses = env.build_symmetric_root_poses(
            initial_distance=initial_distance,
            root_height=root_height,
            lateral_offset=lateral_offset,
            yaw_jitter_deg=yaw_jitter_deg,
        )
        env.apply_root_poses(root_poses)
        return {
            "type": self.name,
            "initial_distance": initial_distance,
            "root_height": root_height,
            "lateral_offset": lateral_offset,
            "yaw_jitter_deg": yaw_jitter_deg,
        }


class RandomizedSymmetricStandResetter(BaseResetter):
    name = "randomized_symmetric_stand"

    def __init__(
        self,
        distance_range: tuple[float, float] = (1.5, 2.5),
        lateral_offset_range: tuple[float, float] = (-0.15, 0.15),
        yaw_jitter_deg_range: tuple[float, float] = (-10.0, 10.0),
        root_height_jitter_range: tuple[float, float] = (-0.02, 0.02),
    ):
        self.distance_range = tuple(float(value) for value in distance_range)
        self.lateral_offset_range = tuple(float(value) for value in lateral_offset_range)
        self.yaw_jitter_deg_range = tuple(float(value) for value in yaw_jitter_deg_range)
        self.root_height_jitter_range = tuple(float(value) for value in root_height_jitter_range)

    def _sample_uniform(self, rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if np.isclose(low, high):
            return float(low)
        return float(rng.uniform(low, high))

    def reset(
        self,
        env: Any,
        rng: np.random.Generator,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        runtime_options = {} if options is None else dict(options)
        initial_distance = float(
            runtime_options.get(
                "initial_distance",
                self._sample_uniform(rng, self.distance_range),
            )
        )
        lateral_offset = float(
            runtime_options.get(
                "lateral_offset",
                self._sample_uniform(rng, self.lateral_offset_range),
            )
        )
        yaw_jitter_deg = float(
            runtime_options.get(
                "yaw_jitter_deg",
                self._sample_uniform(rng, self.yaw_jitter_deg_range),
            )
        )
        root_height = float(
            runtime_options.get(
                "root_height",
                env.DEFAULT_ROOT_HEIGHT + self._sample_uniform(rng, self.root_height_jitter_range),
            )
        )
        env.initial_distance = initial_distance
        root_poses = env.build_symmetric_root_poses(
            initial_distance=initial_distance,
            root_height=root_height,
            lateral_offset=lateral_offset,
            yaw_jitter_deg=yaw_jitter_deg,
        )
        env.apply_root_poses(root_poses)
        return {
            "type": self.name,
            "initial_distance": initial_distance,
            "root_height": root_height,
            "lateral_offset": lateral_offset,
            "yaw_jitter_deg": yaw_jitter_deg,
        }
