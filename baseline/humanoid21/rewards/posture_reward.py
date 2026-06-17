"""Posture diagnostic observer plugin for ``humanoid21``.

Records four per-step posture metrics during simulation:

  1. **joint_deviation** — mean absolute deviation of normalised joint
     positions from the episode-start reference pose (21-DOF average).
  2. **joint_vel** — mean absolute normalised joint angular velocity
     across all 21 joints.
  3. **torso_tilt** — angle (radians) between the torso up-axis and the
     world vertical (0 = perfectly upright, π/2 ≈ horizontal).
  4. **foot_height** — maximum of left / right foot z-position above
     ground (m).

All four arrays have shape ``(T,)`` where *T* is the number of action
steps in the episode.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class PostureRewarder(BaseObserverPlugin):
    """Per-step posture diagnostics (joint deviation, velocity, tilt, foot height)."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._ref_joint_pos: Optional[np.ndarray] = None
        self._joint_deviation: List[float] = []
        self._joint_vel: List[float] = []
        self._torso_tilt: List[float] = []
        self._foot_height: List[float] = []

    # -- lifecycle hooks --------------------------------------------------

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        self._ref_joint_pos = np.asarray(
            core_state["joint_pos_norm"], dtype=np.float32
        ).copy()
        self._joint_deviation.clear()
        self._joint_vel.clear()
        self._torso_tilt.clear()
        self._foot_height.clear()

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state()[self.agent_id]
        static_data = ctx.accessor.get_static_data()[self.agent_id]

        # 1. Joint position deviation from reference standing pose
        joint_pos = np.asarray(core_state["joint_pos_norm"], dtype=np.float32)
        joint_dev = float(np.mean(np.abs(joint_pos - self._ref_joint_pos)))

        # 2. Joint angular velocity (mean absolute)
        joint_vel = np.asarray(core_state["joint_vel_norm"], dtype=np.float32)
        joint_vel_mag = float(np.mean(np.abs(joint_vel)))

        # 3. Torso tilt angle from vertical
        #    uprightness = cos(tilt), so tilt = arccos(uprightness)
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        torso_tilt = float(np.arccos(np.clip(uprightness, -1.0, 1.0)))

        # 4. Foot height above ground (max of both feet)
        body_xpos = derived_state["body_xpos"]
        keypoint_names = static_data["keypoint_body_names"]
        left_name = keypoint_names["foot_left"]
        right_name = keypoint_names["foot_right"]
        left_h = float(body_xpos[left_name][2])
        right_h = float(body_xpos[right_name][2])
        foot_height = max(left_h, right_h)

        self._joint_deviation.append(joint_dev)
        self._joint_vel.append(joint_vel_mag)
        self._torso_tilt.append(torso_tilt)
        self._foot_height.append(foot_height)

    # -- output -----------------------------------------------------------

    def get_output(self) -> Dict[str, np.ndarray]:
        return {
            "joint_deviation": np.asarray(self._joint_deviation, dtype=np.float32),
            "joint_vel": np.asarray(self._joint_vel, dtype=np.float32),
            "torso_tilt": np.asarray(self._torso_tilt, dtype=np.float32),
            "foot_height": np.asarray(self._foot_height, dtype=np.float32),
        }

    # -- blueprint serialization -----------------------------------------

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PostureRewarder":
        return cls(**config)
