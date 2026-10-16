"""MoveTrackObserver — raw kinematics recorder for movement training.

Records ONLY raw data per action step — no scoring, no rewards:

    root_x, root_y   world-frame torso position (m)
    yaw              torso heading (rad, wrapped)
    h_root           torso height (m)

Rewards are computed OFFLINE in the experiment's trajectory builder
(see exp_hier_s3_move_mbs.py): each command segment is evaluated as a
whole — net displacement / net yaw change over the segment vs the
commanded (vx*, vy*, w*) — because per-step velocity signals are too
noisy to reward directly.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework.observer_plugin import BaseObserverPlugin


class MoveTrackObserver(BaseObserverPlugin):
    """Per-step kinematics recorder for one agent."""

    def __init__(self, agent_id: str = "robot_a") -> None:
        self.agent_id = agent_id
        self._root_x = 0.0
        self._root_y = 0.0
        self._yaw = 0.0
        self._h = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_yaw(root_rot: np.ndarray) -> float:
        """root_rot quaternion [w,x,y,z] → heading yaw (rad)."""
        rot = R.from_quat([root_rot[1], root_rot[2], root_rot[3], root_rot[0]])
        fwd = rot.apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(fwd[1], fwd[0]))

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx) -> None:
        self._root_x = self._root_y = self._yaw = self._h = 0.0

    def on_post_action_step(self, ctx) -> None:
        core = ctx.accessor.get_core_state()[self.agent_id]
        pos = np.asarray(core["root_pos"], dtype=np.float64)
        self._root_x = float(pos[0])
        self._root_y = float(pos[1])
        self._h = float(pos[2])
        self._yaw = self._extract_yaw(
            np.asarray(core["root_rot"], dtype=np.float64))

    # ------------------------------------------------------------------
    def get_output(self) -> Dict[str, Any]:
        return {
            "root_x": self._root_x,
            "root_y": self._root_y,
            "yaw": self._yaw,
            "h_root": self._h,
        }
