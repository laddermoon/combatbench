"""Face-opponent observer plugin for humanoid21 end-to-end step 4.

Records the learning agent's torso forward direction (projected to XY
plane) per step.  The experiment uses this together with
ApproachVelocityRewarder's position data to compute the r_face reward
(facing_score × dist_gate).

Forward direction computation:
  1. Get torso body quaternion from derived_state["body_xquat"][torso_name]
  2. Rotate local x-axis [1, 0, 0] by this quaternion → 3D forward vector
  3. Project to XY plane: (forward_x, forward_y)

This is the same approach used in OpponentRelationRewarder.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class FaceOpponentObserver(BaseObserverPlugin):
    """Records torso forward direction (XY projection) per step.

    Outputs:
      forward_x, forward_y — torso forward unit vector in XY plane
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._forward_xy: np.ndarray = np.array([1.0, 0.0], dtype=np.float64)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._forward_xy = np.array([1.0, 0.0], dtype=np.float64)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state([self.agent_id])
        self_derived = derived_state.get(self.agent_id, {})
        static_data = ctx.accessor.get_static_data()
        agent_static = static_data.get(self.agent_id, {})

        torso_body_name = agent_static.get("keypoint_body_names", {}).get("torso")
        body_xquat = self_derived.get("body_xquat", {})

        forward_3d = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if torso_body_name and torso_body_name in body_xquat:
            q = np.asarray(body_xquat[torso_body_name], dtype=np.float64)
            # MuJoCo quat: [w, x, y, z] -> scipy: [x, y, z, w]
            torso_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            forward_3d = torso_rot.apply([1.0, 0.0, 0.0])
        else:
            # Fallback: use root quaternion
            core_state = ctx.accessor.get_core_state()
            self_state = core_state.get(self.agent_id, {})
            q = np.asarray(self_state.get("root_rot", [1, 0, 0, 0]), dtype=np.float64)
            norm = float(np.linalg.norm(q))
            if norm > 1e-8:
                q = q / norm
                torso_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                forward_3d = torso_rot.apply([1.0, 0.0, 0.0])

        # Project to XY plane and normalize
        fxy = forward_3d[:2]
        norm = float(np.linalg.norm(fxy))
        if norm > 1e-8:
            self._forward_xy = fxy / norm
        else:
            self._forward_xy = np.array([1.0, 0.0], dtype=np.float64)

    def get_output(self) -> Dict[str, float]:
        return {
            "forward_x": float(self._forward_xy[0]),
            "forward_y": float(self._forward_xy[1]),
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "FaceOpponentObserver":
        return cls(**config)
