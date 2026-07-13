"""Rollover-only rewarder for ablation: Delta vs PBRS.

Computes a single ``rollover_score`` from torso orientation:
  f_down = -x_world_z  (same as Standup4StageRewarder Stage 0)
  rollover_score = clip((f_down + 1) / 2, 0, 1)

  face up  -> 0.0
  face down -> 1.0

No stage machinery, no height, no contacts.  Pure orientation signal.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class RolloverRewarder(BaseObserverPlugin):
    """Single-signal rollover potential for Delta-vs-PBRS ablation."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._potential: float = 0.0
        self._f_down: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._potential = 0.0
        self._f_down = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        static_data = ctx.accessor.get_static_data()[self.agent_id]

        torso_body_name = static_data["keypoint_body_names"]["torso"]
        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(
            torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z

        self._f_down = float(f_down)
        self._potential = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))

    def get_output(self) -> Dict[str, float]:
        return {
            "potential": self._potential,
            "f_down": self._f_down,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "RolloverRewarder":
        return cls(**config)
