"""Freeze non-target robot during impulse phase.

Resets the specified robot's state to its initial standing pose after
each action step, for a fixed number of steps. This matches the behavior
of RelativeImpulsePlugin's internal sim where non-target robots are
frozen during the impulse application.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BasePlugin
from envs.framework.context import SimContext


class FreezeRobotPlugin(BasePlugin):
    """Freeze a robot to its initial state for N action steps.

    After ``freeze_steps`` action steps, the plugin becomes inactive
    and the robot evolves naturally.
    """

    def __init__(
        self,
        robot_id: str = "robot_b",
        freeze_steps: int = 4,
    ):
        self.robot_id = robot_id
        self.freeze_steps = int(freeze_steps)
        self._initial_state: Optional[Dict[str, Any]] = None
        self._step_count = 0

    @property
    def name(self) -> str:
        return "freeze_robot"

    @property
    def require_mutator(self) -> bool:
        return True

    def to_blueprint(self) -> Dict[str, Any]:
        return {"robot_id": self.robot_id, "freeze_steps": self.freeze_steps}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "FreezeRobotPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._step_count = 0
        core_state = ctx.accessor.get_core_state()
        self._initial_state = {
            k: np.asarray(v).copy()
            for k, v in core_state[self.robot_id].items()
        }

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        if self._step_count >= self.freeze_steps or self._initial_state is None:
            return
        ctx.mutator.set_core_state({
            self.robot_id: self._initial_state,
        })

    def on_post_action_step(self, ctx: SimContext) -> None:
        if self._step_count < self.freeze_steps:
            self._step_count += 1
