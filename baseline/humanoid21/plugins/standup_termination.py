"""Termination plugin for standup training.

Terminates episode early when:
- **Success**: robot height and uprightness exceed thresholds for ``success_hold_steps``.
- **Give-up**: robot stays low for ``stagnation_steps`` consecutive steps
  (avoids wasting rollout budget on hopeless states).

Reads physical state directly from ctx.accessor (no dependency on observer plugins).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BasePlugin, SimContext


class StandupTerminationPlugin(BasePlugin):
    """Early termination for standup-from-fall training."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        success_height: float = 0.60,
        success_uprightness: float = 0.70,
        success_hold_steps: int = 5,
        stagnation_height: float = 0.25,
        stagnation_steps: int = 150,
    ) -> None:
        self.agent_id = str(agent_id)
        self.success_height = float(success_height)
        self.success_uprightness = float(success_uprightness)
        self.success_hold_steps = max(1, int(success_hold_steps))
        self.stagnation_height = float(stagnation_height)
        self.stagnation_steps = max(1, int(stagnation_steps))
        self._success_streak = 0
        self._stagnation_streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standup_termination"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success_height": self.success_height,
            "success_uprightness": self.success_uprightness,
            "success_hold_steps": self.success_hold_steps,
            "stagnation_height": self.stagnation_height,
            "stagnation_steps": self.stagnation_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupTerminationPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._success_streak = 0
        self._stagnation_streak = 0

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        height = float(core_state["root_pos"][2])
        uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

        # Success: standing tall and upright
        if height >= self.success_height and uprightness >= self.success_uprightness:
            self._success_streak += 1
            self._stagnation_streak = 0
            if self._success_streak >= self.success_hold_steps:
                ctx.request_termination("standup_success")
        else:
            self._success_streak = 0

        # Stagnation: stuck on the ground
        if height < self.stagnation_height:
            self._stagnation_streak += 1
            if self._stagnation_streak >= self.stagnation_steps:
                ctx.request_termination("standup_stagnation")
        else:
            self._stagnation_streak = 0
