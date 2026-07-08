"""Termination plugin for 4-stage standup training.

Terminates episode early ONLY when:
- **Give-up**: robot stays low for ``stagnation_steps`` consecutive steps.

Success is NOT terminated early — the episode runs full ``max_steps`` so the
robot practices maintaining Stage 4 balance.  The terminal success bonus is
awarded by the experiment's ``extract_rewards`` based on Stage 4 achievement.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BasePlugin, SimContext


class Standup4StageTerminationPlugin(BasePlugin):
    """Early termination for 4-stage standup-from-fall training."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        stagnation_height: float = 0.25,
        stagnation_steps: int = 150,
    ) -> None:
        self.agent_id = str(agent_id)
        self.stagnation_height = float(stagnation_height)
        self.stagnation_steps = max(1, int(stagnation_steps))
        self._stagnation_streak = 0

    @property
    def name(self) -> str:
        return f"{self.agent_id}_standup_4stage_termination"

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "stagnation_height": self.stagnation_height,
            "stagnation_steps": self.stagnation_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "Standup4StageTerminationPlugin":
        return cls(**config)

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._stagnation_streak = 0

    def on_post_action_step(self, ctx: SimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        height = float(core_state["root_pos"][2])

        # Stagnation: stuck on the ground
        if height < self.stagnation_height:
            self._stagnation_streak += 1
            if self._stagnation_streak >= self.stagnation_steps:
                ctx.request_termination("standup_stagnation")
        else:
            self._stagnation_streak = 0
