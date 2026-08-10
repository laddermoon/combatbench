"""Observer plugin that outputs per-step height, uprightness, and potential φ.

Extends HeightObserver with:
  - Initial φ captured in on_pre_episode (before first action)
  - Pre-computed φ = uprightness * (height / standing_height)
  - initial_phi available in every frame's output for Delta reward calculation

Used by standing balance reward comparison experiments (ST-1 through ST-6).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class HeightPhiObserver(BaseObserverPlugin):
    """Outputs root height, uprightness, and potential φ per step.

    φ = uprightness * (height / standing_height), where standing_height
    defaults to 1.28 m (perfect standing height for the humanoid model).

    The initial φ is captured in on_pre_episode so that Delta reward
    can be computed as r[0] = φ(0) - initial_phi (≈ 0) instead of
    φ(0) - 0 (which would be a large spike since the robot starts standing).
    """

    def __init__(self, agent_id: str = "robot_a", standing_height: float = 1.28):
        self.agent_id = str(agent_id)
        self.standing_height = float(standing_height)
        self._height = 0.0
        self._uprightness = 0.0
        self._initial_phi = 0.0

    def _compute_phi(self, height: float, uprightness: float) -> float:
        return uprightness * (height / self.standing_height)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        self._height = float(core_state["root_pos"][2])
        self._uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )
        self._initial_phi = self._compute_phi(self._height, self._uprightness)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        self._height = float(core_state["root_pos"][2])
        self._uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

    def get_output(self) -> Dict[str, float]:
        phi = self._compute_phi(self._height, self._uprightness)
        return {
            "height": self._height,
            "uprightness": self._uprightness,
            "phi": phi,
            "initial_phi": self._initial_phi,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "standing_height": self.standing_height}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "HeightPhiObserver":
        return cls(**config)
