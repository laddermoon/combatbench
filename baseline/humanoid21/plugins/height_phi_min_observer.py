"""Observer plugin that outputs per-step height, uprightness, and potential φ_A.

Variant of HeightPhiObserver using min(height_ratio, uprightness) instead of
the product form. This avoids over-suppressing φ for legitimate non-upright
postures (bending, crouching) while still detecting "tall but tilted" states.

φ_A = min(height / standing_height, uprightness)

Used by phi_min experiment variants for comparison with original φ.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class HeightPhiMinObserver(BaseObserverPlugin):
    """Outputs root height, uprightness, and potential φ_A per step.

    φ_A = min(height / standing_height, uprightness), where standing_height
    defaults to 1.28 m (perfect standing height for the humanoid model).

    Compared to HeightPhiObserver (product form), this preserves more shaping
    signal for legitimate bent/crouched postures while still gating on
    uprightness when the torso tilts significantly.
    """

    def __init__(self, agent_id: str = "robot_a", standing_height: float = 1.28):
        self.agent_id = str(agent_id)
        self.standing_height = float(standing_height)
        self._height = 0.0
        self._uprightness = 0.0
        self._initial_phi = 0.0

    def _compute_phi(self, height: float, uprightness: float) -> float:
        height_ratio = height / self.standing_height
        return min(height_ratio, uprightness)

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
    def from_blueprint(cls, config: Dict[str, Any]) -> "HeightPhiMinObserver":
        return cls(**config)
