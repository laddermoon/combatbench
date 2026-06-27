"""Minimal observer plugin that outputs per-step root height and uprightness.

Used by standup_v2 experiment to provide a dense height-based reward signal.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class HeightObserver(BaseObserverPlugin):
    """Outputs root height and uprightness per step."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = str(agent_id)
        self._height = 0.0
        self._uprightness = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._height = 0.0
        self._uprightness = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        self._height = float(core_state["root_pos"][2])
        self._uprightness = float(
            np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0]
        )

    def get_output(self) -> Dict[str, float]:
        return {
            "height": self._height,
            "uprightness": self._uprightness,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "HeightObserver":
        return cls(**config)
