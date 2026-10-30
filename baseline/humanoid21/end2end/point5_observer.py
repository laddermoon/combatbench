"""Point5 measurement observer — per-step 5-point world positions.

Wraps ``point5.humanoid21.adapter.point5_from_view`` and exposes
15 scalar coordinates for offline strike scoring.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from baseline.humanoid21.hierarchy.p5_world import (
    f1_from_state, p5_to_world)
from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from point5.humanoid21.adapter import point5_from_view


class Point5Observer(BaseObserverPlugin):
    """Outputs p5_0..14 (head/hr/hl/fr/fl, xyz per point)."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._pos = np.zeros(15, dtype=np.float64)
        self._yaw = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._pos = np.zeros(15, dtype=np.float64)
        self._yaw = 0.0
        self._update(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._update(ctx)

    def _update(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state([self.agent_id])
        st = point5_from_view(ds[self.agent_id], self.agent_id,
                              with_vel=False)
        # Adapter output is local-frame; publish world coordinates so
        # reward/diagnostic consumers stay frame-agnostic.
        self._pos = p5_to_world(st.pos, f1_from_state(st)).reshape(-1)
        # World azimuth (deg) of the torso-facing local frame — scored
        # against the plan's frame-yaw schedule by r_torso_yaw.
        self._yaw = float(st.metadata["frame_yaw_deg"])

    def get_output(self) -> Dict[str, float]:
        out = {f"p5_{i}": float(v) for i, v in enumerate(self._pos)}
        out["p5_yaw"] = self._yaw
        return out

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "Point5Observer":
        return cls(**config)
