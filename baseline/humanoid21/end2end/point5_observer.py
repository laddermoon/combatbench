"""Point5 measurement observer — per-step 5-point world positions.

Wraps ``mocap.point5.humanoid21.adapter.point5_from_view`` and exposes
the flat 15-float vector plus the per-point slices the strike reward
consumes offline in ``build_trajectories``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from mocap.point5.humanoid21.adapter import point5_from_view


class Point5Observer(BaseObserverPlugin):
    """Outputs: ``pos`` (15,) world xyz x 5 points (head/hr/hl/fr/fl)."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self._pos = np.zeros(15, dtype=np.float64)

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._pos = np.zeros(15, dtype=np.float64)
        self._update(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._update(ctx)

    def _update(self, ctx: ReadOnlySimContext) -> None:
        ds = ctx.accessor.get_derived_state([self.agent_id])
        try:
            st = point5_from_view(ds[self.agent_id], self.agent_id,
                                  with_vel=False)
            self._pos = st.pos.reshape(-1)
        except Exception:
            pass

    def get_output(self) -> Dict[str, float]:
        return {f"p5_{i}": float(v) for i, v in enumerate(self._pos)}

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "Point5Observer":
        return cls(**config)
