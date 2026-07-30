"""Wall-contact observer plugin for humanoid21.

Outputs a per-step scalar ``wall_contact`` (1.0 if any non-ground body
contact exists, 0.0 otherwise).  Used by experiments to penalise leaning
against walls / arena boundaries.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class WallContactObserver(BaseObserverPlugin):
    """Detects robot contact with any non-ground environment geom (walls, ceiling)."""

    def __init__(self, agent_id: str = "robot_a", force_threshold: float = 1.0):
        self.agent_id = agent_id
        self.force_threshold = float(force_threshold)
        self._ground_geom_name: str | None = None
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        static_data = ctx.accessor.get_static_data()
        self._ground_geom_name = static_data.get("ground_geom_name", "ground")
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state(["contacts"])
        cv = derived_state.get("contacts")
        if cv is None or cv["ncon"] == 0:
            self._output = 0.0
            return

        static_data = ctx.accessor.get_static_data()
        geom_id_to_name = static_data.get("geom_id_to_name", {})
        ground = self._ground_geom_name or "ground"
        robot_aff = 1 if self.agent_id == "robot_a" else 2

        aff1 = cv["aff1"]
        aff2 = cv["aff2"]
        geom1 = cv["geom1"]
        geom2 = cv["geom2"]
        force_mag = cv["force_mag"]

        for i in range(cv["ncon"]):
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), "")
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), "")
            else:
                continue

            if geom_env == ground:
                continue
            if float(force_mag[i]) < self.force_threshold:
                continue

            self._output = 1.0
            return

        self._output = 0.0

    def get_output(self) -> float:
        return self._output

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "force_threshold": self.force_threshold}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "WallContactObserver":
        return cls(**config)
