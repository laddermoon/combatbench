"""Per-step strike-state observer — mirrors StrikeSettlementPlugin.

The world plugin is the single source of truth for the strike state
machine; this observer simply surfaces the metrics it writes as a
per-step dict for trajectory building and eval:

    phase         0=PRE_HIT 1=SETTLE 2=RETRACT
    locked_hand   0=none 1=left 2=right
    step_damage   locked-hand damage accumulated during this action step
    damage_total  cumulative locked-hand damage (Q)
    retract_done  1 once the retract combo completed
    d_home        locked hand → own torso distance (0 before lock)
    d_opp         locked hand → opp {head,torso} min dist (0 before lock)
    gates         retract-gate bitmask (bit0 upright, bit1 home, bit2 sep)

For the non-learner side (no settlement plugin attached) all fields
stay 0 — the experiment only reads the learner's observer anyway.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StrikeStateObserver(BaseObserverPlugin):
    """Reads ``<agent_id>_strike_*`` metrics written each step."""

    _FIELDS = (
        "phase", "locked_hand", "step_damage", "damage_total",
        "retract_done", "d_home", "d_opp", "gates",
    )

    def __init__(self, agent_id: str = "robot_a") -> None:
        if agent_id not in ("robot_a", "robot_b"):
            raise ValueError(f"agent_id must be robot_a/robot_b, got {agent_id!r}")
        self.agent_id = agent_id
        self._output: Dict[str, float] = {
            f: 0.0 for f in self._FIELDS}

    def _k(self, field: str) -> str:
        return f"{self.agent_id}_strike_{field}"

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = {f: 0.0 for f in self._FIELDS}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        m = ctx.metrics
        self._output = {
            f: float(m.get(self._k(f), 0.0)) for f in self._FIELDS}

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StrikeStateObserver":
        return cls(**config)
