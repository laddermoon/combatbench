"""Net damage reward plugin for humanoid21.

Provides:
  * :class:`NetDamageRewarder` — Per-step net damage reward (damage dealt
    minus damage taken).

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class NetDamageRewarder(BaseObserverPlugin):
    """Per-step net damage reward for curriculum stage 3.

    Requires :class:`CombatScoringPlugin` attached to the runtime so that
    ``ctx.metrics["damage_taken_a"]`` / ``["damage_taken_b"]`` accumulate
    across phy steps. On each action step we read the current totals,
    diff against the previous-step snapshot, and emit::

        net = (opponent damage delta) - (self damage delta)

    Positive when the agent connects, negative when taking the hit.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self._self_key = f"damage_taken_{'a' if self.agent_id == 'robot_a' else 'b'}"
        self._opp_key = f"damage_taken_{'a' if self.opponent_id == 'robot_a' else 'b'}"
        self._prev_self: float = 0.0
        self._prev_opp: float = 0.0
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._prev_self = 0.0
        self._prev_opp = 0.0
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        cur_self = float(ctx.metrics.get(self._self_key, 0.0))
        cur_opp = float(ctx.metrics.get(self._opp_key, 0.0))
        delta_self = max(0.0, cur_self - self._prev_self)
        delta_opp = max(0.0, cur_opp - self._prev_opp)
        self._prev_self = cur_self
        self._prev_opp = cur_opp
        self._output = delta_opp - delta_self

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "NetDamageRewarder":
        return cls(**config)

