"""Observer plugin exposing the StandingTriggeredForcePlugin state machine.

Outputs per step (read from ``ctx.metrics`` written by the plugin's
``on_pre_action_step``, same step):

  - push_phase:        0=WAIT_STAND 1=DELAY 2=PUSHING 3=OBSERVE
  - push_window_clean: inside PUSHING/OBSERVE with no fall recorded yet
  - push_survived:     one-step pulse at OBSERVE exit when the whole
                       PUSHING+OBSERVE window completed without a fall

Motivation: the push fall signal is penalty-only (r_fall onset, wall
lean) — surviving a push window has no direct positive gradient, the
reward arrives diffusely through continued gait income.  These flags
let the experiment pay a dense clean-window bonus (and enable
push-phase-aware analysis in dumps).
"""
from __future__ import annotations

from typing import Any, Dict

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class PushStateObserver(BaseObserverPlugin):
    """Per-agent mirror of the push state machine."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = str(agent_id)
        self._phase: int = 0
        self._clean: bool = False
        self._survived: bool = False

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._phase = 0
        self._clean = False
        self._survived = False

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        m = ctx.metrics
        self._phase = int(m.get(f"{self.agent_id}_push_phase", 0))
        self._clean = bool(
            m.get(f"{self.agent_id}_push_window_clean", False)
        )
        self._survived = bool(
            m.get(f"{self.agent_id}_push_survived", False)
        )

    def get_output(self) -> Dict[str, Any]:
        return {
            "push_phase": self._phase,
            "push_window_clean": self._clean,
            "push_survived": self._survived,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PushStateObserver":
        return cls(**config)
