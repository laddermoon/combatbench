"""Phase observer plugin for staged reward schemes.

Determines whether the robot is in "struggle" or "stability" phase based on
two physical metrics:

  1. **Uprightness** — cos(torso tilt angle).  1.0 = perfectly upright.
  2. **Height** — torso root z-position above ground.

Phase transitions use hysteresis: a configurable number of consecutive
steps must satisfy the target phase condition before the phase actually
switches.  This prevents rapid flickering at boundary states.

Output (per-step dict):
  - ``phase``: "struggle" or "stability"
  - ``is_struggle``: bool (True if struggle)
  - ``is_stability``: bool (True if stability)
  - ``transition``: one of "none", "struggle_to_stability", "stability_to_struggle"
  - ``uprightness``: float
  - ``height``: float
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# Default thresholds — tuned so that normal walking does NOT trigger struggle.
# Uprightness: below this → struggle (cos(0.4 rad) ≈ 0.92, ~23° tilt)
# Height: below this → struggle (normal standing ~0.9m, walking dips to ~0.8m)
DEFAULT_UPRIGHTNESS_THRESHOLD = 0.85
DEFAULT_HEIGHT_THRESHOLD = 0.65
# Hysteresis: need this many consecutive steps to confirm a transition
DEFAULT_STABLE_CONFIRM_STEPS = 10
DEFAULT_STRUGGLE_CONFIRM_STEPS = 3


class PhaseObserver(BaseObserverPlugin):
    """Per-step phase determination based on uprightness and height.

    Phase logic:
      - **Struggle** if uprightness < threshold OR height < threshold.
      - **Stability** if both uprightness >= threshold AND height >= threshold.
      - Hysteresis: must sustain the new condition for ``confirm_steps``
        consecutive steps before the phase actually transitions.

    The thresholds should be lenient enough that normal walking stays in
    stability phase.
    """

    PHASE_STRUGGLE = "struggle"
    PHASE_STABILITY = "stability"

    def __init__(
        self,
        agent_id: str = "robot_a",
        uprightness_threshold: float = DEFAULT_UPRIGHTNESS_THRESHOLD,
        height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
        stable_confirm_steps: int = DEFAULT_STABLE_CONFIRM_STEPS,
        struggle_confirm_steps: int = DEFAULT_STRUGGLE_CONFIRM_STEPS,
    ) -> None:
        self.agent_id = agent_id
        self.uprightness_threshold = float(uprightness_threshold)
        self.height_threshold = float(height_threshold)
        self.stable_confirm_steps = int(stable_confirm_steps)
        self.struggle_confirm_steps = int(struggle_confirm_steps)

        # State
        self._phase: str = self.PHASE_STABILITY
        self._candidate_counter: int = 0
        self._candidate_phase: str = self.PHASE_STABILITY
        self._uprightness: float = 1.0
        self._height: float = 1.0
        self._transition: str = "none"

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._phase = self.PHASE_STABILITY
        self._candidate_counter = 0
        self._candidate_phase = self.PHASE_STABILITY
        self._uprightness = 1.0
        self._height = 1.0
        self._transition = "none"

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        cs = core_state.get(self.agent_id, core_state.get("robot_a", {}))
        derived_state = ctx.accessor.get_derived_state()

        # Uprightness
        upr_raw = derived_state.get(self.agent_id, derived_state.get("robot_a", {}))
        upr = float(np.asarray(upr_raw["uprightness"], dtype=np.float32).reshape(-1)[0])
        self._uprightness = upr

        # Height (root z-position)
        root_pos = cs.get("root_pos", [0.0, 0.0, 1.0])
        self._height = float(root_pos[2])

        # Determine raw condition
        is_struggle_raw = (
            upr < self.uprightness_threshold
            or self._height < self.height_threshold
        )
        raw_phase = self.PHASE_STRUGGLE if is_struggle_raw else self.PHASE_STABILITY

        # Hysteresis: count consecutive steps in the candidate phase
        if raw_phase == self._phase:
            # Same as current — reset candidate
            self._candidate_counter = 0
            self._candidate_phase = self._phase
        else:
            if raw_phase == self._candidate_phase:
                self._candidate_counter += 1
            else:
                self._candidate_phase = raw_phase
                self._candidate_counter = 1

        # Check if we should transition
        confirm = (
            self.stable_confirm_steps
            if raw_phase == self.PHASE_STABILITY
            else self.struggle_confirm_steps
        )
        if self._candidate_counter >= confirm and raw_phase != self._phase:
            old_phase = self._phase
            self._phase = raw_phase
            if old_phase == self.PHASE_STRUGGLE and raw_phase == self.PHASE_STABILITY:
                self._transition = "struggle_to_stability"
            elif old_phase == self.PHASE_STABILITY and raw_phase == self.PHASE_STRUGGLE:
                self._transition = "stability_to_struggle"
            else:
                self._transition = "none"
        else:
            self._transition = "none"

    def get_output(self) -> Dict[str, Any]:
        return {
            "phase": self._phase,
            "is_struggle": self._phase == self.PHASE_STRUGGLE,
            "is_stability": self._phase == self.PHASE_STABILITY,
            "transition": self._transition,
            "uprightness": self._uprightness,
            "height": self._height,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "uprightness_threshold": self.uprightness_threshold,
            "height_threshold": self.height_threshold,
            "stable_confirm_steps": self.stable_confirm_steps,
            "struggle_confirm_steps": self.struggle_confirm_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PhaseObserver":
        return cls(**config)
