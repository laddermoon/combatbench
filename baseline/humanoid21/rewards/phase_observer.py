"""Phase observer plugin for staged reward schemes.

Determines whether the robot is in "struggle" or "stability" phase based on
two physical metrics:

  1. **Uprightness** — cos(torso tilt angle).  1.0 = perfectly upright.
  2. **Height** — torso root z-position above ground.

Phase transitions use two independent hysteresis mechanisms:

  1. **Threshold hysteresis** (Schmitt trigger).  Entering struggle and
     returning to stability use *different* thresholds, leaving a deadband
     in which the current phase is simply held.  This is what kills the
     dominant noise source: a robot drifting slowly across a single
     threshold would otherwise emit a stream of 1-3 step phase runs, each
     carrying a spurious +/-1.0 terminal reward.
  2. **Step hysteresis**.  A configurable number of consecutive steps must
     satisfy the target phase condition before the phase actually switches.
     This filters single-frame spikes.

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


# Struggle entry (falling out of stability) — 25° tilt or torso z < 1.15m.
# Standing torso height is ~1.28m, so 1.15m gives ~0.13m margin.
DEFAULT_UPRIGHTNESS_THRESHOLD = 0.906  # cos(25°) ≈ 0.906
DEFAULT_HEIGHT_THRESHOLD = 1.15        # torso z < 1.15m → struggle (standing ~1.28m)
# Stability recovery (climbing back out of struggle) — stricter, creating a
# deadband of 15°..25° tilt and 1.15..1.20m height where the phase is held.
DEFAULT_STABILITY_UPRIGHTNESS_THRESHOLD = 0.966  # cos(15°) ≈ 0.966
DEFAULT_STABILITY_HEIGHT_THRESHOLD = 1.2
# Hysteresis: need this many consecutive steps to confirm a transition
DEFAULT_STABLE_CONFIRM_STEPS = 5       # 5 consecutive stable steps → stability
DEFAULT_STRUGGLE_CONFIRM_STEPS = 1     # immediate: any single struggle step → struggle


class PhaseObserver(BaseObserverPlugin):
    """Per-step phase determination based on uprightness and height.

    Phase logic (Schmitt trigger + step hysteresis):
      - While in **stability**, drop to struggle if
        ``uprightness < uprightness_threshold`` OR ``height < height_threshold``.
      - While in **struggle**, return to stability only if
        ``uprightness >= stability_uprightness_threshold`` AND
        ``height >= stability_height_threshold``.
      - In between the two threshold sets the current phase is held, so
        boundary dwelling produces no transitions at all.
      - On top of that, the candidate phase must be sustained for
        ``confirm_steps`` consecutive steps before the phase transitions.

    The struggle-entry thresholds should be lenient enough that normal
    walking stays in stability phase.
    """

    PHASE_STRUGGLE = "struggle"
    PHASE_STABILITY = "stability"

    def __init__(
        self,
        agent_id: str = "robot_a",
        uprightness_threshold: float = DEFAULT_UPRIGHTNESS_THRESHOLD,
        height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
        stability_uprightness_threshold: float = DEFAULT_STABILITY_UPRIGHTNESS_THRESHOLD,
        stability_height_threshold: float = DEFAULT_STABILITY_HEIGHT_THRESHOLD,
        stable_confirm_steps: int = DEFAULT_STABLE_CONFIRM_STEPS,
        struggle_confirm_steps: int = DEFAULT_STRUGGLE_CONFIRM_STEPS,
    ) -> None:
        self.agent_id = agent_id
        self.uprightness_threshold = float(uprightness_threshold)
        self.height_threshold = float(height_threshold)
        self.stability_uprightness_threshold = max(
            float(stability_uprightness_threshold), float(uprightness_threshold)
        )
        self.stability_height_threshold = max(
            float(stability_height_threshold), float(height_threshold)
        )
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

        # Determine raw condition using the threshold set for the *opposite*
        # phase, so values inside the deadband hold the current phase.
        if self._phase == self.PHASE_STABILITY:
            leaving = (
                upr < self.uprightness_threshold
                or self._height < self.height_threshold
            )
            raw_phase = self.PHASE_STRUGGLE if leaving else self.PHASE_STABILITY
        else:
            recovered = (
                upr >= self.stability_uprightness_threshold
                and self._height >= self.stability_height_threshold
            )
            raw_phase = self.PHASE_STABILITY if recovered else self.PHASE_STRUGGLE

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
            "stability_uprightness_threshold": self.stability_uprightness_threshold,
            "stability_height_threshold": self.stability_height_threshold,
            "stable_confirm_steps": self.stable_confirm_steps,
            "struggle_confirm_steps": self.struggle_confirm_steps,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PhaseObserver":
        return cls(**config)
