"""A/B variant of `standup_floor04`: scheduled uncertainty floor.

  standup_floor04           : uncertainty_floor=0.4 constant — the hinge
                              relu(floor - U) engages from the start and
                              never releases (U drifts to ~0.31 equilibrium)
  standup_floor04_floorsched: floor 0.4 for updates < FLOOR_DROP_UPDATE,
                              then 0.25 — keep exploration pressure during
                              the approach phase, relax it in the
                              last-mile/precision phase (this run)

Motivation: approach-phase analysis shows σ declines monotonically and the
floor hinge stays engaged the whole time.  Hypothesis: high floor early
aids discovery; lower floor late lets σ collapse for the precise motor
control the stage-4 top demands.

``uncertainty_floor`` is an ExplorationSpec field resolved via
``exploration(update)`` — NOT a CommonParams/PPOParams field, so it is
invisible to ``--param`` patches (the u250 ``--param`` attempt crashed the
first launch).  A scheduled floor therefore needs this hook.

Same seed=42 and all other params as `standup_floor04`.
"""
from __future__ import annotations

from baseline.framework.ppo.experiment import ExplorationSpec

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04FloorSched(StandupFloor04):
    name = "standup_floor04_floorsched"

    floor_drop_update: int = 250
    floor_late: float = 0.25

    def exploration(self, update: int) -> ExplorationSpec:
        floor = (
            self.uncertainty_floor
            if update < self.floor_drop_update
            else self.floor_late
        )
        return ExplorationSpec(
            uncertainty_floor=floor,
            uncertainty_coef=self.uncertainty_coef,
        )


EXPERIMENT_CLASS = StandupFloor04FloorSched
