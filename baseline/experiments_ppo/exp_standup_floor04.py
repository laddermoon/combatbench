"""A/B variant of `standup`: uncertainty floor at 0.4, coef=1.0.

  standup          : floor=0.0            (σ free to collapse)
  standup_floor    : floor=0.3, coef=0.01 (inert — grad 4 orders weak)
  standup_floor_c1 : floor=0.3, coef=1.0  (bites; U decay bent but not
                     pinned — drifts below the floor toward an
                     equilibrium)
  standup_floor04  : floor=0.4, coef=1.0  (this run — higher floor to
                     probe how much U the task actually needs; 0.4 is
                     near init U≈0.46, so the hinge engages almost from
                     the start)

Same seed=42 and all other params as `standup`.
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupFloor04(Standup):
    name = "standup_floor04"

    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 1.0


EXPERIMENT_CLASS = StandupFloor04
