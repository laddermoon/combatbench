"""A/B variant of `standup`: uncertainty floor ENABLED, strong coef.

  standup          : floor=0.0            (σ free to collapse)
  standup_floor    : floor=0.3, coef=0.01 (inert — floor grad ~2e-4 vs
                     policy grad ~4, four orders too weak; U collapsed
                     straight through the floor to 0.166 identical to
                     control)
  standup_floor_c1 : floor=0.3, coef=1.0  (this run — floor grad should
                     be ~100× stronger, same order as policy grad)

Same seed=42 and all other params as `standup`.
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupFloorC1(Standup):
    name = "standup_floor_c1"

    uncertainty_floor: float = 0.3
    uncertainty_coef: float = 1.0


EXPERIMENT_CLASS = StandupFloorC1
