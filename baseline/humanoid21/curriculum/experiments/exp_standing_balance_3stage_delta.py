"""3-stage standing-balance — Delta reward (γ_s = 1).

r_t = φ(t) - φ(t-1)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_3stage_base import (
    StandingBalance3StageBase,
)


class StandingBalance3StageDelta(StandingBalance3StageBase):
    name = "standing_balance_3stage_delta"
    reward_mode = "delta"


EXPERIMENT = StandingBalance3StageDelta()
