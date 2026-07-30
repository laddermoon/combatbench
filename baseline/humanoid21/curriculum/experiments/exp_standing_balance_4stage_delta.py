"""4-stage standing-balance — Delta reward (γ_s = 1).

r_t = φ(t) - φ(t-1)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_4stage_base import (
    StandingBalance4StageBase,
)


class StandingBalance4StageDelta(StandingBalance4StageBase):
    name = "standing_balance_4stage_delta"
    reward_mode = "delta"


EXPERIMENT = StandingBalance4StageDelta()
