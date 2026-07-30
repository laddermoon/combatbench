"""4-stage standing-balance — GS-1.1 reward (γ_s = 1.1).

r_t = 1.1·φ(t) - φ(t-1)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_4stage_base import (
    StandingBalance4StageBase,
)


class StandingBalance4StageGS11(StandingBalance4StageBase):
    name = "standing_balance_4stage_gs11"
    reward_mode = "gs_1p1"


EXPERIMENT = StandingBalance4StageGS11()
