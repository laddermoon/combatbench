"""3-stage standing-balance — Generalized shaping γ_s = 1.1.

r_t = 1.1·φ(t) - φ(t-1)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_3stage_base import (
    StandingBalance3StageBase,
)


class StandingBalance3StageGS11(StandingBalance3StageBase):
    name = "standing_balance_3stage_gs11"
    reward_mode = "gs_1p1"


EXPERIMENT = StandingBalance3StageGS11()
