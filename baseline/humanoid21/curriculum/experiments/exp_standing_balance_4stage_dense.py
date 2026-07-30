"""4-stage standing-balance — Dense reward (γ_s → ∞).

r_t = (1-γ)·φ(t)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_4stage_base import (
    StandingBalance4StageBase,
)


class StandingBalance4StageDense(StandingBalance4StageBase):
    name = "standing_balance_4stage_dense"
    reward_mode = "dense"
    max_updates = 5000


EXPERIMENT = StandingBalance4StageDense()
