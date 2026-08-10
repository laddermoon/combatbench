"""3-stage standing-balance — Dense reward.

r_t = (1-γ)·φ(t)
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.standing_balance_3stage_base import (
    StandingBalance3StageBase,
)


class StandingBalance3StageDense(StandingBalance3StageBase):
    name = "standing_balance_3stage_dense"
    reward_mode = "dense"


EXPERIMENT = StandingBalance3StageDense()
