"""Uncertainty-floor raise variant of `standup_floor04`.

Twin lever of explore_factor: ef widens only the rollout-side sampling
sigma (σ_eff = σ·exp(e·ln3), trained σ untouched); the uncertainty
floor hinge instead pushes the *trained* policy's entropy to stay
>= floor during optimization.  ef=0.5 confirmed positive on all three
seeds — this arm raises floor 0.4 -> 0.5 to test whether retaining
more policy-side entropy produces the same kind of gain, or whether
the benefit is specific to rollout-side widening.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04UFloor05(StandupFloor04):
    name = "standup_floor04_ufloor05"

    uncertainty_floor: float = 0.5  # base is 0.4


EXPERIMENT_CLASS = StandupFloor04UFloor05
