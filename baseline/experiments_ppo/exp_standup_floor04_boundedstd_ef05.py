"""A/B variant of `standup_floor04`: bounded-σ actor + explore_factor=0.5.

  standup_floor04_ef           : unbounded TruncatedNormalPolicy, ef=0.5
                                 (σ_eff = σ × exp(0.5·ln3) ≈ σ×1.73)
  standup_floor04_boundedstd_ef05 : BoundedStdTruncatedNormalPolicy,
                                 ef=0.5 — NOTE the different semantics:
                                 e shifts the raw sigmoid input
                                 additively (v + α·0.5), it does NOT
                                 multiply σ.  At the init point the local
                                 response is calibrated to match the
                                 legacy slope, but the response decays
                                 toward the bounds (σ stays in
                                 (0.05, 2.0) for all e).

Same seed=42 and all other params as `standup_floor04`.  Compares
against `standup_floor04_ef` (unbounded, multiplicative ef) and
`standup_floor04_boundedstd` (bounded, e=0) to isolate the ef effect
under the bounded parameterization.
"""
from __future__ import annotations

from .exp_standup_floor04_ef import StandupFloor04EF


class StandupFloor04BoundedStdEF05(StandupFloor04EF):
    name = "standup_floor04_boundedstd_ef05"

    actor_blueprint = "init_policy_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04BoundedStdEF05
