"""A/B variant of `standup_floor04`: state bounded-σ actor + ef=0.5.

  standup_floor04_ef                       : unbounded σ, ef=0.5 (σ×1.73)
  standup_floor04_boundedstd_ef05          : shared bounded-σ, ef=0.5
  standup_floor04_boundedstd_statesig_ef05 : state-dependent bounded-σ,
                                             ef=0.5 — v(obs) + α·0.5
                                             per state (this run)

NOTE: on bounded policies e is an additive shift on the raw sigmoid
input v, NOT a σ multiplier.  Per-state v means the same e produces
state-dependent σ widening; near-saturated states barely respond
(soft-bound decay), mid-range states get the calibrated response.

Same seed=42 and all other params as `standup_floor04`.

Watch `policy_stats.eff_std_mean` vs `std_mean` (the realized ef lift),
`sigma_state_std`, and `std_upper_saturation_frac` — ef=0.5 pushes
toward the upper bound, so saturation uptake is expected on states
whose v is already high.
"""
from __future__ import annotations

from .exp_standup_floor04_ef import StandupFloor04EF


class StandupFloor04BoundedStdStateSigEF05(StandupFloor04EF):
    name = "standup_floor04_boundedstd_statesig_ef05"

    actor_blueprint = (
        "init_policy_state_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = StandupFloor04BoundedStdStateSigEF05
