"""Policy A/B variant of `standup_floor04`: state-dependent bounded-σ actor.

  standup_floor04                  : TruncatedNormalPolicy — σ = exp(log_std),
                                     global nn.Parameter shared by all states
  standup_floor04_statesig         : StateTruncatedNormalPolicy — σ = f(obs),
                                     unbounded
  standup_floor04_boundedstd       : BoundedStdTruncatedNormalPolicy — shared
                                     σ bounded via sigmoid on log σ
  standup_floor04_boundedstd_statesig : StateBoundedStdTruncatedNormalPolicy —
                                     v = f(obs), σ = exp(r_min + Δr·sigmoid(v))
                                     with σ ∈ (0.05, 2.0); explore_factor
                                     shifts the raw sigmoid input additively
                                     (v + αe) per state (this run)

Same seed=42 and all other params as `standup_floor04`.  Three-way A/B:
vs `standup_floor04` isolates bounded parameterization + state σ;
vs `standup_floor04_boundedstd` isolates state dependence alone;
vs `standup_floor04_statesig` isolates bounding alone.

The σ head half is zero-init with bias = v_init ≈ 0.1644, so at step 0
σ(obs) ≡ e⁻¹ everywhere — the policy starts equivalent to the shared
bounded variant given equal mean weights.

Watch `policy_stats`: `sigma_state_std` (cross-state σ spread — ~0
means the head stays constant/degenerate), `raw_std_min/max` (v range,
sigmoid saturation), `std_position_mean`, `std_lower/upper_saturation_frac`,
`log_std_sensitivity`, `exploration_sensitivity`, `effective_uncertainty`.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04BoundedStdStateSig(StandupFloor04):
    name = "standup_floor04_boundedstd_statesig"

    actor_blueprint = "init_policy_state_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04BoundedStdStateSig
