"""Policy A/B variant of `standup_floor04`: state-σ bounded mixture.

  standup_floor04_mixture_shared_boundedstd   : shared v(K·D,) parameter,
                                                bounded map σ(v),
                                                ef = σ(v+αe)
  standup_floor04_mixture_boundedstd_statesig : same bounded mixture but
                                                v comes from a per-state
                                                head block v(s)
                                                (this run)

A/B against `standup_floor04_mixture_shared_boundedstd` isolates the
σ-source axis (state-dependent vs shared σ) under identical bounded-σ
semantics; A/B against `standup_floor04_boundedstd_statesig` isolates
mixture structure under the same state-bounded σ semantics.

NOTE: U is the marginal Rényi-2 width (L2 metric), NOT the peak metric —
the same uncertainty_floor=0.4 is a DIFFERENT-strength constraint here.

Watch `policy_stats`: `component_weight_k` / `component_overlap`,
`std_position_mean` / `std_*_saturation_frac` (sigmoid position),
`sigma_state_std` (state-dependence of σ), `effective_uncertainty`.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04MixtureBoundedStdStateSig(StandupFloor04):
    name = "standup_floor04_mixture_boundedstd_statesig"

    actor_blueprint = "init_policy_state_mixture_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04MixtureBoundedStdStateSig
