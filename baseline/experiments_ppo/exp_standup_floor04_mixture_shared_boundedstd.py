"""Policy A/B variant of `standup_floor04`: shared bounded-σ mixture.

  standup_floor04_mixture_shared            : shared σ = exp(log_std),
                                              unbounded, ef = σ·3^e
  standup_floor04_mixture_shared_boundedstd : same shared (K·D,) σ but
                                              through the bounded sigmoid
                                              map σ(v)∈(0.05, 2.0) with
                                              additive ef σ(v+αe)
                                              (this run)

A/B against `standup_floor04_mixture_shared` isolates the bounded-σ axis
(bounded map + v+αe exploration); A/B against `standup_floor04_boundedstd`
isolates mixture structure under the same bounded semantics.

NOTE: U is the marginal Rényi-2 width (L2 metric), NOT the peak metric —
the same uncertainty_floor=0.4 is a DIFFERENT-strength constraint here.

Watch `policy_stats`: `component_weight_k` / `component_overlap`,
`std_position_mean` / `std_*_saturation_frac` (sigmoid position),
`effective_uncertainty` (U under the explored σ).
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04MixtureSharedBoundedStd(StandupFloor04):
    name = "standup_floor04_mixture_shared_boundedstd"

    actor_blueprint = "init_policy_shared_mixture_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04MixtureSharedBoundedStd
