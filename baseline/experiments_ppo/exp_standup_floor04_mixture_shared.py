"""Policy A/B variant of `standup_floor04`: shared-σ mixture actor.

  standup_floor04_mixture        : MixtureTruncatedNormalPolicy — K=3
                                   mixture, σ = f(obs) per component
  standup_floor04_mixture_shared : SharedMixtureTruncatedNormalPolicy —
                                   same mixture, but σ is a trainable
                                   (K·D,) parameter shared across states
                                   (this run)

A/B pair against the state-σ mixture — the ONLY difference is the σ
source (parameter vs head); weights, means, sampling, log_prob,
explore_factor scaling (σ·3^e), and the Rényi-2 uncertainty metric are
identical.  Isolates the value of state-dependent σ in the mixture cell.

Same seed=42 and all other params as `standup_floor04`.

Watch `policy_stats`: `effective_components` / `component_weight_k`
(weight usage), `component_overlap` (mode separation).  No
`sigma_state_std` — σ has no state dependence in this cell.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04MixtureShared(StandupFloor04):
    name = "standup_floor04_mixture_shared"

    actor_blueprint = "init_policy_shared_mixture_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04MixtureShared
