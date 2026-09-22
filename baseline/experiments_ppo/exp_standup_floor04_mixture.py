"""Policy A/B variant of `standup_floor04`: mixture truncated-normal actor.

  standup_floor04          : TruncatedNormalPolicy — σ is a global
                             nn.Parameter shared by all states
  standup_floor04_statesig : StateTruncatedNormalPolicy — σ = f(obs)
  standup_floor04_mixture  : MixtureTruncatedNormalPolicy — K=3 diagonal
                             truncated-normal components with shared
                             component index; logits init 0 → π uniform,
                             σ block init w=0/b=-1 → σ ≡ e⁻¹ (this run)

Same seed=42 and all other params as `standup_floor04`.  A/B pair
against train_standup_floor04_ppo_20260920_164819.

NOTE: U is the marginal Rényi-2 width (L2 metric), NOT the peak metric —
the same uncertainty_floor=0.4 is a DIFFERENT-strength constraint here
(see DESIGN_mixture_truncated_normal.md §8).  This run validates
usability; a strictly-controlled A/B needs a matched-U baseline.

Watch `policy_stats`: `effective_components` / `component_weight_k`
(weight usage), `component_overlap` (mode separation, 1 = identical),
`sigma_state_std` (state-dependence of σ).
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04Mixture(StandupFloor04):
    name = "standup_floor04_mixture"

    actor_blueprint = "init_policy_mixture_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04Mixture
