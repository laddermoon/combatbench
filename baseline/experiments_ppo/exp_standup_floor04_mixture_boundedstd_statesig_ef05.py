"""A/B variant of `standup_floor04_mixture_boundedstd_statesig`: ef=0.5.

  mix_bnd_statesig        : MoG K=3 + bounded σ + state-σ, e=0
  mix_bnd_statesig_ef05   : same policy, rollout σ_eff via bounded
                            sigmoid-shift ef=0.5 (this run)

The sweep-recommended combination (RESULTS_truncnorm_sweeps §3): MoG
accelerates convergence, bounded-σ makes ef safe (A4), state-σ adapts
exploration per state.  ef05 was this cell's strongest config
(best-balanced frontier, only cell with 3/3 paired Δu100<0).
"""
from __future__ import annotations

from .exp_standup_floor04_ef import StandupFloor04EF


class StandupFloor04MixtureBoundedStdStateSigEF05(StandupFloor04EF):
    name = "standup_floor04_mixture_boundedstd_statesig_ef05"

    actor_blueprint = "init_policy_state_mixture_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04MixtureBoundedStdStateSigEF05
