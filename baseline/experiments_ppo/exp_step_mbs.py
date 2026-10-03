"""A/B variant of `step`: MoG+bounded-σ+state-σ actor (sweep MVP cell).

  step      : TruncatedNormalPolicy — global σ
  step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy — K=3 MoG,
              bounded σ ∈ (0.05, 2.0) via sigmoid, σ = f(obs)

Warm-start chain: train `standup_floor04_mixture_boundedstd_statesig`
(or its ef05 variant), resume into this experiment.  The step
experiment's built-in phase-dependent explore_factor (σ×0.5 while
low / ×2.0 while standing) maps onto the bounded policy's additive
sigmoid shift — monotone in the same direction, safe by construction
(bounded σ cannot blow up under ef, RESULTS_truncnorm_sweeps A4).
"""
from __future__ import annotations

from .exp_step import Step


class StepMBS(Step):
    name = "step_mbs"

    actor_blueprint = "init_policy_state_mixture_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StepMBS
