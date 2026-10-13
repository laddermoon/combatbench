"""MBS variant of `strike_step`: MoG+bounded-σ+state-σ actor.

  strike_step      : TruncatedNormalPolicy — global σ
  strike_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                     K=3 MoG, bounded σ ∈ (0.05, 2.0), σ = f(obs)

Warm-start chain: ... → attack_prep_step_mbs (s42g, accepted) →
resume into this experiment for single-strike settle/retract training.
"""
from __future__ import annotations

from .exp_strike_step import StrikeStep


class StrikeStepMBS(StrikeStep):
    name = "strike_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = StrikeStepMBS
