"""MBS variant of `punch_power_step`: MoG+bounded-σ+state-σ actor.

  punch_power_step      : TruncatedNormalPolicy — global σ
  punch_power_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                          K=3 MoG, bounded σ ∈ (0.05, 2.0), σ = f(obs)

Warm-start chain: ... → attack_prep_step_mbs (s42g, accepted) →
resume into this experiment for punch-power shaping.
"""
from __future__ import annotations

from .exp_punch_power_step import PunchPowerStep


class PunchPowerStepMBS(PunchPowerStep):
    name = "punch_power_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = PunchPowerStepMBS
