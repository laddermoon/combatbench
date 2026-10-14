"""MBS variant of `strike_simple_step`: MoG+bounded-σ+state-σ actor.

  strike_simple_step      : TruncatedNormalPolicy — global σ
  strike_simple_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy

Warm start: resume from the strike_sparse run's latest checkpoint —
the accumulated hit skill carries over; the extra r_retract critic in
the checkpoint is simply unused (loader tolerates missing/extra
critics).
"""
from __future__ import annotations

from .exp_strike_simple_step import StrikeSimpleStep


class StrikeSimpleStepMBS(StrikeSimpleStep):
    name = "strike_simple_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = StrikeSimpleStepMBS
