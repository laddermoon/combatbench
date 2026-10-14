"""MBS variant of `strike_sparse_step`: MoG+bounded-σ+state-σ actor.

  strike_sparse_step      : TruncatedNormalPolicy — global σ
  strike_sparse_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy

Warm start: follow_s42b final checkpoint (u08565) with --reset-update —
same start as strike_s42e, but no arm-shaping channels.
"""
from __future__ import annotations

from .exp_strike_sparse_step import StrikeSparseStep


class StrikeSparseStepMBS(StrikeSparseStep):
    name = "strike_sparse_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = StrikeSparseStepMBS
