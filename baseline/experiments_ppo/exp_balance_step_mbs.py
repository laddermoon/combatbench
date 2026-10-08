"""A/B variant of `balance_step`: MoG+bounded-σ+state-σ actor.

  balance_step      : TruncatedNormalPolicy — global σ
  balance_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                      K=3 MoG, bounded σ ∈ (0.05, 2.0) via sigmoid,
                      σ = f(obs)

Warm-start chain: standup mbs_ef05 → step_mbs → resume into this
experiment.  Same rationale as exp_step_mbs.
"""
from __future__ import annotations

from .exp_balance_step import BalanceStep


class BalanceStepMBS(BalanceStep):
    name = "balance_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = BalanceStepMBS
