"""A/B variant of `balance_pure`: MoG+bounded-σ+state-σ actor.

  balance_pure      : TruncatedNormalPolicy — global σ
  balance_pure_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                      K=3 MoG, bounded σ ∈ (0.05, 2.0) via sigmoid,
                      σ = f(obs)

Warm-start chain: standup mbs_ef05 → this experiment (obs is natively
96-dim — no zero-padding, the checkpoint fits as-is).
"""
from __future__ import annotations

from .exp_balance_pure import BalancePure


class BalancePureMBS(BalancePure):
    name = "balance_pure_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = BalancePureMBS
