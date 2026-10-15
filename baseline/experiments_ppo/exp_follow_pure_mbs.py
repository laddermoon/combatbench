"""A/B variant of `follow_pure`: MoG+bounded-σ+state-σ actor.

  follow_pure      : TruncatedNormalPolicy — global σ
  follow_pure_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                     K=3 MoG, bounded σ ∈ (0.05, 2.0) via sigmoid,
                     σ = f(obs)

Warm-start chain: standup mbs_ef05 → this experiment (96-dim obs,
native fit).
"""
from __future__ import annotations

from .exp_follow_pure import FollowPure


class FollowPureMBS(FollowPure):
    name = "follow_pure_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = FollowPureMBS
