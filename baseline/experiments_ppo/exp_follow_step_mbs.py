"""A/B variant of `follow_step`: MoG+bounded-σ+state-σ actor.

  follow_step      : TruncatedNormalPolicy — global σ
  follow_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                     K=3 MoG, bounded σ ∈ (0.05, 2.0), σ = f(obs)

Warm-start chain: standup mbs_ef05 → step_mbs → balance_step_mbs →
resume into this experiment.  Same rationale as exp_step_mbs /
exp_balance_step_mbs.
"""
from __future__ import annotations

from .exp_follow_step import FollowStep


class FollowStepMBS(FollowStep):
    name = "follow_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = FollowStepMBS
