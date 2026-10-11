"""MBS variant of `attack_prep_step`: MoG+bounded-σ+state-σ actor.

  attack_prep_step      : TruncatedNormalPolicy — global σ
  attack_prep_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                          K=3 MoG, bounded σ ∈ (0.05, 2.0), σ = f(obs)

Warm-start chain: standup mbs_ef05 → step_mbs → balance_step_mbs →
follow_step_mbs → resume into this experiment.  Same rationale as
exp_follow_step_mbs.
"""
from __future__ import annotations

from .exp_attack_prep_step import AttackPrepStep


class AttackPrepStepMBS(AttackPrepStep):
    name = "attack_prep_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = AttackPrepStepMBS
