"""MBS variant of `fight_step`: MoG+bounded-σ+state-σ actor.

  fight_step      : TruncatedNormalPolicy — global σ
  fight_step_mbs  : StateMixtureBoundedStdTruncatedNormalPolicy —
                    K=3 MoG, bounded σ ∈ (0.05, 2.0), σ = f(obs)

Warm-start chain: ... → follow_step_mbs → attack_prep_step_mbs (s42g,
accepted: real alternating punches) → resume into this experiment.
Same rationale as exp_follow_step_mbs / exp_attack_prep_step_mbs.
"""
from __future__ import annotations

from .exp_fight_step import FightStep


class FightStepMBS(FightStep):
    name = "fight_step_mbs"

    actor_blueprint = (
        "init_policy_state_mixture_bounded_std_truncated_normal.yaml"
    )


EXPERIMENT_CLASS = FightStepMBS
