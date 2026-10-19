"""Merged skill recipe — skill_full_mbs.

One experiment carrying all three stage fixes, trained from the S2
checkpoint over the FULL warmup-generalized start distribution:

  * SkillStepMBS  — stride-settled move rewards (r_move_step /
                    r_drift_step replace dense smoothed r_move/r_drift)
  * SkillGateMBS  — strict launch-pose gate on strikes (θ=1.65 vs
                    clip launch window [0,4], alignment phase feeds
                    r_strike until gated)
  * SkillTransMBS — warmup specs: episodes may start mid-skill from
                    any prior skill (inherited via SkillGateMBS)

Cooperative-MRO merge: build_jobs → warmup (trans) + launch_gate
injection (gate); _build_agent_trajectory → parent's channels then
stride swap (step); on_eval chains train→gate→step metrics.
"""
from __future__ import annotations

from .exp_skill_gate_mbs import SkillGateMBS
from .exp_skill_step_mbs import SkillStepMBS


class SkillFullMBS(SkillStepMBS, SkillGateMBS):
    """stride-settled moves + launch-gated strikes + warmup coverage."""

    name = "skill_full_mbs"


EXPERIMENT_CLASS = SkillFullMBS
