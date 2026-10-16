"""Hierarchy Step 1b — from-scratch gait learning, standing spawn.

Track B (parallel with the s2 migration run): a fresh 65-dim MBS policy
spawns standing and learns the s42i standard cyclic-stepping gait from
the clip reward alone — no fallen-state init, no pushes.  Verifies the
reward can produce the gait without inheriting it.

Same two channels as hier_s1_standup_step_mbs:
    r_potential = 0.01 × φ(t),   aw = 1.0
    r_std_clip  = clip + pose + 0.5·cad + 0.5·h_term,   aw = 3·φ²

Blueprint: baseline/humanoid21/hierarchy/blueprints/hier_s1b_step_scratch_env.yaml
"""
from __future__ import annotations

from pathlib import Path

from .exp_hier_s1_standup_step_mbs import HierS1StandupStepMBS


class HierS1bStepScratch(HierS1StandupStepMBS):
    name = "hier_s1b_step_scratch_mbs"

    max_steps: int = 400
    eval_interval: int = 5

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s1b_step_scratch_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)


EXPERIMENT_CLASS = HierS1bStepScratch
