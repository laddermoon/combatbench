"""Hierarchy Step 4-EE — punch strikes with end-effector observation.

Identical to ``hier_s4_punch_mbs`` except the observation is 78-dim:
57 proprio + 9 command + 12 end-effector ego positions (both hands and
feet, pelvis-relative, /TGT_SCALE).  The 57-dim proprioception carries
no keypoint positions — the policy could see where the strike target
is but not where its own hand currently is, making target-following a
purely open-loop problem.  The ee block closes that loop so the policy
can servo the commanded effector onto the moving target.

Warm-start: expand the 66-dim checkpoint's first-layer weights with
zero columns for the 12 new inputs (see tools/expand_ckpt_obs.py
usage in TRAINING_LOG).
"""
from __future__ import annotations

from pathlib import Path

from .exp_hier_s4_punch_mbs import HierS4PunchMBS


class HierS4PunchEEMBS(HierS4PunchMBS):
    name = "hier_s4_punch_ee_mbs"
    obs_dim: int = 78
    actor_blueprint = "init_policy_mbs_78d.yaml"

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s4_punch_ee_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)


EXPERIMENT_CLASS = HierS4PunchEEMBS
