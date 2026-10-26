"""Guard-96 Step 1 — from-scratch standup + jab_l dynamic guard (96-dim).

96-dim port of ``hier_s1_guard_mbs`` for the new-route sub-skill chain
(bootstrip.md "Follow And AttackPrep": the upper-level router receives
the canonical 96-dim obs and forwards it verbatim to this skill, so the
skill policy must consume raw 96-dim input, NOT the 66-dim
CommandSimulator v2 contract).

Everything else is inherited unchanged: same reward channels, same
jab_l guard clip library, same φ² gate, same capped potential
(_pot_h_cap = 1.24 truncates the standing pull above the clip's own
torso height).  From scratch — no warm-start.

Blueprint: blueprints/hier_s1_guard_96d_env.yaml (Humanoid21Simulator)
"""
from __future__ import annotations

from pathlib import Path

from .exp_hier_s1_guard_mbs import HierS1GuardMBS


class HierS1Guard96D(HierS1GuardMBS):
    """Standup + jab_l dynamic-guard hold under the raw 96-dim contract."""

    name = "hier_s1_guard_96d"

    # --- Network: canonical 96-dim obs (incl. opponent state) ---
    obs_dim: int = 96
    actor_blueprint = "init_policy_mbs_96d.yaml"

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s1_guard_96d_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)


EXPERIMENT_CLASS = HierS1Guard96D
