"""Guard-96 Step 2 — push recovery -> jab_l dynamic guard (96-dim).

96-dim AttackPrep stage 2: warm-start from the ``hier_s1_guard_96d``
stage-1 policy (which already stands up into the jab_l guard), then
train under the 12-level StandingTriggeredForce curriculum
(40N -> 100N) so the policy survives pushes and re-settles into the
guard.  Mirrors the reference run ``hier_s2_balance_s42i_b`` (66-dim),
but on the canonical 96-dim contract the upper router forwards.

Guard deltas vs the stepping S2 (enabled via default-off parent hooks):
    _pot_h_cap = 1.24   r_potential truncated above guard torso height
    _H_TARGET  = 1.21   h_term pull at the clip's own torso height
    cadence_lag = 8     in env yaml — guard load cycle is ~16 steps

Channels (3, inherited):
    r_potential = 0.01 × φ(t) × I(h ≤ 1.24),  aw = 3.0
    r_fall      = fall + wall-lean + push-clean,  aw = 2.0
    r_std_clip  = clip + pose + 0.5·cad + 0.5·h,
                                 aw = 2·φ²·(push_phase != PUSHING)

Blueprint: blueprints/hier_s2_guard_96d_env.yaml
Launch:
    --experiment hier_s2_guard_96d --resume-from <s1 ckpt> --reset-update
"""
from __future__ import annotations

from pathlib import Path

from .exp_hier_s2_balance_mbs import HierS2BalanceMBS


class HierS2Guard96D(HierS2BalanceMBS):
    """Push recovery -> jab_l guard under the raw 96-dim contract."""

    name = "hier_s2_guard_96d"

    # --- Network: canonical 96-dim obs (incl. opponent state) ---
    obs_dim: int = 96
    actor_blueprint = "init_policy_mbs_96d.yaml"

    # Guard height anchors — same values as the S1 guard experiment.
    _H_TARGET: float = 1.21
    _pot_h_cap = 1.24

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s2_guard_96d_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)


EXPERIMENT_CLASS = HierS2Guard96D
