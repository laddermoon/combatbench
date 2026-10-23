"""Hierarchy Step 1 (guard variant) — standup + jab_l dynamic guard hold.

Sibling of ``hier_s1_standup_step_mbs``: same from-scratch standup
protocol (RandomFallenStatePlugin + 4-stage potential + φ² gate), but
the imitation library is the **jab_l dynamic guard** captured from the
strike_kp_2phase best run instead of the cyclic-stepping clips.

Clip provenance: ``train_strike_kp_2phase_ppo_20261021_001547``,
``policy_exports/u09260`` (last export), eval-faithful rollout
(seed 997777, 100 steps) -> ``assets/jab_l_guard_clips.npz``
(7 × 40-frame feature windows, ``extract_standard_clips.py --npz``).
The hold is *dynamic* — foot load alternates with a ~16-step period,
so the cadence term stays ON at cadence_lag=8 (measured off the clip's
load-diff autocorrelation, strongest peak at lag 8).

Channels (unchanged formula, mirror of the stepping variant):
    r_potential = 0.01 × φ(t),                        aw = 1.0
    r_std_clip  = clip + pose + 0.5·cad + 0.5·h,      aw = φ²

Blueprint: blueprints/hier_s1_guard_env.yaml
"""
from __future__ import annotations

from pathlib import Path

from .exp_hier_s1_standup_step_mbs import HierS1StandupStepMBS


class HierS1GuardMBS(HierS1StandupStepMBS):
    """Standup + jab_l dynamic-guard hold under the 66-dim contract."""

    name = "hier_s1_guard_mbs"

    # Torso height target from the recorded clip itself
    # (robot_a mean root_z 1.223 / robot_b 1.207 over the hold).
    _H_TARGET: float = 1.21

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "hierarchy" / "blueprints"
            / "hier_s1_guard_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)


EXPERIMENT_CLASS = HierS1GuardMBS
