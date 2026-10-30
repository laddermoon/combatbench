"""Stand-punch skill — 66-dim, warm-start from hier_s2_balance_s42i_b.

Same machinery as ``punch_p5_mbs`` (multi-part P5 tracking rewards,
world-fixed target + moving guidance track, finite-skill termination)
but a different task configuration:

- Episode start is the plain arena reset pose (== ``stand_p5()``, the
  planner's canonical standing pose — no policy state injection);
- Strikes are planned by ``plan_stand_punch`` (coil -> swing -> land ->
  strike, committed-step full-body jab), retimed to 0.65 s / 13 actions;
- Per-episode options carry a pre-validated local-frame target:
  ``{rid}_sp_hand`` + ``{rid}_sp_t{x,y,z}``, rejection-sampled offline
  in the tight reachable box from STAND_PUNCH.md.

Goal: prove the whole pipeline (plan -> guidance -> tracking reward ->
PPO) can learn a reachable-target strike from standing; isolates the
training side since the planner is independently validated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from functools import lru_cache

from point5.contracts import Point5State
from point5.humanoid21.stand_punch import (
    StandPunchConfig, plan_stand_punch)

from baseline.humanoid21.hierarchy.p5_world import clip_world_samples
from baseline.humanoid21.plugins.stand_punch_target import (
    SEGMENT_DURATIONS, timed_stand_clip)

from .exp_punch_p5_mbs import PunchP5MBS

_ROOT = Path(__file__).resolve().parents[2]
_DT = 0.05


@lru_cache(maxsize=1)
def _measured_stand_p5() -> Point5State:
    """The actual post-reset local P5 pose — deterministic for this env.

    ``stand_p5()`` uses sole landmarks (~8 cm lower ankles than the
    adapter's ankle centres), so offline reachability checks must run
    against the pose the plugin/scorer will actually see at reset.
    """
    from envs.framework.parameterized_blueprint import (
        ParameterizedEnvBlueprint)
    from point5.humanoid21.adapter import point5_from_view
    bp = ParameterizedEnvBlueprint.load(
        _ROOT / "baseline" / "humanoid21" / "hierarchy"
        / "blueprints" / "hier_stand_punch_env.yaml")
    rt = bp.materialize(max_steps=13).build()
    try:
        rt.reset(seed=0, options={})
        return point5_from_view(
            rt.simulator.get_derived_state(["robot_a"])["robot_a"],
            "robot_a", with_vel=False)
    finally:
        rt.close()

# Tight reachable box from STAND_PUNCH.md (~59% plan acceptance).
_TGT_X = (0.95, 1.30)
_TGT_Y = (-0.30, 0.30)
_TGT_Z = (0.75, 1.35)


class StandPunchMBS(PunchP5MBS):
    """Stage: standing-pose P5-guided jab, 66-dim, warm-start s42i_b."""

    name = "stand_punch_mbs"
    # 0.65 s task: stand->coil .10 / coil->swing .20 / swing->land .20 /
    # land->strike .15, thirteen 50-ms actions, no unscored tail.
    max_steps = 13

    _STAND_PLAN = StandPunchConfig()

    # --- env -----------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint)
        return ParameterizedEnvBlueprint.load(
            _ROOT / "baseline" / "humanoid21" / "hierarchy"
            / "blueprints" / "hier_stand_punch_env.yaml")

    # --- jobs ----------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, update,
                   stochastic=True):
        # Skip the parent's stance-relative strike options; sample
        # pre-validated local targets instead.
        jobs = super(PunchP5MBS, self).build_jobs(
            policy_bp, base_seed, n_episodes, update=update,
            stochastic=stochastic)
        rng = np.random.default_rng(base_seed + 555)
        for job in jobs:
            for rid in self._AGENT_IDS:
                hand, tgt = self._sample_target(rng)
                job.episode_options[f"{rid}_sp_hand"] = hand
                for a, v in zip("xyz", tgt):
                    job.episode_options[f"{rid}_sp_t{a}"] = float(v)
        return jobs

    @staticmethod
    def _sample_target(rng, max_tries: int = 64):
        """Rejection-sample a reachable (hand, local target) pair."""
        for _ in range(max_tries):
            hand = "left" if rng.random() < 0.5 else "right"
            tgt = np.array([rng.uniform(*_TGT_X),
                            rng.uniform(*_TGT_Y),
                            rng.uniform(*_TGT_Z)])
            if plan_stand_punch(tgt, hand,
                                initial=_measured_stand_p5()).reachable:
                return hand, tgt
        # Fallback to a known-good central target.
        return "left", np.array([1.05, 0.0, 1.10])

    # --- scoring -------------------------------------------------------

    def _plan_reference(self, st0, target_local, hand, f1):
        plan = plan_stand_punch(target_local, hand, self._STAND_PLAN,
                                initial=st0)
        if not plan.reachable:
            return None
        clip = timed_stand_clip(plan.task, st0.pos)
        return clip_world_samples(clip, f1, _DT)

    @property
    def _load_seconds(self) -> float:
        # Everything before the strike segment is "loading".
        return float(sum(SEGMENT_DURATIONS[:-1]))

    @property
    def _task_seconds(self) -> float:
        return float(sum(SEGMENT_DURATIONS))

EXPERIMENT_CLASS = StandPunchMBS
