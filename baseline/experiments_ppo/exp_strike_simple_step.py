"""Simplified sparse-reward strike variant — NO retract phase.

Same sparse recipe as ``strike_sparse_step`` (follow warm start, no arm
shaping, distance-gated ef exploration, engagement-zone masking), but
the scoring is simplified per user direction:

    first ≥250N hand hit on head/torso
        → SETTLE window (settle_steps=10, ~0.5s):
            damage from the SAME locked hand accumulates
        → window ends → episode terminates "strike_done"

The RETRACT phase is gone entirely — no separation/hand-home combo, no
hold, no retract_timeout, no withheld second-half payment.  r_strike
streams the FULL step_damage as it lands (total = Q), so the reward is
fully dense within the settlement window.

Why: the u234 dump showed retract success was mostly passive — the
learner never learned to back away (median radial −0.60m, it kept
pressing in); and r_retract's gradient (0.03) was drowned by r_face
(~19).  Removing the phase removes the failure mode instead of
scaffolding against it.

Channels: 9 = 8 follow/base + r_strike (r_retract dropped).
"""
from __future__ import annotations

from typing import List

import numpy as np

from baseline.framework.rollout import extract_per_step_field
from baseline.framework.ppo.trajectory import ChannelData, Trajectory

from .exp_follow_step import FollowStep
from .exp_strike_sparse_step import StrikeSparseStep


class StrikeSimpleStep(StrikeSparseStep):
    """Sparse strike with settle-only scoring — no RETRACT."""

    name = "strike_simple_step"

    # Early-stop effectively disabled (user direction: keep training
    # running — manual stop only).  The inherited rule was
    # no_improvement>=200 AND update>=600 which cut s42b at u600.
    _no_improvement_limit: int = 10 ** 9
    _min_updates: int = 10 ** 9

    _channel_names = FollowStep._channel_names + ("r_strike",)
    _channel_gammas = {
        **FollowStep._channel_gammas,
        "r_strike": 0.99,
    }

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        from pathlib import Path
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "strike_simple_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    # ------------------------------------------------------------------
    # Strike channel: stream the full locked-hand damage; no retract
    # masking (the phase cannot occur) and no withheld tail payment.
    # ------------------------------------------------------------------

    def _apply_strike_channels(
        self, trajs: List[Trajectory], episode, agent_id: str,
    ) -> List[Trajectory]:
        strike_key = self._STRIKE_OBS.get(agent_id)
        step_damage = extract_per_step_field(
            episode.observer_outputs, strike_key, "step_damage",
            episode.num_frames)

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            if step_damage is None:
                rew = np.zeros(T, dtype=np.float32)
            else:
                rew = np.asarray(
                    step_damage[:T], dtype=np.float32)
            t.channels["r_strike"] = ChannelData(
                reward=rew,
                is_terminated=ref.is_terminated,
                actor_weight=np.full(
                    T, self.STRIKE_ACTOR_WEIGHT, dtype=np.float32),
            )
        return trajs


EXPERIMENT_CLASS = StrikeSimpleStep
