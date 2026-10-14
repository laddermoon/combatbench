"""Sparse-reward strike variant — follow start, NO arm shaping.

Counterpart to ``strike_step`` (s42e lineage): same env, same
settle/retract event reward, same follow warm start, but the four
attack-prep arm-shaping channels (r_left/right_elbow,
r_left/right_hand_dist — the alternating punch machine) are REMOVED.

Rationale (user direction): instead of scaffolding the punch motion,
let the sparse terminal reward drive discovery — larger rollout batch
(2× episodes/update) plus a higher uncertainty floor keep exploration
alive long enough for the policy to find its own path to a valid
250N+ hit and a successful retract.

Distance zonation (see STRIKE_SPARSE_PLAN.md — the agreed spec; do not
change without user confirmation):

* dist >  D_ENGAGE (0.9m): native follow rewards, all channels live.
* dist <= D_ENGAGE       : task + fall-protection + facing ONLY —
                           r_left_foot / r_right_foot / r_torso masked
                           (r_radial / r_tangential are already off
                           inside FOLLOW_DIST_MAX=0.9 by their own
                           out-zone gate).  Live set: r_strike,
                           r_retract, r_fall, r_potential, r_face.
* RETRACT phase          : unchanged — everything but r_fall /
                           r_strike / r_retract is masked.

Differences vs StrikeStep:
* ``_channel_names`` — 4 arm channels dropped
* engagement-zone masking of feet/torso channels
* ``episodes_per_update`` 512 → 1024 (wider episode coverage per
  update: more seeds/poses/speeds sampled)
* ``uncertainty_floor`` 0.4 → 0.6 (stronger σ floor keeps the policy
  stochastic longer)
"""
from __future__ import annotations

from typing import List

import numpy as np

from baseline.framework.rollout import extract_per_step_field

from .exp_follow_step import FollowStep
from .exp_strike_step import StrikeStep


class StrikeSparseStep(StrikeStep):
    """Follow recipe + strike event reward, no arm shaping."""

    name = "strike_sparse_step"
    agent_used = "random"

    _channel_names = FollowStep._channel_names + (
        "r_strike", "r_retract")
    _channel_gammas = {
        **FollowStep._channel_gammas,
        "r_strike": 0.99,
        "r_retract": 0.99,
    }

    # Engagement distance: inside it, only task + fall-protection +
    # facing channels stay live (plan doc, user-confirmed spec).
    # Aligned with FOLLOW_DIST_MAX so the radial/tangential gates meet
    # the boundary seamlessly.
    D_ENGAGE: float = 0.9
    _CLOSE_MASKED_CHANNELS = (
        "r_left_foot", "r_right_foot", "r_torso",
    )

    # Wider exploration budget: 2× episodes per update and a higher
    # σ floor so discovery doesn't collapse before the sparse event
    # reward is found.
    episodes_per_update: int = 1024
    uncertainty_floor: float = 0.6

    # ------------------------------------------------------------------
    # Trajectories — FollowStep recipe (skip AttackPrepStep's arm
    # channels entirely) + zone masking + strike channels.
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List:
        trajs = FollowStep._build_agent_trajectory(
            self, episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs

        T_full = episode.num_frames
        oo = episode.observer_outputs
        app_key = self._APPROACH_OBS.get(agent_id)
        self_x = extract_per_step_field(oo, app_key, "self_x", T_full)
        self_y = extract_per_step_field(oo, app_key, "self_y", T_full)
        opp_x = extract_per_step_field(oo, app_key, "opp_x", T_full)
        opp_y = extract_per_step_field(oo, app_key, "opp_y", T_full)
        if self_x is not None and opp_x is not None:
            dist = np.hypot(
                np.asarray(opp_x) - np.asarray(self_x),
                np.asarray(opp_y) - np.asarray(self_y),
            )
            close = dist <= self.D_ENGAGE
            for t in trajs:
                T = t.obs.shape[0]
                m = close[:T]
                for cname in self._CLOSE_MASKED_CHANNELS:
                    ch = t.channels.get(cname)
                    if ch is not None:
                        ch.reward[m] = 0.0
                        ch.actor_weight[m] = 0.0

        return self._apply_strike_channels(trajs, episode, agent_id)


EXPERIMENT_CLASS = StrikeSparseStep
