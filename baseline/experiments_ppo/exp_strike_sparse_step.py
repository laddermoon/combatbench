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
* rollout-time ``explore_factor`` is distance-gated: σ×2.0 inside the
  engagement zone, σ×1.0 outside, σ×0.5 while fallen — exploration
  effort goes where the reward signal is sparse (user direction:
  函数化 ef 优于全局 u_floor).
"""
from __future__ import annotations

import math as _math
from typing import List

import numpy as np

from baseline.framework.rollout import extract_per_step_field

from .exp_follow_step import FollowStep
from .exp_strike_step import StrikeStep
from .exp_step import _EF_LOW, _EF_STAND, _H_EF_SWITCH


# --- Distance-gated explore_factor (σ multiplier = 3^ef) ---
# obs[45]  = torso height (m)
# obs[57] / obs[58] = opponent XY relative position in ego frame (m)
#
#   fallen   (h < 1.1):            ef = -0.631 → σ×0.5 (protect standup)
#   standing & dist <= D_ENGAGE:   ef = +0.631 → σ×2.0 (attack-range
#                                   exploration — the sparse-reward zone)
#   standing & dist >  D_ENGAGE:   ef =  0     → σ×1.0 (no extra boost
#                                   outside; follow skills already work)
_D_ENGAGE_EF: float = 0.9          # keep in sync with D_ENGAGE


def _sparse_explore_factor(obs, step):
    """Boost σ only inside engagement range — target exploration at the
    sparse-reward zone instead of flooding the whole state space."""
    if float(obs[45]) < _H_EF_SWITCH:
        return _EF_LOW
    d = _math.hypot(float(obs[57]), float(obs[58]))
    return _EF_STAND if d <= _D_ENGAGE_EF else 0.0


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

    # Wider rollout budget: 2× episodes per update.  Exploration is
    # targeted by _sparse_explore_factor (σ×2 inside engagement range
    # only) — the global uncertainty floor stays at the inherited 0.4:
    # raising it would boost σ everywhere, contradicting the
    # "界外不加强探索" intent.
    episodes_per_update: int = 1024

    def _sampling_spec(self):
        from baseline.framework.rollout import SamplingSpec
        return SamplingSpec(explore_factor=_sparse_explore_factor)

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
