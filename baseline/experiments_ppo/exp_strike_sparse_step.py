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

Channel set = FollowStep's 8 channels + r_strike + r_retract (10).
Trajectory recipe = FollowStep's (arm channels never created) +
``_apply_strike_channels``.

Differences vs StrikeStep:
* ``_channel_names`` — 4 arm channels dropped
* ``episodes_per_update`` 512 → 1024 ("绕的规模" — wider episode
  coverage per update: more seeds/poses/speeds sampled)
* ``uncertainty_floor`` 0.4 → 0.6 (stronger σ floor keeps the policy
  stochastic longer — "探索力度")
"""
from __future__ import annotations

from typing import List

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

    # Wider exploration budget: 2× episodes per update and a higher
    # σ floor so discovery doesn't collapse before the sparse event
    # reward is found.
    episodes_per_update: int = 1024
    uncertainty_floor: float = 0.6

    # ------------------------------------------------------------------
    # Trajectories — FollowStep recipe (skip AttackPrepStep's arm
    # channels entirely) + strike channels.
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List:
        trajs = FollowStep._build_agent_trajectory(
            self, episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs
        return self._apply_strike_channels(trajs, episode, agent_id)


EXPERIMENT_CLASS = StrikeSparseStep
