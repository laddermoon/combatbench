"""Trajectory-based training data structures.

This module defines the atomic units for PPO training in the PPO framework:

- ``RewardChannel``: configuration for a single critic (name, gamma, lambda).
- ``ChannelData``: per-trajectory reward data for one channel.
- ``Trajectory``: the atomic training unit — obs, actions, last_obs, per-channel
  data, importance, mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardChannel:
    """Configuration for a single reward channel (one critic).

    Attributes:
        name: Unique key, e.g. ``"r_stand"``.  Used to index
            ``Trajectory.channels`` and ``critics`` dict.
        gamma: Discount factor for this channel's GAE.
        gae_lambda: GAE λ for this channel.  Different channels can have
            different bias-variance tradeoffs (e.g. sparse terminal rewards
            benefit from high λ, dense shaping from lower λ).
    """

    name: str
    gamma: float
    gae_lambda: float


@dataclass
class ChannelData:
    """Per-trajectory data for one reward channel.

    Attributes:
        reward: ``(T,)`` float32 array of per-step rewards.
        is_terminated: If True, GAE uses ``last_value=0`` (no bootstrap).
            If False, GAE bootstraps from ``V_critic(last_obs)``.
            This is the per-channel termination flag — different channels
            on the same trajectory can have different values.
        actor_weight: Weight for this channel's advantage in the policy
            gradient.  ``0.0`` means the critic is trained but does not
            influence the actor (useful for warming up a new critic).
            Can be a scalar (same weight for all frames) or a ``(T,)``
            array (per-step weight, enabling time-varying channel
            importance within a single trajectory).

            Per-frame L1 normalization: before combining advantages, the
            framework rescales each frame's weights so that
            ``Σ_c |aw_c| = 1``.  This decouples the *relative importance*
            of channels from the *effective learning rate* of the actor.
            Scaling all weights by a constant does not change the
            combined advantage — only the ratio between channels matters.
            Negative values are allowed (they invert the channel's
            advantage direction) and the sign is preserved through
            normalization.
    """

    reward: np.ndarray
    is_terminated: bool
    actor_weight: Union[float, np.ndarray] = 1.0


@dataclass
class Trajectory:
    """Atomic training unit for PPO.

    A trajectory is a contiguous slice of an episode, produced by the
    experiment's ``build_trajectories``.  The PPO buffer concatenates
    trajectories into flat arrays for batched GAE and minibatch updates.

    Attributes:
        obs: ``(T, obs_dim)`` float32 — observations.
        actions: ``(T, act_dim)`` float32 — actions taken.
        last_obs: ``(obs_dim,)`` float32 — observation after the last
            action.  Used for bootstrap value computation.  For mid-episode
            boundaries, this is ``obs[end]`` (the first frame of the next
            segment).  For episode-end, this is ``episode.final_observation``.
        channels: Per-channel data.  A channel absent from this dict means
            the channel is inactive on this trajectory (no critic training,
            no advantage contribution).
        importance: Sample weight for this trajectory — scales both critic
            loss and policy loss.
        explore_intensity: ``(T,)`` float32 — per-frame exploration
            intensity used at rollout time.  Threaded into
            ``evaluate_actions`` so log_prob is computed under the same
            distribution that produced the actions.  When None, defaults
            to 0.0 (neutral) in the buffer.
    """

    obs: np.ndarray
    actions: np.ndarray
    last_obs: np.ndarray
    channels: Dict[str, ChannelData]
    importance: float = 1.0
    explore_intensity: Optional[np.ndarray] = None
