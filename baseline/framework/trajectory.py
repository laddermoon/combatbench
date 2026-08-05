"""V2 trajectory-based training data structures.

This module defines the atomic units for PPO training in the v2 framework:

- ``RewardChannel``: configuration for a single critic (name, gamma, lambda).
- ``ChannelData``: per-trajectory reward data for one channel.
- ``Trajectory``: the atomic training unit — obs, actions, last_obs, per-channel
  data, importance, mode.

The legacy converter ``legacy_to_trajectories`` bridges v1 experiments
(that implement ``extract_rewards`` + ``prepare_segments``) to the v2
``Trajectory`` format, so the PPO trainer can be rewritten to consume
``List[Trajectory]`` exclusively.

The funnel function ``resolve_trajectories`` dispatches to v2
(``build_trajectories``) or v1 (``legacy_to_trajectories``) based on the
experiment type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.common.rollout.episode import Episode

from .experiment import Experiment, Segment


# ---------------------------------------------------------------------------
# Core v2 data structures
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
    """

    reward: np.ndarray
    is_terminated: bool
    actor_weight: float = 1.0


@dataclass
class Trajectory:
    """Atomic training unit for PPO v2.

    A trajectory is a contiguous slice of an episode, produced by the
    experiment's ``build_trajectories`` (v2) or ``legacy_to_trajectories``
    (v1 adapter).  The PPO buffer concatenates trajectories into flat
    arrays for batched GAE and minibatch updates.

    Attributes:
        obs: ``(T, obs_dim)`` float32 — observations.
        actions: ``(T, act_dim)`` float32 — actions taken.
        last_obs: ``(obs_dim,)`` float32 — observation after the last
            action.  Used for bootstrap value computation.  For mid-episode
            boundaries, this is ``obs[end]`` (the first frame of the next
            segment).  For episode-end, this is ``episode.final_observation``.
        channels: Per-channel data.  A channel absent from this dict means
            the channel is inactive on this trajectory (no critic training,
            no advantage contribution).  This replaces v1's ``key_weights``.
        importance: Sample weight for this trajectory — scales both critic
            loss and policy loss.  Replaces v1's ``Segment.weight``.
        mode: Optional actor routing mode (float).  If None, the actor
            computes mode from observation.  Replaces v1's ``Segment.mode``.
        log_prob: ``(T,)`` float32 — old log probabilities, filled by the
            framework's batched ``evaluate_actions`` call.  Experiments
            must leave this as None.
    """

    obs: np.ndarray
    actions: np.ndarray
    last_obs: np.ndarray
    channels: Dict[str, ChannelData]
    importance: float = 1.0
    mode: Optional[float] = None
    log_prob: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Legacy converter: v1 Experiment → List[Trajectory]
# ---------------------------------------------------------------------------

def _resolve_segments(experiment: Experiment, episode: Episode) -> List[Segment]:
    """Try v2 prepare_segments; fall back to v1 prepare_training_segments."""
    segs = experiment.prepare_segments(episode)
    if segs is not None:
        return segs
    # v1 fallback
    raw = experiment.prepare_training_segments(episode)
    result = []
    for seg in raw:
        if isinstance(seg, Segment):
            result.append(seg)
        elif len(seg) == 3:
            result.append(Segment(start=seg[0], end=seg[1], weight=seg[2]))
        elif len(seg) == 4:
            result.append(Segment(start=seg[0], end=seg[1], weight=seg[2], mode=seg[3]))
        else:
            raise ValueError(f"Invalid segment tuple: {seg}")
    return result


def _segment_termination(
    seg: Segment, episode: Episode, T: int,
) -> str:
    """Resolve segment-level termination mode.

    Returns "terminated" or "truncated".
    """
    term_mode = seg.termination
    if term_mode == "truncated":
        return "truncated"
    elif term_mode == "terminated":
        return "terminated"
    else:  # None / "auto"
        if seg.end < T:
            return "terminated"
        else:
            return "terminated" if episode.is_terminated else "truncated"


def _key_termination(
    seg: Segment, key: str, seg_term: str,
) -> str:
    """Resolve per-key termination, falling back to segment-level."""
    if seg.key_termination and key in seg.key_termination:
        kt = seg.key_termination[key]
        if kt == "truncated":
            return "truncated"
        elif kt == "terminated":
            return "terminated"
    return seg_term


def legacy_to_trajectories(
    experiment: Experiment,
    episode: Episode,
    reward_keys: Tuple[str, ...],
    gammas: Dict[str, float],
    stage_weights: Tuple[float, ...],
) -> List[Trajectory]:
    """Convert a v1 Episode into v2 Trajectories.

    This function replaces the v1 pipeline of ``extract_rewards`` +
    ``prepare_segments`` + per-segment slicing with a single pass that
    produces ``Trajectory`` objects directly.

    Args:
        experiment: A v1 ``Experiment`` instance.
        episode: The episode to convert.
        reward_keys: Tuple of reward key names.
        gammas: Per-key gamma dict (unused here, but kept for API symmetry
            with v2 — gamma lives in ``RewardChannel`` in v2).
        stage_weights: Per-key stage weights, used as ``actor_weight`` for
            each channel.  In v1 these are global; in v2 they would be
            baked into ``ChannelData.actor_weight`` by the experiment.

    Returns:
        List of ``Trajectory`` objects, one per segment.  Empty list if
        the episode is skipped (no segments).
    """
    T = episode.num_frames
    ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
    obs_all = episode.observations.get(ep_target)
    acts_all = episode.actions.get(ep_target)
    fin_obs = episode.final_observation.get(ep_target)

    if obs_all is None or acts_all is None or fin_obs is None:
        return []

    segs = _resolve_segments(experiment, episode)
    if not segs:
        return []

    rewards_full = experiment.extract_rewards(episode)

    trajectories: List[Trajectory] = []
    for seg in segs:
        start, end = seg.start, seg.end
        T_seg = end - start
        if T_seg == 0:
            continue

        # Determine last_obs
        if end < T:
            last_obs = np.asarray(obs_all[end], dtype=np.float32)
        else:
            last_obs = np.asarray(fin_obs, dtype=np.float32)

        # Segment-level termination
        seg_term = _segment_termination(seg, episode, T)

        # Per-key active set
        if seg.key_weights is not None:
            active_keys = set(
                k for k, w in seg.key_weights.items() if w > 0.0
            )
        else:
            active_keys = set(reward_keys)

        # Build per-channel data
        channels: Dict[str, ChannelData] = {}
        for key in reward_keys:
            if key not in active_keys:
                continue

            r_full = rewards_full.get(key, np.zeros(T, dtype=np.float32))
            r_seg = np.asarray(r_full[start:end], dtype=np.float32)

            key_term = _key_termination(seg, key, seg_term)
            is_terminated = (key_term == "terminated")

            # actor_weight from stage_weights
            key_idx = reward_keys.index(key)
            aw = float(stage_weights[key_idx]) if key_idx < len(stage_weights) else 1.0

            channels[key] = ChannelData(
                reward=r_seg,
                is_terminated=is_terminated,
                actor_weight=aw,
            )

        trajectories.append(Trajectory(
            obs=np.asarray(obs_all[start:end], dtype=np.float32),
            actions=np.asarray(acts_all[start:end], dtype=np.float32),
            last_obs=last_obs,
            channels=channels,
            importance=float(seg.weight),
            mode=seg.mode,
            log_prob=None,  # framework fills this
        ))

    return trajectories


# ---------------------------------------------------------------------------
# Funnel: dispatch v2 or v1
# ---------------------------------------------------------------------------

def resolve_trajectories(
    experiment: Experiment,
    episode: Episode,
    reward_keys: Tuple[str, ...],
    gammas: Dict[str, float],
    stage_weights: Tuple[float, ...],
) -> List[Trajectory]:
    """Dispatch to v2 ``build_trajectories`` or v1 ``legacy_to_trajectories``.

    This is the single entry point for the PPO trainer.  All experiments
    produce ``List[Trajectory]`` regardless of version.
    """
    # Check for v2 interface
    build_fn = getattr(experiment, "build_trajectories", None)
    if build_fn is not None:
        return build_fn(episode)
    # v1 fallback
    return legacy_to_trajectories(
        experiment, episode, reward_keys, gammas, stage_weights,
    )
