"""``RolloutBatch``: framework-side, algorithm-agnostic rollout container.

This is the **frozen, numpy-only view** of an :class:`AgentTrajectory`,
intended as the single data contract between rollout / algorithm / eval
layers. It lives in :mod:`envs.framework` (not in ``baseline``) on
purpose:

  * The framework already owns the *live* trajectory buffer
    (:class:`AgentTrajectory`) and the per-episode metadata
    (:class:`EpisodeResult`). The frozen RL-style view is the natural
    third member of that family.
  * Downstream algorithm packages (PPO, GRPO, SAC, …) all need the
    same shape contract: ``T+1`` observations, ``T`` actions/rewards,
    Gymnasium-style ``terminated`` / ``truncated`` flags, and an
    optional ``log_probs`` / ``values`` pair. Owning the dataclass
    here means the algorithm packages do not each maintain their own
    translation of trajectories — they just consume
    :meth:`AgentTrajectory.as_rollout_batch`.
  * Import weight: this module imports nothing from ``baseline`` and
    nothing from torch — it is safe to ship across multiprocessing
    boundaries and to reference from worker processes.

Shapes follow the framework convention with
``RolloutConfig.store_initial_observation=True`` (the default in
:class:`EpisodeRunner`):

    ``len(obs) == T + 1``,  ``len(actions) == len(rewards) == T``

The trailing observation doubles as the bootstrap target for truncated
episodes — exposed via :attr:`final_obs` rather than stored as a
separate field, so the alignment cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RolloutBatch:
    """One episode of experience for one agent, as numpy tensors.

    Invariants (checked by :meth:`validate`):

        ``obs.shape[0] == actions.shape[0] + 1 == rewards.shape[0] + 1``

        When ``log_probs`` / ``values`` are provided, their length
        equals ``actions.shape[0]``.

        Exactly one of ``terminated`` / ``truncated`` is true (Gymnasium
        semantics). Use :meth:`AgentTrajectory.to_gymnasium_style` to
        coerce raw framework flags before constructing the batch — the
        framework allows both flags to fire on the same step (e.g.
        KO + timeout), the RL-side view does not.
    """

    agent_id: str
    obs: np.ndarray                     # (T+1, *obs_shape)
    actions: np.ndarray                 # (T,   *action_shape)
    rewards: np.ndarray                 # (T,)
    terminated: bool
    truncated: bool
    log_probs: Optional[np.ndarray] = None  # (T,)
    values: Optional[np.ndarray] = None     # (T,)
    info: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_steps(self) -> int:
        """``T`` — the number of environment steps (== ``len(actions)``)."""
        return int(self.actions.shape[0])

    @property
    def initial_obs(self) -> np.ndarray:
        """``obs[0]`` — observation that produced ``actions[0]``."""
        return self.obs[0]

    @property
    def final_obs(self) -> np.ndarray:
        """``obs[-1]`` — the observation *after* the last action.

        When :attr:`truncated` is True, use this as the bootstrap value
        input for GAE / n-step returns. When :attr:`terminated` is True,
        this observation exists but its value should be treated as 0
        by the critic (no future return).
        """
        return self.obs[-1]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` if shapes / lengths are inconsistent.

        Intended for use in tests and during initial integration; steady
        -state production code can skip this — :meth:`AgentTrajectory.as_rollout_batch`
        produces well-formed batches by construction.
        """
        if self.obs.ndim < 1:
            raise ValueError(
                "obs must be at least 1-D (got shape {})".format(self.obs.shape)
            )
        t_plus_1 = self.obs.shape[0]
        t = self.actions.shape[0]
        if t_plus_1 != t + 1:
            raise ValueError(
                f"obs length ({t_plus_1}) must equal actions length + 1 "
                f"({t + 1}); trajectories must store initial observation."
            )
        if self.rewards.shape[0] != t:
            raise ValueError(
                f"rewards length ({self.rewards.shape[0]}) must equal "
                f"actions length ({t})."
            )
        for name, arr in (("log_probs", self.log_probs), ("values", self.values)):
            if arr is None:
                continue
            if arr.shape[0] != t:
                raise ValueError(
                    f"{name} length ({arr.shape[0]}) must equal actions "
                    f"length ({t}) or be None."
                )
        if self.terminated and self.truncated:
            raise ValueError(
                "terminated and truncated cannot both be True; use "
                "AgentTrajectory.to_gymnasium_style() to coerce framework "
                "flags before constructing a RolloutBatch."
            )
