"""``RolloutBatch``: the single data contract between rollout / algo / eval.

Design rationale (see ``baseline/DESIGN.md`` §3.3):

    - One episode, one agent, one ``RolloutBatch``. Multi-agent rollouts
      are expressed as ``dict[agent_id, list[RolloutBatch]]``.
    - Shapes follow the *framework convention*: with
      ``RolloutConfig.store_initial_observation=True`` (the default in
      ``EpisodeRunner``) there are ``T+1`` observations and ``T`` actions.
      The last observation doubles as the ``final_obs`` for truncation
      bootstrap — exposed via the :attr:`final_obs` property instead of
      being stored as a separate field, so the alignment stays honest.
    - ``log_probs`` / ``values`` are optional — populated when the policy
      exposes ``act_with_extras`` (``RolloutConfig.store_extras=True``)
      or by a post-rollout critic pass. Algorithms that don't need them
      (eval / random baselines) can leave them ``None``.

This file is intentionally small and import-light: it must be usable
inside worker processes without dragging torch in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RolloutBatch:
    """One episode of experience for one agent.

    Invariants (checked by :meth:`validate`):

        ``obs.shape[0] == actions.shape[0] + 1 == rewards.shape[0] + 1``

        When ``log_probs`` / ``values`` are provided, their length equals
        ``actions.shape[0]``.
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
        -state production code can skip this (the collector is supposed
        to produce well-formed batches).
        """
        if self.obs.ndim < 1:
            raise ValueError("obs must be at least 1-D (got shape {})".format(self.obs.shape))
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
            # Not an invariant violation per se, but it almost always
            # signals a plumbing bug. Keep as a warn-level assertion.
            raise ValueError(
                "terminated and truncated cannot both be True; use "
                "exactly one to signal the end of this episode."
            )
