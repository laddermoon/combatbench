"""Base policy ABC for CombatBench.

:class:`BaseCombatPolicy` is the recommended base class for combat
policies loaded via :func:`combatbench.policy.load_util.load_policy`.

It implements the canonical :class:`envs.framework.policy.Policy` Protocol
(see that module for the full contract) and adds a few ergonomics:

* ``observation_space`` / ``action_space`` wiring (gymnasium-compatible)
* ``**kwargs`` passthrough for dynamic configuration via
  :func:`load_policy` query-string parameters
* ``ACTION_DIM`` class attribute used by stock policies (random, etc.)
* a default no-op :meth:`reset`

Policies that do not need any of this can implement the Protocol directly
and still plug into :class:`EpisodeRunner` — structural typing is supported.

Contract summary (see ``envs/framework/policy.py`` for full doc)
----------------------------------------------------------------
* ``act(observation) -> action`` — REQUIRED. Return something coercible
  to ``np.ndarray(dtype=float32)``. Returning ``None`` is NOT allowed.
* ``reset(seed: Optional[int] = None) -> None`` — OPTIONAL. Default
  no-op; override if the policy holds RNG / recurrent state.
* ``act_with_extras(observation) -> (action, extras: dict)`` — OPTIONAL.
  Provide to emit per-step log-probs / values for on-policy RL.
* ``close() -> None`` — OPTIONAL. Release resources.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from gymnasium import spaces


class BaseCombatPolicy(ABC):
    """Abstract base class for all CombatBench combat policies.

    Subclasses MUST override :meth:`act`. Everything else is optional.
    """

    #: Default combat action dimensionality (humanoid21). Subclasses that
    #: target a different action space can override this class attribute
    #: or ignore it entirely.
    ACTION_DIM: int = 21

    def __init__(
        self,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        **kwargs: Any,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.kwargs = dict(kwargs)

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------
    @abstractmethod
    def act(self, observation: Any) -> np.ndarray:
        """Compute and return an action given the current observation.

        Parameters
        ----------
        observation:
            Whatever the bound observer plugin's ``get_output()`` returned.
            Usually a 1D ``np.ndarray`` of features; policies that want a
            richer input should declare a custom observer plugin and bind
            it via :class:`ObserverBinding`.

        Returns
        -------
        action:
            Array-like coercible to ``np.ndarray(dtype=float32)``. Must
            match the environment's expected action dimension. Returning
            ``None`` is NOT allowed.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks (defaults are safe no-ops)
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the policy's internal state at the start of a new episode.

        The ``seed`` argument is a deterministic per-policy child seed
        derived from the runner's ``base_seed`` via
        :class:`numpy.random.SeedSequence` — use it to reseed any
        stochastic components so rollouts are reproducible. Default is
        a no-op; override when the policy holds RNG or recurrent state.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
