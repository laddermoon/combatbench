"""Stochastic policy interface for training.

:class:`StochasticPolicy` is a **standalone** interface (does NOT inherit
:class:`Policy`).  A policy class may implement *both* ``Policy`` and
``StochasticPolicy`` so it can be used:

- As a ``Policy`` (``act()`` → deterministic default behaviour) for
  deployment / competition / evaluation.
- As a ``StochasticPolicy`` (``sample()`` → stochastic sampling with
  exploration control) for training rollouts.

The :class:`ExploratoryPolicy` wrapper consumes a ``StochasticPolicy``
and exposes it as a ``Policy`` to the ``EpisodeRunner``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class StochasticPolicy(ABC):
    """Interface for policies that support stochastic sampling.

    ``sample()`` returns a stochastically sampled action.  The optional
    ``explore_intensity ∈ [-1, 1]`` (0 = neutral) scales the sampling
    distribution; the mapping is policy-defined.
    """

    @abstractmethod
    def sample(
        self,
        observation: Any,
        *,
        explore_intensity: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        """Sample an action with optional exploration intensity.

        Returns ``(action, extra)`` where *extra* is an optional dict
        of policy-defined auxiliary data (e.g. log-prob, value).
        """
        raise NotImplementedError
