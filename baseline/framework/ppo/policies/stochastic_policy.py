"""Stochastic policy interface for training.

:class:`StochasticPolicy` extends the core :class:`Policy` with an
``explore_intensity`` parameter on ``act()``.  This is a **training-layer**
concept: the core ``Policy.act()`` is exploration-agnostic so that
eval / competition / round-runner callers never need to know about it.

Only stochastic policies used in RL training (e.g.
:class:`TruncatedNormalMLPPolicy`) implement this interface.  The
:class:`ExploratoryPolicy` wrapper consumes a ``StochasticPolicy`` and
exposes it as a plain ``Policy`` to the ``EpisodeRunner``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Tuple

from envs.framework.policy import Policy


class StochasticPolicy(Policy):
    """Policy that supports per-step exploration intensity.

    ``explore_intensity ∈ [-1, 1]`` (0 = neutral) scales the sampling
    distribution at action time.  The mapping is policy-defined;
    deterministic policies ignore it.
    """

    @abstractmethod
    def act(
        self,
        observation: Any,
        *,
        explore_intensity: float = 0.0,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        """Compute an action with exploration intensity.

        Same contract as :meth:`Policy.act`, plus ``explore_intensity``
        to control sampling width.
        """
        raise NotImplementedError
