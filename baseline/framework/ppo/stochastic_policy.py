"""Stochastic policy interface for training.

:class:`StochasticPolicy` is a **standalone** interface (does NOT inherit
:class:`Policy`).  A policy class may implement *both* ``Policy`` and
``StochasticPolicy`` so it can be used:

- As a ``Policy`` (``act()`` → deterministic default behaviour) for
  deployment / competition / evaluation.
- As a ``StochasticPolicy`` (``sample()`` → stochastic sampling with
  exploration control) for training rollouts.

The :class:`SamplingPolicy` wrapper consumes a ``StochasticPolicy``
and exposes it as a ``Policy`` to the ``EpisodeRunner``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from baseline.framework.ppo.sampling_context import SamplingContext


class StochasticPolicy(ABC):
    """Interface for policies that support stochastic sampling.

    ``sample()`` returns a stochastically sampled action.  The optional
    ``ctx`` (:class:`SamplingContext`) carries the per-frame sampling
    inputs — ``explore_factor``, ``reference_action``, ``delta_factor``,
    ``delta_mix`` — that the wrapper resolved for this step.  ``None``
    means neutral legacy sampling (equivalent to ``explore_factor=0``);
    the mapping from ctx fields to the sampling distribution is
    policy-defined.
    """

    @abstractmethod
    def sample(
        self,
        observation: Any,
        *,
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        """Sample an action under the given sampling context.

        Returns ``(action, extra)`` where *extra* is an optional dict
        of policy-defined auxiliary data (e.g. log-prob, value).
        """
        raise NotImplementedError
