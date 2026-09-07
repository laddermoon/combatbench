"""ExploratoryPolicy — wraps a StochasticPolicy with per-frame explore_factor.

This wrapper is the bridge between the training layer (which knows about
``explore_factor``) and the core framework (which does not).  It
implements the plain :class:`Policy` interface so that
:class:`EpisodeRunner` never sees ``explore_factor`` — the runner
just calls ``policy.act(obs, want_extra=...)``.

The wrapper:
1. Resolves ``explore_factor`` per frame (constant float or callable
   ``(obs, step) -> float``).
2. Calls ``inner.sample(obs, explore_factor=ef, want_extra=...)``.
3. Merges ``explore_factor`` into the returned ``extra`` dict so it
   travels through ``action_extras`` to recorders and trainers.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from envs.framework.policy import Policy

if TYPE_CHECKING:
    from baseline.framework.ppo.stochastic_policy import StochasticPolicy

#: Per-frame explore_factor: a constant float, or a callable
#: ``(obs, step) -> float``.  Callables must be top-level functions to
#: be picklable across multiprocessing workers.
EfSpec = Union[float, Callable[[np.ndarray, int], float]]


class ExploratoryPolicy(Policy):
    """Wrap a :class:`StochasticPolicy` with per-frame explore_factor.

    Implements :class:`Policy` by delegating to ``inner.sample()``.
    The ``EpisodeRunner`` sees a plain ``Policy`` and is unaware of
    exploration.

    Parameters
    ----------
    inner:
        The stochastic policy to wrap (must implement ``sample()``).
    explore_factor:
        Constant ``float`` or callable ``(obs, step) -> float``.
        Default ``0.0`` (neutral).
    """

    def __init__(
        self,
        inner: "StochasticPolicy",
        explore_factor: EfSpec = 0.0,
    ) -> None:
        if not hasattr(inner, "sample"):
            raise TypeError(
                f"inner must implement sample(); got {type(inner).__name__}"
            )
        self.inner = inner
        self._ef_spec: EfSpec = explore_factor
        self._step: int = 0

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        ef = (
            float(self._ef_spec(observation, self._step))
            if callable(self._ef_spec)
            else float(self._ef_spec)
        )
        self._step += 1
        action, extra = self.inner.sample(
            observation, explore_factor=ef, want_extra=want_extra,
        )
        if extra is not None:
            extra["explore_factor"] = ef
        else:
            extra = {"explore_factor": ef}
        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        self._step = 0
        reset_fn = getattr(self.inner, "reset", None)
        if callable(reset_fn):
            reset_fn(seed)
