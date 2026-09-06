"""ExploratoryPolicy — wraps a StochasticPolicy with per-frame explore_intensity.

This wrapper is the bridge between the training layer (which knows about
``explore_intensity``) and the core framework (which does not).  It
implements the plain :class:`Policy` interface so that
:class:`EpisodeRunner` never sees ``explore_intensity`` — the runner
just calls ``policy.act(obs, want_extra=...)``.

The wrapper:
1. Resolves ``explore_intensity`` per frame (constant float or callable
   ``(obs, step) -> float``).
2. Calls ``inner.act(obs, explore_intensity=ei, want_extra=...)``.
3. Merges ``explore_intensity`` into the returned ``extra`` dict so it
   travels through ``action_extras`` to recorders and trainers.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from envs.framework.policy import Policy

# Lazy import to avoid circular dependency:
# rollout.__init__ → exploratory_policy → ppo.policies → ppo.__init__ → experiment → rollout
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from baseline.framework.ppo.policies.stochastic_policy import StochasticPolicy

#: Per-frame explore_intensity: a constant float, or a callable
#: ``(obs, step) -> float``.  Callables must be top-level functions to
#: be picklable across multiprocessing workers.
EiSpec = Union[float, Callable[[np.ndarray, int], float]]


class ExploratoryPolicy(Policy):
    """Wrap a :class:`StochasticPolicy` with per-frame explore_intensity.

    Parameters
    ----------
    inner:
        The stochastic policy to wrap.
    explore_intensity:
        Constant ``float`` or callable ``(obs, step) -> float``.
        Default ``0.0`` (neutral).
    """

    def __init__(
        self,
        inner: "StochasticPolicy",
        explore_intensity: EiSpec = 0.0,
    ) -> None:
        # Duck-typed: inner must have act(obs, *, explore_intensity, want_extra).
        # We don't isinstance-check to avoid importing StochasticPolicy at
        # module load time (circular dependency via ppo.__init__).
        if not hasattr(inner, "act"):
            raise TypeError(
                f"inner must be a policy with act(); got {type(inner).__name__}"
            )
        self.inner = inner
        self._ei_spec: EiSpec = explore_intensity
        self._step: int = 0

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[Any, Optional[dict]]:
        ei = (
            float(self._ei_spec(observation, self._step))
            if callable(self._ei_spec)
            else float(self._ei_spec)
        )
        self._step += 1
        action, extra = self.inner.act(
            observation, explore_intensity=ei, want_extra=want_extra,
        )
        if extra is not None:
            extra["explore_intensity"] = ei
        else:
            extra = {"explore_intensity": ei}
        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        self._step = 0
        reset_fn = getattr(self.inner, "reset", None)
        if callable(reset_fn):
            reset_fn(seed)
