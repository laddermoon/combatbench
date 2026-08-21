"""NoisyPolicyWrapper — wraps a base Policy and adds Gaussian noise to actions.

Used in state pool collection to generate behaviorally diverse states from
a single trained policy by perturbing its action output with varying noise
levels (sigma).

Compatible with ``PolicyBlueprint`` serialization: the wrapper accepts
``base_cls`` (a policy class descriptor string) and ``base_config`` (init
kwargs for the base policy), so it can be fully specified in a YAML
blueprint without pre-instantiating the base policy.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from envs.framework.policy import Policy, _resolve_policy_class


class NoisyPolicyWrapper(Policy):
    """Wrap a base policy and add i.i.d. Gaussian noise to its actions.

    Parameters
    ----------
    base_cls : str
        Class descriptor for the base policy, e.g.
        ``"file:/path/to/policy.py:ExportedMLPPolicy"`` or
        ``"package.module:PolicyClass"``.
    base_config : dict
        Keyword arguments forwarded to the base policy's ``__init__``.
    sigma : float
        Standard deviation of the Gaussian noise added to actions.
        ``sigma=0`` reproduces the base policy exactly.
    seed : int, optional
        Seed for the internal RNG used for noise generation.
    """

    def __init__(
        self,
        base_cls: str,
        base_config: Optional[dict] = None,
        sigma: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        cls = _resolve_policy_class(base_cls)
        if not issubclass(cls, Policy):
            raise TypeError(
                f"{base_cls} resolves to {cls.__name__}, which does not "
                f"subclass envs.framework.policy.Policy"
            )
        self.base = cls(**(base_config or {}))
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Any]:
        action, extras = self.base.act(observation, want_extra=want_extra)
        action = np.asarray(action, dtype=np.float32)
        if self.sigma > 0:
            noise = self._rng.normal(0.0, self.sigma, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)
        return action, extras

    def reset(self, seed: Optional[int] = None) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset(seed)
        self._rng = np.random.default_rng(seed)
