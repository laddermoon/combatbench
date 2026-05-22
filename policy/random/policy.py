"""Random combat policy.

Uniform random actions in ``[-scale, scale]``. Conforms to the canonical
:class:`envs.framework.policy.Policy` ABC; ``reset(seed)`` reseeds the
internal RNG so rollouts are reproducible from the runner's ``base_seed``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from envs.framework.policy import Policy


class RandomCombatPolicy(Policy):
    """Uniform random action policy, actions in ``[-scale, scale]``."""

    def __init__(
        self,
        scale: float = 0.1,
        seed: Optional[int] = None,
        action_dim: int = 21,
        **_ignored: Any,
    ) -> None:
        # Accept and silently drop unknown kwargs so load_policy query-string
        # parameters that don't apply (e.g. ``model_path``) don't crash.
        self.scale = float(scale)
        self.action_dim = int(action_dim)
        self._init_seed = seed
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        action = self.rng.uniform(-self.scale, self.scale, self.action_dim).astype(np.float32)
        return action, None

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed the internal RNG.

        When the runner supplies a per-episode child seed, use it; otherwise
        fall back to the seed passed at construction time (so a caller that
        never provided a seed still gets fresh randomness each episode).
        """
        self.rng = np.random.default_rng(seed if seed is not None else self._init_seed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale}, action_dim={self.action_dim})"
