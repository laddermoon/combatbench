"""Random combat policy.

Generates uniform random actions in ``[-scale, scale]``. Conforms to the
canonical :class:`envs.framework.policy.Policy` contract; ``reset(seed)``
reseeds the internal RNG so rollouts are reproducible from the runner's
``base_seed``.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from policy.base import BaseCombatPolicy


class RandomCombatPolicy(BaseCombatPolicy):
    """Uniform random action policy, actions in ``[-scale, scale]``."""

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        scale: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(observation_space, action_space, **kwargs)
        self.scale = float(scale)
        self._init_seed = seed
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Any) -> np.ndarray:
        return self.rng.uniform(-self.scale, self.scale, self.ACTION_DIM).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed the internal RNG.

        When the runner supplies a per-episode child seed, use it; otherwise
        fall back to the seed passed at construction time (so a caller that
        never provided a seed still gets fresh randomness each episode).
        """
        self.rng = np.random.default_rng(seed if seed is not None else self._init_seed)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale})"
