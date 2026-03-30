"""
Random Combat Policy

A policy that generates random actions within a specified range.
Useful for baseline comparisons and testing.
"""

from typing import Any, Dict, Optional

import numpy as np

from policy.base import BaseCombatPolicy


class RandomCombatPolicy(BaseCombatPolicy):
    """
    Random action policy for CombatBench.

    Generates random actions uniformly distributed within [-scale, scale].
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        scale: float = 0.1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the random policy.

        Args:
            observation_space: Gymnasium observation space (unused)
            action_space: Gymnasium action space (unused)
            scale: Maximum absolute value of random actions (default: 0.1)
                   Actions will be in [-scale, scale]
            seed: Random seed for reproducibility (default: None)
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate a random action.

        Args:
            obs: Current observation (unused)
            info: Environment info dict (unused)

        Returns:
            action: Random action array with values in [-scale, scale]
        """
        return self.rng.uniform(-self.scale, self.scale, self.ACTION_DIM).astype(np.float32)

    def reset(self) -> None:
        """Reset policy (no-op for random policy)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale})"
