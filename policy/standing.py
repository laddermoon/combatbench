"""
Standing Combat Policy

A policy that returns zero actions, making the robot maintain its current pose.
Useful for testing, debugging, and as a stationary opponent for attacker training.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseCombatPolicy


class StandingCombatPolicy(BaseCombatPolicy):
    """
    Standing still policy for CombatBench.

    Always returns zero actions, which causes the robot to maintain its
    current joint positions (typically the standing pose).
    """

    def __init__(
        self,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the standing policy.

        Args:
            observation_space: Gymnasium observation space (unused)
            action_space: Gymnasium action space (unused)
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(observation_space, action_space, **kwargs)
        self._zero_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Return zero action to maintain current pose.

        Args:
            obs: Current observation (unused)
            info: Environment info dict (unused)

        Returns:
            action: Zero action array (all zeros)
        """
        return self._zero_action.copy()

    def reset(self) -> None:
        """Reset policy (no-op for standing policy)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
