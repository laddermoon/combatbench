"""
No-Op Action Combat Policy

A policy that always returns None to hold the previous action unchanged.
Useful for testing and as a minimal baseline.
"""

from typing import Any, Dict, Optional

import numpy as np

from policy.base import BaseCombatPolicy


class NoOpActionPolicy(BaseCombatPolicy):
    """
    No-op action policy for CombatBench.

    Always returns None to hold the previous action unchanged.
    """

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Return None to hold previous action unchanged.

        Args:
            obs: Current observation (unused)
            info: Environment info dict (unused)

        Returns:
            None, indicating the previous action should be held
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
