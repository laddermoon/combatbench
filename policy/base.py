"""
Base Policy Interface for CombatBench

All combat policies must inherit from BaseCombatPolicy and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces


class BaseCombatPolicy(ABC):
    """
    Abstract base class for all CombatBench combat policies.

    A policy receives observations from the environment and returns actions.
    """

    def __init__(self, observation_space: Optional[spaces.Space] = None, action_space: Optional[spaces.Space] = None, **kwargs: Any):
        self.observation_space = observation_space
        self.action_space = action_space
        self.kwargs = dict(kwargs)

    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Compute and return an action given the current observation.

        Args:
            obs: Current observation from the environment
            info: Optional info dict from the environment

        Returns:
            action: Action array, or None to hold previous action unchanged

        Note:
            Subclasses MUST implement this method.
        """
        pass

    def reset(self) -> None:
        """
        Reset the policy's internal state at the start of a new episode.

        This method is called at the beginning of each episode and should
        reset any internal state (e.g., hidden states, buffers, counters).

        Note:
            Subclasses MAY override this method if they maintain internal state.
            The default implementation does nothing.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
