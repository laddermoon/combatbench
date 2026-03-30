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

    A policy receives observations from the environment and returns
    actions for the 21-DOF humanoid robot.

    Observation Space (127 dims):
        - Joint positions (21) + velocities (21)
        - Root state: height (1) + local orientation (6) + velocities (6)
        - Tactile: feet contact (2) + external forces (6)
        - Opponent observation (64): relative position/velocity, orientation, keypoints

    Action Space (21 dims):
        - Actions are normalized in [-1, 1]
        - Each action dimension corresponds to a controlled joint:
            * Abdomen: abdomen_z, abdomen_y, abdomen_x (3)
            * Right leg: hip_x_right, hip_z_right, hip_y_right, knee_right, ankle_y_right, ankle_x_right (6)
            * Left leg: hip_x_left, hip_z_left, hip_y_left, knee_left, ankle_y_left, ankle_x_left (6)
            * Right arm: shoulder1_right, shoulder2_right, elbow_right (3)
            * Left arm: shoulder1_left, shoulder2_left, elbow_left (3)
    """

    # CombatBench humanoid robot action dimension
    ACTION_DIM = 21

    def __init__(
        self,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        **kwargs
    ):
        """
        Initialize the policy.

        Args:
            observation_space: Gymnasium observation space (optional, for validation)
            action_space: Gymnasium action space (optional, for validation)
            **kwargs: Additional policy-specific parameters
        """
        self.observation_space = observation_space
        self.action_space = action_space

        # Validate action dimension if action_space is provided
        if action_space is not None:
            if hasattr(action_space, 'shape'):
                expected_dim = self.ACTION_DIM
                actual_dim = action_space.shape[0]
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Action space dimension mismatch: "
                        f"expected {expected_dim}, got {actual_dim}"
                    )

    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Compute and return an action given the current observation.

        Args:
            obs: Current observation (127-dim numpy array)
                  Contains proprioception, root state, and opponent information
            info: Optional info dict from the environment containing:
                - scores: Current HP of both robots
                - positions: Robot positions
                - robot_states: Detailed state information
                - relative_metrics: Distance, facing direction, etc.
                - hit_records: Recent collision/damage events

        Returns:
            action: Action array with shape (ACTION_DIM,), values in [-1, 1]
                    Actions will be clipped to [-1, 1] by the environment

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
