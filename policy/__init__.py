"""
CombatBench Policy Module

This module defines the policy interface and provides reference implementations
for CombatBench combat policies.

Policy Interface:
    - __init__(observation_space, action_space, **kwargs): Initialize policy
    - act(obs, info=None) -> np.ndarray: Return action [-1, 1]^ACTION_DIM
    - reset(): Reset policy state at episode start

Available Policies:
    - BaseCombatPolicy: Abstract base class defining the interface
    - RandomCombatPolicy: Random action policy
    - StandingCombatPolicy: Standing still policy (zero actions)
"""

from .base import BaseCombatPolicy
from .random import RandomCombatPolicy
from .standing import StandingCombatPolicy

__all__ = [
    "BaseCombatPolicy",
    "RandomCombatPolicy",
    "StandingCombatPolicy",
]
