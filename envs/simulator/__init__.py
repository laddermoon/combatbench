"""
Simulator module for CombatBench

This module provides the OpenSimulator interface and its implementations.
"""

from .open_simulator import OpenSimulator
from .humanoid21 import Humanoid21Simulator

__all__ = [
    'OpenSimulator',
    'Humanoid21Simulator',
]
