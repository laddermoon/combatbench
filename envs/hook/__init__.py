"""
Hook module for CombatBench

This module provides the hook pattern for modifying simulation states
at specific hook points.
"""

from .base_hook import BaseHook, HookWrapper, InvokeType
from .humanoid21_base_hook import (
    GymEnvironmentAdapter,
    GymHookWrapper,
    TerminationHook,
    VideoRecordingHook,
)

__all__ = [
    # Core Hook framework
    'BaseHook',
    'HookWrapper',
    'InvokeType',

    # Gym adaptation utilities
    'GymEnvironmentAdapter',
    'GymHookWrapper',

    # Basic Hooks
    'TerminationHook',
    'VideoRecordingHook',
]
