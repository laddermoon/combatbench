"""
CombatBench Examples

This package contains example scripts demonstrating the usage of various
Humanoid21 environments.

Each example script runs a few episodes and saves a video.
"""

from .humanoid21_single_agent_example import run_single_agent_example
from .humanoid21_vs_frozen_example import run_vs_frozen_example
from .humanoid21_vs_standing_example import run_vs_standing_example
from .humanoid21_vs_policy_example import run_vs_policy_example
from .humanoid21_non_fall_example import run_non_fall_example
from .humanoid21_fall_example import run_fall_example
from .humanoid21_dual_agent_example import run_dual_agent_example
from .humanoid21_match_example import run_match_example

__all__ = [
    "run_single_agent_example",
    "run_vs_frozen_example",
    "run_vs_standing_example",
    "run_vs_policy_example",
    "run_non_fall_example",
    "run_fall_example",
    "run_dual_agent_example",
    "run_match_example",
]
