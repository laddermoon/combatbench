"""Algorithm primitives: advantage estimation + on-policy update steps.

See ``baseline/DESIGN.md`` §3.6.
"""

from .advantages import (
    compute_gae,
    compute_grpo_advantages,
    compute_returns_to_go,
)
from .ppo import PPOLossOutput, ppo_loss

__all__ = [
    "compute_gae",
    "compute_grpo_advantages",
    "compute_returns_to_go",
    "PPOLossOutput",
    "ppo_loss",
]
