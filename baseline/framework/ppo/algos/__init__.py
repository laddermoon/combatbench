"""PPO algorithm primitives: advantage estimation + on-policy update steps.

Moved from ``baseline/common/algos/`` — these are PPO-specific (GAE,
PPO clipped surrogate loss).  SAC uses Q-learning, not GAE.
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
