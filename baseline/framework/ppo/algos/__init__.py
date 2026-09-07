"""PPO algorithm primitives: advantage estimation + return computation.

Moved from ``baseline/common/algos/`` — these are PPO-specific (GAE,
returns-to-go).  SAC uses Q-learning, not GAE.

P1-6: Removed ``ppo_loss`` / ``PPOLossOutput`` (dead code — the actual
PPO surrogate is inlined in ``trainer.py`` with multi-critic support
that ``ppo_loss`` could not express).  Removed ``compute_grpo_advantages``
(GRPO is unrelated to this framework's multi-channel design).
Kept ``compute_gae`` and ``compute_returns_to_go`` (the latter is
needed for P1-1's λ-invariant EV computation).
"""

from .advantages import (
    compute_gae,
    compute_returns_to_go,
)

__all__ = [
    "compute_gae",
    "compute_returns_to_go",
]
