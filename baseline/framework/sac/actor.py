"""SAC actor boundary types.

The SAC framework owns its actor contract instead of importing the PPO
``TrainablePolicy`` ABC. This module intentionally stays small: the full
S01 policy implementation is adapted in a later phase-2 package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint


class SACActor(Protocol):
    """Minimum structural contract consumed by the SAC training loop."""

    obs_dim: int
    action_dim: int

    def parameters(self): ...

    def sample_action(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiably sample an action and its joint log-density."""
        ...

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic evaluation action."""
        ...

    def to_blueprint(
        self,
        dest_path: Optional[str] = None,
        *,
        stochastic: bool = False,
    ) -> PolicyBlueprint:
        """Export the actor as a deployable policy blueprint."""
        ...

    def export_policy_artifacts(
        self,
        dest_dir: Path,
        *,
        stochastic: bool = False,
    ) -> None:
        """Export additional policy artifacts when supported."""
        ...


__all__ = ["SACActor"]
