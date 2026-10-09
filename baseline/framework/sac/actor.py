"""SAC actor boundary types.

The SAC framework owns its actor contract instead of importing the PPO
``TrainablePolicy`` ABC. This module intentionally stays small; concrete
policy families implement this structural contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint

from .collection import SACBehaviorSpec


class SACActor(Protocol):
    """Structural contract consumed by the SAC training loop (A4.3).

    The eight-cell TN actors implement the full contract below.  The
    legacy ``S01Actor`` (``actor_arch="legacy_tanh"``) only implements the
    trainer-facing subset (``sample_action`` / ``deterministic_action`` /
    export) and is retained for checkpoint compatibility.
    """

    obs_dim: int
    action_dim: int

    def parameters(self): ...

    def sample_action(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiably sample an action and its joint log-density."""
        ...

    def distribution(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return π parameters {mu[B,K,D], sigma[B,K,D], logits[B,K]}."""
        ...

    def expectation_samples(
        self,
        obs: torch.Tensor,
        u_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerated reparameterized samples [B,K,M,D], joint mixture
        log_prob [B,K,M], differentiable integration_weights [B,K,M]."""
        ...

    def sample_behavior(
        self,
        obs: torch.Tensor,
        spec: SACBehaviorSpec,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample a behavior action under exploration factor e∈[-1,1]."""
        ...

    def uncertainty(self, obs: torch.Tensor, kind: str) -> torch.Tensor:
        """Differentiable marginal effective-width U over dims."""
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
