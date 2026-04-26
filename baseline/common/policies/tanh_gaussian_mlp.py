"""Continuous-control actor backbone (Tanh-squashed Gaussian).

Checkpoint / export IO has moved to :mod:`baseline.common.policies.checkpoint`
and is re-exported from this module for backward compatibility with scripts
that imported it from here (pre-PR1). New code should import directly from
:mod:`baseline.common.policies.checkpoint`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .checkpoint import (
    DEFAULT_EXPORT_ACTOR_HIDDEN_DIM,
    build_actor_export_payload,
    build_export_policy_code,
    export_actor_policy_artifacts,
    export_policy_artifacts_from_checkpoint,
)

DEFAULT_LOG_STD_MIN = -4.0
DEFAULT_LOG_STD_MAX = 1.0

__all__ = [
    "DEFAULT_LOG_STD_MIN",
    "DEFAULT_LOG_STD_MAX",
    "DEFAULT_EXPORT_ACTOR_HIDDEN_DIM",
    "TanhGaussianMLPPolicy",
    "build_actor_export_payload",
    "build_export_policy_code",
    "export_actor_policy_artifacts",
    "export_policy_artifacts_from_checkpoint",
]


class TanhGaussianMLPPolicy(nn.Module):
    """Generic continuous-control policy backbone for Box-like actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.expand_as(mean)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw_actions = torch.atanh(clipped_actions)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw_actions) - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob.sum(dim=-1), entropy

    def act_numpy(self, obs: np.ndarray, device: torch.device, deterministic: bool) -> tuple[np.ndarray, Optional[float]]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.deterministic_action(obs_tensor)
                log_prob = None
            else:
                action, log_prob = self.sample_action(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if log_prob is None:
            return action_np, None
        return action_np, float(log_prob.item())


