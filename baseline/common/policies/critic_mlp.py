"""Generic state-value MLP critic.

Two hidden Tanh layers, scalar output. Dim-parameterized so it can be
used with any continuous-control environment that exposes a flat
observation vector — the corresponding actor backbone is
:class:`baseline.common.policies.TanhGaussianMLPPolicy`.
"""
from __future__ import annotations

import torch
from torch import nn


class CriticMLP(nn.Module):
    """V-function approximator for actor-critic methods.

    Args:
        obs_dim: Length of the flat observation vector.
        hidden_dim: Width of both hidden layers.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a scalar value per observation; trailing dim is squeezed."""
        return self.net(obs).squeeze(-1)
