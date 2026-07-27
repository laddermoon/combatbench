"""Critic that subtracts potential φ from its output.

Implements V(s) = V_θ(s) - φ(s), where φ is computed from the observation
vector using the humanoid21 observation layout:

  obs[42:45] = first column of world rotation matrix (c1)
  obs[45:48] = second column of world rotation matrix (c2)
  obs[48]    = root height (Z)

  uprightness = c3[2] = c1[0]*c2[1] - c1[1]*c2[0]
  φ = uprightness * height / standing_height

By subtracting φ for free, V_θ only needs to learn V^Dense(s) — the dense
value function that ST-6's critic learns easily.  This breaks the
"accounting deadlock" where the critic must reconstruct -φ from the noisy
Delta reward signal.

Used by the killer falsification experiment for ST-4.
"""
from __future__ import annotations

import torch
from torch import nn


class PhiSubtractedCritic(nn.Module):
    """V(s) = V_θ(s) - φ(s), with φ computed from obs.

    The MLP backbone is identical to CriticMLP (2-layer Tanh, scalar output).
    The φ subtraction is a fixed (non-learnable) operation.

    Args:
        obs_dim: Length of the flat observation vector (default 96).
        hidden_dim: Width of both hidden layers (default 256).
        standing_height: Nominal standing height for φ normalization (default 1.28).
    """

    def __init__(
        self,
        obs_dim: int = 96,
        hidden_dim: int = 256,
        standing_height: float = 1.28,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.standing_height = float(standing_height)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _compute_phi(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute φ = uprightness * (height / standing_height) from obs.

        obs layout (humanoid21):
          [42:45] = c1 (1st col of world rot mat)
          [45:48] = c2 (2nd col of world rot mat)
          [48]    = height (Z)

        uprightness = c3[2] = c1[0]*c2[1] - c1[1]*c2[0]
        """
        c1 = obs[:, 42:45]  # (N, 3)
        c2 = obs[:, 45:48]  # (N, 3)
        height = obs[:, 48]  # (N,)
        uprightness = c1[:, 0] * c2[:, 1] - c1[:, 1] * c2[:, 0]  # (N,)
        return uprightness * height / self.standing_height

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V_θ(s) - φ(s), squeezed to 1-D."""
        v_theta = self.net(obs).squeeze(-1)
        phi = self._compute_phi(obs)
        return v_theta - phi
