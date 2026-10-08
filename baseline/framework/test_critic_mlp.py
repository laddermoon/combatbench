"""Sanity tests for ``CriticMLP``: shape & dim parameterization."""
from __future__ import annotations

import torch

from baseline.framework.critic_mlp import CriticMLP


def test_scalar_output_per_observation():
    critic = CriticMLP(obs_dim=8, hidden_dim=16)
    obs = torch.zeros(5, 8)
    out = critic(obs)
    assert out.shape == (5,)


def test_dim_parameterized_no_default_obs_dim():
    # Different obs_dim / hidden_dim should both work.
    for obs_dim, hidden_dim in [(4, 8), (37, 64), (1, 2)]:
        critic = CriticMLP(obs_dim=obs_dim, hidden_dim=hidden_dim)
        out = critic(torch.zeros(3, obs_dim))
        assert out.shape == (3,)
        assert critic.obs_dim == obs_dim
        assert critic.hidden_dim == hidden_dim
