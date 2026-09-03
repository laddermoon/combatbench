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


def test_humanoid21_back_compat_alias():
    """The old ``baseline.humanoid21.base.Critic`` symbol must still exist
    and produce the same outputs as the canonical CriticMLP."""
    from baseline.humanoid21.base import Critic

    torch.manual_seed(0)
    critic_old = Critic(obs_dim=12, hidden_dim=32)
    torch.manual_seed(0)
    critic_new = CriticMLP(obs_dim=12, hidden_dim=32)

    obs = torch.randn(4, 12)
    out_old = critic_old(obs)
    out_new = critic_new(obs)
    assert out_old.shape == out_new.shape == (4,)
    # Same seed → same init → same output.
    torch.testing.assert_close(out_old, out_new)
