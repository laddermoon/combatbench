"""Unit tests for SAC trainer (sac_update_v2).

Tests:
- Q critic forward pass shapes.
- Multi-head Q grouping.
- sac_update_v2 runs without error.
- Action gradient normalization stats update.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from baseline.framework.sac.experiment import SACParams, SACRewardChannel
from baseline.framework.sac.networks import MultiHeadQCritic, QTrunkHeads
from baseline.framework.sac.trainer import GradNormStats, sac_update_v2


# ---------------------------------------------------------------------------
# Simple actor for testing (implements sample_action)
# ---------------------------------------------------------------------------

class SimpleActor(nn.Module):
    """Minimal actor for testing: tanh-squashed Gaussian."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def sample_action(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)


def test_qtrunk_heads_forward():
    """Test QTrunkHeads forward pass."""
    obs_dim, action_dim, hidden_dim = 8, 3, 16
    channels = ("r_a", "r_b")
    net = QTrunkHeads(obs_dim, action_dim, hidden_dim, channels)

    obs = torch.randn(4, obs_dim)
    act = torch.randn(4, action_dim)

    q_a = net(obs, act, "r_a")
    q_b = net(obs, act, "r_b")
    assert q_a.shape == (4,)
    assert q_b.shape == (4,)

    all_q = net.forward_all(obs, act)
    assert set(all_q.keys()) == {"r_a", "r_b"}

    print("test_qtrunk_heads_forward: PASS")


def test_multihead_q_critic_grouping():
    """Test that channels are correctly grouped by trunk_group."""
    channels = (
        SACRewardChannel(name="r_a", gamma=0.99, trunk_group="g1"),
        SACRewardChannel(name="r_b", gamma=0.99, trunk_group="g1"),
        SACRewardChannel(name="r_c", gamma=0.95, trunk_group="g2"),
    )

    critic = MultiHeadQCritic(
        obs_dim=8, action_dim=3, channels=channels,
        hidden_dim=16, layer_norm=False, critic_lr=1e-3,
        device=torch.device("cpu"),
    )

    # Should have 2 groups
    assert len(critic.groups) == 2
    assert critic.n_networks == 4  # 2 groups × 2 (Q1+Q2)

    # Channel to group mapping
    assert critic.channel_to_group["r_a"] == "g1"
    assert critic.channel_to_group["r_b"] == "g1"
    assert critic.channel_to_group["r_c"] == "g2"

    # Forward pass
    obs = torch.randn(4, 8)
    act = torch.randn(4, 3)
    q_a = critic.q1_forward(obs, act, "r_a")
    q_c = critic.q1_forward(obs, act, "r_c")
    assert q_a.shape == (4,)
    assert q_c.shape == (4,)

    # forward_all
    all_q = critic.q1_forward_all(obs, act)
    assert set(all_q.keys()) == {"r_a", "r_b", "r_c"}

    print("test_multihead_q_critic_grouping: PASS")


def test_multihead_auto_group_by_gamma():
    """Test auto-grouping by gamma when trunk_group is None."""
    channels = (
        SACRewardChannel(name="r_a", gamma=0.99),
        SACRewardChannel(name="r_b", gamma=0.99),
        SACRewardChannel(name="r_c", gamma=0.95),
    )

    critic = MultiHeadQCritic(
        obs_dim=8, action_dim=3, channels=channels,
        hidden_dim=16, layer_norm=False, critic_lr=1e-3,
        device=torch.device("cpu"),
    )

    # r_a and r_b should share a group (same gamma), r_c separate
    assert critic.channel_to_group["r_a"] == critic.channel_to_group["r_b"]
    assert critic.channel_to_group["r_a"] != critic.channel_to_group["r_c"]

    print("test_multihead_auto_group_by_gamma: PASS")


def test_sac_update_runs():
    """Test that sac_update_v2 runs without error."""
    obs_dim, action_dim = 8, 3
    channels = (
        SACRewardChannel(name="r_a", gamma=0.99, n_step=1),
        SACRewardChannel(name="r_b", gamma=0.99, n_step=1),
    )

    actor = SimpleActor(obs_dim, action_dim)
    critic = MultiHeadQCritic(
        obs_dim=obs_dim, action_dim=action_dim, channels=channels,
        hidden_dim=32, layer_norm=False, critic_lr=1e-3,
        device=torch.device("cpu"),
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)

    sp = SACParams(use_grad_norm=False)  # Use naive mode for simple test

    # Create a fake batch
    B = 16
    batch = {
        "obs": torch.randn(B, obs_dim),
        "actions": torch.randn(B, action_dim),
        "sample_weights": torch.ones(B),
    }
    for ch in ("r_a", "r_b"):
        batch[f"rewards_{ch}"] = torch.randn(B, 1)
        batch[f"dones_{ch}"] = torch.zeros(B, 1)
        batch[f"next_obs_{ch}"] = torch.randn(B, obs_dim)
        batch[f"valid_steps_{ch}"] = torch.ones(B)
        batch[f"actor_weights_{ch}"] = torch.ones(B)

    stats = sac_update_v2(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        batch=batch,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
    )

    assert "actor_loss" in stats
    assert "alpha" in stats
    assert "q1_loss_r_a" in stats
    assert "q1_loss_r_b" in stats

    print("test_sac_update_runs: PASS")


def test_grad_norm_stats():
    """Test GradNormStats running statistics."""
    stats = GradNormStats(("r_a", "r_b"), ema_decay=0.9)

    assert not stats.initialized

    # First update
    stats.update({"r_a": 1.0, "r_b": 2.0})
    assert stats.initialized
    assert abs(stats.scale("r_a") - (1.0 + 1e-6)) < 0.01
    assert abs(stats.scale("r_b") - (2.0 + 1e-6)) < 0.01

    # Second update (EMA)
    stats.update({"r_a": 3.0, "r_b": 4.0})
    # EMA: 0.9 * 1.0 + 0.1 * 9.0 = 0.9 + 0.9 = 1.8
    # scale = sqrt(1.8) ≈ 1.34
    assert 1.0 < stats.scale("r_a") < 1.5

    print("test_grad_norm_stats: PASS")


def test_sac_update_with_grad_norm():
    """Test that sac_update_v2 works with gradient normalization."""
    obs_dim, action_dim = 8, 3
    channels = (
        SACRewardChannel(name="r_a", gamma=0.99, n_step=1),
        SACRewardChannel(name="r_b", gamma=0.99, n_step=1),
    )

    actor = SimpleActor(obs_dim, action_dim)
    critic = MultiHeadQCritic(
        obs_dim=obs_dim, action_dim=action_dim, channels=channels,
        hidden_dim=32, layer_norm=False, critic_lr=1e-3,
        device=torch.device("cpu"),
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)

    sp = SACParams(use_grad_norm=True, grad_norm_est_interval=1)
    grad_norm_stats = GradNormStats(("r_a", "r_b"))

    B = 16
    batch = {
        "obs": torch.randn(B, obs_dim),
        "actions": torch.randn(B, action_dim),
        "sample_weights": torch.ones(B),
    }
    for ch in ("r_a", "r_b"):
        batch[f"rewards_{ch}"] = torch.randn(B, 1)
        batch[f"dones_{ch}"] = torch.zeros(B, 1)
        batch[f"next_obs_{ch}"] = torch.randn(B, obs_dim)
        batch[f"valid_steps_{ch}"] = torch.ones(B)
        batch[f"actor_weights_{ch}"] = torch.ones(B)

    stats = sac_update_v2(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        batch=batch,
        channels=channels,
        sp=sp,
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
        grad_norm_stats=grad_norm_stats,
        grad_norm_step=0,
    )

    assert "grad_share_r_a" in stats
    assert "grad_share_r_b" in stats
    assert "grad_scale_r_a" in stats
    assert "grad_scale_r_b" in stats

    # Gradient shares should sum to ~1.0
    total_share = stats["grad_share_r_a"] + stats["grad_share_r_b"]
    assert abs(total_share - 1.0) < 0.1, f"Shares sum to {total_share}, expected ~1.0"

    print("test_sac_update_with_grad_norm: PASS")


def test_soft_update():
    """Test that soft target update changes target parameters."""
    channels = (
        SACRewardChannel(name="r_a", gamma=0.99),
    )

    critic = MultiHeadQCritic(
        obs_dim=8, action_dim=3, channels=channels,
        hidden_dim=16, layer_norm=False, critic_lr=1e-3,
        device=torch.device("cpu"),
    )

    # Get a target parameter before update
    target_param = next(critic.groups[
        list(critic.groups.keys())[0]
    ].q1_target.parameters())

    before = target_param.data.clone()

    # Modify the online network
    online_param = next(critic.groups[
        list(critic.groups.keys())[0]
    ].q1.parameters())
    online_param.data.fill_(1.0)

    # Soft update with tau=0.1
    critic.soft_update(0.1)

    after = target_param.data.clone()

    # Target should have moved toward the online network
    assert not torch.allclose(before, after), "Target should change after soft update"

    print("test_soft_update: PASS")


if __name__ == "__main__":
    test_qtrunk_heads_forward()
    test_multihead_q_critic_grouping()
    test_multihead_auto_group_by_gamma()
    test_sac_update_runs()
    test_grad_norm_stats()
    test_sac_update_with_grad_norm()
    test_soft_update()
    print("\nAll trainer tests passed!")
