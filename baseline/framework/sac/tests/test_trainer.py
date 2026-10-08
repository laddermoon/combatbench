"""Permanent tests for S01 actor and standard Shannon SAC update."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline.framework.sac.experiment import SACParams, SACRewardChannel
from baseline.framework.sac.networks import MultiHeadQCritic
from baseline.framework.sac.s01_actor import S01Actor
from baseline.framework.sac.trainer import (
    SACTrainerError,
    compute_critic_targets,
    sac_update,
    validate_sac_channels,
)


def _batch(B: int = 8, obs_dim: int = 6, action_dim: int = 3, C: int = 2):
    torch.manual_seed(4)
    terminated = torch.zeros(B, dtype=torch.bool)
    truncated = torch.zeros(B, dtype=torch.bool)
    bootstrap = torch.ones(B)
    terminated[-1] = True
    bootstrap[-1] = 0.0
    return {
        "obs": torch.randn(B, obs_dim),
        "actions": torch.empty(B, action_dim).uniform_(-0.8, 0.8),
        "next_obs": torch.randn(B, obs_dim),
        "rewards": torch.randn(B, C),
        "channel_valid": torch.ones(B, C, dtype=torch.bool),
        "terminated": terminated,
        "truncated": truncated,
        "bootstrap": bootstrap,
        "actor_gate": torch.ones(B, C),
        "actor_weight": torch.full((B, C), 1.0 / C),
        "actor_gate_next": torch.ones(B, C),
        "actor_weight_next": torch.full((B, C), 1.0 / C),
        "sample_weight": torch.ones(B),
        "policy_action": torch.zeros(B, action_dim),
        "sample_ids": torch.arange(B, dtype=torch.int64),
        "source_keys": [f"src:{i}" for i in range(B)],
    }


def _models(C: int = 2, obs_dim: int = 6, action_dim: int = 3):
    torch.manual_seed(5)
    channels = tuple(
        SACRewardChannel(name=f"r{i}", gamma=0.99) for i in range(C)
    )
    actor = S01Actor(
        obs_dim, action_dim, hidden_dim=16,
        log_std_min=-3.0, log_std_max=-0.1, init_log_std=-0.5,
    )
    critic = MultiHeadQCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        channels=channels,
        hidden_dim=16,
        layer_norm=False,
        critic_lr=1e-3,
        device=torch.device("cpu"),
    )
    return actor, critic, channels


def test_s01_actor_samples_bounded_actions_and_consistent_log_prob() -> None:
    actor = S01Actor(
        6, 3, hidden_dim=16,
        log_std_min=-3.0, log_std_max=-0.1, init_log_std=-0.5,
        seed=17,
    )
    obs = torch.randn(32, 6)
    action, log_prob = actor.sample_action(obs)
    assert action.shape == (32, 3)
    assert torch.isfinite(log_prob).all()
    assert (action.abs() <= 1.0).all()

    evaluated, entropy = actor.evaluate_actions(obs, action)
    assert torch.allclose(evaluated, log_prob, atol=2e-4, rtol=2e-4)
    assert entropy.shape == (32,)
    deterministic = actor.deterministic_action(obs)
    assert torch.equal(deterministic, actor.deterministic_action(obs))


def test_s01_export_blueprint_builds_runtime_policy(tmp_path) -> None:
    actor = S01Actor(6, 3, hidden_dim=16, seed=3)
    bp = actor.to_blueprint(str(tmp_path / "policy"), stochastic=False)
    runtime = bp.build()
    obs = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
    action, extra = runtime.act(obs, want_extra=True)
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert extra == {"log_prob": None}
    expected = actor.deterministic_action(
        torch.as_tensor(obs).unsqueeze(0)
    ).detach().squeeze(0).numpy()
    np.testing.assert_allclose(action, expected, atol=1e-6)


def test_multihead_q_critic_uses_independent_groups_and_target_sync() -> None:
    channels = (
        SACRewardChannel(name="ra", gamma=0.99),
        SACRewardChannel(name="rb", gamma=0.99),
        SACRewardChannel(name="rc", gamma=0.99),
    )
    critic = MultiHeadQCritic(
        6, 3, channels, hidden_dim=16, layer_norm=False,
        critic_lr=1e-3, device=torch.device("cpu"),
    )
    assert len(critic.groups) == 3
    assert critic.n_networks == 6
    q1 = critic.q1_forward(torch.randn(4, 6), torch.randn(4, 3), "ra")
    assert q1.shape == (4,)

    group = critic.groups["channel_ra"]
    target_param = next(group.q1_target.parameters())
    before = target_param.detach().clone()
    next(group.q1.parameters()).data.fill_(1.0)
    critic.soft_update(0.1)
    assert not torch.equal(before, target_param.detach())

    with pytest.raises(ValueError, match="shared critic trunk"):
        MultiHeadQCritic(
            6,
            3,
            (
                SACRewardChannel(name="a", gamma=0.99, trunk_group="shared"),
                SACRewardChannel(name="b", gamma=0.99, trunk_group="shared"),
            ),
            hidden_dim=16,
            layer_norm=False,
            critic_lr=1e-3,
            device=torch.device("cpu"),
        )


def test_critic_target_uses_bootstrap_and_shannon_entropy() -> None:
    actor, critic, channels = _models(C=1)
    batch = _batch(B=4, C=1)
    batch["bootstrap"] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    batch["terminated"] = batch["bootstrap"] == 0
    alpha = torch.tensor(0.5)

    next_actions = torch.zeros(4, 3)
    next_log_probs = torch.full((4,), 2.0)
    actor.sample_action = lambda obs: (next_actions, next_log_probs)
    critic.q1_target_forward = lambda obs, act, ch: torch.full((4,), 3.0)
    critic.q2_target_forward = lambda obs, act, ch: torch.full((4,), 4.0)

    targets = compute_critic_targets(
        actor, critic, batch, channels,
        alpha=alpha, reward_scale=2.0, device=torch.device("cpu"),
    )["r0"]
    expected = batch["rewards"][:, 0] * 2.0 + 0.99 * batch["bootstrap"] * (3.0 - 0.5 * 2.0)
    torch.testing.assert_close(targets, expected)


def test_critic_target_selects_one_weighted_twin_pair() -> None:
    actor, critic, channels = _models(C=2)
    batch = _batch(B=2, C=2)
    batch["bootstrap"] = torch.ones(2)
    batch["terminated"] = torch.zeros(2, dtype=torch.bool)
    batch["actor_weight_next"] = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    batch["actor_gate_next"] = batch["actor_weight_next"] * 10.0
    actor.sample_action = lambda obs: (torch.zeros(2, 3), torch.zeros(2))
    critic.q1_target_forward = (
        lambda obs, act, ch: torch.tensor([0.0, 10.0])
        if ch == "r0" else torch.tensor([10.0, 0.0])
    )
    critic.q2_target_forward = (
        lambda obs, act, ch: torch.tensor([10.0, 0.0])
        if ch == "r0" else torch.tensor([0.0, 10.0])
    )
    target_info = {}

    targets = compute_critic_targets(
        actor, critic, batch, channels,
        alpha=torch.tensor(0.0), reward_scale=1.0,
        device=torch.device("cpu"), target_info=target_info,
    )

    assert target_info["pair_index"].tolist() == [0, 0]
    expected_r0 = batch["rewards"][:, 0] + 0.99 * torch.tensor([0.0, 10.0])
    expected_r1 = batch["rewards"][:, 1] + 0.99 * torch.tensor([10.0, 0.0])
    torch.testing.assert_close(targets["r0"], expected_r0)
    torch.testing.assert_close(targets["r1"], expected_r1)


def test_sac_update_mutates_actor_alpha_and_target() -> None:
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-3)
    batch = _batch(B=16)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    target_before = {
        k: v.detach().clone()
        for k, v in next(iter(critic.groups.values())).q1_target.state_dict().items()
    }
    alpha_before = log_alpha.detach().clone()

    stats = sac_update(
        actor,
        critic,
        actor_optimizer,
        log_alpha,
        alpha_optimizer,
        batch,
        channels,
        SACParams(use_grad_norm=False),
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
    )

    assert stats["critic_loss"] >= 0
    assert "q1_loss_r0" in stats
    assert "td_abs_mean_r0" in stats
    assert "actor_weight_mean_r0" in stats
    assert any(not torch.equal(p, b) for p, b in zip(actor.parameters(), actor_before))
    assert not torch.equal(log_alpha.detach(), alpha_before)
    target_after = next(iter(critic.groups.values())).q1_target.state_dict()
    assert any(
        not torch.equal(target_after[k], target_before[k])
        for k in target_before
    )


def test_trainer_rejects_unsupported_or_invalid_inputs() -> None:
    actor, critic, channels = _models(C=1)
    opt = torch.optim.Adam(actor.parameters())
    log_alpha = torch.tensor(0.0, requires_grad=True)
    batch = _batch(B=4, C=1)

    with pytest.raises(SACTrainerError, match="n_step"):
        sac_update(
            actor, critic, opt, log_alpha, None,
            batch,
            (SACRewardChannel(name="r0", gamma=0.99, n_step=2),),
            SACParams(use_grad_norm=False), 1.0, torch.device("cpu"),
        )
    with pytest.raises(SACTrainerError, match="gradient"):
        sac_update(
            actor, critic, opt, log_alpha, None,
            batch, channels, SACParams(use_grad_norm=True),
            1.0, torch.device("cpu"),
        )
    bad = dict(batch)
    bad.pop("source_keys")
    with pytest.raises(SACTrainerError, match="source_keys"):
        sac_update(
            actor, critic, opt, log_alpha, None,
            bad, channels, SACParams(use_grad_norm=False),
            1.0, torch.device("cpu"),
        )
    with pytest.raises(SACTrainerError, match="twin"):
        validate_sac_channels(
            (SACRewardChannel(name="r0", gamma=0.99, n_critics=3),)
        )
    with pytest.raises(SACTrainerError, match="common cohort gamma"):
        validate_sac_channels(
            (
                SACRewardChannel(name="r0", gamma=0.99),
                SACRewardChannel(name="r1", gamma=0.9),
            )
        )


def test_zero_actor_weight_does_not_freeze_that_channel_critic() -> None:
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    batch = _batch(B=16)
    batch["actor_gate"] = torch.tensor([[1.0, 0.0]]).repeat(16, 1)
    batch["actor_weight"] = torch.tensor([[1.0, 0.0]]).repeat(16, 1)
    batch["actor_gate_next"] = batch["actor_gate"].clone()
    batch["actor_weight_next"] = batch["actor_weight"].clone()
    group1 = critic.groups["channel_r1"]
    before = [
        p.detach().clone() for p in group1.q1.parameters()
    ]

    stats = sac_update(
        actor,
        critic,
        actor_optimizer,
        log_alpha,
        None,
        batch,
        channels,
        SACParams(use_grad_norm=False, tau=0.0),
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
    )

    assert stats["critic_updated_r1"] == 1.0
    assert stats["actor_weight_mean_r1"] == 0.0
    assert any(
        not torch.equal(p, before_p)
        for p, before_p in zip(group1.q1.parameters(), before)
    )


def test_actor_and_alpha_use_only_all_channel_valid_rows() -> None:
    actor, critic, channels = _models(C=2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    batch = _batch(B=8)
    batch["channel_valid"][:4, 1] = False

    stats = sac_update(
        actor,
        critic,
        actor_optimizer,
        log_alpha,
        None,
        batch,
        channels,
        SACParams(use_grad_norm=False, tau=0.0),
        grad_clip_norm=1.0,
        device=torch.device("cpu"),
    )

    assert stats["actor_valid_count"] == 4.0
    assert stats["critic_updated_r0"] == 1.0
    assert stats["critic_updated_r1"] == 1.0

    no_actor_rows = dict(batch)
    no_actor_rows["channel_valid"] = torch.tensor(
        [[True, False]] * 4 + [[False, True]] * 4,
        dtype=torch.bool,
    )
    with pytest.raises(SACTrainerError, match="actor-valid"):
        sac_update(
            actor, critic, actor_optimizer, log_alpha, None,
            no_actor_rows, channels, SACParams(use_grad_norm=False),
            1.0, torch.device("cpu"),
        )
