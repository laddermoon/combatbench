"""Tests for ``compute_gae``, ``compute_returns_to_go``, ``compute_grpo_advantages``.

Pin the contract from ``baseline/DESIGN.md`` §3.6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.framework.ppo.algos import (
    compute_gae,
    compute_grpo_advantages,
    compute_returns_to_go,
)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
class TestGAE:
    def test_lam_one_recovers_monte_carlo(self):
        # When lam=1 and last_value=0, returns should equal the discounted
        # cumulative reward and advantages = returns - values.
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        gamma = 0.5
        adv, ret = compute_gae(
            rewards, values, last_value=0.0, gamma=gamma, lam=1.0,
        )
        expected_ret = compute_returns_to_go(rewards, gamma=gamma, last_value=0.0)
        np.testing.assert_allclose(ret, expected_ret, atol=1e-6)
        np.testing.assert_allclose(adv, expected_ret - values, atol=1e-6)

    def test_lam_zero_recovers_td0(self):
        # When lam=0, advantage[t] = r_t + gamma*V[t+1] - V[t].
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        gamma = 0.9
        last_value = 8.0
        adv, _ = compute_gae(
            rewards, values, last_value=last_value, gamma=gamma, lam=0.0,
        )
        # Hand compute deltas:
        # delta[2] = 3 + 0.9*8 - 7 = 3.2
        # delta[1] = 2 + 0.9*7 - 6 = 2.3
        # delta[0] = 1 + 0.9*6 - 5 = 1.4
        np.testing.assert_allclose(adv, [1.4, 2.3, 3.2], atol=1e-5)

    def test_returns_equals_advantages_plus_values(self):
        rewards = np.array([0.5, 1.0, -0.2, 2.0], dtype=np.float32)
        values = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        adv, ret = compute_gae(
            rewards, values, last_value=0.0, gamma=0.99, lam=0.95,
        )
        np.testing.assert_allclose(ret, adv + values, atol=1e-6)

    def test_terminated_uses_zero_bootstrap(self):
        # When the caller passes last_value=0 (terminated convention),
        # the final-step advantage equals delta[T-1] = r[T-1] - V[T-1].
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        adv, _ = compute_gae(
            rewards, values, last_value=0.0, gamma=0.99, lam=0.0,
        )
        # delta[2] = 1.0 + 0.99*0 - 0.5 = 0.5
        assert adv[-1] == pytest.approx(0.5, abs=1e-5)

    def test_truncated_uses_critic_bootstrap(self):
        # Same as above but with last_value=2.0 (truncated convention).
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        adv, _ = compute_gae(
            rewards, values, last_value=2.0, gamma=0.99, lam=0.0,
        )
        # delta[2] = 1.0 + 0.99*2.0 - 0.5 = 2.48
        assert adv[-1] == pytest.approx(2.48, abs=1e-5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_gae(np.zeros(3), np.zeros(4))

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_gae(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_gamma_lam_validation(self):
        rewards = np.zeros(3)
        values = np.zeros(3)
        with pytest.raises(ValueError, match="gamma"):
            compute_gae(rewards, values, gamma=1.5)
        with pytest.raises(ValueError, match="lam"):
            compute_gae(rewards, values, lam=-0.1)


# ---------------------------------------------------------------------------
# Returns-to-go
# ---------------------------------------------------------------------------
class TestReturnsToGo:
    def test_sums_correctly(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gamma = 0.5
        ret = compute_returns_to_go(rewards, gamma=gamma, last_value=0.0)
        # ret[2] = 3
        # ret[1] = 2 + 0.5*3 = 3.5
        # ret[0] = 1 + 0.5*3.5 = 2.75
        np.testing.assert_allclose(ret, [2.75, 3.5, 3.0], atol=1e-6)

    def test_bootstrap_propagates(self):
        rewards = np.zeros(3, dtype=np.float32)
        gamma = 0.5
        ret = compute_returns_to_go(rewards, gamma=gamma, last_value=4.0)
        # ret[2] = 0 + 0.5*4 = 2
        # ret[1] = 0 + 0.5*2 = 1
        # ret[0] = 0 + 0.5*1 = 0.5
        np.testing.assert_allclose(ret, [0.5, 1.0, 2.0], atol=1e-6)


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
class TestGRPOAdvantages:
    def test_centered_within_group(self):
        # Three episodes with returns [0, 5, 10]; mean=5, std≈4.082.
        rewards = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([5.0, 0.0], dtype=np.float32),
            np.array([10.0, 0.0], dtype=np.float32),
        ]
        adv = compute_grpo_advantages(
            rewards, gamma=1.0, broadcast_to_steps=False,
        )
        # Sum of centered values must be ~0.
        assert abs(sum(adv)) < 1e-5
        # Highest-return episode has advantage > 0; lowest has < 0.
        assert adv[2] > 0
        assert adv[0] < 0

    def test_broadcast_to_steps_matches_episode_lengths(self):
        rewards = [
            np.zeros(3, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        # All-zero rewards → all returns zero → std=eps → advantages all zero
        # but we still want shapes.
        adv = compute_grpo_advantages(rewards)
        assert [a.shape[0] for a in adv] == [3, 7, 2]
        for a in adv:
            np.testing.assert_allclose(a, 0.0, atol=1e-3)

    def test_within_episode_advantage_is_constant(self):
        # GRPO is a per-episode scalar; broadcasting it across steps means
        # all values within one episode should be identical.
        rewards = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float32),
        ]
        adv = compute_grpo_advantages(rewards, gamma=1.0)
        for a in adv:
            assert a.std() == pytest.approx(0.0, abs=1e-6)

    def test_gamma_discounts_within_episode(self):
        # For one-step-each episodes, gamma should not matter.
        rewards = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        a05 = compute_grpo_advantages(rewards, gamma=0.5, broadcast_to_steps=False)
        a10 = compute_grpo_advantages(rewards, gamma=1.0, broadcast_to_steps=False)
        np.testing.assert_allclose(a05, a10, atol=1e-6)

    def test_empty_group_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_grpo_advantages([])

    def test_2d_per_episode_rewards_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_grpo_advantages([np.zeros((3, 2))])
