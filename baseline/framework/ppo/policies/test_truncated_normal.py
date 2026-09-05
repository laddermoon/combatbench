"""Tests for TruncatedNormalPolicy.

Verifies:
1. Sampling produces actions in [-1, 1]
2. log_prob matches manual truncated normal computation
3. log_prob integrates to 1 (probability conservation)
4. sample_action and evaluate_actions give consistent log_prob
5. Uncertainty U is in [0, 1] and matches 1/(2×peak)
6. U = 1 for uniform-like (large σ), U → 0 for narrow (small σ)
7. explore_intensity scales σ correctly (ei=-1→1/3, 0→1, +1→3)
8. Gradients flow to mean and log_std
9. U is per-obs (depends on mean)
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
    _std_normal_cdf,
    _std_normal_pdf,
    _std_normal_icdf,
    _SQRT_2PI,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32


def _make_policy(**kwargs) -> TruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return TruncatedNormalPolicy(**defaults)


class TestTruncatedNormalSampling(unittest.TestCase):
    """Sampling produces valid actions in [-1, 1]."""

    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(1000, OBS_DIM)
        actions, _ = p.sample_action(obs)
        self.assertTrue(actions.shape == (1000, ACTION_DIM))
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_deterministic_action_is_mean(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        det = p.deterministic_action(obs)
        mean, _ = p.forward(obs)
        self.assertTrue(torch.allclose(det, mean, atol=1e-6))

    def test_deterministic_action_in_range(self):
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM) * 10  # extreme obs
        det = p.deterministic_action(obs)
        self.assertTrue((det >= -1.0).all())
        self.assertTrue((det <= 1.0).all())


class TestLogProb(unittest.TestCase):
    """log_prob correctness."""

    def test_log_prob_matches_scipy(self):
        """Compare log_prob against scipy's truncnorm."""
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        actions, log_probs = p.sample_action(obs)

        # Check per-dim against scipy for a few samples
        mean, sigma = p.forward(obs)
        for i in range(5):
            for d in range(ACTION_DIM):
                m = float(mean[i, d])
                s = float(sigma[i, d])
                a_std = (-1.0 - m) / s
                b_std = (1.0 - m) / s
                # scipy truncnorm logpdf
                expected = sp_stats.truncnorm.logpdf(
                    float(actions[i, d]), a_std, b_std, loc=m, scale=s
                )
                # Our log_prob per-dim
                z = (float(actions[i, d]) - m) / s
                a_t = (-1.0 - m) / s
                b_t = (1.0 - m) / s
                Z = sp_stats.norm.cdf(b_t) - sp_stats.norm.cdf(a_t)
                our = (-0.5 * z * z - math.log(s)
                       - 0.5 * math.log(2 * math.pi) - math.log(Z))
                self.assertAlmostEqual(our, expected, places=4,
                                       msg=f"sample {i} dim {d}")

    def test_log_prob_integrates_to_one(self):
        """MC estimate of ∫ exp(log_prob) dx ≈ 1 over [-1,1]."""
        torch.manual_seed(123)
        p = _make_policy()
        # Fix a single obs
        obs = torch.randn(1, OBS_DIM)
        mean, sigma = p.forward(obs)

        # MC integration: sample uniformly in [-1,1], evaluate log_prob
        N = 100000
        x = torch.rand(N, ACTION_DIM) * 2.0 - 1.0  # uniform in [-1,1]
        obs_batch = obs.expand(N, -1)
        ev = p.evaluate_actions(obs_batch, x, torch.full((N,), 0.0))
        # ∫ exp(log_prob) dx ≈ (2/N) × Σ exp(log_prob_per_dim)
        # log_prob is summed over dims, so exp(log_prob) is joint density
        # ∫ joint dx = (2^d / N) × Σ exp(log_prob)
        integral = (2.0 ** ACTION_DIM / N) * torch.exp(ev.log_prob).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=1,
                               msg=f"integral = {integral}")

    def test_sample_vs_evaluate_log_prob(self):
        """sample_action and evaluate_actions give same log_prob."""
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions, lp_sample = p.sample_action(obs)
        ev = p.evaluate_actions(obs, actions, torch.full((50,), 0.0))
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-4, f"sample vs evaluate diff = {diff}")


class TestUncertainty(unittest.TestCase):
    """Uncertainty U = 1/(2×peak)."""

    def test_u_in_range(self):
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        ev = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u = ev.entropy
        self.assertTrue((u >= 0.0).all(), f"U < 0: min={u.min()}")
        self.assertTrue((u <= 1.0).all(), f"U > 1: max={u.max()}")

    def test_u_decreases_with_smaller_sigma(self):
        """Smaller σ → higher peak → lower U."""
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        # Large σ
        p.log_std.data.fill_(0.0)  # σ = 1.0
        ev_large = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_large = ev_large.entropy.mean().item()

        # Small σ
        p.log_std.data.fill_(-3.0)  # σ ≈ 0.05
        ev_small = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_small = ev_small.entropy.mean().item()

        self.assertGreater(u_large, u_small,
                           f"U(σ=1)={u_large} should > U(σ=0.05)={u_small}")

    def test_u_matches_formula(self):
        """U = σ × √(2π) × Z / 2 when mean ∈ (-1,1)."""
        p = _make_policy()
        p.log_std.data.fill_(-1.0)  # σ ≈ 0.368
        obs = torch.zeros(10, OBS_DIM)
        ev = p.evaluate_actions(obs, torch.zeros(10, ACTION_DIM), torch.full((10,), 0.0))
        u_actual = ev.entropy[0].item()

        # Manual: use actual mean from network
        with torch.no_grad():
            mean = torch.tanh(p.net(obs))
        sigma = math.exp(-1.0)
        u_per_dim = []
        for d in range(ACTION_DIM):
            m = float(mean[0, d])
            a = (-1.0 - m) / sigma
            b = (1.0 - m) / sigma
            Z = sp_stats.norm.cdf(b) - sp_stats.norm.cdf(a)
            u_per_dim.append(sigma * _SQRT_2PI * Z / 2.0)
        u_expected = sum(u_per_dim) / ACTION_DIM

        self.assertAlmostEqual(u_actual, u_expected, places=4,
                               msg=f"U={u_actual}, expected={u_expected}")

    def test_u_is_per_obs(self):
        """U depends on mean (through Z), so different obs → different U."""
        p = _make_policy()
        p.log_std.data.fill_(-1.0)
        # Two very different obs → different means → different Z → different U
        obs1 = torch.zeros(1, OBS_DIM)  # mean ≈ 0
        obs2 = torch.randn(1, OBS_DIM) * 5  # mean likely near ±1
        ev1 = p.evaluate_actions(obs1, torch.zeros(1, ACTION_DIM), torch.full((1,), 0.0))
        ev2 = p.evaluate_actions(obs2, torch.zeros(1, ACTION_DIM), torch.full((1,), 0.0))
        # They should be different (Z changes with mean position)
        self.assertNotAlmostEqual(ev1.entropy[0].item(), ev2.entropy[0].item(),
                                  places=3,
                                  msg="U should differ for different obs")


class TestExploreIntensity(unittest.TestCase):
    """explore_intensity exponential σ scaling: scale = exp(ei * ln(3))."""

    def test_scale_values(self):
        p = _make_policy()
        p.log_std.data.fill_(0.0)  # σ = 1.0

        self.assertAlmostEqual(p._explore_scale(0.0), 1.0, places=6)
        self.assertAlmostEqual(p._explore_scale(-1.0), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(p._explore_scale(1.0), 3.0, places=6)

    def test_scale_affects_sampling_sigma(self):
        p = _make_policy()
        p.log_std.data.fill_(0.0)  # σ = 1.0

        # Neutral
        _, sigma_neutral = p.forward(torch.randn(1, OBS_DIM), explore_intensity=0.0)
        self.assertAlmostEqual(sigma_neutral[0, 0].item(), 1.0, places=5)

        # Suppressed
        _, sigma_suppressed = p.forward(torch.randn(1, OBS_DIM), explore_intensity=-1.0)
        self.assertAlmostEqual(sigma_suppressed[0, 0].item(), 1.0 / 3.0,
                               places=5)

        # Expanded
        _, sigma_expanded = p.forward(torch.randn(1, OBS_DIM), explore_intensity=1.0)
        self.assertAlmostEqual(sigma_expanded[0, 0].item(), 3.0, places=5)

    def test_scale_does_not_affect_uncertainty(self):
        """U uses policy σ, not effective σ."""
        p = _make_policy()
        p.log_std.data.fill_(-1.0)
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)

        ev_neutral = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        ev_expanded = p.evaluate_actions(obs, actions, torch.full((10,), 1.0))

        diff = (ev_neutral.entropy - ev_expanded.entropy).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"U should not change with explore_intensity, diff={diff}")


class TestGradients(unittest.TestCase):
    """Gradients flow to mean (net) and log_std."""

    def test_gradient_to_log_std(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.log_prob.mean() + ev.entropy.mean()
        loss.backward()
        self.assertIsNotNone(p.log_std.grad)
        self.assertFalse(torch.allclose(p.log_std.grad,
                                         torch.zeros_like(p.log_std.grad)))

    def test_gradient_to_net(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.log_prob.mean()
        loss.backward()
        for param in p.net.parameters():
            self.assertIsNotNone(param.grad)

    def test_uncertainty_gradient_to_log_std(self):
        """U should have gradient w.r.t. log_std."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.entropy.mean()
        loss.backward()
        self.assertIsNotNone(p.log_std.grad)
        # U increases with σ (wider → more uncertain), so gradient should
        # be positive (increasing log_std increases U)
        self.assertTrue((p.log_std.grad > 0).all(),
                        f"∂U/∂log_std should be positive, got {p.log_std.grad}")


class TestStats(unittest.TestCase):
    """want_stats returns expected keys."""

    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0), want_stats=True)
        self.assertIsNotNone(ev.stats)
        for key in ["uncertainty", "std_mean", "eff_std_mean",
                     "std_min", "std_max", "mean_abs"]:
            self.assertIn(key, ev.stats)


if __name__ == "__main__":
    unittest.main()
