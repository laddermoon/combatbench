"""Tests for StateTruncatedNormalPolicy.

Verifies:
1. Sampling produces actions in [-1, 1]
2. log_prob matches manual truncated normal computation (scipy)
3. log_prob integrates to 1 (probability conservation)
4. sample_action and evaluate_actions give consistent log_prob
5. Uncertainty U is in [0, 1] and matches 1/(2×peak)
6. U increases with σ; U → ~1 for very large σ (reachable ceiling)
7. explore_factor scales σ correctly (ei=-1→1/3, 0→1, +1→3)
8. Gradients flow to the σ half of head and to the trunk
9. σ is state-dependent (different obs → different σ)
10. Degenerate equivalence: with baseline weights copied in and
    σ-head constant -1, outputs match TruncatedNormalPolicy bit-for-bit
11. Export: strict loading, self-contained policy.py, parity
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.state_truncated_normal_mlp import (
    StateTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
    _SQRT_2PI,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32


def _make_policy(**kwargs) -> StateTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return StateTruncatedNormalPolicy(**defaults)


def _set_const_log_std(p: StateTruncatedNormalPolicy, value: float) -> None:
    """Force σ(obs) = exp(value) for every obs by zeroing the σ half of
    the head weight and pinning its bias — the state-dependent analog of
    ``p.log_std.data.fill_(value)`` on the baseline."""
    d = p.action_dim
    with torch.no_grad():
        p.head.weight[d:, :].zero_()
        p.head.bias[d:].fill_(value)


def _policy_sigma(p: StateTruncatedNormalPolicy, obs: torch.Tensor) -> torch.Tensor:
    """Policy σ (no explore scale) for a batch of obs."""
    _, sigma = p._head_forward(obs)
    return sigma


class TestSampling(unittest.TestCase):
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

    def test_init_sigma_matches_baseline(self):
        """At init, σ = e^-1 everywhere (σ-head w=0, b=-1)."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        sigma = _policy_sigma(p, obs)
        self.assertTrue(
            torch.allclose(sigma, torch.full_like(sigma, math.exp(-1.0)),
                           atol=1e-6),
            f"init σ should be e^-1, got {sigma[0]}",
        )


class TestLogProb(unittest.TestCase):
    """log_prob correctness."""

    def test_log_prob_matches_scipy(self):
        """Compare log_prob against scipy's truncnorm."""
        torch.manual_seed(42)
        p = _make_policy()
        # Non-constant σ: give the σ head random weights so per-sample σ
        # actually varies.
        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.1)
        obs = torch.randn(100, OBS_DIM)
        actions, log_probs = p.sample_action(obs)

        mean, sigma = p.forward(obs)
        for i in range(5):
            for d in range(ACTION_DIM):
                m = float(mean[i, d])
                s = float(sigma[i, d])
                a_std = (-1.0 - m) / s
                b_std = (1.0 - m) / s
                expected = sp_stats.truncnorm.logpdf(
                    float(actions[i, d]), a_std, b_std, loc=m, scale=s
                )
                z = (float(actions[i, d]) - m) / s
                Z = sp_stats.norm.cdf(b_std) - sp_stats.norm.cdf(a_std)
                our = (-0.5 * z * z - math.log(s)
                       - 0.5 * math.log(2 * math.pi) - math.log(Z))
                self.assertAlmostEqual(our, expected, places=4,
                                       msg=f"sample {i} dim {d}")

    def test_log_prob_integrates_to_one(self):
        """MC estimate of ∫ exp(log_prob) dx ≈ 1 over [-1,1]."""
        torch.manual_seed(123)
        p = _make_policy()
        obs = torch.randn(1, OBS_DIM)

        N = 100000
        x = torch.rand(N, ACTION_DIM) * 2.0 - 1.0  # uniform in [-1,1]
        obs_batch = obs.expand(N, -1)
        ev = p.evaluate_actions(obs_batch, x, torch.full((N,), 0.0))
        # log_prob is summed over dims, so exp(log_prob) is joint density
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
        u = ev.uncertainty
        self.assertTrue((u >= 0.0).all(), f"U < 0: min={u.min()}")
        self.assertTrue((u <= 1.0).all(), f"U > 1: max={u.max()}")

    def test_u_increases_with_sigma(self):
        """Larger σ → lower peak → higher U."""
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)

        _set_const_log_std(p, 0.0)  # σ = 1.0
        ev_large = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_large = ev_large.uncertainty.mean().item()

        _set_const_log_std(p, -3.0)  # σ ≈ 0.05
        ev_small = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_small = ev_small.uncertainty.mean().item()

        self.assertGreater(u_large, u_small,
                           f"U(σ=1)={u_large} should > U(σ=0.05)={u_small}")

    def test_u_matches_formula(self):
        """U = σ × √(2π) × Z / 2 when mean ∈ (-1,1)."""
        p = _make_policy()
        _set_const_log_std(p, -1.0)  # σ ≈ 0.368
        obs = torch.zeros(10, OBS_DIM)
        ev = p.evaluate_actions(obs, torch.zeros(10, ACTION_DIM), torch.full((10,), 0.0))
        u_actual = ev.uncertainty[0].item()

        with torch.no_grad():
            mean = torch.tanh(p.head(p.trunk(obs))[:, :ACTION_DIM])
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
        """U varies across obs — through both Z(mean) and σ(obs)."""
        torch.manual_seed(0)
        p = _make_policy()
        # State-dependent σ: non-zero σ-head weights so σ varies with obs.
        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.5)
        obs = torch.randn(200, OBS_DIM)
        ev = p.evaluate_actions(obs, torch.zeros(200, ACTION_DIM), torch.full((200,), 0.0))
        u = ev.uncertainty
        self.assertGreater(
            float(u.std().item()), 1e-3,
            f"U should vary across obs, std={u.std().item()}",
        )

    def test_large_sigma_no_nan_and_u_bounded(self):
        """σ has no business bound — verify large σ stays finite and U ≤ 1.

        With σ → large, the truncated normal → uniform on [-1,1], so
        U → 1 from below.  log_std = 10 → σ ≈ 22026 must not produce
        NaN/Inf in log_prob or U.
        """
        p = _make_policy()
        _set_const_log_std(p, 10.0)  # σ ≈ 22026
        obs = torch.randn(50, OBS_DIM)
        actions = torch.rand(50, ACTION_DIM) * 2 - 1
        ev = p.evaluate_actions(obs, actions, torch.full((50,), 0.0))
        self.assertTrue(torch.isfinite(ev.log_prob).all(),
                        f"log_prob has non-finite values at σ=22026")
        u = ev.uncertainty
        self.assertTrue(torch.isfinite(u).all(), "U non-finite at σ=22026")
        self.assertTrue((u <= 1.0).all(), f"U > 1 at σ=22026: max={u.max()}")
        # σ=22026 should be very close to uniform → U very close to 1
        self.assertGreater(u.mean().item(), 0.9,
                           f"U(σ=22026)={u.mean()} should approach 1")

    def test_extreme_log_std_clamped(self):
        """raw_log_std beyond ±20 is clamped (numerical safety only)."""
        p = _make_policy()
        _set_const_log_std(p, 50.0)  # beyond _LOG_STD_SAFE_MAX=20
        obs = torch.randn(10, OBS_DIM)
        sigma = _policy_sigma(p, obs)
        self.assertTrue(
            torch.allclose(sigma, torch.full_like(sigma, math.exp(20.0))),
            "σ should be clamped to e^20",
        )


class TestExploreIntensity(unittest.TestCase):
    """explore_factor exponential σ scaling: scale = exp(ei * ln(3))."""

    def test_scale_values(self):
        p = _make_policy()
        self.assertAlmostEqual(p._explore_scale(0.0), 1.0, places=6)
        self.assertAlmostEqual(p._explore_scale(-1.0), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(p._explore_scale(1.0), 3.0, places=6)

    def test_scale_affects_sampling_sigma(self):
        p = _make_policy()
        _set_const_log_std(p, 0.0)  # σ = 1.0
        obs = torch.randn(1, OBS_DIM)

        _, sigma_neutral = p.forward(obs, explore_factor=0.0)
        self.assertAlmostEqual(sigma_neutral[0, 0].item(), 1.0, places=5)

        _, sigma_suppressed = p.forward(obs, explore_factor=-1.0)
        self.assertAlmostEqual(sigma_suppressed[0, 0].item(), 1.0 / 3.0,
                               places=5)

        _, sigma_expanded = p.forward(obs, explore_factor=1.0)
        self.assertAlmostEqual(sigma_expanded[0, 0].item(), 3.0, places=5)

    def test_scale_is_multiplicative_on_state_sigma(self):
        """ei scaling applies on top of state-dependent σ — the ratio
        σ_eff/σ_policy is exactly exp(ei·ln3) per frame regardless of
        what the state-dependent σ is."""
        p = _make_policy()
        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.3)
        obs = torch.randn(20, OBS_DIM)
        _, sigma_p = p._head_forward(obs)

        ei = torch.full((20,), 0.7)
        _, sigma_eff = p.forward(obs, explore_factor=ei)
        ratio = (sigma_eff / sigma_p)
        expected = math.exp(0.7 * math.log(3.0))
        self.assertTrue(
            torch.allclose(ratio, torch.full_like(ratio, expected), atol=1e-5),
            f"σ_eff/σ_policy should be {expected} uniformly, got {ratio[:3,0]}",
        )

    def test_scale_does_not_affect_uncertainty(self):
        """U uses policy σ, not effective σ."""
        p = _make_policy()
        _set_const_log_std(p, -1.0)
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)

        ev_neutral = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        ev_expanded = p.evaluate_actions(obs, actions, torch.full((10,), 1.0))

        diff = (ev_neutral.uncertainty - ev_expanded.uncertainty).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"U should not change with explore_factor, diff={diff}")


class TestStateDependence(unittest.TestCase):
    """σ is a function of obs — the single variable this policy adds."""

    def test_sigma_varies_across_obs(self):
        """With non-zero σ-head weights, different obs → different σ."""
        p = _make_policy()
        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.5)
        obs = torch.randn(200, OBS_DIM)
        sigma = _policy_sigma(p, obs)  # (B, action_dim)
        # Per-dim std across the batch should be clearly non-zero
        per_dim_std = sigma.std(dim=0)
        self.assertTrue(
            (per_dim_std > 1e-3).all(),
            f"σ should vary across obs, per-dim std={per_dim_std}",
        )

    def test_init_is_state_independent(self):
        """At init (σ-head w=0, b=-1), σ is constant across obs —
        matching the baseline's global σ exactly."""
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        sigma = _policy_sigma(p, obs)
        self.assertTrue(
            torch.allclose(sigma, sigma[:1].expand_as(sigma), atol=1e-7),
            "init σ should be constant across obs",
        )


class TestGradients(unittest.TestCase):
    """Gradients flow to the σ half of head and to the trunk."""

    def test_gradient_to_sigma_head(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.log_prob.mean() + ev.uncertainty.mean()
        loss.backward()
        d = ACTION_DIM
        self.assertIsNotNone(p.head.weight.grad)
        self.assertFalse(torch.allclose(
            p.head.weight.grad[d:, :],
            torch.zeros_like(p.head.weight.grad[d:, :]),
        ), "σ half of head.weight got zero gradient")

    def test_gradient_to_mean_path(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.log_prob.mean()
        loss.backward()
        for param in p.trunk.parameters():
            self.assertIsNotNone(param.grad)
        self.assertIsNotNone(p.head.weight.grad)
        self.assertFalse(torch.allclose(
            p.head.weight.grad[:ACTION_DIM, :],
            torch.zeros_like(p.head.weight.grad[:ACTION_DIM, :]),
        ), "mean half of head.weight got zero gradient")

    def test_uncertainty_gradient_to_sigma_head(self):
        """U should have positive gradient w.r.t. σ (wider → more uncertain)."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.uncertainty.mean()
        loss.backward()
        d = ACTION_DIM
        # σ = exp(log_std): ∂U/∂bias[d:] should be positive
        # (increasing bias → increasing σ → increasing U)
        self.assertIsNotNone(p.head.bias.grad)
        self.assertTrue(
            (p.head.bias.grad[d:] > 0).all(),
            f"∂U/∂(σ-bias) should be positive, got {p.head.bias.grad[d:]}",
        )

class TestStats(unittest.TestCase):
    """want_stats returns expected keys."""

    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0), want_stats=True)
        self.assertIsNotNone(ev.stats)
        for key in ["uncertainty", "std_mean", "eff_std_mean",
                    "std_min", "std_max", "std_std", "mean_abs"]:
            self.assertIn(key, ev.stats)

    def test_std_std_zero_at_init(self):
        """std_std ≈ 0 at init (σ constant), > 0 once σ-head is trained."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions = torch.zeros(50, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.full((50,), 0.0), want_stats=True)
        self.assertAlmostEqual(ev.stats["std_std"], 0.0, places=6)

        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.5)
        ev2 = p.evaluate_actions(obs, actions, torch.full((50,), 0.0), want_stats=True)
        self.assertGreater(ev2.stats["std_std"], 1e-3)


class TestDegenerateEquivalence(unittest.TestCase):
    """Core correctness test: copy baseline weights → identical outputs.

    TruncatedNormalPolicy:
        net = Linear(obs,h)→Tanh→Linear(h,h)→Tanh→Linear(h,D); mean=tanh(net(obs))
        log_std = param (init -1)

    StateTruncatedNormalPolicy:
        trunk = net[:4]; head = Linear(h, 2D); mean=tanh(head[:D]); σ=exp(head[D:])

    Wiring net[4] into head[:D] and pinning head[D:] to bias -1 makes the
    two policies compute the SAME function — every downstream quantity
    (mean, σ, log_prob, U, samples) must be bit-identical.  This proves
    the port changed only the parameterization of σ, nothing else.
    """

    def _make_equivalent_pair(self):
        torch.manual_seed(7)
        base = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        state = _make_policy()
        d = ACTION_DIM
        with torch.no_grad():
            # Trunk: first two Linear layers of baseline net.
            state.trunk[0].weight.copy_(base.net[0].weight)
            state.trunk[0].bias.copy_(base.net[0].bias)
            state.trunk[2].weight.copy_(base.net[2].weight)
            state.trunk[2].bias.copy_(base.net[2].bias)
            # Head mean half ← baseline final Linear.
            state.head.weight[:d].copy_(base.net[4].weight)
            state.head.bias[:d].copy_(base.net[4].bias)
            # σ half → constant log_std = -1 (matches baseline param).
            state.head.weight[d:].zero_()
            state.head.bias[d:].fill_(-1.0)
            base.log_std.fill_(-1.0)
        return base, state

    def test_forward_bit_identical(self):
        base, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        m_b, s_b = base.forward(obs)
        m_s, s_s = state.forward(obs)
        torch.testing.assert_close(m_s, m_b, rtol=0, atol=0)
        torch.testing.assert_close(s_s, s_b, rtol=0, atol=0)

    def test_forward_bit_identical_with_explore_factor(self):
        base, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        ei = torch.linspace(-1.0, 1.0, 64)
        m_b, s_b = base.forward(obs, explore_factor=ei)
        m_s, s_s = state.forward(obs, explore_factor=ei)
        torch.testing.assert_close(m_s, m_b, rtol=0, atol=0)
        torch.testing.assert_close(s_s, s_b, rtol=0, atol=0)

    def test_sample_action_bit_identical(self):
        base, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.manual_seed(123)
        a_b, lp_b = base.sample_action(obs)
        torch.manual_seed(123)
        a_s, lp_s = state.sample_action(obs)
        torch.testing.assert_close(a_s, a_b, rtol=0, atol=0)
        torch.testing.assert_close(lp_s, lp_b, rtol=0, atol=0)

    def test_evaluate_actions_bit_identical(self):
        base, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.rand(64, ACTION_DIM) * 2 - 1
        ei = torch.full((64,), 0.0)
        ev_b = base.evaluate_actions(obs, actions, ei)
        ev_s = state.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(ev_s.log_prob, ev_b.log_prob, rtol=0, atol=0)
        torch.testing.assert_close(ev_s.uncertainty, ev_b.uncertainty, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Export: strict loading + self-contained policy.py
# ---------------------------------------------------------------------------

class TestExportStrictLoading(unittest.TestCase):
    """Exported policy loading must be strict and validated (P0-5)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="stn_p0_5_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        return bp, self._model_path

    def _load_payload(self):
        return torch.load(self._model_path, map_location="cpu")

    def _save_payload(self, payload):
        torch.save(payload, self._model_path)

    def test_export_payload_has_format_metadata(self):
        self._make_export()
        p = self._load_payload()
        self.assertEqual(p["format_version"], 1)
        self.assertEqual(p["policy_class"], "StateTruncatedNormalPolicy")
        self.assertEqual(p["arch"]["obs_dim"], OBS_DIM)
        self.assertEqual(p["arch"]["action_dim"], ACTION_DIM)
        self.assertEqual(p["arch"]["hidden_dim"], HIDDEN_DIM)
        self.assertIn("state_dict_keys", p)
        self.assertIsInstance(p["state_dict_keys"], list)

    def test_export_roundtrip_exact(self):
        """Export → reload produces identical actions on non-zero input."""
        torch.manual_seed(123)
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)  # non-zero!
        expected = p.act(obs)[0]
        loaded = bp.build()
        actual = loaded.act(obs)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_export_rejects_missing_keys(self):
        bp, _ = self._make_export()
        payload = self._load_payload()
        sd = dict(payload["state_dict"])
        removed = "trunk.0.weight"
        sd.pop(removed)
        payload["state_dict"] = sd
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn(removed, str(ctx.exception),
                      f"Error should mention missing key {removed}")

    def test_export_rejects_extra_keys(self):
        bp, _ = self._make_export()
        payload = self._load_payload()
        sd = dict(payload["state_dict"])
        sd["nonexistent.layer.weight"] = torch.zeros(4, 4)
        payload["state_dict"] = sd
        self._save_payload(payload)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_rejects_wrong_format_version(self):
        bp, _ = self._make_export()
        payload = self._load_payload()
        payload["format_version"] = 999
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("format version", str(ctx.exception).lower())

    def test_export_rejects_wrong_policy_class(self):
        bp, _ = self._make_export()
        payload = self._load_payload()
        payload["policy_class"] = "TruncatedNormalPolicy"
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("class mismatch", str(ctx.exception).lower())

    def test_export_no_silent_param_swallowing(self):
        """Unknown kwargs must raise TypeError."""
        bp, _ = self._make_export()
        with self.assertRaises(TypeError):
            bp.build(unknown_param=True)


class TestExportSelfContained(unittest.TestCase):
    """Exported policy.py must not import from baseline.* or envs.* (P0-6)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="stn_p0_6_test_")

    def test_export_policy_py_has_no_repo_imports(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to_blueprint(dest_path=self._tmp)
        policy_code = (Path(self._tmp) / "policy.py").read_text()
        for forbidden in [
            "from baseline", "import baseline",
            "from envs", "import envs",
        ]:
            self.assertNotIn(forbidden, policy_code,
                             f"Exported policy.py must not contain '{forbidden}'")

    def test_export_has_manifest(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to_blueprint(dest_path=self._tmp)
        import json
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["policy_class"], "StateTruncatedNormalPolicy")
        self.assertEqual(manifest["exported_class"], "ExportedStateTruncNormPolicy")
        self.assertIn("arch", manifest)
        self.assertIn("files", manifest)

    def test_export_works_without_repo_on_path(self):
        """Exported policy loads and runs with no baseline.* on sys.path."""
        import subprocess
        import sys

        torch.manual_seed(42)
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]

        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import ExportedStateTruncNormPolicy; "
            "import numpy as np; "
            "p = ExportedStateTruncNormPolicy(); "
            "obs = np.array({obs!r}, dtype=np.float32); "
            "a, _ = p.act(obs); "
            "print(repr(a.tolist()))"
        ).format(
            tmp=self._tmp,
            obs=obs.tolist(),
        )
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                 "HOME": "/root",
                 "LD_LIBRARY_PATH": "/usr/local/lib",
                 "PYTHONPATH": self._tmp},  # only the export dir
        )
        if result.returncode != 0:
            self.fail(
                f"Subprocess failed (no repo on path):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        actual = np.array(eval(result.stdout.strip()), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                   err_msg="Exported policy output differs from training-side")


class TestExportParity(unittest.TestCase):
    """Training-side and export-side must agree bit-for-bit."""

    def test_parity_act_deterministic(self):
        torch.manual_seed(999)
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        # Make σ genuinely state-dependent before exporting.
        with torch.no_grad():
            p.head.weight[ACTION_DIM:, :].normal_(0, 0.3)
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="stn_parity_"))
        loaded = bp.build()
        for seed in range(10):
            torch.manual_seed(seed)
            obs = torch.randn(OBS_DIM).numpy().astype(np.float32)
            expected = p.act(obs)[0]
            actual = loaded.act(obs)[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                       err_msg=f"Parity failed on seed {seed}")

    def test_parity_sample_stochastic(self):
        torch.manual_seed(777)
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="stn_parity_"))
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        torch.manual_seed(12345)
        expected_action, expected_lp = p.sample(obs, explore_factor=0.5, want_extra=True)
        torch.manual_seed(12345)
        actual_action, actual_lp = loaded.sample(obs, explore_factor=0.5, want_extra=True)
        np.testing.assert_allclose(actual_action, expected_action, rtol=0, atol=0,
                                   err_msg="Sampled action parity failed")
        self.assertAlmostEqual(actual_lp["log_prob"], expected_lp["log_prob"], places=6,
                               msg="log_prob parity failed")


class TestDeviceSync(unittest.TestCase):
    """.to(device) must keep self.device in sync (P0-4)."""

    def test_device_follows_to(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        self.assertEqual(p.device.type, "cpu")
        p_cpu = p.to("cpu")
        self.assertEqual(p_cpu.device.type, "cpu")
        self.assertEqual(p.device.type, "cpu")

    def test_act_after_to_cpu(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to("cpu")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    @unittest.skipIf(not torch.cuda.is_available(), "需要 GPU")
    def test_act_after_to_cuda(self):
        p = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to("cuda")
        self.assertEqual(p.device.type, "cuda")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


if __name__ == "__main__":
    unittest.main()
