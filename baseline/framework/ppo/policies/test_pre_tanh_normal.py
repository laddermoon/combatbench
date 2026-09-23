"""Tests for PreTanhNormalPolicy.

Verifies (per DESIGN_pre_tanh_normal.md §11):
1. Action density is the transformed Gaussian: p_A(a) =
   N(atanh(a); mu, sigma^2) / (1 - a^2) — scipy reference, grid
   integration, sample-vs-evaluate consistency
2. Shared per-dim sigma (no state-dependent sigma head)
3. Uncertainty = closed-form action-space L2 width
   U_d = 2*sqrt(pi)*sigma / (1 + exp(sigma^2)*cosh(2*mu)),
   checked against z-space quadrature; sigma* is the unique maximizer
4. explore_factor moves (mu, r) radially toward theta*: e=0 identity,
   U_e non-decreasing in e, fixed-point invariance
5. Numerical guards: |a| > a_safe, non-finite actions, e out of range,
   and unsupported-parameter tail risk all fail loudly
6. Gradients reach the net and log_std; float64 gradcheck
7. Export: strict loading, self-containment, parity
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.pre_tanh_normal_mlp import (
    PreTanhNormalPolicy,
    _ACTION_SAFE,
    _COVERAGE_LOG_STD_STAR,
    _COVERAGE_U_MAX,
    _Z_SAFE,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32

_SIGMA_STAR = math.exp(_COVERAGE_LOG_STD_STAR)


def _make_policy(**kwargs) -> PreTanhNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return PreTanhNormalPolicy(**defaults)


def _set_const_policy(
    p: PreTanhNormalPolicy,
    mu,
    log_std,
) -> None:
    """Force a state-independent policy: mu = const, sigma = const."""
    with torch.no_grad():
        for m in p.net.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                m.bias.zero_()
        p.net[-1].bias[:] = torch.as_tensor(mu, dtype=torch.float32)
        p.log_std.data = torch.as_tensor(log_std, dtype=torch.float32)


def _pdf_ref(mu, sigma, a):
    """Reference action density p_A(a) = N(atanh a; mu, sig^2)/(1-a^2)."""
    a = np.asarray(a, dtype=np.float64)
    z = np.arctanh(np.clip(a, -1.0 + 1e-12, 1.0 - 1e-12))
    return sp_stats.norm.pdf(z, loc=mu, scale=sigma) / (1.0 - a * a)


def _u_ref(mus, sigmas):
    """Reference U: z-space quadrature of R = ∫ q(z)^2 cosh^2(z) dz."""
    us = []
    for mu, sig in zip(mus, sigmas):
        lo = min(mu - 12.0 * sig, -20.0)
        hi = max(mu + 12.0 * sig, 20.0)
        z = np.linspace(lo, hi, 400001)
        q = sp_stats.norm.pdf(z, loc=mu, scale=sig)
        r_int = np.trapezoid(q * q * np.cosh(z) ** 2, z)
        us.append(1.0 / (2.0 * r_int))
    return float(np.mean(us))


class TestSampling(unittest.TestCase):
    """Sampling produces valid, scoreable actions."""

    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(1000, OBS_DIM)
        actions, _ = p.sample_action(obs)
        self.assertEqual(actions.shape, (1000, ACTION_DIM))
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())
        self.assertTrue((actions.abs() <= _ACTION_SAFE).all())

    def test_actions_in_range_with_explore(self):
        p = _make_policy()
        obs = torch.randn(500, OBS_DIM)
        for e in (-1.0, -0.5, 0.5, 1.0):
            actions, _ = p.sample_action(obs, explore_factor=e)
            self.assertTrue((actions.abs() <= _ACTION_SAFE).all())

    def test_act_is_tanh_mu(self):
        p = _make_policy()
        mu = np.array([0.5, -1.2, 0.0, 2.0])
        _set_const_policy(p, mu, np.full(ACTION_DIM, -1.0))
        obs = torch.randn(10, OBS_DIM)
        det = p.deterministic_action(obs)
        torch.testing.assert_close(
            det,
            torch.tanh(
                torch.as_tensor(mu, dtype=torch.float32)
            ).expand(10, -1),
            rtol=0, atol=1e-6,
        )

    def test_init(self):
        """log_std is a per-dim shared parameter, init sigma = e^-1."""
        p = _make_policy()
        self.assertEqual(tuple(p.log_std.shape), (ACTION_DIM,))
        torch.testing.assert_close(
            p.log_std.data, torch.full((ACTION_DIM,), -1.0),
        )

    def test_shared_sigma_not_state_dependent(self):
        """Same e → identical effective sigma across different obs."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        mu_e, sigma_e = p.forward(obs, explore_factor=0.0)
        self.assertEqual(sigma_e.shape[-1], ACTION_DIM)
        row_spread = (sigma_e.max(dim=0).values
                      - sigma_e.min(dim=0).values)
        torch.testing.assert_close(
            row_spread, torch.zeros_like(row_spread), rtol=0, atol=0,
        )

    def test_empirical_moments_match_quadrature(self):
        """Sampled E[A], Var[A] match numerical integration."""
        torch.manual_seed(0)
        p = _make_policy()
        mu = np.array([0.4, -0.6, 0.0, 1.0])
        sig = np.array([0.3, 0.6, 0.9, 0.5])
        _set_const_policy(p, mu, np.log(sig))
        n = 40000
        obs = torch.zeros(n, OBS_DIM)
        with torch.no_grad():
            samples, _ = p.sample_action(obs)
        samples = samples.numpy().astype(np.float64)
        for d in range(ACTION_DIM):
            lo, hi = mu[d] - 12 * sig[d], mu[d] + 12 * sig[d]
            z = np.linspace(lo, hi, 200001)
            q = sp_stats.norm.pdf(z, loc=mu[d], scale=sig[d])
            a = np.tanh(z)
            m_ref = np.trapezoid(a * q, z)
            m2_ref = np.trapezoid(a * a * q, z)
            self.assertAlmostEqual(
                samples[:, d].mean(), m_ref, delta=0.02,
                msg=f"dim {d} mean: {samples[:, d].mean()} vs {m_ref}",
            )
            self.assertAlmostEqual(
                samples[:, d].var(), m2_ref - m_ref ** 2, delta=0.02,
                msg=f"dim {d} var",
            )


class TestLogProb(unittest.TestCase):
    """Transformed-density log_prob correctness."""

    def test_log_prob_matches_scipy(self):
        p = _make_policy()
        mu = np.array([0.3, -0.8, 0.5, -0.2])
        sig = np.array([0.4, 0.7, 1.2, 0.9])
        _set_const_policy(p, mu, np.log(sig))
        obs = torch.zeros(30, OBS_DIM)
        actions = torch.rand(30, ACTION_DIM) * 1.8 - 0.9
        ev = p.evaluate_actions(obs, actions, torch.zeros(30))
        ref = np.ones(30)
        for d in range(ACTION_DIM):
            ref *= _pdf_ref(mu[d], sig[d], actions[:, d].numpy())
        np.testing.assert_allclose(
            np.exp(ev.log_prob.detach().numpy()), ref,
            rtol=1e-5, atol=1e-8,
            err_msg="action density differs from scipy reference",
        )

    def test_log_prob_integrates_to_one(self):
        """Grid integral of the 2-D action density ≈ 1."""
        p = _make_policy(action_dim=2)
        mu = np.array([0.3, -0.5])
        sig = np.array([0.5, 0.8])
        _set_const_policy(p, mu, np.log(sig))
        n = 500
        g = np.linspace(-0.999, 0.999, n)
        xx, yy = np.meshgrid(g, g)
        pts = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1),
            dtype=torch.float32,
        )
        obs = torch.zeros(len(pts), OBS_DIM)
        ev = p.evaluate_actions(obs, pts, torch.zeros(len(pts)))
        dg = g[1] - g[0]
        integral = float(ev.log_prob.exp().sum()) * dg * dg
        self.assertAlmostEqual(
            integral, 1.0, places=2,
            msg=f"action density integral = {integral}",
        )

    def test_sample_vs_evaluate_log_prob(self):
        """sample_action scores the stored float32 action; evaluating
        the same action must reproduce the same log_prob."""
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions, lp_sample = p.sample_action(obs)
        ev = p.evaluate_actions(obs, actions, torch.zeros(50))
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-5, f"sample vs evaluate diff = {diff}")

    def test_sample_vs_evaluate_log_prob_with_explore(self):
        torch.manual_seed(7)
        p = _make_policy()
        obs = torch.randn(40, OBS_DIM)
        for e in (-1.0, -0.5, 0.5, 1.0):
            actions, lp_sample = p.sample_action(obs, explore_factor=e)
            ev = p.evaluate_actions(
                obs, actions, torch.full((40,), e),
            )
            diff = (lp_sample - ev.log_prob).abs().max().item()
            self.assertLess(diff, 1e-5, f"e={e}: diff = {diff}")


class TestUncertainty(unittest.TestCase):
    """U = 1/(2·∫p_A^2) per dim — closed-form vs quadrature."""

    def _u(self, mu, sig):
        p = _make_policy()
        _set_const_policy(p, mu, np.log(sig))
        obs = torch.zeros(4, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(4, ACTION_DIM), torch.zeros(4),
        )
        return float(ev.uncertainty[0])

    def test_u_in_range(self):
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(100, ACTION_DIM), torch.zeros(100),
        )
        u = ev.uncertainty
        self.assertTrue((u >= 0.0).all(), f"U < 0: min={u.min()}")
        self.assertTrue((u <= 1.0).all(), f"U > 1: max={u.max()}")

    def test_u_matches_quadrature(self):
        mus = np.array([0.0, 0.5, -0.8, 1.5])
        sigs = np.array([0.3, 0.6, 1.0, 0.368])
        u = self._u(mus, sigs)
        ref = _u_ref(mus, sigs)
        self.assertAlmostEqual(u, ref, places=4,
                               msg=f"U={u}, quadrature={ref}")

    def test_u_extremes(self):
        """Narrow sigma → U≈0; huge sigma → boundary mixture → U≈0;
        sigma* at mu=0 gives the family max U≈0.985."""
        mu = np.zeros(ACTION_DIM)
        self.assertLess(
            self._u(mu, np.full(ACTION_DIM, 1e-3)), 0.01,
        )
        self.assertLess(
            self._u(mu, np.full(ACTION_DIM, 30.0)), 0.01,
        )
        u_star = self._u(mu, np.full(ACTION_DIM, _SIGMA_STAR))
        self.assertAlmostEqual(u_star, _COVERAGE_U_MAX, places=4)

    def test_sigma_star_is_maximizer(self):
        """U(0, sigma*) exceeds U at perturbed (mu, sigma)."""
        mu0 = np.zeros(ACTION_DIM)
        u_star = self._u(mu0, np.full(ACTION_DIM, _SIGMA_STAR))
        for sig in (0.3, 0.6, 1.2, 2.0):
            self.assertGreater(
                u_star, self._u(mu0, np.full(ACTION_DIM, sig)) + 1e-6,
                f"U(0,{sig}) should be below U_max",
            )
        for m in (0.3, -0.5, 1.0):
            self.assertGreater(
                u_star,
                self._u(np.full(ACTION_DIM, m),
                        np.full(ACTION_DIM, _SIGMA_STAR)) + 1e-6,
                f"U({m},sigma*) should be below U_max",
            )

    def test_u_independent_of_actions_and_explore(self):
        p = _make_policy()
        obs = torch.randn(20, OBS_DIM)
        a1 = torch.zeros(20, ACTION_DIM)
        a2 = torch.rand(20, ACTION_DIM) * 2 - 1
        ev1 = p.evaluate_actions(obs, a1, torch.zeros(20))
        ev2 = p.evaluate_actions(obs, a2, torch.ones(20))
        torch.testing.assert_close(
            ev1.uncertainty, ev2.uncertainty, rtol=0, atol=0,
        )

    def test_u_varies_with_state_via_mu(self):
        """Shared sigma does NOT mean constant U: state-dependent mu
        moves U through the tanh geometry."""
        torch.manual_seed(0)
        p = _make_policy()
        obs = torch.randn(200, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(200, ACTION_DIM), torch.zeros(200),
        )
        self.assertGreater(float(ev.uncertainty.std()), 1e-6)


class TestExploreFactor(unittest.TestCase):
    """Coverage-radial mapping: mu_e = c·mu, r_e = r* + c(r - r*)."""

    def test_e0_is_identity(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        mu, r = p._policy_params(obs)
        mu_e, r_e = p._explored_params(mu, r, torch.zeros(10))
        torch.testing.assert_close(mu_e, mu, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            r_e, r.expand(10, -1), rtol=0, atol=1e-12,
        )

    def test_mapping_values(self):
        p = _make_policy()
        mu = torch.tensor([[1.2, -0.6]], dtype=torch.float64)
        r = torch.tensor([math.log(0.5), math.log(2.0)],
                         dtype=torch.float64)
        for e in (-1.0, -0.5, 0.5, 1.0):
            c = math.exp(-e * math.log(3.0))
            mu_e, r_e = p._explored_params(
                mu, r, torch.tensor([e], dtype=torch.float64),
            )
            torch.testing.assert_close(
                mu_e, mu * c, rtol=0, atol=1e-12,
            )
            torch.testing.assert_close(
                r_e,
                (_COVERAGE_LOG_STD_STAR
                 + c * (r - _COVERAGE_LOG_STD_STAR)).expand(1, -1),
                rtol=0, atol=1e-12,
            )

    def test_coverage_monotone_in_e(self):
        """U_e non-decreasing as e grows, for a grid of (mu, sigma)."""
        p = _make_policy()
        es = np.linspace(-1.0, 1.0, 21)
        for mu_val in (-1.5, -0.4, 0.0, 0.7, 2.0):
            for sig_val in (0.1, 0.3, 0.86, 1.5, 4.0):
                mu = torch.full((1, 1), mu_val, dtype=torch.float64)
                r = torch.tensor([math.log(sig_val)], dtype=torch.float64)
                us = []
                for e in es:
                    mu_e, r_e = p._explored_params(
                        mu, r, torch.tensor([e], dtype=torch.float64),
                    )
                    us.append(float(p._action_uncertainty(mu_e, r_e)))
                diffs = np.diff(us)
                self.assertTrue(
                    (diffs >= -1e-12).all(),
                    f"mu={mu_val} sigma={sig_val}: non-monotone "
                    f"U_e {us}",
                )

    def test_coverage_strictly_increases_off_optimum(self):
        p = _make_policy()
        mu = torch.tensor([[1.0]], dtype=torch.float64)
        r = torch.tensor([math.log(2.0)], dtype=torch.float64)
        mu_e0, r_e0 = p._explored_params(mu, r, 0.0)
        mu_e1, r_e1 = p._explored_params(mu, r, 1.0)
        u0 = float(p._action_uncertainty(mu_e0, r_e0))
        u1 = float(p._action_uncertainty(mu_e1, r_e1))
        self.assertGreater(u1, u0 + 0.01)

    def test_fixed_point_invariant(self):
        """At theta* = (0, r*) every e is a no-op."""
        p = _make_policy()
        mu = torch.zeros(4, 1, dtype=torch.float64)
        r = torch.tensor([_COVERAGE_LOG_STD_STAR], dtype=torch.float64)
        for e in (-1.0, -0.5, 0.5, 1.0):
            mu_e, r_e = p._explored_params(
                mu, r, torch.full((4,), e, dtype=torch.float64),
            )
            torch.testing.assert_close(
                mu_e, mu, rtol=0, atol=1e-12,
            )
            torch.testing.assert_close(
                r_e, r.expand(4, -1), rtol=0, atol=1e-12,
            )

    def test_e_out_of_range_raises(self):
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        for e in (-1.5, 1.5):
            with self.assertRaises(ValueError):
                p.sample_action(obs, explore_factor=e)
        with self.assertRaises(ValueError):
            p.evaluate_actions(
                obs, torch.zeros(4, ACTION_DIM),
                torch.full((4,), 1.5),
            )

    def test_sample_differs_from_act(self):
        """Stochastic sampling actually explores — differs from the
        deterministic median act()."""
        p = _make_policy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        a1, _ = p.act(obs)
        torch.manual_seed(123)
        a2, _ = p.sample(obs, explore_factor=0.0)
        self.assertFalse(np.allclose(a1, a2, atol=1e-7),
                         "sample should differ from deterministic act")

    def test_reset_reproducibility(self):
        """Policy.reset(seed) reseeds sampling — the framework's
        per-episode reproducibility contract."""
        p = _make_policy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        p.reset(42)
        a1, _ = p.sample(obs)
        a2, _ = p.sample(obs)
        p.reset(42)
        a3, _ = p.sample(obs)
        np.testing.assert_allclose(a3, a1, rtol=0, atol=0)
        self.assertFalse(np.allclose(a2, a1, atol=1e-8),
                         "consecutive samples should differ")
        p.reset(99)
        a4, _ = p.sample(obs)
        self.assertFalse(np.allclose(a4, a1, atol=1e-8),
                         "different seed should give different action")


class TestGuards(unittest.TestCase):
    """Fail-loud behavior at the numeric boundary (DESIGN §8)."""

    def test_boundary_action_raises(self):
        p = _make_policy()
        obs = torch.zeros(4, OBS_DIM)
        for bad in (1.0, -1.0):
            actions = torch.zeros(4, ACTION_DIM)
            actions[0, 0] = bad
            with self.assertRaises(RuntimeError):
                p.evaluate_actions(obs, actions, torch.zeros(4))

    def test_action_beyond_safe_margin_raises(self):
        p = _make_policy()
        obs = torch.zeros(4, OBS_DIM)
        actions = torch.zeros(4, ACTION_DIM)
        # float32 spacing near 1.0 is ~6e-8 — a larger offset is needed
        # to land strictly above _ACTION_SAFE after quantization.
        actions[0, 0] = 1.0 - (1.0 - _ACTION_SAFE) * 0.25
        with self.assertRaises(RuntimeError):
            p.evaluate_actions(obs, actions, torch.zeros(4))

    def test_nonfinite_action_raises(self):
        p = _make_policy()
        obs = torch.zeros(4, OBS_DIM)
        actions = torch.zeros(4, ACTION_DIM)
        actions[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            p.evaluate_actions(obs, actions, torch.zeros(4))
        actions[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            p.evaluate_actions(obs, actions, torch.zeros(4))

    def test_unsafe_sampling_params_raise(self):
        """mu far into saturation with small sigma: the effective
        distribution's mass past z_safe exceeds the budget → raise,
        no clip, no resample."""
        p = _make_policy()
        _set_const_policy(
            p, np.full(ACTION_DIM, 6.5),
            np.full(ACTION_DIM, math.log(0.3)),
        )
        obs = torch.zeros(4, OBS_DIM)
        with self.assertRaises(RuntimeError):
            p.sample_action(obs)

    def test_evaluate_scores_safe_actions_under_unsafe_params(self):
        """evaluate_actions checks the ACTIONS, not the sampling
        support: stored actions inside the range remain scoreable."""
        p = _make_policy()
        _set_const_policy(
            p, np.full(ACTION_DIM, 6.5),
            np.full(ACTION_DIM, math.log(0.3)),
        )
        obs = torch.zeros(4, OBS_DIM)
        actions = torch.zeros(4, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.zeros(4))
        self.assertTrue(torch.isfinite(ev.log_prob).all())

    def test_tail_risk_value(self):
        """_tail_risk matches the Gaussian tail formula."""
        mu = torch.tensor([[0.0]], dtype=torch.float64)
        sig = torch.tensor([[0.5]], dtype=torch.float64)
        p_tail = PreTanhNormalPolicy._tail_risk(mu, sig)
        z = _Z_SAFE / 0.5
        expect = 2.0 * sp_stats.norm.sf(z)
        self.assertAlmostEqual(float(p_tail), expect, places=12)


class TestGradients(unittest.TestCase):
    """log_prob and U gradients reach the net and log_std."""

    def test_log_prob_gradient_completeness(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        (-ev.log_prob.mean()).backward()
        self.assertIsNotNone(p.log_std.grad)
        self.assertFalse(
            torch.allclose(p.log_std.grad,
                           torch.zeros_like(p.log_std.grad)),
            "log_std got zero log_prob gradient",
        )
        last = p.net[-1]
        self.assertIsNotNone(last.weight.grad)
        self.assertFalse(
            torch.allclose(last.weight.grad,
                           torch.zeros_like(last.weight.grad)),
            "mean head got zero log_prob gradient",
        )

    def test_uncertainty_gradient_completeness(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
        )
        ev.uncertainty.mean().backward()
        self.assertIsNotNone(p.log_std.grad)
        self.assertFalse(
            torch.allclose(p.log_std.grad,
                           torch.zeros_like(p.log_std.grad)),
            "log_std got zero U gradient",
        )
        self.assertIsNotNone(p.net[-1].weight.grad)
        self.assertFalse(
            torch.allclose(p.net[-1].weight.grad,
                           torch.zeros_like(p.net[-1].weight.grad)),
            "mean head got zero U gradient",
        )

    def test_trunk_receives_gradient(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        (-ev.log_prob.mean() + ev.uncertainty.mean()).backward()
        for param in p.net[0].parameters():
            self.assertIsNotNone(param.grad)

    def test_gradcheck_u_and_log_prob(self):
        """Finite-difference check of U and log_prob wrt mu, r, e."""
        p = _make_policy()
        acts = (torch.rand(4, 2, dtype=torch.float64) * 2 - 1) * 0.9
        mu = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        r = torch.randn(2, dtype=torch.float64,
                        requires_grad=True) * 0.5 - 0.3
        e = (torch.rand(4, dtype=torch.float64,
                        requires_grad=True) * 2 - 1) * 0.9

        def fn(m_, r_, e_):
            mu_e, r_e = p._explored_params(m_, r_, e_)
            lp = p._action_log_prob(acts, mu_e, r_e)
            u = p._action_uncertainty(mu_e, r_e)
            return lp.sum() + u.sum()

        self.assertTrue(torch.autograd.gradcheck(
            fn, (mu, r, e), raise_exception=True,
        ))

    def test_native_u_gradcheck(self):
        p = _make_policy()
        mu = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        r = (torch.randn(3, dtype=torch.float64,
                         requires_grad=True) * 0.5 - 0.3)

        def fn(m_, r_):
            return p._action_uncertainty(m_, r_).sum()

        self.assertTrue(torch.autograd.gradcheck(
            fn, (mu, r), raise_exception=True,
        ))


class TestBufferScale(unittest.TestCase):
    """The PPOBuffer callsite: one evaluate_actions over all frames."""

    def test_full_buffer_scale(self):
        p = _make_policy()
        B = 204800
        obs = torch.randn(B, OBS_DIM)
        actions = torch.tanh(torch.randn(B, ACTION_DIM))
        with torch.no_grad():
            ev = p.evaluate_actions(
                obs, actions, torch.zeros(B), want_stats=True,
            )
        self.assertEqual(ev.log_prob.shape, (B,))
        self.assertTrue(torch.isfinite(ev.log_prob).all())
        self.assertTrue(torch.isfinite(ev.uncertainty).all())
        self.assertIsNotNone(ev.stats)


class TestStats(unittest.TestCase):
    """want_stats keys and effective-vs-policy semantics."""

    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertIsNotNone(ev.stats)
        for key in [
            "uncertainty", "effective_uncertainty",
            "latent_std_mean", "latent_std_min", "latent_std_max",
            "effective_latent_std_mean", "effective_latent_std_min",
            "effective_latent_std_max",
            "latent_mean_abs", "effective_latent_mean_abs",
            "coverage_distance",
            "near_boundary_probability",
            "effective_near_boundary_probability",
            "effective_unsafe_tail_max",
        ]:
            self.assertIn(key, ev.stats, f"missing stat {key}")

    def test_effective_stats_reflect_e(self):
        """Native stats are e-independent; effective stats move."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions = torch.zeros(50, ACTION_DIM)
        ev0 = p.evaluate_actions(
            obs, actions, torch.zeros(50), want_stats=True,
        )
        ev1 = p.evaluate_actions(
            obs, actions, torch.ones(50), want_stats=True,
        )
        self.assertAlmostEqual(
            ev0.stats["uncertainty"], ev1.stats["uncertainty"], places=7,
        )
        self.assertAlmostEqual(
            ev0.stats["latent_std_mean"],
            ev1.stats["latent_std_mean"], places=7,
        )
        self.assertNotAlmostEqual(
            ev0.stats["effective_uncertainty"],
            ev1.stats["effective_uncertainty"], places=4,
        )

    def test_stats_at_init(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        s = ev.stats
        self.assertAlmostEqual(
            s["latent_std_mean"], math.exp(-1.0), places=5,
        )
        self.assertLess(s["effective_unsafe_tail_max"], 1e-12)


class TestExport(unittest.TestCase):
    """Strict loading, self-containment, and parity."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ptn_export_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = _make_policy()
        return p.to_blueprint(dest_path=self._tmp)

    def test_payload_metadata(self):
        self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["policy_class"], "PreTanhNormalPolicy",
        )
        self.assertEqual(payload["arch"]["action_dim"], ACTION_DIM)
        self.assertEqual(
            payload["distribution_kind"],
            "tanh_diagonal_normal_shared_std_v1",
        )
        self.assertEqual(
            payload["uncertainty_kind"], "marginal_renyi2_width_v1",
        )
        self.assertEqual(
            payload["exploration_kind"], "coverage_radial_logscale_v1",
        )

    def test_manifest(self):
        self._make_export()
        import json
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(
            manifest["policy_class"], "PreTanhNormalPolicy",
        )
        self.assertEqual(
            manifest["exported_class"], "ExportedPreTanhNormalPolicy",
        )
        self.assertEqual(
            manifest["exploration_kind"], "coverage_radial_logscale_v1",
        )

    def test_rejects_missing_and_extra_keys(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        sd = dict(payload["state_dict"])
        sd.pop("net.0.weight")
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(Exception):
            bp.build()
        sd["bogus.weight"] = torch.zeros(2, 2)
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(Exception):
            bp.build()

    def test_rejects_wrong_format_version(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["format_version"] = 999
        torch.save(payload, self._model_path)
        with self.assertRaises(Exception):
            bp.build()

    def test_rejects_wrong_policy_class(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "StateTruncatedNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(Exception):
            bp.build()

    def test_rejects_wrong_kinds(self):
        bp = self._make_export()
        for key in ("distribution_kind", "uncertainty_kind",
                    "exploration_kind"):
            payload = torch.load(self._model_path, map_location="cpu")
            payload[key] = "bogus"
            torch.save(payload, self._model_path)
            with self.assertRaises(Exception, msg=key):
                bp.build()

    def test_no_repo_imports(self):
        self._make_export()
        code = (Path(self._tmp) / "policy.py").read_text()
        for forbidden in [
            "from baseline", "import baseline", "from envs", "import envs",
        ]:
            self.assertNotIn(forbidden, code)

    def test_parity_act_and_sample(self):
        torch.manual_seed(999)
        p = _make_policy()
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="ptn_par_"))
        loaded = bp.build()
        for seed in range(10):
            torch.manual_seed(seed)
            obs = torch.randn(OBS_DIM).numpy().astype(np.float32)
            expected = p.act(obs)[0]
            actual = loaded.act(obs)[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for e in (0.0, 0.5, -0.7):
            torch.manual_seed(12345)
            a_exp, lp_exp = p.sample(
                obs, explore_factor=e, want_extra=True,
            )
            torch.manual_seed(12345)
            a_act, lp_act = loaded.sample(
                obs, explore_factor=e, want_extra=True,
            )
            np.testing.assert_allclose(a_act, a_exp, rtol=0, atol=0)
            self.assertAlmostEqual(
                lp_act["log_prob"], lp_exp["log_prob"], places=6,
            )

    def test_parity_evaluate(self):
        torch.manual_seed(3)
        p = _make_policy()
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="ptn_ev_"))
        loaded = bp.build()
        obs = torch.randn(8, OBS_DIM)
        actions = torch.tanh(torch.randn(8, ACTION_DIM))
        ei = torch.linspace(-1.0, 1.0, 8)
        ev = p.evaluate_actions(obs, actions, ei)
        lp_exp = loaded.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(
            lp_exp, ev.log_prob.detach(), rtol=0, atol=1e-10,
        )

    def test_works_without_repo_on_path(self):
        import subprocess
        import sys

        torch.manual_seed(42)
        p = _make_policy()
        p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]
        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import ExportedPreTanhNormalPolicy; "
            "import numpy as np; "
            "p = ExportedPreTanhNormalPolicy(); "
            "obs = np.array({obs!r}, dtype=np.float32); "
            "a, _ = p.act(obs); "
            "print(repr(a.tolist()))"
        ).format(tmp=self._tmp, obs=obs.tolist())
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                 "HOME": "/root",
                 "PYTHONPATH": self._tmp},
        )
        if result.returncode != 0:
            self.fail(f"subprocess failed:\n{result.stderr}")
        actual = np.array(eval(result.stdout.strip()), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


class TestBlueprint(unittest.TestCase):
    def test_blueprint_loads(self):
        from envs.framework.policy import PolicyBlueprint
        bp = PolicyBlueprint.load(
            "baseline/humanoid21/blueprints/"
            "init_policy_pre_tanh_normal.yaml"
        )
        policy = bp.build()
        self.assertIsInstance(policy, PreTanhNormalPolicy)
        self.assertEqual(policy.obs_dim, 96)
        self.assertEqual(policy.action_dim, 21)


class TestDeviceSync(unittest.TestCase):
    def test_act_after_to_cpu(self):
        p = _make_policy()
        p.to("cpu")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))


if __name__ == "__main__":
    unittest.main()
