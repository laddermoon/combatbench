"""Tests for MixtureTruncatedNormalPolicy.

Verifies (per DESIGN_mixture_truncated_normal.md §12):
1. Sampling produces actions in [-1, 1] with a shared component index
2. log_prob is the full mixture density (scipy truncnorm reference,
   integration to 1, sample-vs-evaluate consistency)
3. Mixture semantics: duplicate-component invariance, permutation
   invariance, dominant-component degeneration
4. Uncertainty U = normalized marginal Rényi-2 width, checked against
   numeric integration; independent of actions and explore_factor
5. explore_factor scales only component σ (π and μ untouched)
6. Gradients reach all three head blocks (logits / mean / log_std)
7. K=1 degenerate equivalence with StateTruncatedNormalPolicy
   (distribution-level; U is the L2 metric, not the peak metric)
8. Export: strict loading incl. K validation, self-contained policy.py,
   subprocess load without repo on sys.path, act/sample parity
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import special as sp_special
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.mixture_truncated_normal_mlp import (
    MixtureTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.state_truncated_normal_mlp import (
    StateTruncatedNormalPolicy,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
K = 3


def _make_policy(**kwargs) -> MixtureTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_components=K,
        device="cpu",
    )
    defaults.update(kwargs)
    return MixtureTruncatedNormalPolicy(**defaults)


def _set_const_mixture(
    p: MixtureTruncatedNormalPolicy,
    logits,
    mus,
    log_stds,
) -> None:
    """Force a state-independent mixture: π=softmax(logits), μ=mus,
    σ=exp(log_stds).  All head weights zeroed; biases carry the values.
    """
    K_, D = p.num_components, p.action_dim
    mus_t = torch.as_tensor(mus, dtype=torch.float32).reshape(K_, D)
    ls_t = torch.as_tensor(log_stds, dtype=torch.float32).reshape(K_, D)
    with torch.no_grad():
        p.head.weight.zero_()
        p.head.bias.zero_()
        p.head.bias[:K_] = torch.as_tensor(logits, dtype=torch.float32)
        p.head.bias[K_:K_ + K_ * D] = torch.atanh(
            mus_t.clamp(-0.9999, 0.9999)
        ).flatten()
        p.head.bias[K_ + K_ * D:] = ls_t.flatten()


def _mixture_pdf_ref(logits, mus, sigmas, actions):
    """scipy reference: p(a) = Σ_k π_k Π_d truncnorm(a_d; μ_kd, σ_kd)."""
    logits = np.asarray(logits, dtype=np.float64)
    pi = np.exp(logits - sp_special.logsumexp(logits))
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    K_, D = mus.shape
    total = np.zeros(actions.shape[0])
    for k in range(K_):
        comp = np.ones(actions.shape[0])
        for d in range(D):
            m, s = mus[k, d], sigmas[k, d]
            comp *= sp_stats.truncnorm.pdf(
                actions[:, d], (-1.0 - m) / s, (1.0 - m) / s,
                loc=m, scale=s,
            )
        total += pi[k] * comp
    return total


def _u_ref(logits, mus, sigmas):
    """Reference U: numeric integration of each marginal's squared density."""
    logits = np.asarray(logits, dtype=np.float64)
    pi = np.exp(logits - sp_special.logsumexp(logits))
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    K_, D = mus.shape
    grid = np.linspace(-1.0, 1.0, 20001)
    us = []
    for d in range(D):
        p_d = np.zeros_like(grid)
        for k in range(K_):
            m, s = mus[k, d], sigmas[k, d]
            p_d += pi[k] * sp_stats.truncnorm.pdf(
                grid, (-1.0 - m) / s, (1.0 - m) / s, loc=m, scale=s,
            )
        r_d = np.trapezoid(p_d * p_d, grid)
        us.append(1.0 / (2.0 * r_d))
    return float(np.mean(us))


class TestSampling(unittest.TestCase):
    """Sampling produces valid actions; component index is shared."""

    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(1000, OBS_DIM)
        actions, _ = p.sample_action(obs)
        self.assertEqual(actions.shape, (1000, ACTION_DIM))
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_shared_component_index(self):
        """Two well-separated components: each sample should sit near ONE
        component on ALL dims — never a cross-component mix."""
        torch.manual_seed(0)
        p = _make_policy(num_components=2)
        hi = 0.5 * np.ones(ACTION_DIM)
        lo = -0.5 * np.ones(ACTION_DIM)
        _set_const_mixture(
            p,
            logits=[0.0, 0.0],
            mus=np.stack([lo, hi]),
            log_stds=np.full((2, ACTION_DIM), math.log(0.02)),
        )
        obs = torch.randn(2000, OBS_DIM)
        actions, _ = p.sample_action(obs)
        frac_all_pos = ((actions > 0).all(dim=-1)).float().mean().item()
        frac_all_neg = ((actions < 0).all(dim=-1)).float().mean().item()
        frac_mixed = 1.0 - frac_all_pos - frac_all_neg
        # Shared index: ~50/50 all-pos / all-neg, ~0% mixed.  Per-dim
        # mixing would give ~1/16 all-pos and mostly-mixed samples.
        self.assertGreater(frac_all_pos, 0.4)
        self.assertGreater(frac_all_neg, 0.4)
        self.assertLess(frac_mixed, 0.01)

    def test_deterministic_action_is_argmax_component_mean(self):
        p = _make_policy()
        _set_const_mixture(
            p,
            logits=[0.0, 2.0, -1.0],
            mus=np.stack([
                -0.5 * np.ones(ACTION_DIM),
                0.3 * np.ones(ACTION_DIM),
                0.0 * np.ones(ACTION_DIM),
            ]),
            log_stds=np.full((K, ACTION_DIM), -1.0),
        )
        obs = torch.randn(10, OBS_DIM)
        det = p.deterministic_action(obs)
        self.assertTrue(
            torch.allclose(det, torch.full_like(det, 0.3), atol=1e-4),
            f"act should be component-1 mean (max weight), got {det[0]}",
        )

    def test_init(self):
        """At init: π uniform, σ ≡ e⁻¹, components not identical."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        log_pi, mean, sigma = p._head_forward(obs)
        self.assertTrue(torch.allclose(
            log_pi.exp(), torch.full_like(log_pi, 1.0 / K), atol=1e-6,
        ), "init π should be uniform")
        self.assertTrue(torch.allclose(
            sigma, torch.full_like(sigma, math.exp(-1.0)), atol=1e-6,
        ), "init σ should be e^-1")
        diffs = [
            float((mean[:, i] - mean[:, j]).abs().max())
            for i in range(K) for j in range(i + 1, K)
        ]
        self.assertTrue(all(d > 1e-6 for d in diffs),
                        "components should differ at init (symmetry broken)")


class TestLogProb(unittest.TestCase):
    """Full-mixture log_prob correctness."""

    def test_log_prob_matches_scipy_mixture(self):
        torch.manual_seed(42)
        p = _make_policy()
        logits = [0.5, -0.3, 0.8]
        mus = np.array([
            [-0.4, 0.1, 0.6, -0.2],
            [0.5, -0.5, 0.0, 0.4],
            [0.0, 0.7, -0.6, 0.1],
        ])
        sigmas = np.array([
            [0.2, 0.5, 0.3, 1.0],
            [0.8, 0.1, 0.6, 0.4],
            [1.5, 0.3, 0.2, 0.9],
        ])
        _set_const_mixture(p, logits, mus, np.log(sigmas))
        obs = torch.randn(30, OBS_DIM)
        actions = torch.rand(30, ACTION_DIM) * 2 - 1
        ev = p.evaluate_actions(
            obs, actions, torch.full((30,), 0.0),
        )
        ref = _mixture_pdf_ref(logits, mus, sigmas, actions.numpy())
        np.testing.assert_allclose(
            np.exp(ev.log_prob.detach().numpy()), ref, rtol=1e-4, atol=1e-6,
            err_msg="mixture density differs from scipy reference",
        )

    def test_log_prob_integrates_to_one(self):
        """Grid integral of the 2-D mixture density ≈ 1."""
        p = _make_policy(num_components=3, action_dim=2)
        logits = [0.4, -0.2, 0.9]
        mus = np.array([[-0.5, 0.3], [0.4, -0.4], [0.8, 0.7]])
        sigmas = np.array([[0.15, 0.4], [0.3, 0.2], [0.6, 0.5]])
        _set_const_mixture(p, logits, mus, np.log(sigmas))
        n = 400
        g = np.linspace(-1.0, 1.0, n)
        xx, yy = np.meshgrid(g, g)
        pts = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32,
        )
        obs = torch.zeros(len(pts), OBS_DIM)
        ev = p.evaluate_actions(obs, pts, torch.zeros(len(pts)))
        integral = (4.0 / (n - 1) ** 2) * float(ev.log_prob.exp().sum())
        self.assertAlmostEqual(integral, 1.0, places=2,
                               msg=f"mixture integral = {integral}")

    def test_sample_vs_evaluate_log_prob(self):
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions, lp_sample = p.sample_action(obs)
        ev = p.evaluate_actions(obs, actions, torch.full((50,), 0.0))
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-4, f"sample vs evaluate diff = {diff}")

    def test_empirical_moments_match_reference(self):
        """Sampled moments and component frequency match the scipy
        marginal — an independent reference, not just internal
        consistency."""
        torch.manual_seed(0)
        p = _make_policy(num_components=2)
        logits = [math.log(0.6), math.log(0.4)]
        mus = np.array([
            [-0.5, 0.3, 0.6, -0.1],
            [0.4, -0.6, -0.2, 0.5],
        ])
        sigmas = np.array([
            [0.1, 0.3, 0.2, 0.4],
            [0.25, 0.15, 0.35, 0.2],
        ])
        _set_const_mixture(p, logits, mus, np.log(sigmas))
        n = 40000
        obs = torch.zeros(n, OBS_DIM)
        with torch.no_grad():
            samples, _ = p.sample_action(obs)
        samples = samples.numpy().astype(np.float64)

        pi = np.exp(np.asarray(logits) - sp_special.logsumexp(logits))
        for d in range(ACTION_DIM):
            # Marginal moments of the mixture per dim.
            m_ref = 0.0
            m2_ref = 0.0
            for k in range(2):
                mu, s = mus[k, d], sigmas[k, d]
                a_, b_ = (-1.0 - mu) / s, (1.0 - mu) / s
                tn = sp_stats.truncnorm(a_, b_, loc=mu, scale=s)
                m_ref += pi[k] * tn.mean()
                m2_ref += pi[k] * (tn.var() + tn.mean() ** 2)
            var_ref = m2_ref - m_ref ** 2
            self.assertAlmostEqual(
                samples[:, d].mean(), m_ref, delta=0.02,
                msg=f"dim {d} mean: {samples[:, d].mean()} vs {m_ref}",
            )
            self.assertAlmostEqual(
                samples[:, d].var(), var_ref, delta=0.02,
                msg=f"dim {d} var: {samples[:, d].var()} vs {var_ref}",
            )
        # Component frequency: nearest-mean assignment over all dims
        # recovers π0 (components are well separated, σ ≤ 0.4).
        d0 = ((samples - mus[0]) ** 2).sum(axis=1)
        d1 = ((samples - mus[1]) ** 2).sum(axis=1)
        frac0 = (d0 < d1).mean()
        self.assertAlmostEqual(frac0, 0.6, delta=0.02)


class TestMixtureSemantics(unittest.TestCase):
    """Mixture identities: duplication, permutation, degeneration."""

    def test_duplicate_component_invariance(self):
        """Splitting a component into two identical ones with split
        weight changes neither log_prob nor U."""
        p2 = _make_policy(num_components=2)
        p3 = _make_policy(num_components=3)
        mus = np.array([[0.2] * ACTION_DIM, [-0.6] * ACTION_DIM])
        ls = np.full((2, ACTION_DIM), -1.0)
        _set_const_mixture(p2, [0.0, 0.0], mus, ls)
        # K=3: [0.3/0.7 split of comp0] + comp1 → π = [0.15, 0.35, 0.5]
        mus3 = np.concatenate([mus[:1], mus[:1], mus[1:]])
        ls3 = np.concatenate([ls[:1], ls[:1], ls[1:]])
        _set_const_mixture(
            p3, [math.log(0.15), math.log(0.35), math.log(0.5)],
            mus3, ls3,
        )
        obs = torch.randn(20, OBS_DIM)
        actions = torch.rand(20, ACTION_DIM) * 2 - 1
        e = torch.zeros(20)
        ev2 = p2.evaluate_actions(obs, actions, e)
        ev3 = p3.evaluate_actions(obs, actions, e)
        torch.testing.assert_close(
            ev3.log_prob, ev2.log_prob, rtol=0, atol=1e-5,
        )
        torch.testing.assert_close(
            ev3.uncertainty, ev2.uncertainty, rtol=0, atol=1e-5,
        )

    def test_permutation_invariance(self):
        p = _make_policy()
        q = _make_policy()
        logits = [0.3, -0.7, 1.1]
        mus = np.array([
            [-0.4, 0.2, 0.5, -0.1],
            [0.6, -0.3, 0.0, 0.4],
            [0.1, 0.7, -0.5, 0.2],
        ])
        ls = np.array([
            [-2.0, -0.5, -1.2, 0.0],
            [-0.3, -2.5, -0.8, -1.0],
            [0.4, -1.3, -1.7, -0.2],
        ])
        _set_const_mixture(p, logits, mus, ls)
        perm = [2, 0, 1]
        _set_const_mixture(
            q, [logits[i] for i in perm], mus[perm], ls[perm],
        )
        obs = torch.randn(20, OBS_DIM)
        actions = torch.rand(20, ACTION_DIM) * 2 - 1
        e = torch.zeros(20)
        ev_p = p.evaluate_actions(obs, actions, e)
        ev_q = q.evaluate_actions(obs, actions, e)
        torch.testing.assert_close(
            ev_q.log_prob, ev_p.log_prob, rtol=0, atol=1e-5,
        )
        torch.testing.assert_close(
            ev_q.uncertainty, ev_p.uncertainty, rtol=0, atol=1e-5,
        )

    def test_dominant_weight_degenerates(self):
        """π → one-hot ⇒ mixture behaves like that component."""
        p = _make_policy()
        mus = np.array([
            [-0.5] * ACTION_DIM,
            [0.3] * ACTION_DIM,
            [0.9] * ACTION_DIM,
        ])
        ls = np.full((K, ACTION_DIM), -1.5)
        _set_const_mixture(p, [0.0, 30.0, -30.0], mus, ls)
        obs = torch.randn(10, OBS_DIM)
        det = p.deterministic_action(obs)
        self.assertTrue(torch.allclose(
            det, torch.full_like(det, 0.3), atol=1e-4,
        ))
        # log_prob ≈ component-1 conditional density
        actions = torch.rand(10, ACTION_DIM) * 2 - 1
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        log_pi, mean, sigma = p._head_forward(obs)
        a = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6).unsqueeze(1)
        log_Z = p._log_trunc_Z(mean, sigma)
        z = (a - mean) / sigma
        comp = (-0.5 * z * z - torch.log(sigma)
                - 0.5 * math.log(2 * math.pi) - log_Z).sum(-1)
        torch.testing.assert_close(
            ev.log_prob, comp[:, 1], rtol=0, atol=1e-4,
        )


class TestUncertainty(unittest.TestCase):
    """U = 1/(2·∫p_d²) per dim, averaged — marginal Rényi-2 width."""

    def test_u_in_range(self):
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(100, ACTION_DIM), torch.zeros(100),
        )
        u = ev.uncertainty
        self.assertTrue((u >= 0.0).all(), f"U < 0: min={u.min()}")
        self.assertTrue((u <= 1.0).all(), f"U > 1: max={u.max()}")

    def test_u_matches_numeric_integration(self):
        """Closed-form U vs trapezoid integration of the marginal."""
        p = _make_policy()
        logits = [0.5, -0.3, 0.8]
        mus = np.array([
            [-0.4, 0.1, 0.6, -0.2],
            [0.5, -0.5, 0.0, 0.4],
            [0.0, 0.7, -0.6, 0.1],
        ])
        sigmas = np.array([
            [0.2, 0.5, 0.3, 1.0],
            [0.8, 0.1, 0.6, 0.4],
            [1.5, 0.3, 0.2, 0.9],
        ])
        _set_const_mixture(p, logits, mus, np.log(sigmas))
        obs = torch.zeros(5, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(5, ACTION_DIM), torch.zeros(5),
        )
        ref = _u_ref(logits, mus, sigmas)
        self.assertAlmostEqual(
            float(ev.uncertainty[0]), ref, places=3,
            msg=f"U={float(ev.uncertainty[0])}, numeric={ref}",
        )

    def test_u_separated_peaks_double(self):
        """Two equal-weight well-separated narrow components ≈ 2× the
        single-component U on each dim."""
        p1 = _make_policy(num_components=1)
        p2 = _make_policy(num_components=2)
        s = math.log(0.05)
        _set_const_mixture(
            p1, [0.0], np.full((1, ACTION_DIM), 0.3), np.full((1, ACTION_DIM), s),
        )
        _set_const_mixture(
            p2, [0.0, 0.0],
            np.stack([-0.3 * np.ones(ACTION_DIM), 0.3 * np.ones(ACTION_DIM)]),
            np.full((2, ACTION_DIM), s),
        )
        obs = torch.zeros(4, OBS_DIM)
        z = torch.zeros(4)
        u1 = p1.evaluate_actions(obs, torch.zeros(4, ACTION_DIM), z).uncertainty
        u2 = p2.evaluate_actions(obs, torch.zeros(4, ACTION_DIM), z).uncertainty
        ratio = (u2 / u1).mean().item()
        self.assertAlmostEqual(ratio, 2.0, places=2,
                               msg=f"separated-pair U ratio = {ratio}")

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

    def test_u_extremes(self):
        """Wide σ → U≈1 (uniform); narrow σ → U≈0."""
        p = _make_policy(num_components=1)
        obs = torch.zeros(4, OBS_DIM)
        z = torch.zeros(4)
        _set_const_mixture(
            p, [0.0], np.zeros((1, ACTION_DIM)),
            np.full((1, ACTION_DIM), 10.0),
        )
        u_wide = p.evaluate_actions(
            obs, torch.zeros(4, ACTION_DIM), z,
        ).uncertainty
        self.assertGreater(float(u_wide.mean()), 0.99)
        _set_const_mixture(
            p, [0.0], np.zeros((1, ACTION_DIM)),
            np.full((1, ACTION_DIM), -6.0),
        )
        u_narrow = p.evaluate_actions(
            obs, torch.zeros(4, ACTION_DIM), z,
        ).uncertainty
        self.assertLess(float(u_narrow.mean()), 0.01)

    def test_u_no_nan_at_extremes(self):
        p = _make_policy()
        _set_const_mixture(
            p, [0.0, 5.0, -5.0],
            np.stack([
                -0.999 * np.ones(ACTION_DIM),
                np.zeros(ACTION_DIM),
                0.999 * np.ones(ACTION_DIM),
            ]),
            np.array([
                np.full(ACTION_DIM, -19.0),
                np.full(ACTION_DIM, 19.0),
                np.full(ACTION_DIM, -19.0),
            ]),
        )
        obs = torch.randn(20, OBS_DIM)
        actions = torch.rand(20, ACTION_DIM) * 2 - 1
        ev = p.evaluate_actions(obs, actions, torch.zeros(20))
        self.assertTrue(torch.isfinite(ev.log_prob).all())
        self.assertTrue(torch.isfinite(ev.uncertainty).all())


class TestExploreFactor(unittest.TestCase):
    """explore_factor scales only component σ."""

    def test_scale_values(self):
        p = _make_policy()
        self.assertAlmostEqual(p._explore_scale(0.0), 1.0, places=6)
        self.assertAlmostEqual(p._explore_scale(-1.0), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(p._explore_scale(1.0), 3.0, places=6)

    def test_effective_sigma_per_component(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        log_pi, mean, sigma_p = p._head_forward(obs)
        for e, expect in [(-1.0, 1.0 / 3.0), (0.0, 1.0), (1.0, 3.0)]:
            eff = p._effective_sigma(sigma_p, e)
            torch.testing.assert_close(
                eff / sigma_p,
                torch.full_like(eff, expect), rtol=0, atol=1e-6,
            )

    def test_batched_explore_factor(self):
        """Per-frame e broadcasts (B,) → (B,1,1) over (B,K,D)."""
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        _, _, sigma_p = p._head_forward(obs)
        ei = torch.linspace(-1.0, 1.0, 8)
        eff = p._effective_sigma(sigma_p, ei)
        expect = torch.exp(ei * math.log(3.0)).view(-1, 1, 1)
        torch.testing.assert_close(eff, sigma_p * expect, rtol=0, atol=1e-6)

    def test_explore_factor_leaves_weights_and_means(self):
        """e affects only σ_eff; the scored mixture uses the same π, μ."""
        p = _make_policy()
        _set_const_mixture(
            p, [0.2, -0.4, 0.7],
            np.array([
                [-0.4, 0.1, 0.6, -0.2],
                [0.5, -0.5, 0.0, 0.4],
                [0.0, 0.7, -0.6, 0.1],
            ]),
            np.zeros((K, ACTION_DIM)),
        )
        obs = torch.zeros(5, OBS_DIM)
        actions = torch.rand(5, ACTION_DIM) * 2 - 1
        for e in (-1.0, 1.0):
            ev = p.evaluate_actions(
                obs, actions, torch.full((5,), e),
            )
            sigmas_scaled = np.exp(np.zeros((K, ACTION_DIM)) + e * math.log(3))
            ref = _mixture_pdf_ref(
                [0.2, -0.4, 0.7],
                np.array([
                    [-0.4, 0.1, 0.6, -0.2],
                    [0.5, -0.5, 0.0, 0.4],
                    [0.0, 0.7, -0.6, 0.1],
                ]),
                sigmas_scaled,
                actions.numpy(),
            )
            np.testing.assert_allclose(
                np.exp(ev.log_prob.detach().numpy()), ref, rtol=1e-4, atol=1e-6,
                err_msg=f"e={e}: mixture density mismatch",
            )


class TestGradients(unittest.TestCase):
    """All three head blocks receive gradient from both log_prob and U."""

    def _grad_blocks(self, p):
        """Row-block views of the leaf ``head.weight.grad`` after backward."""
        K_, D = p.num_components, p.action_dim
        g = p.head.weight.grad
        return {
            "logits": g[:K_],
            "mean": g[K_:K_ + K_ * D],
            "log_std": g[K_ + K_ * D:],
        }

    def test_log_prob_gradient_completeness(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        (-ev.log_prob.mean()).backward()
        self.assertIsNotNone(p.head.weight.grad)
        for name, g in self._grad_blocks(p).items():
            self.assertFalse(
                torch.allclose(g, torch.zeros_like(g)),
                f"{name} block got zero log_prob gradient",
            )

    def test_uncertainty_gradient_completeness(self):
        """U reaches σ/mean blocks; logits get grad when components differ."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
        )
        ev.uncertainty.mean().backward()
        self.assertIsNotNone(p.head.weight.grad)
        K_, D = p.num_components, p.action_dim
        self.assertFalse(torch.allclose(
            p.head.weight.grad[K_ + K_ * D:], torch.zeros(K_ * D, HIDDEN_DIM),
        ), "log_std block got zero U gradient")

    def test_trunk_receives_gradient(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        (-ev.log_prob.mean() + ev.uncertainty.mean()).backward()
        for param in p.trunk.parameters():
            self.assertIsNotNone(param.grad)

    def test_gradcheck_log_prob_and_u(self):
        """Finite-difference check of the closed-form overlap integral
        and mixture log_prob on an asymmetric example (float64)."""
        torch.manual_seed(0)
        p = _make_policy(num_components=3, action_dim=2)
        acts = (torch.rand(4, 2, dtype=torch.float64) * 2 - 1) * 0.9
        logits = torch.randn(4, 3, dtype=torch.float64,
                             requires_grad=True)
        raw_mean = torch.randn(4, 3, 2, dtype=torch.float64,
                               requires_grad=True)
        raw_ls = torch.randn(4, 3, 2, dtype=torch.float64,
                             requires_grad=True)

        def fn(lg, rm, rl):
            log_pi = torch.log_softmax(lg, dim=-1)
            mean = torch.tanh(rm)
            sigma = rl.clamp(-20.0, 20.0).exp()
            lp = p._mixture_log_prob(acts, mean, sigma, log_pi)
            u = p._marginal_uncertainty(log_pi, mean, sigma)
            return lp.sum() + u.sum()

        self.assertTrue(torch.autograd.gradcheck(
            fn, (logits, raw_mean, raw_ls), raise_exception=True,
        ))

    def test_identical_components_zero_logits_u_gradient(self):
        """Fully coincident components: U w.r.t. logits is exactly zero —
        the distribution is invariant under weight reallocation."""
        p = _make_policy()
        _set_const_mixture(
            p, [0.0, 0.0, 0.0],
            np.zeros((K, ACTION_DIM)),
            np.zeros((K, ACTION_DIM)),
        )
        logits = torch.zeros(2, K, requires_grad=True)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.zeros(2, K, ACTION_DIM)
        sigma = torch.ones(2, K, ACTION_DIM)
        u = p._marginal_uncertainty(log_pi, mean, sigma)
        u.sum().backward()
        self.assertTrue(torch.allclose(
            logits.grad, torch.zeros_like(logits.grad), atol=1e-6,
        ), f"coincident-component logits grad should be 0, "
           f"got {logits.grad}")


class TestBufferScale(unittest.TestCase):
    """The PPOBuffer callsite: one evaluate_actions over all frames."""

    def test_full_buffer_scale(self):
        p = _make_policy()
        B = 204800
        obs = torch.randn(B, OBS_DIM)
        actions = torch.rand(B, ACTION_DIM) * 2 - 1
        with torch.no_grad():
            ev = p.evaluate_actions(
                obs, actions, torch.zeros(B), want_stats=True,
            )
        self.assertEqual(ev.log_prob.shape, (B,))
        self.assertTrue(torch.isfinite(ev.log_prob).all())
        self.assertTrue(torch.isfinite(ev.uncertainty).all())
        self.assertIsNotNone(ev.stats)


class TestStats(unittest.TestCase):
    """want_stats keys and init values."""

    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertIsNotNone(ev.stats)
        for key in [
            "uncertainty", "std_mean", "eff_std_mean", "std_min", "std_max",
            "mixture_weight_entropy", "effective_components",
            "max_component_weight", "sigma_state_std", "component_overlap",
            "component_weight_0", "component_weight_1", "component_weight_2",
        ]:
            self.assertIn(key, ev.stats, f"missing stat {key}")

    def test_stats_at_init(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        s = ev.stats
        self.assertAlmostEqual(
            s["mixture_weight_entropy"], math.log(K), places=5,
        )
        self.assertAlmostEqual(s["effective_components"], float(K), places=4)
        self.assertAlmostEqual(s["max_component_weight"], 1.0 / K, places=5)
        self.assertAlmostEqual(s["std_mean"], math.exp(-1.0), places=4)
        # Components near-identical at init → overlap ≈ 1
        self.assertGreater(s["component_overlap"], 0.9)


class TestDegenerateEquivalence(unittest.TestCase):
    """K=1 reduces to StateTruncatedNormalPolicy at distribution level.

    The σ branch init (w=0, b=-1) and shared trunk/mean wiring make the
    mixture with one component compute the same density.  U is the L2
    metric by design — checked against its own formula, NOT the old
    peak metric (see DESIGN §8).
    """

    def _make_pair(self):
        torch.manual_seed(7)
        state = StateTruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        mix = _make_policy(num_components=1)
        D = ACTION_DIM
        with torch.no_grad():
            mix.trunk[0].weight.copy_(state.trunk[0].weight)
            mix.trunk[0].bias.copy_(state.trunk[0].bias)
            mix.trunk[2].weight.copy_(state.trunk[2].weight)
            mix.trunk[2].bias.copy_(state.trunk[2].bias)
            # mixture head layout for K=1: [0]=logit, [1:1+D]=mean,
            # [1+D:1+2D]=log_std
            mix.head.weight[1:1 + D].copy_(state.head.weight[:D])
            mix.head.bias[1:1 + D].copy_(state.head.bias[:D])
            mix.head.weight[1 + D:].zero_()
            mix.head.bias[1 + D:].fill_(-1.0)
            mix.head.weight[0].zero_()
            mix.head.bias[0].zero_()
        return state, mix

    def test_act_bit_identical(self):
        state, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            mix.deterministic_action(obs), state.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_log_prob_within_tol(self):
        """log_prob matches up to erf-sum vs CDF-diff rounding (~1e-7)."""
        state, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.rand(64, ACTION_DIM) * 2 - 1
        ei = torch.linspace(-1.0, 1.0, 64)
        ev_s = state.evaluate_actions(obs, actions, ei)
        ev_m = mix.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(
            ev_m.log_prob, ev_s.log_prob, rtol=0, atol=1e-5,
        )

    def test_sample_close(self):
        """Same seed → same u stream; erf-space vs CDF-space icdf differ
        only by float rounding."""
        state, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.manual_seed(123)
        a_s, _ = state.sample_action(obs)
        torch.manual_seed(123)
        a_m, _ = mix.sample_action(obs)
        torch.testing.assert_close(a_m, a_s, rtol=0, atol=1e-5)


class TestExport(unittest.TestCase):
    """Strict loading, self-containment, and parity."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="mtn_export_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = _make_policy()
        return p.to_blueprint(dest_path=self._tmp)

    def test_payload_metadata(self):
        self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["policy_class"], "MixtureTruncatedNormalPolicy")
        self.assertEqual(payload["arch"]["num_components"], K)

    def test_manifest(self):
        self._make_export()
        import json
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(
            manifest["policy_class"], "MixtureTruncatedNormalPolicy",
        )
        self.assertEqual(
            manifest["exported_class"], "ExportedMixtureTruncNormPolicy",
        )
        self.assertEqual(
            manifest["uncertainty_kind"], "marginal_renyi2_width_v1",
        )
        self.assertEqual(manifest["arch"]["num_components"], K)

    def test_rejects_wrong_k(self):
        """Wrong K in payload → wrong head width → strict load error."""
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["arch"]["num_components"] = K + 1
        payload["num_components"] = K + 1
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_rejects_missing_and_extra_keys(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        sd = dict(payload["state_dict"])
        sd.pop("trunk.0.weight")
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()
        sd["bogus.weight"] = torch.zeros(2, 2)
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_rejects_wrong_format_version(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["format_version"] = 999
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("format version", str(ctx.exception).lower())

    def test_rejects_wrong_policy_class(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "StateTruncatedNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("class mismatch", str(ctx.exception).lower())

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
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="mtn_par_"))
        loaded = bp.build()
        for seed in range(10):
            torch.manual_seed(seed)
            obs = torch.randn(OBS_DIM).numpy().astype(np.float32)
            expected = p.act(obs)[0]
            actual = loaded.act(obs)[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        torch.manual_seed(12345)
        a_exp, lp_exp = p.sample(obs, explore_factor=0.5, want_extra=True)
        torch.manual_seed(12345)
        a_act, lp_act = loaded.sample(
            obs, explore_factor=0.5, want_extra=True,
        )
        np.testing.assert_allclose(a_act, a_exp, rtol=0, atol=0)
        self.assertAlmostEqual(
            lp_act["log_prob"], lp_exp["log_prob"], places=6,
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
            "from policy import ExportedMixtureTruncNormPolicy; "
            "import numpy as np; "
            "p = ExportedMixtureTruncNormPolicy(); "
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
