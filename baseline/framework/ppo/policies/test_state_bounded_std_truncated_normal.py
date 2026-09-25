"""Tests for StateBoundedStdTruncatedNormalPolicy.

Verifies (per DESIGN_bounded_std_truncated_normal.md §3.2/§11):
1. Architecture: trunk + head(2D) split [raw_mean | v]; σ half zero-init
   with bias=v_init → σ(obs) ≡ init_std at step 0
2. Degenerate equivalence: copying the shared variant's mean weights and
   raw_std into the state policy yields bit-identical forward/sample/
   evaluate outputs at all explore factors
3. State dependence: non-zero σ-head weights make σ vary across states
   (sigma_state_std > 0) while staying bounded
4. explore_factor: v_e = v + αe elementwise on (B, D) v — monotone σ in e
   per state, e=0 identity, range validated
5. Distribution parity vs scipy truncnorm with per-state σ;
   sample/evaluate consistency
6. Gradients flow to both head halves and the trunk
7. Export: strict load, std_source="state" metadata, parity, no-repo
   subprocess, reset(seed) replay
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.state_bounded_std_truncated_normal_mlp import (
    StateBoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _SQRT_2PI,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32

SIGMA_MIN = 0.05
SIGMA_MAX = 2.0
INIT_STD = math.exp(-1.0)


def _make_policy(**kwargs) -> StateBoundedStdTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return StateBoundedStdTruncatedNormalPolicy(**defaults)


def _make_shared(**kwargs) -> BoundedStdTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return BoundedStdTruncatedNormalPolicy(**defaults)


def _copy_shared_weights(
    state_p: StateBoundedStdTruncatedNormalPolicy,
    shared_p: BoundedStdTruncatedNormalPolicy,
) -> None:
    """Copy shared variant weights so the state policy degenerates to it.

    trunk[0]/trunk[2] take net[0]/net[2]; the mean half of head takes
    net[4]; the v half bias takes raw_std (weights already zero).
    """
    d = state_p.action_dim
    sd = state_p.state_dict()
    sd["trunk.0.weight"].copy_(shared_p.net[0].weight)
    sd["trunk.0.bias"].copy_(shared_p.net[0].bias)
    sd["trunk.2.weight"].copy_(shared_p.net[2].weight)
    sd["trunk.2.bias"].copy_(shared_p.net[2].bias)
    sd["head.weight"][:d].copy_(shared_p.net[4].weight)
    sd["head.bias"][:d].copy_(shared_p.net[4].bias)
    sd["head.bias"][d:].copy_(shared_p.raw_std.data)
    state_p.load_state_dict(sd)


class TestConfig(unittest.TestCase):
    """Config validation inherited from the shared variant."""

    def test_defaults_resolve(self):
        p = _make_policy()
        self.assertEqual(p.sigma_min, SIGMA_MIN)
        self.assertEqual(p.sigma_max, SIGMA_MAX)
        self.assertAlmostEqual(p.init_std, INIT_STD, places=15)
        self.assertAlmostEqual(p.explore_alpha, 1.19933891045474, places=10)

    def test_rejects_bad_bounds(self):
        for kw in (
            dict(sigma_min=0.0),
            dict(sigma_min=-0.1),
            dict(sigma_min=0.5, sigma_max=0.5),
            dict(sigma_min=0.5, init_std=0.4),   # init below min
            dict(init_std=3.0),                # init above max
            dict(sigma_min=float("nan")),
            dict(sigma_max=float("inf")),
        ):
            with self.assertRaises(ValueError, msg=str(kw)):
                _make_policy(**kw)

    def test_rejects_bad_alpha(self):
        for a in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=str(a)):
                _make_policy(explore_alpha=a)


class TestInit(unittest.TestCase):
    """σ head zero-init → constant σ = e⁻¹ at step 0."""

    def test_init_sigma_is_init_std_everywhere(self):
        torch.manual_seed(0)
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        mean, sigma = p.forward(obs)
        self.assertTrue(
            torch.allclose(
                sigma, torch.full_like(sigma, INIT_STD), atol=1e-6
            )
        )
        self.assertEqual(sigma.shape, (64, ACTION_DIM))

    def test_init_v_bias_is_v_init(self):
        p = _make_policy()
        d = p.action_dim
        self.assertTrue(
            torch.allclose(
                p.head.bias[d:],
                torch.full((d,), p._v_init),
            )
        )
        self.assertTrue((p.head.weight[d:] == 0).all())

    def test_sigma_state_std_zero_at_init(self):
        torch.manual_seed(0)
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.rand(64, ACTION_DIM) * 2 - 1,
            torch.zeros(64), want_stats=True,
        )
        self.assertEqual(ev.stats["sigma_state_std"], 0.0)


class TestDegenerateEquivalence(unittest.TestCase):
    """Copy shared weights → bit-identical outputs at every e."""

    def setUp(self):
        torch.manual_seed(0)
        self.shared = _make_shared()
        self.state = _make_policy()
        _copy_shared_weights(self.state, self.shared)
        torch.manual_seed(1)
        self.obs = torch.randn(32, OBS_DIM)

    def test_forward_bit_identical(self):
        for e in (0.0, 0.5, -1.0, 1.0):
            m1, s1 = self.state.forward(self.obs, explore_factor=e)
            m2, s2 = self.shared.forward(self.obs, explore_factor=e)
            self.assertTrue(torch.equal(m1, m2), f"mean differs at e={e}")
            self.assertTrue(torch.equal(s1, s2), f"sigma differs at e={e}")

    def test_evaluate_bit_identical(self):
        acts = torch.rand(32, ACTION_DIM) * 2 - 1
        for e in (0.0, 0.7, -0.3):
            ef = torch.full((32,), e)
            ev1 = self.state.evaluate_actions(self.obs, acts, ef)
            ev2 = self.shared.evaluate_actions(self.obs, acts, ef)
            self.assertTrue(torch.equal(ev1.log_prob, ev2.log_prob))
            self.assertTrue(
                torch.equal(ev1.uncertainty, ev2.uncertainty)
            )

    def test_sample_bit_identical(self):
        for e in (0.0, 0.5, -0.5):
            self.state.reset(9)
            a1, lp1 = self.state.sample_action(
                self.obs, explore_factor=e,
            )
            self.shared.reset(9)
            a2, lp2 = self.shared.sample_action(
                self.obs, explore_factor=e,
            )
            self.assertTrue(torch.equal(a1, a2), f"action differs e={e}")
            self.assertTrue(torch.equal(lp1, lp2))


class TestStateDependence(unittest.TestCase):
    """Non-zero σ-head weights → σ varies across states, still bounded."""

    def _perturbed(self) -> StateBoundedStdTruncatedNormalPolicy:
        torch.manual_seed(2)
        p = _make_policy()
        d = p.action_dim
        with torch.no_grad():
            p.head.weight[d:].normal_(0.0, 0.5)
        return p

    def test_sigma_varies_across_states(self):
        p = self._perturbed()
        obs = torch.randn(256, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.rand(256, ACTION_DIM) * 2 - 1,
            torch.zeros(256), want_stats=True,
        )
        self.assertGreater(ev.stats["sigma_state_std"], 0.0)

    def test_sigma_bounded_under_state_variation(self):
        p = self._perturbed()
        obs = torch.randn(256, OBS_DIM) * 5  # large obs → large |v|
        for e in (0.0, 1.0, -1.0):
            _, sigma = p.forward(obs, explore_factor=e)
            self.assertTrue(
                (sigma >= SIGMA_MIN - 1e-6).all()
                and (sigma <= SIGMA_MAX + 1e-6).all()
            )

    def test_v_gradient_flows_to_head_and_trunk(self):
        p = self._perturbed()
        obs = torch.randn(16, OBS_DIM, requires_grad=False)
        mean, sigma = p.forward(obs)
        sigma.sum().backward()
        d = p.action_dim
        self.assertIsNotNone(p.head.weight.grad)
        self.assertGreater(
            p.head.weight.grad[d:].abs().sum().item(), 0.0
        )
        self.assertGreater(
            p.trunk[0].weight.grad.abs().sum().item(), 0.0
        )


class TestExploreFactor(unittest.TestCase):
    """Additive v_e = v + αe on per-state v."""

    def setUp(self):
        torch.manual_seed(3)
        self.p = _make_policy()
        d = self.p.action_dim
        with torch.no_grad():
            self.p.head.weight[d:].normal_(0.0, 0.5)
        self.obs = torch.randn(16, OBS_DIM)

    def test_e0_identity(self):
        m0, s0 = self.p.forward(self.obs, explore_factor=0.0)
        ev = self.p.evaluate_actions(
            self.obs, torch.rand(16, ACTION_DIM) * 2 - 1,
            torch.zeros(16),
        )
        # e=0 tensor path must reproduce scalar path exactly
        mt, st = self.p.forward(
            self.obs, explore_factor=torch.zeros(16)
        )
        self.assertTrue(torch.equal(m0, mt))
        self.assertTrue(torch.equal(s0, st))

    def test_sigma_monotone_in_e_per_state(self):
        es = torch.linspace(-1, 1, 41)
        prev = None
        for e in es:
            _, sigma = self.p.forward(
                self.obs, explore_factor=float(e)
            )
            if prev is not None:
                self.assertTrue(
                    (sigma >= prev - 1e-7).all(),
                    f"σ decreased at e={e}",
                )
            prev = sigma

    def test_tensor_e_per_frame(self):
        """(B,) e broadcast: row i must equal scalar-e forward on frame i."""
        ef = torch.linspace(-1, 1, 16)
        m, sigma = self.p.forward(self.obs, explore_factor=ef)
        _, s_lo = self.p.forward(self.obs, explore_factor=-1.0)
        _, s_hi = self.p.forward(self.obs, explore_factor=1.0)
        self.assertTrue((sigma >= s_lo - 1e-7).all())
        self.assertTrue((sigma <= s_hi + 1e-7).all())
        for i in (0, 5, 15):
            mi, si = self.p.forward(
                self.obs, explore_factor=torch.full((16,), float(ef[i]))
            )
            self.assertTrue(torch.equal(sigma[i : i + 1], si[i : i + 1]))

    def test_out_of_range_e_raises(self):
        for e in (1.5, -2.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=str(e)):
                self.p.forward(self.obs, explore_factor=e)
        with self.assertRaises(ValueError):
            self.p.forward(
                self.obs, explore_factor=torch.full((16,), 2.0)
            )

    def test_negative_v_direction(self):
        """Regression guard vs v·3^e: for v<0, e>0 must still raise σ."""
        p = _make_policy()
        d = p.action_dim
        with torch.no_grad():
            p.head.bias[d:].fill_(-2.0)  # v << 0
        obs = torch.randn(4, OBS_DIM)
        _, s_lo = p.forward(obs, explore_factor=-1.0)
        _, s_hi = p.forward(obs, explore_factor=1.0)
        self.assertTrue((s_hi > s_lo).all())


class TestLogProb(unittest.TestCase):
    """Density matches scipy truncnorm with the per-state effective σ."""

    def test_log_prob_matches_scipy(self):
        torch.manual_seed(4)
        p = _make_policy()
        d = p.action_dim
        with torch.no_grad():
            p.head.weight[d:].normal_(0.0, 0.5)
        obs = torch.randn(8, OBS_DIM)
        acts = torch.rand(8, ACTION_DIM) * 1.6 - 0.8
        mean, sigma = p.forward(obs)
        ev = p.evaluate_actions(obs, acts, torch.zeros(8))
        m_np = mean.detach().numpy()
        s_np = sigma.detach().numpy()
        a_np = acts.numpy()
        ref = 0.0
        for i in range(8):
            for j in range(ACTION_DIM):
                tn = sp_stats.truncnorm(
                    (-1.0 - m_np[i, j]) / s_np[i, j],
                    (1.0 - m_np[i, j]) / s_np[i, j],
                    loc=m_np[i, j],
                    scale=s_np[i, j],
                )
                ref += tn.logpdf(a_np[i, j])
        self.assertAlmostEqual(
            float(ev.log_prob.sum()), float(ref), places=4,
        )

    def test_sample_vs_evaluate_log_prob(self):
        torch.manual_seed(5)
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        torch.manual_seed(6)
        acts, lp_sample = p.sample_action(obs)
        ev = p.evaluate_actions(obs, acts, torch.zeros(8))
        self.assertTrue(
            torch.allclose(ev.log_prob, lp_sample, atol=1e-5)
        )


class TestUncertainty(unittest.TestCase):
    """U uses policy σ (e=0); monotone in e via σ for fixed μ."""

    def test_u_ignores_explore_factor(self):
        torch.manual_seed(7)
        p = _make_policy()
        obs = torch.randn(16, OBS_DIM)
        acts = torch.rand(16, ACTION_DIM) * 2 - 1
        u0 = p.evaluate_actions(obs, acts, torch.zeros(16)).uncertainty
        u1 = p.evaluate_actions(
            obs, acts, torch.full((16,), 0.8)
        ).uncertainty
        self.assertTrue(torch.equal(u0, u1))

    def test_u_matches_closed_form(self):
        torch.manual_seed(8)
        p = _make_policy()
        d = p.action_dim
        with torch.no_grad():
            p.head.weight[d:].normal_(0.0, 0.3)
        obs = torch.randn(8, OBS_DIM)
        acts = torch.rand(8, ACTION_DIM) * 2 - 1
        ev = p.evaluate_actions(obs, acts, torch.zeros(8))
        mean, sigma = p.forward(obs)
        # U = σ·√(2π)·Z/2 per dim, mean over dims
        from scipy.stats import norm as _n
        m_np = mean.detach().numpy()
        s_np = sigma.detach().numpy()
        u = (
            s_np * _SQRT_2PI
            * (_n.cdf((1 - m_np) / s_np) - _n.cdf((-1 - m_np) / s_np))
            / 2.0
        ).mean(axis=-1)
        self.assertTrue(
            np.allclose(
                ev.uncertainty.detach().numpy(), u, atol=1e-5,
            )
        )


class TestSampling(unittest.TestCase):
    def test_actions_in_range(self):
        torch.manual_seed(10)
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        a, _ = p.sample_action(obs)
        self.assertTrue((a >= -1.0).all() and (a <= 1.0).all())

    def test_actions_in_range_under_exploration(self):
        torch.manual_seed(11)
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        for e in (-1.0, 1.0):
            a, _ = p.sample_action(obs, explore_factor=e)
            self.assertTrue((a >= -1.0).all() and (a <= 1.0).all())

    def test_reset_replay(self):
        torch.manual_seed(12)
        p = _make_policy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        p.reset(123)
        a1, _ = p.sample(obs, explore_factor=0.4)
        p.reset(123)
        a2, _ = p.sample(obs, explore_factor=0.4)
        self.assertTrue(np.array_equal(a1, a2))

    def test_state_dict_keys(self):
        p = _make_policy()
        keys = set(p.state_dict().keys())
        self.assertEqual(
            keys,
            {
                "trunk.0.weight", "trunk.0.bias",
                "trunk.2.weight", "trunk.2.bias",
                "head.weight", "head.bias",
            },
        )


class TestStats(unittest.TestCase):
    def test_stats_keys(self):
        torch.manual_seed(13)
        p = _make_policy()
        obs = torch.randn(16, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.rand(16, ACTION_DIM) * 2 - 1,
            torch.zeros(16), want_stats=True,
        )
        for k in (
            "uncertainty", "std_mean", "eff_std_mean", "std_min",
            "std_max", "mean_abs",
            "effective_uncertainty", "eff_std_min", "eff_std_max",
            "raw_std_min", "raw_std_max", "std_position_mean",
            "std_lower_saturation_frac", "std_upper_saturation_frac",
            "log_std_sensitivity", "exploration_sensitivity",
            "sigma_state_std",
        ):
            self.assertIn(k, ev.stats, f"missing stat {k}")


class TestExport(unittest.TestCase):
    """Self-contained export: strict load, metadata, parity."""

    def setUp(self):
        torch.manual_seed(14)
        self.p = _make_policy()
        d = self.p.action_dim
        with torch.no_grad():
            self.p.head.weight[d:].normal_(0.0, 0.3)
        self.dir = tempfile.mkdtemp()
        self.p.to_blueprint(self.dir)
        import importlib.util
        s = importlib.util.spec_from_file_location(
            "exported_pol", Path(self.dir) / "policy.py",
        )
        self.mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(self.mod)

    def test_payload_metadata(self):
        payload = torch.load(
            Path(self.dir) / "model.pt", map_location="cpu",
        )
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["policy_class"],
            "StateBoundedStdTruncatedNormalPolicy",
        )
        self.assertEqual(
            payload["distribution_kind"],
            "bounded_std_diagonal_truncated_normal_v1",
        )
        self.assertEqual(payload["std_source"], "state")
        self.assertEqual(
            payload["std_parameterization"], "sigmoid_log_std_v1",
        )
        self.assertEqual(
            payload["exploration_kind"], "raw_std_additive_shift_v1",
        )
        self.assertAlmostEqual(payload["sigma_min"], SIGMA_MIN)
        self.assertAlmostEqual(payload["sigma_max"], SIGMA_MAX)
        self.assertAlmostEqual(payload["init_std"], INIT_STD)
        self.assertAlmostEqual(
            payload["explore_alpha"], 1.19933891045474, places=10,
        )

    def test_export_act_parity(self):
        ep = self.mod.ExportedStateBoundedStdTruncNormPolicy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        a_t, _ = self.p.act(obs)
        a_e, _ = ep.act(obs)
        self.assertTrue(np.array_equal(a_t, a_e))

    def test_export_sample_parity(self):
        ep = self.mod.ExportedStateBoundedStdTruncNormPolicy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for e in (0.0, 0.5, -0.7):
            self.p.reset(21)
            a_t, lp_t = self.p.sample(
                obs, explore_factor=e, want_extra=True,
            )
            ep.reset(21)
            a_e, lp_e = ep.sample(
                obs, explore_factor=e, want_extra=True,
            )
            self.assertTrue(np.array_equal(a_t, a_e), f"e={e}")
            self.assertAlmostEqual(lp_t["log_prob"], lp_e["log_prob"])

    def test_export_rejects_wrong_source(self):
        payload = torch.load(
            Path(self.dir) / "model.pt", map_location="cpu",
        )
        payload["std_source"] = "shared"
        bad = Path(self.dir) / "bad.pt"
        torch.save(payload, bad)
        with self.assertRaises(RuntimeError):
            self.mod.ExportedStateBoundedStdTruncNormPolicy(
                model_path=str(bad)
            )

    def test_export_rejects_missing_config(self):
        payload = torch.load(
            Path(self.dir) / "model.pt", map_location="cpu",
        )
        del payload["sigma_min"]
        bad = Path(self.dir) / "bad.pt"
        torch.save(payload, bad)
        with self.assertRaises(RuntimeError):
            self.mod.ExportedStateBoundedStdTruncNormPolicy(
                model_path=str(bad)
            )

    def test_export_no_repo_imports(self):
        src = Path(self.dir, "policy.py").read_text()
        for forbidden in (
            "from baseline", "import baseline",
            "from envs", "import envs",
        ):
            self.assertNotIn(forbidden, src)

    def test_export_reset_replay(self):
        ep = self.mod.ExportedStateBoundedStdTruncNormPolicy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        ep.reset(99)
        a1, _ = ep.sample(obs, explore_factor=0.3)
        ep.reset(99)
        a2, _ = ep.sample(obs, explore_factor=0.3)
        self.assertTrue(np.array_equal(a1, a2))

    def test_export_works_without_repo(self):
        """Load the export in a subprocess with no repo on sys.path."""
        script = (
            "import sys, importlib.util, numpy as np\n"
            f"spec = importlib.util.spec_from_file_location('p', r'{Path(self.dir) / 'policy.py'}')\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "pol = m.ExportedStateBoundedStdTruncNormPolicy()\n"
            "obs = np.zeros(%d, dtype=np.float32)\n" % OBS_DIM +
            "a, _ = pol.act(obs); assert a.shape == (%d,)\n" % ACTION_DIM +
            "a, _ = pol.sample(obs, explore_factor=0.5)\n"
            "assert a.shape == (%d,) and np.all(np.abs(a) <= 1.0)\n" % ACTION_DIM +
            "print('OK')\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd="/tmp",
            env={"PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr[-2000:])
        self.assertIn("OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
