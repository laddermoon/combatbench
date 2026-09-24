"""Tests for BoundedStdTruncatedNormalPolicy.

Verifies (per DESIGN_bounded_std_truncated_normal.md §11):
1. Bounded σ: exp(r_min + Δr·sigmoid(v)) stays within (σ_min, σ_max)
2. Init: σ(e=0) = init_std = e⁻¹; derived alpha matches the analytic
   local-response calibration
3. explore_factor: additive v_e = v + αe — σ monotone in e for BOTH
   signs of v (regression guard against the rejected v·3^e mapping),
   e=0 identity, μ invariant, range validated
4. Distribution: log_prob matches scipy truncnorm with effective σ;
   sample/evaluate parity; density integrates to 1
5. Uncertainty U = 1/(2×peak) with policy σ: in [0,1], independent of
   e, monotone in σ for fixed μ
6. Gradients flow to raw_std and net
7. Export: strict load, kind/config validation, no-repo subprocess
   parity, reset(seed) replay
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
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


def _make_policy(**kwargs) -> BoundedStdTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return BoundedStdTruncatedNormalPolicy(**defaults)


class TestConfig(unittest.TestCase):
    """Constructor validation — fail loud on bad bounds/alpha."""

    def test_defaults_resolve(self):
        p = _make_policy()
        self.assertEqual(p.sigma_min, SIGMA_MIN)
        self.assertEqual(p.sigma_max, SIGMA_MAX)
        self.assertAlmostEqual(p.init_std, INIT_STD, places=15)
        # Analytic calibration: alpha = ln3 / (Δr·p0(1−p0))
        r_min, r_max = math.log(SIGMA_MIN), math.log(SIGMA_MAX)
        p0 = (-1.0 - r_min) / (r_max - r_min)
        expected_alpha = math.log(3.0) / ((r_max - r_min) * p0 * (1 - p0))
        self.assertAlmostEqual(p.explore_alpha, expected_alpha, places=12)
        self.assertAlmostEqual(p.explore_alpha, 1.19933891045474, places=10)

    def test_init_std_via_sigmoid_inverse(self):
        p = _make_policy()
        sigma = p.policy_sigma()
        self.assertAlmostEqual(
            float(sigma[0]), INIT_STD, places=5,
            msg=f"σ(init)={float(sigma[0])}, expected {INIT_STD}",
        )
        self.assertAlmostEqual(
            float(p.raw_std[0]), 0.1644220033, places=5,
        )

    def test_explicit_alpha(self):
        p = _make_policy(explore_alpha=2.5)
        self.assertEqual(p.explore_alpha, 2.5)

    def test_rejects_bad_bounds(self):
        for kw in (
            dict(sigma_min=0.0),
            dict(sigma_min=-0.1),
            dict(sigma_min=2.0, sigma_max=1.0),
            dict(init_std=0.01),          # below sigma_min
            dict(init_std=5.0),           # above sigma_max
            dict(sigma_min=float("nan")),
            dict(sigma_max=float("inf")),
        ):
            with self.assertRaises(ValueError, msg=str(kw)):
                _make_policy(**kw)

    def test_rejects_bad_alpha(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"alpha={bad}"):
                _make_policy(explore_alpha=bad)


class TestBoundedSigma(unittest.TestCase):
    """σ always inside (σ_min, σ_max), monotone in v."""

    def test_sigma_bounds_extreme_v(self):
        """σ stays in [σ_min, σ_max] up to float32 rounding.

        At saturated v the sigmoid lands exactly on 0/1, so σ =
        exp(r_min/r_max) — which may round 1 ulp past the bound in
        float32.  The spec verifies the closed interval with a rounding
        tolerance, not a strict open interval.
        """
        p = _make_policy()
        tol = 1e-7
        for v in (-50.0, -10.0, 0.0, 10.0, 50.0):
            p.raw_std.data.fill_(v)
            s = p.policy_sigma()
            self.assertTrue((s >= SIGMA_MIN - tol).all(),
                            f"v={v}: σ={s.min().item()}")
            self.assertTrue((s <= SIGMA_MAX + tol).all(),
                            f"v={v}: σ={s.max().item()}")

    def test_sigma_monotone_in_v(self):
        p = _make_policy()
        vs = torch.linspace(-8, 8, 65)
        sigmas = torch.stack([
            p._bounded_sigma(torch.full((ACTION_DIM,), float(v)))
            for v in vs
        ])[:, 0]
        diffs = sigmas[1:] - sigmas[:-1]
        self.assertTrue((diffs >= 0).all(), "σ must be non-decreasing in v")
        # Strictly increasing away from saturation.
        mid = (vs > -4) & (vs < 4)
        mid_sig = sigmas[:-1][mid[:-1]]
        self.assertTrue((sigmas[1:][mid[:-1]] > mid_sig).all())

    def test_sigma_approaches_bounds(self):
        p = _make_policy()
        lo = p._bounded_sigma(torch.tensor([-30.0]))
        hi = p._bounded_sigma(torch.tensor([30.0]))
        self.assertAlmostEqual(float(lo), SIGMA_MIN, places=4)
        self.assertAlmostEqual(float(hi), SIGMA_MAX, places=3)


class TestExploreFactor(unittest.TestCase):
    """Additive exploration on raw v: v_e = v + α·e."""

    def _eff_sigma(self, p, obs, e):
        _, sigma = p.forward(obs, explore_factor=e)
        return sigma

    def test_e0_is_identity(self):
        p = _make_policy()
        obs = torch.randn(16, OBS_DIM)
        _, s0 = p.forward(obs, explore_factor=0.0)
        pol = p.policy_sigma().expand_as(s0)
        self.assertTrue(torch.equal(s0, pol),
                        "e=0 must exactly reproduce policy σ")

    def test_sigma_monotone_in_e(self):
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        es = np.linspace(-1.0, 1.0, 21)
        sigmas = torch.stack([
            self._eff_sigma(p, obs, float(e)) for e in es
        ])
        diffs = sigmas[1:] - sigmas[:-1]
        self.assertTrue(
            (diffs >= -1e-7).all(),
            f"σ must be non-decreasing in e, min diff={diffs.min()}",
        )

    def test_additive_direction_negative_v(self):
        """Regression guard for the rejected v·3^e mapping.

        With v < 0, e=+1 must INCREASE σ — a multiplicative map would
        push v further negative and decrease σ.
        """
        p = _make_policy()
        p.raw_std.data.fill_(-2.0)          # sigmoid ≈ 0.12, σ near min
        obs = torch.randn(4, OBS_DIM)
        s_neg = self._eff_sigma(p, obs, -1.0)
        s_0 = self._eff_sigma(p, obs, 0.0)
        s_pos = self._eff_sigma(p, obs, 1.0)
        self.assertTrue((s_neg < s_0).all())
        self.assertTrue((s_0 < s_pos).all(),
                        "e=+1 must increase σ even when v < 0")

    def test_e1_is_not_sigma_times_3(self):
        """Additive semantics: σ(e=1) ≠ 3σ — it is bounded."""
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        s_pos = self._eff_sigma(p, obs, 1.0)
        s_0 = self._eff_sigma(p, obs, 0.0)
        self.assertTrue((s_pos < 3.0 * s_0).all())
        self.assertTrue((s_pos <= SIGMA_MAX).all())
        # Documented init values: σ(e=±1) ≈ 0.1315 / 0.9436.
        self.assertAlmostEqual(float(s_pos[0, 0]), 0.9436326, places=4)
        s_neg = self._eff_sigma(p, obs, -1.0)
        self.assertAlmostEqual(float(s_neg[0, 0]), 0.1314986, places=4)

    def test_mean_invariant_under_e(self):
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        m0, _ = p.forward(obs, explore_factor=0.0)
        for e in (-1.0, -0.3, 0.7, 1.0):
            me, _ = p.forward(obs, explore_factor=e)
            self.assertTrue(torch.equal(m0, me),
                            f"μ changed under e={e}")

    def test_per_frame_tensor_e(self):
        p = _make_policy()
        obs = torch.randn(6, OBS_DIM)
        e = torch.linspace(-1, 1, 6)
        _, sigma = p.forward(obs, explore_factor=e)
        self.assertEqual(sigma.shape, (6, ACTION_DIM))
        diffs = sigma[1:] - sigma[:-1]
        self.assertTrue((diffs > 0).all())

    def test_out_of_range_e_raises(self):
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        for bad in (-2.0, 1.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"e={bad}"):
                p.forward(obs, explore_factor=bad)
        with self.assertRaises(ValueError):
            p.forward(obs, explore_factor=torch.tensor([0.0, 1.2, 0.0]))
        with self.assertRaises(ValueError):
            p.forward(obs, explore_factor=torch.tensor([0.0, float("nan")]))


class TestLogProb(unittest.TestCase):
    """log_prob correctness against scipy with effective σ."""

    def test_log_prob_matches_scipy(self):
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        e = torch.rand(50) * 2 - 1
        actions, log_probs = p.sample_action(obs, explore_factor=e)
        mean, sigma = p.forward(obs, explore_factor=e)
        for i in range(5):
            for d in range(ACTION_DIM):
                m = float(mean[i, d])
                s = float(sigma[i, d])
                expected = sp_stats.truncnorm.logpdf(
                    float(actions[i, d]),
                    (-1.0 - m) / s, (1.0 - m) / s,
                    loc=m, scale=s,
                )
                z = (float(actions[i, d]) - m) / s
                Z = sp_stats.norm.cdf((1.0 - m) / s) - sp_stats.norm.cdf(
                    (-1.0 - m) / s
                )
                our = (-0.5 * z * z - math.log(s)
                       - 0.5 * math.log(2 * math.pi) - math.log(Z))
                self.assertAlmostEqual(our, expected, places=4,
                                       msg=f"sample {i} dim {d}")

    def test_log_prob_integrates_to_one(self):
        torch.manual_seed(123)
        p = _make_policy()
        obs = torch.randn(1, OBS_DIM)
        N = 100000
        x = torch.rand(N, ACTION_DIM) * 2.0 - 1.0
        obs_batch = obs.expand(N, -1)
        ev = p.evaluate_actions(obs_batch, x, torch.full((N,), 0.0))
        integral = (2.0 ** ACTION_DIM / N) * torch.exp(ev.log_prob).sum().item()
        self.assertAlmostEqual(integral, 1.0, places=1)

    def test_sample_vs_evaluate_log_prob(self):
        torch.manual_seed(42)
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        e = torch.rand(50) * 2 - 1
        actions, lp_sample = p.sample_action(obs, explore_factor=e)
        ev = p.evaluate_actions(obs, actions, e)
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-4)


class TestUncertainty(unittest.TestCase):
    """U = 1/(2×peak) with policy σ (no explore shift)."""

    def test_u_in_range(self):
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(100, ACTION_DIM), torch.zeros(100),
        )
        self.assertTrue((ev.uncertainty >= 0.0).all())
        self.assertTrue((ev.uncertainty <= 1.0).all())

    def test_u_ignores_explore_factor(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)
        u0 = p.evaluate_actions(obs, actions, torch.zeros(10)).uncertainty
        u1 = p.evaluate_actions(obs, actions, torch.ones(10)).uncertainty
        self.assertLess((u0 - u1).abs().max().item(), 1e-5)

    def test_u_matches_formula(self):
        p = _make_policy()
        obs = torch.zeros(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
        )
        with torch.no_grad():
            mean = torch.tanh(p.net(obs))
        sigma = float(p.policy_sigma()[0])
        u_per_dim = []
        for d in range(ACTION_DIM):
            m = float(mean[0, d])
            Z = sp_stats.norm.cdf((1.0 - m) / sigma) - sp_stats.norm.cdf(
                (-1.0 - m) / sigma
            )
            u_per_dim.append(sigma * _SQRT_2PI * Z / 2.0)
        self.assertAlmostEqual(
            ev.uncertainty[0].item(), sum(u_per_dim) / ACTION_DIM, places=4,
        )

    def test_u_monotone_in_sigma(self):
        """For fixed μ, U increases with σ (through the bounded map)."""
        p = _make_policy()
        obs = torch.zeros(8, OBS_DIM)
        actions = torch.zeros(8, ACTION_DIM)
        us = []
        for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
            p.raw_std.data.fill_(v)
            ev = p.evaluate_actions(obs, actions, torch.zeros(8))
            us.append(ev.uncertainty.mean().item())
        for a, b in zip(us, us[1:]):
            self.assertGreater(b, a, f"U not increasing: {us}")

    def test_u_monotone_in_e(self):
        """Effective U (σ_e) increases with e — via stats diagnostic."""
        p = _make_policy()
        obs = torch.zeros(8, OBS_DIM)
        actions = torch.zeros(8, ACTION_DIM)
        us = []
        for e in (-1.0, -0.5, 0.0, 0.5, 1.0):
            ev = p.evaluate_actions(
                obs, actions, torch.full((8,), e), want_stats=True,
            )
            us.append(ev.stats["effective_uncertainty"])
        for a, b in zip(us, us[1:]):
            self.assertGreater(b, a, f"U_eff not increasing: {us}")


class TestGradients(unittest.TestCase):
    """Gradients flow to raw_std and net."""

    def test_gradient_to_raw_std(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        loss = ev.log_prob.mean() + ev.uncertainty.mean()
        loss.backward()
        self.assertIsNotNone(p.raw_std.grad)
        self.assertFalse(
            torch.allclose(p.raw_std.grad, torch.zeros_like(p.raw_std.grad)),
        )

    def test_uncertainty_gradient_to_raw_std(self):
        """U increases with σ, and σ increases with v → ∂U/∂v > 0."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
        )
        ev.uncertainty.mean().backward()
        self.assertIsNotNone(p.raw_std.grad)
        self.assertTrue((p.raw_std.grad > 0).all())

    def test_gradient_to_net(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        ev.log_prob.mean().backward()
        for param in p.net.parameters():
            self.assertIsNotNone(param.grad)


class TestSampling(unittest.TestCase):
    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(1000, OBS_DIM)
        actions, _ = p.sample_action(obs)
        self.assertEqual(actions.shape, (1000, ACTION_DIM))
        self.assertTrue((actions >= -1.0).all())
        self.assertTrue((actions <= 1.0).all())

    def test_actions_in_range_under_exploration(self):
        p = _make_policy()
        obs = torch.randn(500, OBS_DIM)
        for e in (-1.0, 1.0):
            actions, _ = p.sample_action(obs, explore_factor=e)
            self.assertTrue((actions >= -1.0).all())
            self.assertTrue((actions <= 1.0).all())

    def test_deterministic_action_is_mean(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        det = p.deterministic_action(obs)
        mean, _ = p.forward(obs)
        self.assertTrue(torch.allclose(det, mean, atol=1e-6))

    def test_reset_replay(self):
        """reset(seed) must make repeated samples identical."""
        p = _make_policy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        p.reset(7)
        a1, _ = p.sample(obs, explore_factor=0.5, want_extra=True)
        p.reset(7)
        a2, _ = p.sample(obs, explore_factor=0.5, want_extra=True)
        np.testing.assert_array_equal(a1, a2)

    def test_state_dict_has_no_log_std(self):
        p = _make_policy()
        self.assertIn("raw_std", p.state_dict())
        self.assertNotIn("log_std", p.state_dict())


class TestStats(unittest.TestCase):
    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertIsNotNone(ev.stats)
        for key in (
            "uncertainty", "std_mean", "eff_std_mean", "std_min",
            "std_max", "mean_abs",
            "effective_uncertainty", "eff_std_min", "eff_std_max",
            "raw_std_min", "raw_std_max", "std_position_mean",
            "std_lower_saturation_frac", "std_upper_saturation_frac",
            "log_std_sensitivity", "exploration_sensitivity",
        ):
            self.assertIn(key, ev.stats, msg=key)


# ---------------------------------------------------------------------------
# Export: strict loading, metadata validation, self-contained parity
# ---------------------------------------------------------------------------

class TestExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="bounded_std_export_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = _make_policy()
        bp = p.to_blueprint(dest_path=self._tmp)
        return p, bp

    def _load_payload(self):
        return torch.load(self._model_path, map_location="cpu")

    def test_payload_metadata(self):
        self._make_export()
        payload = self._load_payload()
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["policy_class"], "BoundedStdTruncatedNormalPolicy",
        )
        self.assertEqual(
            payload["distribution_kind"],
            "bounded_std_diagonal_truncated_normal_v1",
        )
        self.assertEqual(payload["std_source"], "shared")
        self.assertEqual(
            payload["std_parameterization"], "sigmoid_log_std_v1",
        )
        self.assertEqual(
            payload["exploration_kind"], "raw_std_additive_shift_v1",
        )
        self.assertEqual(payload["sigma_min"], SIGMA_MIN)
        self.assertEqual(payload["sigma_max"], SIGMA_MAX)
        self.assertAlmostEqual(payload["init_std"], INIT_STD, places=15)
        self.assertAlmostEqual(
            payload["explore_alpha"], 1.19933891045474, places=10,
        )

    def test_export_act_parity(self):
        torch.manual_seed(123)
        p, bp = self._make_export()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]
        loaded = bp.build()
        actual = loaded.act(obs)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_export_sample_parity(self):
        torch.manual_seed(123)
        p, bp = self._make_export()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        p.reset(9)
        expected, extra_p = p.sample(
            obs, explore_factor=0.4, want_extra=True,
        )
        loaded = bp.build()
        loaded.reset(9)
        actual, extra_l = loaded.sample(
            obs, explore_factor=0.4, want_extra=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(
            extra_p["log_prob"], extra_l["log_prob"], places=6,
        )

    def test_export_rejects_missing_keys(self):
        _, bp = self._make_export()
        payload = self._load_payload()
        sd = dict(payload["state_dict"])
        sd.pop("net.0.weight")
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_rejects_wrong_class(self):
        _, bp = self._make_export()
        payload = self._load_payload()
        payload["policy_class"] = "TruncatedNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_rejects_wrong_kind(self):
        _, bp = self._make_export()
        for key in ("distribution_kind", "std_parameterization",
                    "exploration_kind", "std_source"):
            payload = self._load_payload()
            payload[key] = "bogus"
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            # restore for next iteration
            p, bp2 = self._make_export()

    def test_export_rejects_missing_config(self):
        _, bp = self._make_export()
        for key in ("sigma_min", "sigma_max", "explore_alpha"):
            payload = self._load_payload()
            del payload[key]
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            self._make_export()

    def test_export_rejects_invalid_config(self):
        _, bp = self._make_export()
        payload = self._load_payload()
        payload["sigma_min"] = 3.0   # > sigma_max
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()
        payload = self._load_payload()
        payload["sigma_min"] = SIGMA_MIN
        payload["explore_alpha"] = -1.0
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_no_repo_imports(self):
        self._make_export()
        code = (Path(self._tmp) / "policy.py").read_text()
        for forbidden in (
            "from baseline", "import baseline",
            "from envs", "import envs",
        ):
            self.assertNotIn(forbidden, code)

    def test_export_manifest(self):
        self._make_export()
        import json
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(
            manifest["policy_class"], "BoundedStdTruncatedNormalPolicy",
        )
        self.assertEqual(
            manifest["exported_class"], "ExportedBoundedStdTruncNormPolicy",
        )
        self.assertEqual(
            manifest["distribution_kind"],
            "bounded_std_diagonal_truncated_normal_v1",
        )
        self.assertEqual(manifest["sigma_min"], SIGMA_MIN)

    def test_export_works_without_repo(self):
        """Subprocess with no repo on sys.path loads + runs the export."""
        import subprocess
        import sys

        torch.manual_seed(42)
        p, _ = self._make_export()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]

        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import ExportedBoundedStdTruncNormPolicy; "
            "import numpy as np; "
            "p = ExportedBoundedStdTruncNormPolicy(); "
            "obs = np.array({obs!r}, dtype=np.float32); "
            "a, _ = p.act(obs); "
            "print(repr(a.tolist()))"
        ).format(tmp=self._tmp, obs=obs.tolist())
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        import ast
        actual = np.array(ast.literal_eval(result.stdout.strip()),
                         dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
