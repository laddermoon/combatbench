"""Tests for SharedMixtureBoundedStdTruncatedNormalPolicy (MoG=yes,
state-σ=no, bounded=yes).

Same mixture as ``SharedMixtureTruncatedNormalPolicy`` with the σ axis
swapped to the bounded sigmoid map: shared (K,D) raw control ``v``,
σ(v) = exp(r_min + Δr·sigmoid(v)) ∈ (σ_min, σ_max), and explore shifts
v additively (σ_eff = σ(v + αe)) instead of multiplicatively.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.framework.ppo.policies.bounded_std_truncated_normal_mlp import (
    BoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.shared_mixture_bounded_std_truncated_normal_mlp import (
    SharedMixtureBoundedStdTruncatedNormalPolicy,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
K = 3
SIGMA_MIN, SIGMA_MAX = 0.05, 2.0
INIT_STD = math.exp(-1.0)
ALPHA = 1.19933891045474


def _make_policy(**kwargs) -> SharedMixtureBoundedStdTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_components=K,
        device="cpu",
    )
    defaults.update(kwargs)
    return SharedMixtureBoundedStdTruncatedNormalPolicy(**defaults)


class TestInit(unittest.TestCase):
    def test_init_sigma_is_e_minus_1(self):
        """v_init inverts the map → σ(v_init) == e⁻¹ everywhere."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        sigma = p._policy_sigma(raw)
        self.assertTrue(torch.allclose(
            sigma, torch.full_like(sigma, INIT_STD), atol=1e-6,
        ))

    def test_init_pi_uniform_components_distinct(self):
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        log_pi, mean, _ = p._forward_raw(obs)
        self.assertTrue(torch.allclose(
            log_pi.exp(), torch.full_like(log_pi, 1.0 / K), atol=1e-6,
        ))
        diffs = [
            float((mean[:, i] - mean[:, j]).abs().max())
            for i in range(K) for j in range(i + 1, K)
        ]
        self.assertTrue(all(d > 0 for d in diffs))

    def test_layout(self):
        """Head has no σ block; raw_std is a (K,D) parameter at v_init."""
        p = _make_policy()
        self.assertEqual(p.head.out_features, K + K * ACTION_DIM)
        self.assertEqual(tuple(p.raw_std.shape), (K, ACTION_DIM))
        self.assertFalse(hasattr(p, "log_std"))
        self.assertAlmostEqual(p.explore_alpha, ALPHA, places=10)

    def test_rejects_bad_bounds(self):
        for kw in (dict(sigma_min=0.5, sigma_max=0.4),
                   dict(sigma_min=-0.1),
                   dict(init_std=5.0)):
            with self.assertRaises(ValueError, msg=str(kw)):
                _make_policy(**kw)

    def test_rejects_bad_alpha(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _make_policy(explore_alpha=bad)


class TestBoundedSigmaMap(unittest.TestCase):
    def test_sigma_stays_in_bounds(self):
        """σ ∈ [σ_min, σ_max] — bounds are closed: float32 sigmoid
        saturates to exactly 0/1 at |v| ~ 50."""
        p = _make_policy()
        with torch.no_grad():
            p.raw_std.uniform_(-50.0, 50.0)
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        sigma = p._policy_sigma(raw)
        # exp(ln σ_min) can land 1 ulp below σ_min in float32.
        self.assertTrue((sigma >= SIGMA_MIN - 1e-7).all())
        self.assertTrue((sigma <= SIGMA_MAX).all())
        # Moderate v stays strictly inside.
        with torch.no_grad():
            p.raw_std.uniform_(-3.0, 3.0)
        _, _, raw2 = p._forward_raw(obs)
        sigma2 = p._policy_sigma(raw2)
        self.assertTrue((sigma2 > SIGMA_MIN).all())
        self.assertTrue((sigma2 < SIGMA_MAX).all())

    def test_explore_monotone_per_component(self):
        """e>0 widens every component's σ; e<0 shrinks (elementwise)."""
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        s0 = p._explored_sigma(raw, 0.0)
        s_hi = p._explored_sigma(raw, 0.5)
        s_lo = p._explored_sigma(raw, -0.5)
        self.assertTrue((s_hi > s0).all())
        self.assertTrue((s_lo < s0).all())

    def test_e0_is_identity(self):
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        torch.testing.assert_close(
            p._explored_sigma(raw, 0.0), p._policy_sigma(raw),
            rtol=0, atol=0,
        )

    def test_batched_explore_factor(self):
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        ei = torch.linspace(-1.0, 1.0, 8)
        eff = p._explored_sigma(raw, ei)
        # Row i uses v + α·ei[i] — matches scalar call per row.
        for i in (0, 4, 7):
            eff_i = p._explored_sigma(raw[i:i + 1], float(ei[i]))
            torch.testing.assert_close(eff[i:i + 1], eff_i, rtol=0, atol=0)

    def test_ei_out_of_range_raises(self):
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        for bad in (1.5, -2.0, torch.full((4,), 1.2), float("nan")):
            with self.assertRaises(ValueError, msg=str(bad)):
                p.sample_action(obs, explore_factor=bad)
        with self.assertRaises(ValueError):
            p.evaluate_actions(
                obs, torch.zeros(4, ACTION_DIM), torch.full((4,), 1.2),
            )

    def test_alpha_calibration_local_response(self):
        """d(log σ)/de at init ≈ ln 3 — matches the multiplicative slope."""
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        eps = 1e-3
        s_hi = p._explored_sigma(raw, eps).log()
        s_lo = p._explored_sigma(raw, -eps).log()
        slope = ((s_hi - s_lo) / (2 * eps)).mean().item()
        self.assertAlmostEqual(slope, math.log(3.0), places=3)


class TestSampling(unittest.TestCase):
    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        for e in (0.0, 0.5, -1.0):
            a, _ = p.sample_action(obs, explore_factor=e)
            self.assertTrue((a >= -1.0).all() and (a <= 1.0).all())

    def test_sample_vs_evaluate_log_prob(self):
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions, lp_sample = p.sample_action(obs)
        ev = p.evaluate_actions(obs, actions, torch.zeros(50))
        self.assertLess(
            (lp_sample - ev.log_prob).abs().max().item(), 1e-4,
        )

    def test_reset_replay(self):
        p = _make_policy()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        p.reset(123)
        a1, _ = p.sample(obs, explore_factor=0.4)
        p.reset(123)
        a2, _ = p.sample(obs, explore_factor=0.4)
        np.testing.assert_array_equal(a1, a2)


class TestUncertainty(unittest.TestCase):
    def test_u_ignores_explore_factor(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.rand(10, ACTION_DIM) * 2 - 1
        ev1 = p.evaluate_actions(obs, actions, torch.zeros(10))
        ev2 = p.evaluate_actions(obs, actions, 0.5 * torch.ones(10))
        self.assertTrue(torch.equal(ev1.uncertainty, ev2.uncertainty))

    def test_u_in_unit_interval(self):
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(50, ACTION_DIM), torch.zeros(50),
        )
        self.assertTrue(((ev.uncertainty > 0) & (ev.uncertainty <= 1)).all())


class TestStats(unittest.TestCase):
    def test_stats_keys(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        for key in [
            # mixture diagnostics
            "uncertainty", "std_mean", "eff_std_mean", "std_min", "std_max",
            "mixture_weight_entropy", "effective_components",
            "max_component_weight", "component_overlap",
            "component_weight_0", "component_weight_1", "component_weight_2",
            # bounded-σ diagnostics
            "effective_uncertainty", "eff_std_min", "eff_std_max",
            "raw_std_min", "raw_std_max", "std_position_mean",
            "std_lower_saturation_frac", "std_upper_saturation_frac",
            "log_std_sensitivity", "exploration_sensitivity",
        ]:
            self.assertIn(key, ev.stats, f"missing stat {key}")

    def test_no_sigma_state_std(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertNotIn("sigma_state_std", ev.stats)

    def test_saturation_counters(self):
        """Pushing v to extremes moves the saturation fractions."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        with torch.no_grad():
            p.raw_std.fill_(50.0)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertGreater(ev.stats["std_upper_saturation_frac"], 0.99)
        self.assertAlmostEqual(ev.stats["std_max"], SIGMA_MAX, places=5)


class TestK1Equivalence(unittest.TestCase):
    """K=1 bounded shared mixture ≡ BoundedStdTruncatedNormalPolicy at
    distribution level (U metric differs by design — not compared)."""

    def _make_pair(self):
        torch.manual_seed(7)
        single = BoundedStdTruncatedNormalPolicy(
            OBS_DIM, ACTION_DIM, HIDDEN_DIM,
        )
        mix = _make_policy(num_components=1)
        D = ACTION_DIM
        with torch.no_grad():
            mix.trunk[0].weight.copy_(single.net[0].weight)
            mix.trunk[0].bias.copy_(single.net[0].bias)
            mix.trunk[2].weight.copy_(single.net[2].weight)
            mix.trunk[2].bias.copy_(single.net[2].bias)
            mix.head.weight[1:1 + D].copy_(single.net[4].weight)
            mix.head.bias[1:1 + D].copy_(single.net[4].bias)
            mix.head.weight[0].zero_()
            mix.head.bias[0].zero_()
            mix.raw_std.copy_(single.raw_std.view(1, D))
        return single, mix

    def test_act_bit_identical(self):
        single, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            mix.deterministic_action(obs), single.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_log_prob_within_tol(self):
        """Same σ map → density equal up to erf/CDF icdf rounding."""
        single, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.rand(64, ACTION_DIM) * 2 - 1
        ei = torch.linspace(-1.0, 1.0, 64)
        ev_s = single.evaluate_actions(obs, actions, ei)
        ev_m = mix.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(
            ev_m.log_prob, ev_s.log_prob, rtol=0, atol=1e-5,
        )

    def test_sample_close(self):
        single, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        single.reset(123)
        a_s, _ = single.sample_action(obs, explore_factor=0.3)
        mix.reset(123)
        a_m, _ = mix.sample_action(obs, explore_factor=0.3)
        torch.testing.assert_close(a_m, a_s, rtol=0, atol=1e-5)


class TestExport(unittest.TestCase):
    """Strict loading, metadata + bounded-config validation."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="smbtn_export_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = _make_policy()
        return p.to_blueprint(dest_path=self._tmp)

    def test_payload_metadata(self):
        self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["policy_class"],
            "SharedMixtureBoundedStdTruncatedNormalPolicy")
        self.assertEqual(payload["arch"]["num_components"], K)
        self.assertEqual(
            payload["distribution_kind"],
            "bounded_std_mixture_truncated_normal_v1")
        self.assertEqual(payload["std_source"], "shared")
        self.assertEqual(
            payload["std_parameterization"], "sigmoid_log_std_v1")
        self.assertEqual(
            payload["uncertainty_kind"], "marginal_renyi2_width_v1")
        self.assertEqual(
            payload["exploration_kind"], "raw_std_additive_shift_v1")
        self.assertEqual(payload["sigma_min"], SIGMA_MIN)
        self.assertEqual(payload["sigma_max"], SIGMA_MAX)
        self.assertAlmostEqual(payload["init_std"], INIT_STD, places=15)
        self.assertAlmostEqual(payload["explore_alpha"], ALPHA, places=10)

    def test_manifest(self):
        self._make_export()
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(
            manifest["policy_class"],
            "SharedMixtureBoundedStdTruncatedNormalPolicy")
        self.assertEqual(
            manifest["exported_class"],
            "ExportedSharedMixtureBoundedStdTruncNormPolicy")
        self.assertEqual(manifest["std_source"], "shared")
        self.assertEqual(manifest["explore_alpha"], ALPHA)

    def test_rejects_wrong_policy_class(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "MixtureTruncatedNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_rejects_wrong_metadata(self):
        bp = self._make_export()
        for key in ("distribution_kind", "std_parameterization",
                    "exploration_kind", "std_source"):
            payload = torch.load(self._model_path, map_location="cpu")
            payload[key] = "bogus"
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            self._make_export()

    def test_rejects_missing_metadata(self):
        bp = self._make_export()
        for key in ("distribution_kind", "std_parameterization",
                    "exploration_kind", "std_source"):
            payload = torch.load(self._model_path, map_location="cpu")
            del payload[key]
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            self._make_export()

    def test_rejects_missing_config(self):
        bp = self._make_export()
        for key in ("sigma_min", "sigma_max", "explore_alpha"):
            payload = torch.load(self._model_path, map_location="cpu")
            del payload[key]
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            self._make_export()

    def test_rejects_invalid_config(self):
        bp = self._make_export()
        for key, val in (("sigma_min", 5.0), ("sigma_min", -0.5),
                         ("explore_alpha", -1.0)):
            payload = torch.load(self._model_path, map_location="cpu")
            payload[key] = val
            torch.save(payload, self._model_path)
            with self.assertRaises(RuntimeError, msg=key):
                bp.build()
            self._make_export()

    def test_rejects_missing_state_dict_key(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        sd = dict(payload["state_dict"])
        sd.pop("raw_std")
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_act_parity(self):
        p = _make_policy()
        bp = p.to_blueprint(dest_path=self._tmp)
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        np.testing.assert_allclose(
            loaded.act(obs)[0], p.act(obs)[0], rtol=0, atol=0)

    def test_export_sample_parity(self):
        p = _make_policy()
        bp = p.to_blueprint(dest_path=self._tmp)
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for e in (0.0, 0.5, -0.7):
            p.reset(9)
            a_t, lp_t = p.sample(obs, explore_factor=e, want_extra=True)
            loaded.reset(9)
            a_e, lp_e = loaded.sample(obs, explore_factor=e, want_extra=True)
            self.assertTrue(np.array_equal(a_t, a_e), f"e={e}")
            self.assertAlmostEqual(lp_t["log_prob"], lp_e["log_prob"])

    def test_export_works_without_repo(self):
        """Subprocess with no repo on sys.path loads + runs the export."""
        import subprocess
        import sys

        p = _make_policy()
        p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]

        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import "
            "ExportedSharedMixtureBoundedStdTruncNormPolicy as P; "
            "import numpy as np; "
            "p = P(); "
            "obs = np.array({obs!r}, dtype=np.float32); "
            "a, _ = p.act(obs); "
            "print(repr(a.tolist()))"
        ).format(tmp=self._tmp, obs=obs.tolist())
        result = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=30,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                 "HOME": "/root",
                 "LD_LIBRARY_PATH": "/usr/local/lib",
                 "PYTHONPATH": self._tmp},
        )
        if result.returncode != 0:
            self.fail(
                f"Subprocess failed (no repo on path):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        actual = np.array(eval(result.stdout.strip()), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
