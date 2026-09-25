"""Tests for StateMixtureBoundedStdTruncatedNormalPolicy (MoG=yes,
state-σ=yes, bounded=yes).

The last cell of the 2×2×2 family: same bounded shared-σ mixture as
``SharedMixtureBoundedStdTruncatedNormalPolicy`` but the σ control v is
a per-state head block (w=0, b=v_init → σ ≡ e⁻¹ at init) instead of a
shared parameter.  The key test is degenerate equivalence — copy the
shared sibling's trunk + logit/mean head rows, pin v head to a constant
and raw_std to the same value → bit-identical behaviour.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.framework.ppo.policies.shared_mixture_bounded_std_truncated_normal_mlp import (
    SharedMixtureBoundedStdTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.state_mixture_bounded_std_truncated_normal_mlp import (
    StateMixtureBoundedStdTruncatedNormalPolicy,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
K = 3
SIGMA_MIN, SIGMA_MAX = 0.05, 2.0
INIT_STD = math.exp(-1.0)
ALPHA = 1.19933891045474


def _make_policy(**kwargs) -> StateMixtureBoundedStdTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_components=K,
        device="cpu",
    )
    defaults.update(kwargs)
    return StateMixtureBoundedStdTruncatedNormalPolicy(**defaults)


class TestInit(unittest.TestCase):
    def test_init_sigma_is_e_minus_1(self):
        """v block w=0, b=v_init → σ ≡ e⁻¹ for every obs."""
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
        """Head carries the v block (K + 2K·D); no raw_std parameter."""
        p = _make_policy()
        self.assertEqual(p.head.out_features, K + 2 * K * ACTION_DIM)
        self.assertFalse(hasattr(p, "raw_std"))
        self.assertAlmostEqual(p.explore_alpha, ALPHA, places=10)
        n = K + K * ACTION_DIM
        self.assertTrue((p.head.weight[n:] == 0).all())
        self.assertTrue(
            torch.allclose(
                p.head.bias[n:], torch.full_like(p.head.bias[n:], p._v_init),
            )
        )


class TestBoundedSigmaMap(unittest.TestCase):
    def test_sigma_stays_in_bounds(self):
        p = _make_policy()
        with torch.no_grad():
            p.head.weight[K + K * ACTION_DIM:].normal_(0, 5.0)
            p.head.bias[K + K * ACTION_DIM:].uniform_(-50.0, 50.0)
        obs = torch.randn(64, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        sigma = p._policy_sigma(raw)
        # exp(ln σ_min) can land 1 ulp below σ_min in float32.
        self.assertTrue((sigma >= SIGMA_MIN - 1e-7).all())
        self.assertTrue((sigma <= SIGMA_MAX).all())

    def test_explore_monotone_per_component(self):
        p = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        s0 = p._explored_sigma(raw, 0.0)
        s_hi = p._explored_sigma(raw, 0.5)
        s_lo = p._explored_sigma(raw, -0.5)
        self.assertTrue((s_hi > s0).all())
        self.assertTrue((s_lo < s0).all())

    def test_ei_out_of_range_raises(self):
        p = _make_policy()
        obs = torch.randn(4, OBS_DIM)
        with self.assertRaises(ValueError):
            p.sample_action(obs, explore_factor=1.5)
        with self.assertRaises(ValueError):
            p.evaluate_actions(
                obs, torch.zeros(4, ACTION_DIM), torch.full((4,), -1.2),
            )


class TestSampling(unittest.TestCase):
    def test_actions_in_range(self):
        p = _make_policy()
        obs = torch.randn(64, OBS_DIM)
        for e in (0.0, 0.5, -1.0):
            a, _ = p.sample_action(obs, explore_factor=e)
            self.assertTrue((a >= -1.0).all() and (a <= 1.0).all())

    def test_sample_vs_evaluate_log_prob(self):
        p = _make_policy()
        with torch.no_grad():
            p.head.weight[K + K * ACTION_DIM:].normal_(0, 0.3)
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
            # state-σ diagnostic
            "sigma_state_std",
        ]:
            self.assertIn(key, ev.stats, f"missing stat {key}")

    def test_sigma_state_std_zero_at_init(self):
        """v head w=0 → σ constant → 0; >0 once the v head varies."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(50, ACTION_DIM), torch.zeros(50),
            want_stats=True,
        )
        self.assertAlmostEqual(ev.stats["sigma_state_std"], 0.0, places=6)
        with torch.no_grad():
            p.head.weight[K + K * ACTION_DIM:].normal_(0, 0.5)
        ev2 = p.evaluate_actions(
            obs, torch.zeros(50, ACTION_DIM), torch.zeros(50),
            want_stats=True,
        )
        self.assertGreater(ev2.stats["sigma_state_std"], 1e-3)


class TestDegenerateEquivalence(unittest.TestCase):
    """state-bounded mixture ≡ shared-bounded mixture at constant v.

    Copy trunk + logit/mean head rows from the shared sibling, pin the
    v head to a constant (w=0, b=v0) and raw_std to the same v0 →
    bit-identical behaviour.
    """

    def _make_equivalent_pair(self, v0: float = 0.1644):
        torch.manual_seed(7)
        shared = SharedMixtureBoundedStdTruncatedNormalPolicy(
            OBS_DIM, ACTION_DIM, HIDDEN_DIM, num_components=K,
        )
        state = _make_policy()
        with torch.no_grad():
            state.trunk[0].weight.copy_(shared.trunk[0].weight)
            state.trunk[0].bias.copy_(shared.trunk[0].bias)
            state.trunk[2].weight.copy_(shared.trunk[2].weight)
            state.trunk[2].bias.copy_(shared.trunk[2].bias)
            # state head's first K+K·D rows = shared head (logits+mean).
            n = K + K * ACTION_DIM
            state.head.weight[:n].copy_(shared.head.weight)
            state.head.bias[:n].copy_(shared.head.bias)
            # Pin σ control: state v head → v0; shared raw_std → v0.
            state.head.weight[n:].zero_()
            state.head.bias[n:].fill_(v0)
            shared.raw_std.fill_(v0)
        return shared, state

    def test_forward_bit_identical(self):
        for v0 in (0.1644, -0.5, 0.8):
            shared, state = self._make_equivalent_pair(v0)
            obs = torch.randn(64, OBS_DIM)
            lp_s, m_s, raw_s = state._forward_raw(obs)
            lp_h, m_h, raw_h = shared._forward_raw(obs)
            torch.testing.assert_close(lp_s, lp_h, rtol=0, atol=0)
            torch.testing.assert_close(m_s, m_h, rtol=0, atol=0)
            torch.testing.assert_close(
                state._policy_sigma(raw_s), shared._policy_sigma(raw_h),
                rtol=0, atol=0,
            )

    def test_act_bit_identical(self):
        shared, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            state.deterministic_action(obs), shared.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_sample_action_bit_identical(self):
        for e in (0.0, 0.5, -0.7):
            shared, state = self._make_equivalent_pair()
            obs = torch.randn(64, OBS_DIM)
            shared.reset(123)
            a_h, lp_h = shared.sample_action(obs, explore_factor=e)
            state.reset(123)
            a_s, lp_s = state.sample_action(obs, explore_factor=e)
            torch.testing.assert_close(a_s, a_h, rtol=0, atol=0)
            torch.testing.assert_close(lp_s, lp_h, rtol=0, atol=0)

    def test_evaluate_actions_bit_identical(self):
        shared, state = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.rand(64, ACTION_DIM) * 2 - 1
        ei = torch.linspace(-1.0, 1.0, 64)
        ev_h = shared.evaluate_actions(obs, actions, ei)
        ev_s = state.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(ev_s.log_prob, ev_h.log_prob, rtol=0, atol=0)
        torch.testing.assert_close(
            ev_s.uncertainty, ev_h.uncertainty, rtol=0, atol=0,
        )


class TestExport(unittest.TestCase):
    """Strict loading, metadata + bounded-config validation."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="smbtn_state_export_test_")
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
            "StateMixtureBoundedStdTruncatedNormalPolicy")
        self.assertEqual(payload["arch"]["num_components"], K)
        self.assertEqual(
            payload["distribution_kind"],
            "bounded_std_mixture_truncated_normal_v1")
        self.assertEqual(payload["std_source"], "state")
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
            "StateMixtureBoundedStdTruncatedNormalPolicy")
        self.assertEqual(
            manifest["exported_class"],
            "ExportedStateMixtureBoundedStdTruncNormPolicy")
        self.assertEqual(manifest["std_source"], "state")
        self.assertEqual(manifest["explore_alpha"], ALPHA)

    def test_rejects_wrong_policy_class(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "SharedMixtureBoundedStdTruncatedNormalPolicy"
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

    def test_export_act_parity(self):
        p = _make_policy()
        with torch.no_grad():
            p.head.weight[K + K * ACTION_DIM:].normal_(0, 0.3)
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
            "ExportedStateMixtureBoundedStdTruncNormPolicy as P; "
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
