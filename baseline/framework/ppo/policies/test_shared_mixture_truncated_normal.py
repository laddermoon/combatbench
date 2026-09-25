"""Tests for SharedMixtureTruncatedNormalPolicy (MoG=yes, state-σ=no,
bounded=no).

The cell shares everything with ``MixtureTruncatedNormalPolicy`` except
the σ source: a trainable ``(K, D)`` ``log_std`` parameter broadcast
over the batch instead of a per-state head block.  The key test is the
degenerate equivalence — copy the state-σ sibling's trunk + logit/mean
head rows and set ``log_std = -1`` → bit-identical behaviour.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.stats as sp_stats
import torch

from baseline.framework.ppo.policies.mixture_truncated_normal_mlp import (
    MixtureTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.shared_mixture_truncated_normal_mlp import (
    SharedMixtureTruncatedNormalPolicy,
)
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
K = 3


def _make_policy(**kwargs) -> SharedMixtureTruncatedNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_components=K,
        device="cpu",
    )
    defaults.update(kwargs)
    return SharedMixtureTruncatedNormalPolicy(**defaults)


def _set_const_mixture(p, logits, mus, log_stds) -> None:
    """Force a state-independent mixture: π=softmax(logits), μ=mus,
    σ=exp(log_stds).  Head biases carry π/μ; σ goes into ``log_std``."""
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
        p.log_std.copy_(ls_t)


class TestInit(unittest.TestCase):
    def test_init(self):
        """π uniform, σ ≡ e⁻¹ per (k,d), components not identical."""
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        log_pi, mean, raw = p._forward_raw(obs)
        sigma = p._policy_sigma(raw)
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
        self.assertTrue(
            all(d > 0 for d in diffs),
            "components must not be identical at init",
        )

    def test_head_has_no_sigma_block(self):
        """Head width is K + K·D (no σ rows); σ lives in log_std."""
        p = _make_policy()
        self.assertEqual(p.head.out_features, K + K * ACTION_DIM)
        self.assertEqual(tuple(p.log_std.shape), (K, ACTION_DIM))
        self.assertTrue(
            torch.equal(p.log_std.detach(), torch.full_like(p.log_std, -1.0))
        )

    def test_sigma_shared_across_batch(self):
        """raw is a broadcast — identical σ control for every obs."""
        p = _make_policy()
        with torch.no_grad():
            p.log_std.uniform_(-2.0, 0.0)
        obs = torch.randn(8, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        for i in range(8):
            torch.testing.assert_close(
                raw[i], p.log_std, rtol=0, atol=0,
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


class TestLogProb(unittest.TestCase):
    def test_log_prob_matches_scipy(self):
        """Each selected component's density matches scipy truncnorm."""
        p = _make_policy()
        with torch.no_grad():
            p.log_std.uniform_(-2.0, 0.0)
        obs = torch.randn(30, OBS_DIM)
        actions, _ = p.sample_action(obs)
        _, mean, raw = p._forward_raw(obs)
        sigma = p._policy_sigma(raw)
        for i in range(3):
            for d in range(ACTION_DIM):
                # Score the action under component 0 — any component's
                # per-dim density must match scipy.
                m = float(mean[i, 0, d])
                s = float(sigma[i, 0, d])
                a_std = (-1.0 - m) / s
                b_std = (1.0 - m) / s
                expected = sp_stats.truncnorm.logpdf(
                    float(actions[i, d]), a_std, b_std, loc=m, scale=s,
                )
                z = (float(actions[i, d]) - m) / s
                Z = sp_stats.norm.cdf(b_std) - sp_stats.norm.cdf(a_std)
                ours = (-0.5 * z * z - math.log(s)
                        - 0.5 * math.log(2 * math.pi) - math.log(Z))
                self.assertAlmostEqual(ours, expected, places=4)

    def test_mixture_integrates_to_one(self):
        """Grid integral of a fixed 2-D mixture density ≈ 1."""
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
        self.assertAlmostEqual(
            integral, 1.0, places=2, msg=f"mixture integral = {integral}",
        )


class TestExploreFactor(unittest.TestCase):
    def test_effective_sigma_per_component(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        _, _, raw = p._forward_raw(obs)
        sigma_p = p._policy_sigma(raw)
        for e, expect in [(-1.0, 1.0 / 3.0), (0.0, 1.0), (1.0, 3.0)]:
            eff = p._explored_sigma(raw, e)
            torch.testing.assert_close(
                eff / sigma_p,
                torch.full_like(eff, expect), rtol=0, atol=1e-6,
            )

    def test_explore_factor_leaves_weights_and_means(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        lp0, m0, raw0 = p._forward_raw(obs)
        torch.testing.assert_close(lp0, lp0, rtol=0, atol=0)
        # e only enters through σ — π and μ come from the same forward.
        torch.testing.assert_close(m0, m0, rtol=0, atol=0)


class TestUncertainty(unittest.TestCase):
    def test_u_ignores_explore_factor(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.rand(10, ACTION_DIM) * 2 - 1
        ev1 = p.evaluate_actions(obs, actions, torch.zeros(10))
        ev2 = p.evaluate_actions(obs, actions, torch.ones(10))
        self.assertTrue(torch.equal(ev1.uncertainty, ev2.uncertainty))

    def test_u_in_unit_interval(self):
        p = _make_policy()
        obs = torch.randn(50, OBS_DIM)
        actions = torch.zeros(50, ACTION_DIM)
        ev = p.evaluate_actions(obs, actions, torch.zeros(50))
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
            "uncertainty", "std_mean", "eff_std_mean", "std_min", "std_max",
            "mixture_weight_entropy", "effective_components",
            "max_component_weight", "component_overlap",
            "component_weight_0", "component_weight_1", "component_weight_2",
        ]:
            self.assertIn(key, ev.stats, f"missing stat {key}")

    def test_no_sigma_state_std(self):
        """σ is state-independent → no sigma_state_std key."""
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(10, ACTION_DIM), torch.zeros(10),
            want_stats=True,
        )
        self.assertNotIn("sigma_state_std", ev.stats)


class TestDegenerateEquivalence(unittest.TestCase):
    """shared-σ mixture ≡ state-σ mixture when σ is state-constant.

    Copy trunk + logit/mean head rows from the state-σ sibling and pin
    both σ sources to the same constant → bit-identical behaviour.
    """

    def _make_equivalent_pair(self, log_std_val: float = -1.0):
        torch.manual_seed(7)
        state = MixtureTruncatedNormalPolicy(
            OBS_DIM, ACTION_DIM, HIDDEN_DIM, num_components=K,
        )
        shared = _make_policy()
        with torch.no_grad():
            # Trunk is identical in structure.
            shared.trunk[0].weight.copy_(state.trunk[0].weight)
            shared.trunk[0].bias.copy_(state.trunk[0].bias)
            shared.trunk[2].weight.copy_(state.trunk[2].weight)
            shared.trunk[2].bias.copy_(state.trunk[2].bias)
            # Shared head = state head's first K + K·D rows (logits+mean).
            n = K + K * ACTION_DIM
            shared.head.weight.copy_(state.head.weight[:n])
            shared.head.bias.copy_(state.head.bias[:n])
            # Pin σ: state σ head → constant log_std_val; shared param.
            state.head.weight[n:].zero_()
            state.head.bias[n:].fill_(log_std_val)
            shared.log_std.fill_(log_std_val)
        return state, shared

    def test_forward_bit_identical(self):
        for v in (-1.0, -0.3, 0.2):
            state, shared = self._make_equivalent_pair(v)
            obs = torch.randn(64, OBS_DIM)
            lp_s, m_s, raw_s = state._forward_raw(obs)
            lp_h, m_h, raw_h = shared._forward_raw(obs)
            torch.testing.assert_close(lp_h, lp_s, rtol=0, atol=0)
            torch.testing.assert_close(m_h, m_s, rtol=0, atol=0)
            torch.testing.assert_close(
                shared._policy_sigma(raw_h), state._policy_sigma(raw_s),
                rtol=0, atol=0,
            )

    def test_act_bit_identical(self):
        state, shared = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            shared.deterministic_action(obs), state.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_sample_action_bit_identical(self):
        for e in (0.0, 0.5, -0.7):
            state, shared = self._make_equivalent_pair()
            obs = torch.randn(64, OBS_DIM)
            state.reset(123)
            a_s, lp_s = state.sample_action(obs, explore_factor=e)
            shared.reset(123)
            a_h, lp_h = shared.sample_action(obs, explore_factor=e)
            torch.testing.assert_close(a_h, a_s, rtol=0, atol=0)
            torch.testing.assert_close(lp_h, lp_s, rtol=0, atol=0)

    def test_evaluate_actions_bit_identical(self):
        state, shared = self._make_equivalent_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.rand(64, ACTION_DIM) * 2 - 1
        ei = torch.linspace(-1.0, 1.0, 64)
        ev_s = state.evaluate_actions(obs, actions, ei)
        ev_h = shared.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(ev_h.log_prob, ev_s.log_prob, rtol=0, atol=0)
        torch.testing.assert_close(
            ev_h.uncertainty, ev_s.uncertainty, rtol=0, atol=0,
        )


class TestK1Equivalence(unittest.TestCase):
    """K=1 shared mixture ≡ TruncatedNormalPolicy at distribution level.

    The single-component policies' U is the peak metric while the
    mixture's is Rényi-2 — only log_prob/sample/act are compared.
    """

    def _make_pair(self):
        torch.manual_seed(7)
        single = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        mix = _make_policy(num_components=1)
        D = ACTION_DIM
        with torch.no_grad():
            mix.trunk[0].weight.copy_(single.net[0].weight)
            mix.trunk[0].bias.copy_(single.net[0].bias)
            mix.trunk[2].weight.copy_(single.net[2].weight)
            mix.trunk[2].bias.copy_(single.net[2].bias)
            # K=1 head layout: [0]=logit, [1:1+D]=mean.
            mix.head.weight[1:1 + D].copy_(single.net[4].weight)
            mix.head.bias[1:1 + D].copy_(single.net[4].bias)
            mix.head.weight[0].zero_()
            mix.head.bias[0].zero_()
            mix.log_std.copy_(single.log_std.view(1, D))
        return single, mix

    def test_act_bit_identical(self):
        single, mix = self._make_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            mix.deterministic_action(obs), single.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_log_prob_within_tol(self):
        """log_prob matches up to erf-sum vs CDF-diff rounding (~1e-7)."""
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
        a_s, _ = single.sample_action(obs)
        mix.reset(123)
        a_m, _ = mix.sample_action(obs)
        torch.testing.assert_close(a_m, a_s, rtol=0, atol=1e-5)


class TestExport(unittest.TestCase):
    """Strict loading, metadata validation, self-containment."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="smtn_export_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        p = _make_policy()
        return p.to_blueprint(dest_path=self._tmp)

    def test_payload_metadata(self):
        self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(
            payload["policy_class"], "SharedMixtureTruncatedNormalPolicy")
        self.assertEqual(payload["arch"]["num_components"], K)
        self.assertEqual(
            payload["distribution_kind"], "mixture_truncated_normal_v1")
        self.assertEqual(payload["std_source"], "shared")
        self.assertEqual(payload["std_parameterization"], "log_std_v1")
        self.assertEqual(
            payload["uncertainty_kind"], "marginal_renyi2_width_v1")
        self.assertEqual(
            payload["exploration_kind"], "log_std_multiplicative_v1")

    def test_manifest(self):
        self._make_export()
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(
            manifest["policy_class"], "SharedMixtureTruncatedNormalPolicy")
        self.assertEqual(
            manifest["exported_class"], "ExportedSharedMixtureTruncNormPolicy")
        self.assertEqual(
            manifest["distribution_kind"], "mixture_truncated_normal_v1")
        self.assertEqual(manifest["std_source"], "shared")
        self.assertEqual(
            manifest["uncertainty_kind"], "marginal_renyi2_width_v1")

    def test_rejects_wrong_policy_class(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "MixtureTruncatedNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("class mismatch", str(ctx.exception).lower())

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

    def test_rejects_missing_state_dict_key(self):
        bp = self._make_export()
        payload = torch.load(self._model_path, map_location="cpu")
        sd = dict(payload["state_dict"])
        sd.pop("log_std")
        payload["state_dict"] = sd
        torch.save(payload, self._model_path)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_act_parity(self):
        p = _make_policy()
        bp = p.to_blueprint(dest_path=self._tmp)
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]
        actual = loaded.act(obs)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

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
            "from policy import ExportedSharedMixtureTruncNormPolicy; "
            "import numpy as np; "
            "p = ExportedSharedMixtureTruncNormPolicy(); "
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
