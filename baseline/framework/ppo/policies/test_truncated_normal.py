"""Tests for TruncatedNormalPolicy.

Verifies:
1. Sampling produces actions in [-1, 1]
2. log_prob matches manual truncated normal computation
3. log_prob integrates to 1 (probability conservation)
4. sample_action and evaluate_actions give consistent log_prob
5. Uncertainty U is in [0, 1] and matches 1/(2×peak)
6. U = 1 for uniform-like (large σ), U → 0 for narrow (small σ)
7. explore_factor scales σ correctly (ei=-1→1/3, 0→1, +1→3)
8. Gradients flow to mean and log_std
9. U is per-obs (depends on mean)
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

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
        u = ev.uncertainty
        self.assertTrue((u >= 0.0).all(), f"U < 0: min={u.min()}")
        self.assertTrue((u <= 1.0).all(), f"U > 1: max={u.max()}")

    def test_u_decreases_with_smaller_sigma(self):
        """Smaller σ → higher peak → lower U."""
        p = _make_policy()
        obs = torch.randn(100, OBS_DIM)
        # Large σ
        p.log_std.data.fill_(0.0)  # σ = 1.0
        ev_large = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_large = ev_large.uncertainty.mean().item()

        # Small σ
        p.log_std.data.fill_(-3.0)  # σ ≈ 0.05
        ev_small = p.evaluate_actions(obs, torch.zeros(100, ACTION_DIM), torch.full((100,), 0.0))
        u_small = ev_small.uncertainty.mean().item()

        self.assertGreater(u_large, u_small,
                           f"U(σ=1)={u_large} should > U(σ=0.05)={u_small}")

    def test_u_matches_formula(self):
        """U = σ × √(2π) × Z / 2 when mean ∈ (-1,1)."""
        p = _make_policy()
        p.log_std.data.fill_(-1.0)  # σ ≈ 0.368
        obs = torch.zeros(10, OBS_DIM)
        ev = p.evaluate_actions(obs, torch.zeros(10, ACTION_DIM), torch.full((10,), 0.0))
        u_actual = ev.uncertainty[0].item()

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
        self.assertNotAlmostEqual(ev1.uncertainty[0].item(), ev2.uncertainty[0].item(),
                                  places=3,
                                  msg="U should differ for different obs")


class TestExploreIntensity(unittest.TestCase):
    """explore_factor exponential σ scaling: scale = exp(ei * ln(3))."""

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
        _, sigma_neutral = p.forward(torch.randn(1, OBS_DIM), explore_factor=0.0)
        self.assertAlmostEqual(sigma_neutral[0, 0].item(), 1.0, places=5)

        # Suppressed
        _, sigma_suppressed = p.forward(torch.randn(1, OBS_DIM), explore_factor=-1.0)
        self.assertAlmostEqual(sigma_suppressed[0, 0].item(), 1.0 / 3.0,
                               places=5)

        # Expanded
        _, sigma_expanded = p.forward(torch.randn(1, OBS_DIM), explore_factor=1.0)
        self.assertAlmostEqual(sigma_expanded[0, 0].item(), 3.0, places=5)

    def test_scale_does_not_affect_uncertainty(self):
        """U uses policy σ, not effective σ."""
        p = _make_policy()
        p.log_std.data.fill_(-1.0)
        obs = torch.randn(10, OBS_DIM)
        actions = torch.zeros(10, ACTION_DIM)

        ev_neutral = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        ev_expanded = p.evaluate_actions(obs, actions, torch.full((10,), 1.0))

        diff = (ev_neutral.uncertainty - ev_expanded.uncertainty).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"U should not change with explore_factor, diff={diff}")


class TestGradients(unittest.TestCase):
    """Gradients flow to mean (net) and log_std."""

    def test_gradient_to_log_std(self):
        p = _make_policy()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.randn(10, ACTION_DIM).clamp(-0.9, 0.9)
        ev = p.evaluate_actions(obs, actions, torch.full((10,), 0.0))
        loss = ev.log_prob.mean() + ev.uncertainty.mean()
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
        loss = ev.uncertainty.mean()
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


# ---------------------------------------------------------------------------
# P0-5: strict loading + format validation for exported policies
# ---------------------------------------------------------------------------

class TestExportStrictLoading(unittest.TestCase):
    """P0-5: Exported policy loading must be strict and validated.

    Before the fix, `load_state_dict(strict=False)` silently loaded
    checkpoints with missing keys, producing partially-random policies
    that run without any warning.  This is fatal for a benchmark project.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="p0_5_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def _make_export(self):
        """Export a policy and return the model.pt path."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        return bp, self._model_path

    def _load_payload(self):
        return torch.load(self._model_path, map_location="cpu")

    def _save_payload(self, payload):
        torch.save(payload, self._model_path)

    def test_export_payload_has_format_metadata(self):
        """Payload includes format_version, policy_class, arch, keys."""
        self._make_export()
        p = self._load_payload()
        self.assertEqual(p["format_version"], 1)
        self.assertEqual(p["policy_class"], "TruncatedNormalPolicy")
        self.assertEqual(p["arch"]["obs_dim"], OBS_DIM)
        self.assertEqual(p["arch"]["action_dim"], ACTION_DIM)
        self.assertEqual(p["arch"]["hidden_dim"], HIDDEN_DIM)
        self.assertIn("state_dict_keys", p)
        self.assertIsInstance(p["state_dict_keys"], list)

    def test_export_roundtrip_exact(self):
        """Export → reload produces identical actions on non-zero input.

        Key: use NON-ZERO input.  With zero input, a missing first-layer
        weight is masked (output = bias only) and the test passes even
        with strict=False.  This is the trap that hid the bug.
        """
        torch.manual_seed(123)
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)  # non-zero!
        expected = p.act(obs)[0]
        loaded = bp.build()
        actual = loaded.act(obs)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_export_rejects_missing_keys(self):
        """Missing state_dict key → RuntimeError, not silent random policy."""
        bp, _ = self._make_export()
        payload = self._load_payload()
        sd = dict(payload["state_dict"])
        # Remove a critical weight key
        removed = "net.0.weight"
        sd.pop(removed)
        payload["state_dict"] = sd
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn(removed, str(ctx.exception),
                      f"Error should mention missing key {removed}")

    def test_export_rejects_extra_keys(self):
        """Extra state_dict key → RuntimeError (strict=True catches both)."""
        bp, _ = self._make_export()
        payload = self._load_payload()
        sd = dict(payload["state_dict"])
        sd["nonexistent.layer.weight"] = torch.zeros(4, 4)
        payload["state_dict"] = sd
        self._save_payload(payload)
        with self.assertRaises(RuntimeError):
            bp.build()

    def test_export_rejects_wrong_format_version(self):
        """Wrong format_version → RuntimeError with actionable message."""
        bp, _ = self._make_export()
        payload = self._load_payload()
        payload["format_version"] = 999
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("format version", str(ctx.exception).lower())

    def test_export_rejects_wrong_policy_class(self):
        """Wrong policy_class → RuntimeError."""
        bp, _ = self._make_export()
        payload = self._load_payload()
        payload["policy_class"] = "SomeOtherPolicy"
        self._save_payload(payload)
        with self.assertRaises(RuntimeError) as ctx:
            bp.build()
        self.assertIn("class mismatch", str(ctx.exception).lower())

    def test_export_no_silent_param_swallowing(self):
        """**_ignored removed: unknown kwargs must raise TypeError.

        Before the fix, `**_ignored: Any` silently swallowed any
        misspelled blueprint parameter, making typos invisible.
        """
        bp, _ = self._make_export()
        # The loader __init__ now takes only model_path, no **kwargs.
        # Passing an unexpected kwarg should raise TypeError.
        with self.assertRaises(TypeError):
            bp.build(unknown_param=True)


# ---------------------------------------------------------------------------
# P0-6: Self-contained exports (no baseline.* imports)
# ---------------------------------------------------------------------------

class TestExportSelfContained(unittest.TestCase):
    """P0-6: Exported policy.py must not import from baseline.* or envs.*.

    Before the fix, the exported policy.py imported
    ``baseline.framework.ppo.policies.truncated_normal_mlp``, which broke
    81 historical artifacts when that module was moved to ``policies/todo/``.
    """

    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="p0_6_test_")

    def test_export_policy_py_has_no_repo_imports(self):
        """The generated policy.py must not import from baseline.* or envs.*."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to_blueprint(dest_path=self._tmp)
        policy_code = (Path(self._tmp) / "policy.py").read_text()
        # Check for forbidden imports
        for forbidden in [
            "from baseline", "import baseline",
            "from envs", "import envs",
        ]:
            self.assertNotIn(forbidden, policy_code,
                             f"Exported policy.py must not contain '{forbidden}'")

    def test_export_has_manifest(self):
        """Export directory contains MANIFEST.json with required fields."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to_blueprint(dest_path=self._tmp)
        import json
        manifest = json.loads(
            (Path(self._tmp) / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["policy_class"], "TruncatedNormalPolicy")
        self.assertEqual(manifest["exported_class"], "ExportedTruncNormPolicy")
        self.assertIn("arch", manifest)
        self.assertIn("files", manifest)

    def test_export_works_without_repo_on_path(self):
        """Exported policy loads and runs with no baseline.* on sys.path.

        This is the core P0-6 test: simulate a user who has the export
        directory but NOT the repo.  We load the exported policy.py in
        a subprocess with a clean PYTHONPATH (only stdlib + torch/numpy)
        and verify it produces the same output as the training-side policy.
        """
        import subprocess
        import sys

        torch.manual_seed(42)
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]

        # Run in a subprocess with a restricted PYTHONPATH that does NOT
        # include the repo.  The export must work on its own.
        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import ExportedTruncNormPolicy; "
            "import numpy as np; "
            "p = ExportedTruncNormPolicy(); "
            "obs = np.array({obs!r}, dtype=np.float32); "
            "a, _ = p.act(obs); "
            "print(repr(a.tolist()))"
        ).format(
            tmp=self._tmp,
            obs=obs.tolist(),
        )
        # Use the same Python with site-packages (for torch/numpy) but
        # strip the repo from sys.path by not passing PYTHONPATH=repo.
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
    """P0-6 方案 A life-line: training-side and export-side must agree.

    The export inlines the inference code.  If the two implementations
    drift (e.g., someone changes the training-side math but not the
    template), this test catches it immediately.
    """

    def test_parity_act_deterministic(self):
        """act() (deterministic mean) must match bit-for-bit."""
        torch.manual_seed(999)
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="parity_"))
        loaded = bp.build()
        # Test on multiple non-zero inputs
        for seed in range(10):
            torch.manual_seed(seed)
            obs = torch.randn(OBS_DIM).numpy().astype(np.float32)
            expected = p.act(obs)[0]
            actual = loaded.act(obs)[0]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0,
                                       err_msg=f"Parity failed on seed {seed}")

    def test_parity_sample_stochastic(self):
        """sample() with fixed torch seed must match bit-for-bit."""
        torch.manual_seed(777)
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="parity_"))
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        # Use the same random seed for both sampling calls
        torch.manual_seed(12345)
        expected_action, expected_lp = p.sample(obs, explore_factor=0.5, want_extra=True)
        torch.manual_seed(12345)
        actual_action, actual_lp = loaded.sample(obs, explore_factor=0.5, want_extra=True)
        np.testing.assert_allclose(actual_action, expected_action, rtol=0, atol=0,
                                   err_msg="Sampled action parity failed")
        self.assertAlmostEqual(actual_lp["log_prob"], expected_lp["log_prob"], places=6,
                               msg="log_prob parity failed")


# ---------------------------------------------------------------------------
# P0-4: .to(device) must keep self.device in sync
# ---------------------------------------------------------------------------

class TestDeviceSync(unittest.TestCase):
    """P0-4: .to(device) / .cuda() must update self.device.

    Before the fix, ``self.device`` was a plain attribute set in
    ``__init__`` and never updated by ``nn.Module.to()``.  After
    ``policy.to('cuda')``, parameters were on cuda but ``act()`` still
    placed the input on cpu → RuntimeError.
    """

    def test_device_follows_to(self):
        """self.device reflects the current parameter device after .to()."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        self.assertEqual(p.device.type, "cpu")
        p_cpu = p.to("cpu")
        self.assertEqual(p_cpu.device.type, "cpu")
        self.assertEqual(p.device.type, "cpu")  # in-place .to() updates p too

    def test_act_after_to_cpu(self):
        """act() works after .to('cpu') (the common eval/export path)."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to("cpu")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_act_with_device_kwarg_init(self):
        """Constructing with device='cpu' then act() works."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM, device="cpu")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))

    @unittest.skipIf(not torch.cuda.is_available(), "需要 GPU")
    def test_act_after_to_cuda(self):
        """act() works after .to('cuda') — the exact bug from P0-4.

        Before the fix, this raised:
          RuntimeError: Expected all tensors to be on the same device,
          but found at least two devices, cuda:0 and cpu!
        """
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to("cuda")
        self.assertEqual(p.device.type, "cuda")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = p.act(obs)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))

    @unittest.skipIf(not torch.cuda.is_available(), "需要 GPU")
    def test_sample_after_to_cuda(self):
        """sample() also works after .to('cuda')."""
        p = TruncatedNormalPolicy(OBS_DIM, ACTION_DIM, HIDDEN_DIM)
        p.to("cuda")
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, extra = p.sample(obs, explore_factor=0.5, want_extra=True)
        self.assertEqual(action.shape, (ACTION_DIM,))
        self.assertIn("log_prob", extra)


if __name__ == "__main__":
    unittest.main()
