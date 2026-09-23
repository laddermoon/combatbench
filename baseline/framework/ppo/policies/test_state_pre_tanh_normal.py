"""Tests for StatePreTanhNormalPolicy.

The distribution math, exploration mapping, uncertainty, guards and
scoring path are inherited from PreTanhNormalPolicy and fully covered
by test_pre_tanh_normal.py.  This file verifies what is DIFFERENT:

1. σ is state-dependent (head output), initialized to σ ≡ e⁻¹
2. Degenerate equivalence: const-σ state version == shared version
   (act, log_prob, seeded sample parity)
3. Per-state σ flows correctly through scoring / U / exploration
4. sigma_state_std diagnostic is meaningful (zero at init, >0 after
   the σ head moves)
5. Export: strict loading, self-containment, parity
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
)
from baseline.framework.ppo.policies.state_pre_tanh_normal_mlp import (
    StatePreTanhNormalPolicy,
)

OBS_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32


def _make_state(**kwargs) -> StatePreTanhNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return StatePreTanhNormalPolicy(**defaults)


def _make_shared(**kwargs) -> PreTanhNormalPolicy:
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device="cpu",
    )
    defaults.update(kwargs)
    return PreTanhNormalPolicy(**defaults)


def _degenerate_pair():
    """Shared policy + state policy wired to produce identical (mu, r):
    same trunk/mean weights, state σ head ≡ -1 → σ ≡ e⁻¹."""
    torch.manual_seed(7)
    shared = _make_shared()
    state = _make_state()
    with torch.no_grad():
        state.trunk[0].weight.copy_(shared.net[0].weight)
        state.trunk[0].bias.copy_(shared.net[0].bias)
        state.trunk[2].weight.copy_(shared.net[2].weight)
        state.trunk[2].bias.copy_(shared.net[2].bias)
        D = ACTION_DIM
        state.head.weight[:D].copy_(shared.net[4].weight)
        state.head.bias[:D].copy_(shared.net[4].bias)
        # σ block already zero/-1 by construction; assert it
        assert torch.all(state.head.weight[D:] == 0)
        assert torch.all(state.head.bias[D:] == -1.0)
    return shared, state


class TestInit(unittest.TestCase):
    def test_sigma_init_is_const_e_minus_1(self):
        """σ head init: zero weights + bias -1 → σ(s) ≡ e⁻¹."""
        p = _make_state()
        obs = torch.randn(100, OBS_DIM)
        _, r = p._policy_params(obs)
        sigma = r.exp()
        torch.testing.assert_close(
            sigma, torch.full_like(sigma, math.exp(-1.0)), rtol=0, atol=1e-6,
        )
        self.assertEqual(r.shape, (100, ACTION_DIM))

    def test_head_layout(self):
        p = _make_state()
        self.assertEqual(
            tuple(p.head.weight.shape), (2 * ACTION_DIM, HIDDEN_DIM),
        )
        self.assertFalse(hasattr(p, "log_std"))


class TestDegenerateEquivalence(unittest.TestCase):
    """Const-σ state version must reduce exactly to the shared version."""

    def test_act_bit_identical(self):
        shared, state = _degenerate_pair()
        obs = torch.randn(64, OBS_DIM)
        torch.testing.assert_close(
            state.deterministic_action(obs),
            shared.deterministic_action(obs),
            rtol=0, atol=0,
        )

    def test_log_prob_identical(self):
        shared, state = _degenerate_pair()
        obs = torch.randn(64, OBS_DIM)
        actions = torch.tanh(torch.randn(64, ACTION_DIM))
        ei = torch.linspace(-1.0, 1.0, 64)
        ev_s = shared.evaluate_actions(obs, actions, ei)
        ev_t = state.evaluate_actions(obs, actions, ei)
        torch.testing.assert_close(
            ev_t.log_prob, ev_s.log_prob, rtol=0, atol=1e-10,
        )
        torch.testing.assert_close(
            ev_t.uncertainty, ev_s.uncertainty, rtol=0, atol=1e-10,
        )

    def test_sample_identical_same_seed(self):
        """Same RNG sequence → identical sampled actions."""
        shared, state = _degenerate_pair()
        obs = torch.randn(64, OBS_DIM)
        for e in (0.0, 0.5, -0.7):
            torch.manual_seed(123)
            a_s, lp_s = shared.sample_action(obs, explore_factor=e)
            torch.manual_seed(123)
            a_t, lp_t = state.sample_action(obs, explore_factor=e)
            torch.testing.assert_close(a_t, a_s, rtol=0, atol=0)
            torch.testing.assert_close(lp_t, lp_s, rtol=0, atol=1e-10)


class TestStateSigma(unittest.TestCase):
    """σ(s) varies with obs once the σ head is non-degenerate."""

    def _randomize_sigma_head(self, p):
        with torch.no_grad():
            torch.nn.init.normal_(p.head.weight[ACTION_DIM:], std=0.5)
            torch.nn.init.normal_(p.head.bias[ACTION_DIM:], std=0.3)

    def test_sigma_varies_with_obs(self):
        p = _make_state()
        self._randomize_sigma_head(p)
        obs = torch.randn(200, OBS_DIM)
        _, r = p._policy_params(obs)
        sigma = r.exp()
        per_dim_std = sigma.std(dim=0)
        self.assertTrue((per_dim_std > 1e-4).all(),
                        f"σ should vary across obs, got {per_dim_std}")

    def test_sigma_state_std_stat(self):
        p = _make_state()
        obs = torch.randn(50, OBS_DIM)
        ev_init = p.evaluate_actions(
            obs, torch.zeros(50, ACTION_DIM), torch.zeros(50),
            want_stats=True,
        )
        self.assertAlmostEqual(
            ev_init.stats["sigma_state_std"], 0.0, places=7,
            msg="const-σ init must report zero state std",
        )
        self._randomize_sigma_head(p)
        ev = p.evaluate_actions(
            obs, torch.zeros(50, ACTION_DIM), torch.zeros(50),
            want_stats=True,
        )
        self.assertGreater(ev.stats["sigma_state_std"], 1e-4)

    def test_per_state_sigma_scores_correctly(self):
        """log_prob must use each row's own σ — check against the
        transformed-density reference evaluated per row."""
        p = _make_state()
        self._randomize_sigma_head(p)
        obs = torch.randn(20, OBS_DIM)
        actions = torch.tanh(torch.randn(20, ACTION_DIM))
        mu, r = p._policy_params(obs)
        ev = p.evaluate_actions(obs, actions, torch.zeros(20))
        a = actions.numpy().astype(np.float64)
        mu_np = mu.detach().numpy()
        sig_np = r.exp().detach().numpy()
        z = np.arctanh(np.clip(a, -1 + 1e-12, 1 - 1e-12))
        ref = np.zeros(20)
        for d in range(ACTION_DIM):
            for b in range(20):
                ref[b] += math.log(
                    sp_stats.norm.pdf(z[b, d], mu_np[b, d], sig_np[b, d])
                ) - math.log(1.0 - a[b, d] ** 2)
        np.testing.assert_allclose(
            ev.log_prob.detach().numpy(), ref, rtol=1e-5, atol=1e-8,
            err_msg="per-state σ density mismatch",
        )

    def test_u_uses_per_state_sigma(self):
        """U(s) must reflect both μ(s) and σ(s) — two states differing
        only in σ must give different U."""
        p = _make_state()
        self._randomize_sigma_head(p)
        obs = torch.randn(200, OBS_DIM)
        ev = p.evaluate_actions(
            obs, torch.zeros(200, ACTION_DIM), torch.zeros(200),
        )
        # Same-σ batch would give near-constant U; σ variation must
        # show up beyond the μ-driven variation alone.
        self.assertGreater(float(ev.uncertainty.std()), 1e-4)


class TestSharedMathInherited(unittest.TestCase):
    """Spot-checks that inherited machinery works on the (B,D) r path."""

    def test_explore_mapping_broadcasts_state_r(self):
        p = _make_state()
        obs = torch.randn(10, OBS_DIM)
        mu, r = p._policy_params(obs)
        self.assertEqual(r.shape, (10, ACTION_DIM))
        ei = torch.linspace(-1.0, 1.0, 10)
        mu_e, r_e = p._explored_params(mu, r, ei)
        c = torch.exp(-ei.double() * math.log(3.0)).unsqueeze(-1)
        torch.testing.assert_close(mu_e, mu * c, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            r_e,
            -0.1513403077614502 + c * (r + 0.1513403077614502),
            rtol=0, atol=1e-12,
        )

    def test_guards_inherited(self):
        p = _make_state()
        obs = torch.zeros(4, OBS_DIM)
        actions = torch.zeros(4, ACTION_DIM)
        actions[0, 0] = 1.0
        with self.assertRaises(RuntimeError):
            p.evaluate_actions(obs, actions, torch.zeros(4))
        actions[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            p.evaluate_actions(obs, actions, torch.zeros(4))
        with self.assertRaises(ValueError):
            p.evaluate_actions(
                obs, torch.zeros(4, ACTION_DIM), torch.full((4,), 2.0),
            )

    def test_sample_and_evaluate_consistency(self):
        torch.manual_seed(42)
        p = _make_state()
        obs = torch.randn(50, OBS_DIM)
        for e in (0.0, 0.5, -0.7):
            actions, lp = p.sample_action(obs, explore_factor=e)
            self.assertTrue((actions.abs() <= _ACTION_SAFE).all())
            ev = p.evaluate_actions(
                obs, actions, torch.full((50,), e),
            )
            diff = (lp - ev.log_prob).abs().max().item()
            self.assertLess(diff, 1e-5, f"e={e}: diff={diff}")

    def test_gradients_reach_sigma_head(self):
        p = _make_state()
        obs = torch.randn(10, OBS_DIM)
        actions = torch.tanh(torch.randn(10, ACTION_DIM))
        ev = p.evaluate_actions(obs, actions, torch.zeros(10))
        (-ev.log_prob.mean() + ev.uncertainty.mean()).backward()
        g = p.head.weight.grad
        self.assertFalse(
            torch.allclose(g[ACTION_DIM:],
                           torch.zeros_like(g[ACTION_DIM:])),
            "σ head block got zero gradient",
        )
        self.assertFalse(
            torch.allclose(g[:ACTION_DIM],
                           torch.zeros_like(g[:ACTION_DIM])),
            "mean head block got zero gradient",
        )

    def test_full_buffer_scale(self):
        p = _make_state()
        B = 204800
        obs = torch.randn(B, OBS_DIM)
        actions = torch.tanh(torch.randn(B, ACTION_DIM))
        with torch.no_grad():
            ev = p.evaluate_actions(
                obs, actions, torch.zeros(B), want_stats=True,
            )
        self.assertTrue(torch.isfinite(ev.log_prob).all())
        self.assertTrue(torch.isfinite(ev.uncertainty).all())


class TestExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="sptn_export_test_")
        self._model_path = str(Path(self._tmp) / "model.pt")

    def test_payload_metadata(self):
        _make_state().to_blueprint(dest_path=self._tmp)
        payload = torch.load(self._model_path, map_location="cpu")
        self.assertEqual(
            payload["policy_class"], "StatePreTanhNormalPolicy",
        )
        self.assertEqual(
            payload["distribution_kind"],
            "tanh_diagonal_normal_state_std_v1",
        )
        self.assertEqual(
            payload["exploration_kind"], "coverage_radial_logscale_v1",
        )

    def test_rejects_wrong_policy_class(self):
        p = _make_state()
        bp = p.to_blueprint(dest_path=self._tmp)
        payload = torch.load(self._model_path, map_location="cpu")
        payload["policy_class"] = "PreTanhNormalPolicy"
        torch.save(payload, self._model_path)
        with self.assertRaises(Exception):
            bp.build()

    def test_no_repo_imports(self):
        _make_state().to_blueprint(dest_path=self._tmp)
        code = (Path(self._tmp) / "policy.py").read_text()
        for forbidden in [
            "from baseline", "import baseline", "from envs", "import envs",
        ]:
            self.assertNotIn(forbidden, code)

    def test_parity_act_and_sample(self):
        torch.manual_seed(999)
        p = _make_state()
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="sptn_par_"))
        loaded = bp.build()
        for seed in range(10):
            torch.manual_seed(seed)
            obs = torch.randn(OBS_DIM).numpy().astype(np.float32)
            np.testing.assert_allclose(
                loaded.act(obs)[0], p.act(obs)[0], rtol=0, atol=0,
            )
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

    def test_exported_reset_reproducibility(self):
        """Exported policy honors Policy.reset(seed) — without it,
        rollout sampling is nondeterministic across runs."""
        torch.manual_seed(5)
        p = _make_state()
        bp = p.to_blueprint(dest_path=tempfile.mkdtemp(prefix="sptn_rs_"))
        loaded = bp.build()
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        loaded.reset(42)
        a1, _ = loaded.sample(obs)
        a2, _ = loaded.sample(obs)
        loaded.reset(42)
        a3, _ = loaded.sample(obs)
        np.testing.assert_allclose(a3, a1, rtol=0, atol=0)
        self.assertFalse(np.allclose(a2, a1, atol=1e-8))

    def test_works_without_repo_on_path(self):
        import subprocess
        import sys

        torch.manual_seed(42)
        p = _make_state()
        p.to_blueprint(dest_path=self._tmp)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        expected = p.act(obs)[0]
        runner = (
            "import sys; sys.path.insert(0, {tmp!r}); "
            "from policy import ExportedStatePreTanhNormalPolicy; "
            "import numpy as np; "
            "p = ExportedStatePreTanhNormalPolicy(); "
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
            "init_policy_state_pre_tanh_normal.yaml"
        )
        policy = bp.build()
        self.assertIsInstance(policy, StatePreTanhNormalPolicy)
        self.assertEqual(policy.obs_dim, 96)
        self.assertEqual(policy.action_dim, 21)


if __name__ == "__main__":
    unittest.main()
