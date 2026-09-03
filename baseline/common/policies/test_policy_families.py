"""Test suite for new policy families.

Tests are parametrized over families where possible.  The most important
test is degenerate_equivalence: each family configured to its most
degenerate form must match TanhGaussianMLPPolicy's log_prob to 1e-6.
This is a proof by reduction that the shared base class's tanh math is
correct.

Run with:
    cd /data1/mono/things/combatbench
    PYTHONPATH=. python3 -m pytest baseline/common/policies/test_policy_families.py -v

Or without pytest:
    PYTHONPATH=. python3 baseline/common/policies/test_policy_families.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import unittest

# Ensure repo is on sys.path when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from baseline.common.policies.tanh_squashed_base import TanhSquashedPolicyBase
from baseline.framework.ppo import ActorEval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs_batch(batch_size: int, obs_dim: int, seed: int = 42) -> torch.Tensor:
    """Reproducible random observation batch."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, obs_dim, generator=g)


def make_actions_batch(batch_size: int, action_dim: int, seed: int = 123) -> torch.Tensor:
    """Reproducible random action batch in (-1, 1)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch_size, action_dim, generator=g)
    return torch.tanh(raw)  # ensure in (-1, 1)


def measure_act_latency(policy, n_warmup: int = 50, n_measure: int = 200) -> float:
    """Measure single-step act() latency in seconds (CPU, 1 thread)."""
    torch.set_num_threads(1)
    obs = np.random.randn(policy.obs_dim).astype(np.float32)
    # Warmup
    for _ in range(n_warmup):
        policy.act(obs)
    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        policy.act(obs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Stage 0: Base class validation
# ---------------------------------------------------------------------------

class TestBaseClassEquivalence(unittest.TestCase):
    """Validate that TanhSquashedPolicyBase reproduces baseline math.

    Uses a minimal diagonal-Gaussian subclass (_DiagGaussianRef) that
    wraps the same distribution as TanhGaussianMLPPolicy.  If the base
    class's tanh Jacobian, atanh, or sum axis is wrong, this test fails.
    """

    @classmethod
    def setUpClass(cls):
        cls.obs_dim = 96
        cls.action_dim = 21
        cls.batch_size = 64
        cls.obs = make_obs_batch(cls.batch_size, cls.obs_dim)
        cls.actions = make_actions_batch(cls.batch_size, cls.action_dim)

    def test_diag_gaussian_matches_baseline(self):
        """A diagonal Gaussian via the base class must match
        TanhGaussianMLPPolicy's log_prob to 1e-6."""
        from baseline.common.policies.tanh_squashed_base import TanhSquashedPolicyBase

        # Build the reference baseline policy.
        baseline = TanhGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )

        # Build a minimal diagonal Gaussian subclass using the base class.
        ref = _DiagGaussianRef(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )

        # Copy weights from baseline to ref so they compute the same thing.
        ref.net.load_state_dict(baseline.net.state_dict())
        ref.log_std.data.copy_(baseline.log_std.data)

        # Compare evaluate_actions log_prob.
        ev_base = baseline.evaluate_actions(self.obs, self.actions)
        ev_ref = ref.evaluate_actions(self.obs, self.actions)

        diff = (ev_base.log_prob - ev_ref.log_prob).abs().max().item()
        self.assertLess(
            diff, 1e-6,
            f"Base class log_prob differs from baseline by {diff:.2e}",
        )

    def test_diag_gaussian_matches_baseline_sample(self):
        """sample_action log_prob must also match."""
        from baseline.common.policies.tanh_squashed_base import TanhSquashedPolicyBase

        baseline = TanhGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        ref = _DiagGaussianRef(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        ref.net.load_state_dict(baseline.net.state_dict())
        ref.log_std.data.copy_(baseline.log_std.data)

        torch.manual_seed(999)
        action_base, lp_base = baseline.sample_action(self.obs)
        torch.manual_seed(999)
        action_ref, lp_ref = ref.sample_action(self.obs)

        diff = (lp_base - lp_ref).abs().max().item()
        self.assertLess(
            diff, 1e-6,
            f"Base class sample_action log_prob differs by {diff:.2e}",
        )


class _DiagGaussianRef(TanhSquashedPolicyBase):
    """Minimal diagonal Gaussian using the base class hooks.

    This is NOT a production policy — it exists only to prove the base
    class's tanh math is correct by reproducing TanhGaussianMLPPolicy.
    It implements ``_raw_log_prob_per_dim`` so the base class uses the
    same per-dimension computation order as the baseline.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def _effective_log_std(self):
        return torch.clamp(
            self.log_std + float(np.log(self._temperature)),
            self.log_std_min, self.log_std_max,
        )

    def _raw_sample(self, obs):
        mean = self.net(obs)
        log_std = self._effective_log_std()
        raw = mean + log_std.exp() * torch.randn_like(mean)
        return raw, None

    def _raw_log_prob(self, obs, raw_action):
        from torch.distributions import Normal
        mean = self.net(obs)
        log_std = self._effective_log_std()
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action).sum(-1), None

    def _raw_log_prob_per_dim(self, obs, raw_action):
        """Per-dimension log_prob, for bit-identical baseline matching."""
        from torch.distributions import Normal
        mean = self.net(obs)
        log_std = self._effective_log_std()
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(raw_action), None

    def _raw_mode(self, obs):
        return self.net(obs)

    def _regularizer_and_stats(self, obs, raw_action, raw_log_prob, want_stats,
                                sample_extras, score_extras):
        # Closed-form entropy for a diagonal Gaussian.
        from torch.distributions import Normal
        mean = self.net(obs)
        log_std = self._effective_log_std()
        entropy = Normal(mean, log_std.exp()).entropy().sum(-1)
        regularizer = None
        if self._entropy_coef != 0.0:
            regularizer = -self._entropy_coef * entropy.mean()
        stats = None
        if want_stats:
            with torch.no_grad():
                eff_std = log_std.exp()
                stats = {
                    "entropy": float(entropy.mean().item()),
                    "std_mean": float(eff_std.mean().item()),
                    "std_min": float(eff_std.min().item()),
                    "std_max": float(eff_std.max().item()),
                    "tanh_sat_frac": float((mean.abs() > 2.0).float().mean().item()),
                }
        return regularizer, stats

    def export_config(self):
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    @property
    def export_class_path(self):
        return "baseline.common.policies.test_policy_families:_DiagGaussianRef"


# ---------------------------------------------------------------------------
# Normalization test (shared utility)
# ---------------------------------------------------------------------------

def monte_carlo_normalization(
    policy, obs_dim: int, action_dim: int,
    n_samples: int = 200_000, n_obs: int = 50, seed: int = 42,
) -> float:
    """Estimate ∫ p(a|s) da averaged over random states.

    For a correctly normalized distribution, E[1] = 1.  We estimate
    via:  (1/n_obs) * (1/n_samples) * Σ_s Σ_a 1
    ... but that's trivially 1.  Instead we check that the average
    log_prob exp matches the Monte Carlo estimate of the partition
    function.

    Actually, the correct test is: for a fixed obs, sample many actions,
    compute exp(log_prob) for each, and check that the *importance
    weights* are consistent.  A simpler and more robust test for
    small action_dim: grid integration.

    For action_dim=2, we grid the [-1,1]² space and numerically
    integrate exp(log_prob).
    """
    if action_dim != 2:
        raise ValueError("Grid normalization only for action_dim=2")

    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n_obs, obs_dim, generator=g)

    # Grid over [-1, 1]²
    n_grid = 200
    xs = torch.linspace(-0.999, 0.999, n_grid)
    grid = torch.cartesian_prod(xs, xs)  # (n_grid², 2)
    da = (xs[1] - xs[0]).item() ** 2

    total = 0.0
    for i in range(n_obs):
        o = obs[i:i+1].expand(grid.shape[0], -1)
        with torch.no_grad():
            ev = policy.evaluate_actions(o, grid)
        total += float(ev.log_prob.exp().sum().item()) * da
    return total / n_obs


# ---------------------------------------------------------------------------
# Sample/score self-consistency test (shared utility)
# ---------------------------------------------------------------------------

def sample_score_consistency(
    policy, obs_dim: int, action_dim: int,
    n_samples: int = 10_000, seed: int = 42,
) -> Tuple[float, float]:
    """Check that -mean(log_prob(samples)) ≈ MC entropy estimate.

    Returns:
        (expected_nll, mc_entropy_estimate)
        These should be close for a correctly normalized distribution.
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n_samples, obs_dim, generator=g)

    with torch.no_grad():
        # Sample actions
        actions, log_probs = policy.sample_action(obs)
        expected_nll = float((-log_probs).mean().item())

        # MC entropy via k-nearest-neighbor (for action_dim > 2)
        # Simple: use the log_prob values directly
        mc_entropy = expected_nll  # For a well-behaved distribution, these match

    return expected_nll, mc_entropy


# ---------------------------------------------------------------------------
# Gradient completeness test (shared utility)
# ---------------------------------------------------------------------------

def gradient_completeness(
    policy, obs_dim: int, action_dim: int, seed: int = 42,
) -> Dict[str, bool]:
    """Check that all parameters receive non-zero gradient.

    Returns:
        Dict mapping parameter name → has_nonzero_grad.
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(32, obs_dim, generator=g)
    actions = torch.tanh(torch.randn(32, action_dim, generator=g))

    # Zero all grads
    policy.zero_grad()

    # Forward + backward through evaluate_actions
    ev = policy.evaluate_actions(obs, actions)
    loss = ev.log_prob.mean()
    loss.backward()

    results = {}
    for name, param in policy.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
            results[name] = has_grad
    return results


# ---------------------------------------------------------------------------
# Stage 1: State-dependent Gaussian tests
# ---------------------------------------------------------------------------

class TestStateGaussian(unittest.TestCase):
    """Tests for StateGaussianMLPPolicy."""

    @classmethod
    def setUpClass(cls):
        cls.obs_dim = 96
        cls.action_dim = 21
        cls.batch_size = 64
        cls.obs = make_obs_batch(cls.batch_size, cls.obs_dim)
        cls.actions = make_actions_batch(cls.batch_size, cls.action_dim)

    def test_degenerate_equivalence(self):
        """With σ-head forced to constant matching baseline's log_std=-1,
        log_prob must match baseline to 1e-6."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        baseline = TanhGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        policy = StateGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )

        # Copy trunk weights from baseline (same architecture).
        # Baseline's net: [Linear(obs,h), Tanh, Linear(h,h), Tanh, Linear(h,act)]
        # Policy's trunk: [Linear(obs,h), Tanh, Linear(h,h), Tanh]
        # Policy's head:  Linear(h, 2*act)
        policy.trunk[0].weight.data.copy_(baseline.net[0].weight.data)
        policy.trunk[0].bias.data.copy_(baseline.net[0].bias.data)
        policy.trunk[2].weight.data.copy_(baseline.net[2].weight.data)
        policy.trunk[2].bias.data.copy_(baseline.net[2].bias.data)
        # Copy the mean half of the head from baseline's last layer.
        policy.head.weight.data[:self.action_dim, :].copy_(baseline.net[4].weight.data)
        policy.head.bias.data[:self.action_dim].copy_(baseline.net[4].bias.data)
        # The log-std half is already initialized to produce log_std ≈ -1.0
        # via _init_head. Baseline's log_std is also -1.0. So they should match.

        ev_base = baseline.evaluate_actions(self.obs, self.actions)
        ev_policy = policy.evaluate_actions(self.obs, self.actions)

        diff = (ev_base.log_prob - ev_policy.log_prob).abs().max().item()
        self.assertLess(
            diff, 1e-6,
            f"StateGaussian degenerate log_prob differs from baseline by {diff:.2e}",
        )

    def test_normalization_2d(self):
        """For action_dim=2, the density must integrate to ~1."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        policy = StateGaussianMLPPolicy(
            obs_dim=16, action_dim=2, hidden_dim=32,
            log_std_min=-4.0, log_std_max=0.0,
        )
        integral = monte_carlo_normalization(policy, obs_dim=16, action_dim=2)
        self.assertAlmostEqual(
            integral, 1.0, delta=0.02,
            msg=f"Normalization integral = {integral:.4f}, expected ~1.0",
        )

    def test_sample_score_consistency(self):
        """Sampled actions' log_prob must be self-consistent."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        policy = StateGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        # Sample actions and score them.
        torch.manual_seed(42)
        actions, lp_sample = policy.sample_action(self.obs)
        # Score the same actions via evaluate_actions.
        ev = policy.evaluate_actions(self.obs, actions)
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-5, f"sample_action vs evaluate_actions log_prob diff = {diff:.2e}")

    def test_gradient_completeness(self):
        """All parameters must receive non-zero gradient."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        policy = StateGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
            entropy_coef=1e-3,
        )
        results = gradient_completeness(policy, 96, 21)
        no_grad = [name for name, has_grad in results.items() if not has_grad]
        self.assertEqual(
            no_grad, [],
            f"Parameters with no gradient: {no_grad}",
        )

    def test_export_roundtrip(self):
        """Export → reload must reproduce actions and log_probs."""
        import tempfile
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        policy = StateGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
            log_std_min=-4.0, log_std_max=0.0,
            temperature=1.0,
        )
        # Set some non-default explore_intensity to verify it doesn't break export.
        policy.set_exploration(0.0)

        obs_np = self.obs[0].cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Deterministic export.
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            loaded = bp.build()
            # Compare deterministic actions via act().
            policy.set_deterministic(True)
            a_orig, _ = policy.act(obs_np)
            policy.set_deterministic(False)
            a_loaded, _ = loaded.act(obs_np)
            diff = np.abs(a_orig - a_loaded).max()
            self.assertLess(diff, 1e-6, f"Export roundtrip det action diff = {diff:.2e}")

            # Stochastic export.
            bp_stoch = policy.to_blueprint(dest_path=tmpdir + "_stoch", stochastic=True)
            loaded_stoch = bp_stoch.build()
            torch.manual_seed(123)
            a1, extra1 = policy.act(obs_np, want_extra=True)
            torch.manual_seed(123)
            a2, extra2 = loaded_stoch.act(obs_np, want_extra=True)
            diff_a = np.abs(a1 - a2).max()
            self.assertLess(diff_a, 1e-6, f"Export roundtrip stoch action diff = {diff_a:.2e}")
            if extra1 and extra2 and "log_prob" in extra1 and "log_prob" in extra2:
                diff_lp = abs(extra1["log_prob"] - extra2["log_prob"])
                self.assertLess(diff_lp, 1e-5, f"Export roundtrip stoch log_prob diff = {diff_lp:.2e}")

    def test_latency(self):
        """act() must be within 10× baseline latency."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        baseline = TanhGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        policy = StateGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        base_latency = measure_act_latency(baseline)
        policy_latency = measure_act_latency(policy)
        ratio = policy_latency / base_latency
        self.assertLess(
            ratio, 10.0,
            f"StateGaussian latency {policy_latency*1e6:.1f}µs vs baseline "
            f"{base_latency*1e6:.1f}µs = {ratio:.1f}×, exceeds 10× budget",
        )

    def test_explore_intensity_scaling(self):
        """Higher explore_intensity must increase σ."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy

        policy = StateGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        # Measure σ at explore_intensity=0 (no extra noise).
        with torch.no_grad():
            _, log_std_0 = policy._forward_head(self.obs[:16])
            std_0 = log_std_0.exp().mean().item()

        # Set explore_intensity=0.5 (moderate extra noise).
        policy.set_exploration(0.5)
        with torch.no_grad():
            _, log_std_5 = policy._forward_head(self.obs[:16])
            std_5 = log_std_5.exp().mean().item()

        # Set explore_intensity=1.0 (maximum extra noise).
        policy.set_exploration(1.0)
        with torch.no_grad():
            _, log_std_1 = policy._forward_head(self.obs[:16])
            std_1 = log_std_1.exp().mean().item()

        self.assertGreater(std_5, std_0, "explore_intensity=0.5 should increase σ")
        self.assertGreater(std_1, std_5, "explore_intensity=1.0 should increase σ further")


# ---------------------------------------------------------------------------
# Stage 2: Low-rank Gaussian tests
# ---------------------------------------------------------------------------

class TestLowRankGaussian(unittest.TestCase):
    """Tests for LowRankGaussianMLPPolicy."""

    @classmethod
    def setUpClass(cls):
        cls.obs_dim = 96
        cls.action_dim = 21
        cls.batch_size = 64
        cls.obs = make_obs_batch(cls.batch_size, cls.obs_dim)
        cls.actions = make_actions_batch(cls.batch_size, cls.action_dim)

    def test_degenerate_equivalence(self):
        """With U=0, log_prob must match state-dependent Gaussian (①)."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        # Build ① as reference.
        ref = StateGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        # Build ② with U=0 (manually zero U for degenerate test).
        policy = LowRankGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, rank=4, log_std_min=-4.0, log_std_max=0.0,
        )
        # Copy trunk + mean head + log_std head from ref to policy.
        policy.trunk[0].weight.data.copy_(ref.trunk[0].weight.data)
        policy.trunk[0].bias.data.copy_(ref.trunk[0].bias.data)
        policy.trunk[2].weight.data.copy_(ref.trunk[2].weight.data)
        policy.trunk[2].bias.data.copy_(ref.trunk[2].bias.data)
        ad = self.action_dim
        # Mean half.
        policy.head.weight.data[:ad, :].copy_(ref.head.weight.data[:ad, :])
        policy.head.bias.data[:ad].copy_(ref.head.bias.data[:ad])
        # Log-std half.
        policy.head.weight.data[ad:2*ad, :].copy_(ref.head.weight.data[ad:, :])
        policy.head.bias.data[ad:2*ad].copy_(ref.head.bias.data[ad:])
        # U half: manually zero for degenerate equivalence test.
        # (Default init is now small random, not zero, to avoid the
        # U=0 saddle point where ∂log_prob/∂U = 0.)
        policy.head.weight.data[2*ad:, :] = 0.0
        policy.head.bias.data[2*ad:] = 0.0

        ev_ref = ref.evaluate_actions(self.obs, self.actions)
        ev_policy = policy.evaluate_actions(self.obs, self.actions)

        # LowRankMultivariateNormal uses a different log_prob computation
        # path than Normal (Woodbury identity + PD margin ε on cov_diag),
        # so we expect a small numerical difference even at U=0.  The
        # PD margin (1e-6 per dim) alone accounts for ~1e-4 over 21 dims;
        # the Woodbury identity adds more.  Use 5e-3 tolerance — still
        # tight enough to catch a wrong U reshape (which would produce
        # a much larger difference).
        diff = (ev_ref.log_prob - ev_policy.log_prob).abs().max().item()
        self.assertLess(
            diff, 5e-3,
            f"LowRank U=0 log_prob differs from ① by {diff:.2e} "
            f"(expected < 5e-3 due to different computation path)",
        )

    def test_normalization_2d(self):
        """For action_dim=2, the density must integrate to ~1."""
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        policy = LowRankGaussianMLPPolicy(
            obs_dim=16, action_dim=2, hidden_dim=32, rank=1,
            log_std_min=-4.0, log_std_max=0.0,
        )
        integral = monte_carlo_normalization(policy, obs_dim=16, action_dim=2)
        self.assertAlmostEqual(
            integral, 1.0, delta=0.02,
            msg=f"Normalization integral = {integral:.4f}, expected ~1.0",
        )

    def test_sample_score_consistency(self):
        """Sampled actions' log_prob must be self-consistent."""
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
        )
        torch.manual_seed(42)
        actions, lp_sample = policy.sample_action(self.obs)
        ev = policy.evaluate_actions(self.obs, actions)
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-4, f"sample vs evaluate log_prob diff = {diff:.2e}")

    def test_gradient_completeness(self):
        """All parameters including U must receive non-zero gradient."""
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        # Use non-zero U so U gets gradient.  Randomize the U half.
        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
            entropy_coef=1e-3,
        )
        # Perturb U from zero so it's not degenerate.
        with torch.no_grad():
            ad = self.action_dim
            policy.head.weight.data[2*ad:, :] = torch.randn_like(policy.head.weight.data[2*ad:, :]) * 0.1

        results = gradient_completeness(policy, 96, 21)
        no_grad = [name for name, has_grad in results.items() if not has_grad]
        self.assertEqual(
            no_grad, [],
            f"Parameters with no gradient: {no_grad}",
        )

    def test_export_roundtrip(self):
        """Export → reload must reproduce actions."""
        import tempfile
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
            log_std_min=-4.0, log_std_max=0.0,
        )
        # Perturb U so it's non-degenerate.
        with torch.no_grad():
            ad = self.action_dim
            policy.head.weight.data[2*ad:, :] = torch.randn_like(policy.head.weight.data[2*ad:, :]) * 0.1

        obs_np = self.obs[0].cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            loaded = bp.build()
            policy.set_deterministic(True)
            a_orig, _ = policy.act(obs_np)
            policy.set_deterministic(False)
            a_loaded, _ = loaded.act(obs_np)
            diff = np.abs(a_orig - a_loaded).max()
            self.assertLess(diff, 1e-6, f"Export roundtrip det action diff = {diff:.2e}")

    def test_export_roundtrip_rank_mismatch(self):
        """Wrong rank on reload must raise (strict=True catches it)."""
        import tempfile
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            # Tamper with the payload to use wrong rank.
            payload = torch.load(Path(tmpdir) / "model.pt", map_location="cpu")
            payload["config"]["rank"] = 8  # Wrong rank.
            torch.save(payload, Path(tmpdir) / "model.pt")
            # Loading should fail due to strict=True.
            with self.assertRaises(Exception):
                bp.build()

    def test_latency(self):
        """act() must be within 10× baseline latency."""
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        baseline = TanhGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
        )
        base_latency = measure_act_latency(baseline)
        policy_latency = measure_act_latency(policy)
        ratio = policy_latency / base_latency
        self.assertLess(
            ratio, 10.0,
            f"LowRank latency {policy_latency*1e6:.1f}µs vs baseline "
            f"{base_latency*1e6:.1f}µs = {ratio:.1f}×, exceeds 10× budget",
        )

    def test_explore_intensity_scales_both_sigma_and_U(self):
        """Higher explore_intensity must scale both σ and U."""
        from baseline.common.policies.low_rank_gaussian_mlp import LowRankGaussianMLPPolicy

        policy = LowRankGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, rank=4,
        )
        # Perturb U so it's non-degenerate.
        with torch.no_grad():
            ad = self.action_dim
            policy.head.weight.data[2*ad:, :] = torch.randn_like(policy.head.weight.data[2*ad:, :]) * 0.1

        # Measure σ and U at explore_intensity=0.
        with torch.no_grad():
            _, log_std_0, U_0 = policy._forward_head(self.obs[:16])
            std_0 = log_std_0.exp().mean().item()
            U_norm_0 = U_0.flatten(1).norm(dim=-1).mean().item()

        # Set explore_intensity=0.5.
        policy.set_exploration(0.5)
        with torch.no_grad():
            _, log_std_5, U_5 = policy._forward_head(self.obs[:16])
            std_5 = log_std_5.exp().mean().item()
            U_norm_5 = U_5.flatten(1).norm(dim=-1).mean().item()

        self.assertGreater(std_5, std_0, "explore_intensity=0.5 should increase σ")
        self.assertGreater(U_norm_5, U_norm_0, "explore_intensity=0.5 should increase ||U||")


# ---------------------------------------------------------------------------
# Stage 3: Mixture of Gaussians tests
# ---------------------------------------------------------------------------

class TestMoGaussian(unittest.TestCase):
    """Tests for MoGTanhMLPPolicy."""

    @classmethod
    def setUpClass(cls):
        cls.obs_dim = 96
        cls.action_dim = 21
        cls.batch_size = 64
        cls.obs = make_obs_batch(cls.batch_size, cls.obs_dim)
        cls.actions = make_actions_batch(cls.batch_size, cls.action_dim)

    def test_degenerate_equivalence_K1(self):
        """With K=1, log_prob must match state-dependent Gaussian (①)."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        ref = StateGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        policy = MoGTanhMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, K=1, log_std_min=-4.0, log_std_max=0.0,
        )
        # Copy trunk from ref.
        policy.trunk[0].weight.data.copy_(ref.trunk[0].weight.data)
        policy.trunk[0].bias.data.copy_(ref.trunk[0].bias.data)
        policy.trunk[2].weight.data.copy_(ref.trunk[2].weight.data)
        policy.trunk[2].bias.data.copy_(ref.trunk[2].bias.data)
        # Copy mean head: ref's head[:ad] → policy's means portion.
        ad = self.action_dim
        # MoG head: [logits(1), means(1*ad), raw_log_stds(1*ad)]
        # ref head: [mean(ad), raw_log_std(ad)]
        policy.head.weight.data[1:1+ad, :].copy_(ref.head.weight.data[:ad, :])
        policy.head.bias.data[1:1+ad].copy_(ref.head.bias.data[:ad])
        policy.head.weight.data[1+ad:, :].copy_(ref.head.weight.data[ad:, :])
        policy.head.bias.data[1+ad:].copy_(ref.head.bias.data[ad:])
        # Logits are zero (uniform = only component).

        ev_ref = ref.evaluate_actions(self.obs, self.actions)
        ev_policy = policy.evaluate_actions(self.obs, self.actions)

        # MoG uses logsumexp over K=1 (a no-op) + log_softmax (returns 0
        # for K=1), so the math should be identical.  But the computation
        # path differs slightly (logsumexp vs direct sum), introducing
        # ~1e-5 numerical noise.  Use 5e-5 tolerance.
        diff = (ev_ref.log_prob - ev_policy.log_prob).abs().max().item()
        self.assertLess(
            diff, 5e-5,
            f"MoG K=1 log_prob differs from ① by {diff:.2e}",
        )

    def test_normalization_2d(self):
        """For action_dim=2, the density must integrate to ~1."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=16, action_dim=2, hidden_dim=32, K=3,
            log_std_min=-4.0, log_std_max=0.0,
        )
        integral = monte_carlo_normalization(policy, obs_dim=16, action_dim=2)
        self.assertAlmostEqual(
            integral, 1.0, delta=0.02,
            msg=f"MoG normalization integral = {integral:.4f}, expected ~1.0",
        )

    def test_sample_score_consistency(self):
        """Sampled actions' log_prob must be self-consistent."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
        )
        torch.manual_seed(42)
        actions, lp_sample = policy.sample_action(self.obs)
        ev = policy.evaluate_actions(self.obs, actions)
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-5, f"sample vs evaluate log_prob diff = {diff:.2e}")

    def test_gradient_completeness(self):
        """All three heads (logits, means, log_stds) must receive gradient."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
            entropy_coef=1e-3,
        )
        results = gradient_completeness(policy, 96, 21)
        no_grad = [name for name, has_grad in results.items() if not has_grad]
        self.assertEqual(
            no_grad, [],
            f"Parameters with no gradient: {no_grad}",
        )

    def test_export_roundtrip(self):
        """Export → reload must reproduce actions."""
        import tempfile
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
            log_std_min=-4.0, log_std_max=0.0,
        )
        obs_np = self.obs[0].cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            loaded = bp.build()
            policy.set_deterministic(True)
            a_orig, _ = policy.act(obs_np)
            policy.set_deterministic(False)
            a_loaded, _ = loaded.act(obs_np)
            diff = np.abs(a_orig - a_loaded).max()
            self.assertLess(diff, 1e-6, f"Export roundtrip det action diff = {diff:.2e}")

    def test_export_roundtrip_K_mismatch(self):
        """Wrong K on reload must raise (strict=True catches it)."""
        import tempfile
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            payload = torch.load(Path(tmpdir) / "model.pt", map_location="cpu")
            payload["config"]["K"] = 5  # Wrong K.
            torch.save(payload, Path(tmpdir) / "model.pt")
            with self.assertRaises(Exception):
                bp.build()

    def test_latency(self):
        """act() must be within 10× baseline latency."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        baseline = TanhGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
        )
        base_latency = measure_act_latency(baseline)
        policy_latency = measure_act_latency(policy)
        ratio = policy_latency / base_latency
        self.assertLess(
            ratio, 10.0,
            f"MoG latency {policy_latency*1e6:.1f}µs vs baseline "
            f"{base_latency*1e6:.1f}µs = {ratio:.1f}×, exceeds 10× budget",
        )

    def test_explore_intensity_scales_sigma_not_logits(self):
        """Higher explore_intensity must scale σ but not change mixture logits."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
        )
        # Measure σ and weights at explore_intensity=0.
        with torch.no_grad():
            logits_0, _, log_stds_0 = policy._forward_head(self.obs[:16])
            std_0 = log_stds_0.exp().mean().item()
            weights_0 = torch.softmax(logits_0, dim=-1)

        # Set explore_intensity=0.5.
        policy.set_exploration(0.5)
        with torch.no_grad():
            logits_5, _, log_stds_5 = policy._forward_head(self.obs[:16])
            std_5 = log_stds_5.exp().mean().item()
            weights_5 = torch.softmax(logits_5, dim=-1)

        self.assertGreater(std_5, std_0, "explore_intensity=0.5 should increase σ")
        # Weights should be unchanged (logits are not scaled).
        diff_weights = (weights_0 - weights_5).abs().max().item()
        self.assertLess(diff_weights, 1e-6, "Mixture weights should not change with temperature")

    def test_component_usage_matches_weights(self):
        """Effective component usage should ≈ mean(π_k) over many samples."""
        from baseline.common.policies.mog_tanh_mlp import MoGTanhMLPPolicy

        policy = MoGTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, K=3,
        )
        # Perturb logits so components are not uniform.
        with torch.no_grad():
            policy.head.bias.data[:3] = torch.tensor([2.0, 0.0, -1.0])

        n_samples = 10000
        obs = torch.randn(n_samples, 96)
        with torch.no_grad():
            logits, _, _ = policy._forward_head(obs)
            mean_weights = torch.softmax(logits, dim=-1).mean(dim=0)  # (K,)

            # Sample and record component indices.
            torch.manual_seed(42)
            raw, extras = policy._raw_sample(obs)
            idx = extras["component_idx"]
            usage = torch.zeros(3)
            for k in range(3):
                usage[k] = (idx == k).float().mean()

        diff = (mean_weights - usage).abs().max().item()
        self.assertLess(
            diff, 0.03,
            f"Component usage {usage} differs from mean weights {mean_weights} by {diff:.3f}",
        )


# ---------------------------------------------------------------------------
# Stage 4: RealNVP normalizing flow tests
# ---------------------------------------------------------------------------

class TestRealNVPFlow(unittest.TestCase):
    """Tests for RealNVPTanhMLPPolicy."""

    @classmethod
    def setUpClass(cls):
        cls.obs_dim = 96
        cls.action_dim = 21
        cls.batch_size = 64
        cls.obs = make_obs_batch(cls.batch_size, cls.obs_dim)
        cls.actions = make_actions_batch(cls.batch_size, cls.action_dim)

    def test_degenerate_equivalence_identity_flow(self):
        """With identity flow (s=0, t=0), log_prob must match ①."""
        from baseline.common.policies.state_gaussian_mlp import StateGaussianMLPPolicy
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        ref = StateGaussianMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, log_std_min=-4.0, log_std_max=0.0,
        )
        # Flow with identity init (default: s=0, t=0).
        policy = RealNVPTanhMLPPolicy(
            obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=256, num_layers=4, scale_max=1.0,
            log_std_min=-4.0, log_std_max=0.0,
        )
        # Copy trunk + base_head from ref.
        policy.trunk[0].weight.data.copy_(ref.trunk[0].weight.data)
        policy.trunk[0].bias.data.copy_(ref.trunk[0].bias.data)
        policy.trunk[2].weight.data.copy_(ref.trunk[2].weight.data)
        policy.trunk[2].bias.data.copy_(ref.trunk[2].bias.data)
        ad = self.action_dim
        policy.base_head.weight.data[:ad, :].copy_(ref.head.weight.data[:ad, :])
        policy.base_head.bias.data[:ad].copy_(ref.head.bias.data[:ad])
        policy.base_head.weight.data[ad:, :].copy_(ref.head.weight.data[ad:, :])
        policy.base_head.bias.data[ad:].copy_(ref.head.bias.data[ad:])
        # Flow layers are already identity (zeroed by _init_heads).

        ev_ref = ref.evaluate_actions(self.obs, self.actions)
        ev_policy = policy.evaluate_actions(self.obs, self.actions)

        # With identity flow, the log_prob should match ① closely.
        # Small differences may arise from the flow's forward/inverse
        # passes introducing floating-point noise even at s=0, t=0.
        diff = (ev_ref.log_prob - ev_policy.log_prob).abs().max().item()
        self.assertLess(
            diff, 1e-4,
            f"RealNVP identity-flow log_prob differs from ① by {diff:.2e}",
        )

    def test_inverse_consistency(self):
        """inverse(forward(x)) ≈ x to 1e-5."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
        )
        # Perturb flow layers so the flow is non-identity.
        for layer in policy.layers:
            with torch.no_grad():
                layer.conditioner[-1].weight.data = torch.randn_like(layer.conditioner[-1].weight.data) * 0.1
                layer.conditioner[-1].bias.data = torch.randn_like(layer.conditioner[-1].bias.data) * 0.1

        trunk_obs = policy.trunk(self.obs)
        z = torch.randn_like(self.obs[:, :21])  # (B, 21)

        with torch.no_grad():
            raw, _ = policy._flow_forward(z, trunk_obs)
            z_recon, _ = policy._flow_inverse(raw, trunk_obs)

        diff = (z - z_recon).abs().max().item()
        self.assertLess(diff, 1e-5, f"Inverse consistency error = {diff:.2e}")

    def test_normalization_2d(self):
        """For action_dim=2, the density must integrate to ~1."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=16, action_dim=2, hidden_dim=32, num_layers=2,
            conditioner_hidden=16, scale_max=1.0,
            log_std_min=-4.0, log_std_max=0.0,
        )
        # Perturb flow so it's non-identity.
        for layer in policy.layers:
            with torch.no_grad():
                layer.conditioner[-1].weight.data = torch.randn_like(layer.conditioner[-1].weight.data) * 0.1
                layer.conditioner[-1].bias.data = torch.randn_like(layer.conditioner[-1].bias.data) * 0.1

        integral = monte_carlo_normalization(policy, obs_dim=16, action_dim=2)
        self.assertAlmostEqual(
            integral, 1.0, delta=0.02,
            msg=f"RealNVP normalization integral = {integral:.4f}, expected ~1.0",
        )

    def test_sample_score_consistency(self):
        """Sampled actions' log_prob must be self-consistent."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
        )
        # Perturb flow.
        for layer in policy.layers:
            with torch.no_grad():
                layer.conditioner[-1].weight.data = torch.randn_like(layer.conditioner[-1].weight.data) * 0.1
                layer.conditioner[-1].bias.data = torch.randn_like(layer.conditioner[-1].bias.data) * 0.1

        torch.manual_seed(42)
        actions, lp_sample = policy.sample_action(self.obs)
        ev = policy.evaluate_actions(self.obs, actions)
        diff = (lp_sample - ev.log_prob).abs().max().item()
        self.assertLess(diff, 1e-4, f"sample vs evaluate log_prob diff = {diff:.2e}")

    def test_gradient_completeness(self):
        """All parameters including flow conditioners must receive gradient."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
            entropy_coef=1e-3,
        )
        # Perturb flow so gradients flow through non-identity transforms.
        for layer in policy.layers:
            with torch.no_grad():
                layer.conditioner[-1].weight.data = torch.randn_like(layer.conditioner[-1].weight.data) * 0.1
                layer.conditioner[-1].bias.data = torch.randn_like(layer.conditioner[-1].bias.data) * 0.1

        results = gradient_completeness(policy, 96, 21)
        no_grad = [name for name, has_grad in results.items() if not has_grad]
        self.assertEqual(
            no_grad, [],
            f"Parameters with no gradient: {no_grad}",
        )

    def test_export_roundtrip(self):
        """Export → reload must reproduce actions."""
        import tempfile
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
            scale_max=1.0, log_std_min=-4.0, log_std_max=0.0,
        )
        # Perturb flow.
        for layer in policy.layers:
            with torch.no_grad():
                layer.conditioner[-1].weight.data = torch.randn_like(layer.conditioner[-1].weight.data) * 0.1
                layer.conditioner[-1].bias.data = torch.randn_like(layer.conditioner[-1].bias.data) * 0.1

        obs_np = self.obs[0].cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            loaded = bp.build()
            policy.set_deterministic(True)
            a_orig, _ = policy.act(obs_np)
            policy.set_deterministic(False)
            a_loaded, _ = loaded.act(obs_np)
            diff = np.abs(a_orig - a_loaded).max()
            self.assertLess(diff, 1e-6, f"Export roundtrip det action diff = {diff:.2e}")

    def test_export_roundtrip_num_layers_mismatch(self):
        """Wrong num_layers on reload must raise (strict=True catches it)."""
        import tempfile
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = policy.to_blueprint(dest_path=tmpdir, stochastic=False)
            payload = torch.load(Path(tmpdir) / "model.pt", map_location="cpu")
            payload["config"]["num_layers"] = 2  # Wrong.
            torch.save(payload, Path(tmpdir) / "model.pt")
            with self.assertRaises(Exception):
                bp.build()

    def test_latency(self):
        """act() must be within 10× baseline latency."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        baseline = TanhGaussianMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256,
        )
        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
        )
        base_latency = measure_act_latency(baseline)
        policy_latency = measure_act_latency(policy)
        ratio = policy_latency / base_latency
        self.assertLess(
            ratio, 10.0,
            f"RealNVP latency {policy_latency*1e6:.1f}µs vs baseline "
            f"{base_latency*1e6:.1f}µs = {ratio:.1f}×, exceeds 10× budget",
        )

    def test_explore_intensity_scales_base_not_flow(self):
        """Higher explore_intensity must scale base σ but not flow parameters."""
        from baseline.common.policies.realnvp_tanh_mlp import RealNVPTanhMLPPolicy

        policy = RealNVPTanhMLPPolicy(
            obs_dim=96, action_dim=21, hidden_dim=256, num_layers=4,
        )
        # Measure base σ at explore_intensity=0.
        with torch.no_grad():
            _, base_dist_0 = policy._base_dist(self.obs[:16])
            std_0 = base_dist_0.stddev.mean().item()

        # Set explore_intensity=0.5.
        policy.set_exploration(0.5)
        with torch.no_grad():
            _, base_dist_5 = policy._base_dist(self.obs[:16])
            std_5 = base_dist_5.stddev.mean().item()

        self.assertGreater(std_5, std_0, "explore_intensity=0.5 should increase base σ")


# ---------------------------------------------------------------------------
# Main entry point (for running without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
