"""Tests for OU exploration noise in TanhSquashedPolicyBase and
FixedSigmaGaussianMLPPolicy.

Test categories (matching the implementation plan):

1. Degenerate equivalence: noise_scale=0 → identical to baseline.
2. Sampling/scoring consistency: sample with shift, score with shift → match.
3. OU statistics: steady-state variance ≈ 1, lag-1 autocorrelation ≈ exp(-1/τ).
4. Reset determinism: same seed → same sequence; reset → x_0 = 0.
5. Export round-trip: OU params survive export/reload; reset forwards.
6. End-to-end log_prob consistency: PPOBuffer recomputes matching log_prob.
7. Inconsistent input rejection: mixed noise_shift presence → raise.
8. Backward compatibility: existing tests still pass (verified separately).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal

import pytest

from baseline.framework.ppo.policies import (
    FixedSigmaGaussianMLPPolicy,
    TanhGaussianMLPPolicy,
)
from baseline.framework.ppo.policies.tanh_squashed_base import TanhSquashedPolicyBase
from baseline.framework.ppo import ActorEval


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 96
ACTION_DIM = 21
HIDDEN_DIM = 64  # small for fast tests


def _make_policy(**kwargs):
    """Create a FixedSigmaGaussianMLPPolicy with test defaults."""
    defaults = dict(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
    )
    defaults.update(kwargs)
    return FixedSigmaGaussianMLPPolicy(**defaults)


def _make_baseline(**kwargs):
    """Create a TanhGaussianMLPPolicy with matching architecture."""
    defaults = dict(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
    )
    defaults.update(kwargs)
    return TanhGaussianMLPPolicy(**defaults)


# ---------------------------------------------------------------------------
# 1. Degenerate equivalence: noise_scale=0 → identical to baseline
# ---------------------------------------------------------------------------

class TestDegenerateEquivalence:

    def test_state_dict_keys_match(self):
        """FixedSigma and baseline must have identical state_dict keys."""
        fp = _make_policy()
        bp = _make_baseline()
        assert set(fp.state_dict().keys()) == set(bp.state_dict().keys())

    def test_checkpoint_load_strict(self):
        """Baseline checkpoint loads into FixedSigma with strict=True."""
        bp = _make_baseline()
        fp = _make_policy()
        fp.load_state_dict(bp.state_dict(), strict=True)

    def test_evaluate_actions_identical(self):
        """With noise_scale=0, evaluate_actions is bit-identical to baseline."""
        bp = _make_baseline()
        fp = _make_policy(noise_scale=0.0)
        fp.load_state_dict(bp.state_dict(), strict=True)

        obs = torch.randn(8, OBS_DIM)
        acts = torch.tanh(torch.randn(8, ACTION_DIM))

        with torch.no_grad():
            ev1 = bp.evaluate_actions(obs, acts, torch.full((8,), 0.5), want_stats=True)
            ev2 = fp.evaluate_actions(obs, acts, torch.full((8,), 0.5), want_stats=True)

        assert torch.allclose(ev1.log_prob, ev2.log_prob, atol=1e-6)
        # Stats are not compared: the baseline uses closed-form entropy
        # while the OU policy uses a sample-based estimate, so the
        # entropy_raw values differ.  The log_prob equivalence is the
        # meaningful test.

    def test_deterministic_action_identical(self):
        """With noise_scale=0, deterministic_action matches baseline."""
        bp = _make_baseline()
        fp = _make_policy(noise_scale=0.0)
        fp.load_state_dict(bp.state_dict(), strict=True)

        obs = torch.randn(4, OBS_DIM)
        with torch.no_grad():
            d1 = bp.deterministic_action(obs)
            d2 = fp.deterministic_action(obs)
        assert torch.allclose(d1, d2, atol=1e-6)

    def test_sample_action_identical_with_same_seed(self):
        """With noise_scale=0 and same RNG seed, sample_action matches."""
        bp = _make_baseline()
        fp = _make_policy(noise_scale=0.0)
        fp.load_state_dict(bp.state_dict(), strict=True)

        obs = torch.randn(4, OBS_DIM)
        torch.manual_seed(123)
        a1, lp1 = bp.sample_action(obs)
        torch.manual_seed(123)
        a2, lp2 = fp.sample_action(obs)

        assert torch.allclose(a1, a2, atol=1e-6)
        assert torch.allclose(lp1, lp2, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Sampling/scoring consistency (THE critical test)
# ---------------------------------------------------------------------------

class TestSamplingScoringConsistency:
    """Verify that sample_action and evaluate_actions agree when noise_shift
    is used.  This is the single most important test — if these disagree,
    PPO's importance ratios are silently wrong.
    """

    def test_sample_then_score_matches(self):
        """sample_action(obs, noise_shift=s) → (a, lp).
        evaluate_actions(obs, a, noise_shift=s) → lp2.
        lp and lp2 must be identical.
        """
        policy = _make_policy(noise_scale=0.3, noise_tau_steps=10.0)
        obs = torch.randn(16, OBS_DIM)
        shift = torch.randn(16, ACTION_DIM) * 0.3

        with torch.no_grad():
            action, lp_sample = policy.sample_action(obs, noise_shift=shift)
            ev = policy.evaluate_actions(obs, action, torch.full((16,), 0.5), noise_shift=shift)
            lp_eval = ev.log_prob

        assert torch.allclose(lp_sample, lp_eval, atol=1e-5), (
            f"sample_action log_prob != evaluate_actions log_prob. "
            f"max diff: {(lp_sample - lp_eval).abs().max().item()}"
        )

    def test_score_without_shift_differs(self):
        """Scoring without the shift must give a DIFFERENT log_prob,
        confirming the shift actually affects the computation."""
        policy = _make_policy(noise_scale=0.3, noise_tau_steps=10.0)
        obs = torch.randn(16, OBS_DIM)
        shift = torch.randn(16, ACTION_DIM) * 0.3

        with torch.no_grad():
            action, _ = policy.sample_action(obs, noise_shift=shift)
            ev_with = policy.evaluate_actions(obs, action, torch.full((16,), 0.5), noise_shift=shift)
            ev_without = policy.evaluate_actions(obs, action, torch.full((16,), 0.5))

        # They should differ — the shift changes which raw_action is scored.
        assert not torch.allclose(ev_with.log_prob, ev_without.log_prob, atol=1e-3)

    def test_zero_shift_matches_no_shift(self):
        """A zero shift must be identical to no shift at all."""
        policy = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        acts = torch.tanh(torch.randn(8, ACTION_DIM))

        with torch.no_grad():
            ev_none = policy.evaluate_actions(obs, acts, torch.full((8,), 0.5))
            ev_zero = policy.evaluate_actions(
                obs, acts, torch.full((8,), 0.5),
                noise_shift=torch.zeros(8, ACTION_DIM),
            )
        assert torch.allclose(ev_none.log_prob, ev_zero.log_prob, atol=1e-6)

    def test_per_dim_path_matches(self):
        """The per-dim log_prob path (used by FixedSigma) must also
        produce consistent results with shift."""
        policy = _make_policy()
        obs = torch.randn(8, OBS_DIM)
        shift = torch.randn(8, ACTION_DIM) * 0.2

        with torch.no_grad():
            action, lp_sample = policy.sample_action(obs, noise_shift=shift)
            ev = policy.evaluate_actions(obs, action, torch.full((8,), 0.5), noise_shift=shift)

        assert torch.allclose(lp_sample, ev.log_prob, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. OU statistics: steady-state variance and autocorrelation
# ---------------------------------------------------------------------------

class TestOUStatistics:

    def test_steady_state_variance(self):
        """The AR(1) process x_t should have Var(x) ≈ 1 at steady state."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=1.0)
        policy.reset(seed=42)

        n_steps = 5000
        samples = np.zeros((n_steps, ACTION_DIM), dtype=np.float32)
        for t in range(n_steps):
            shift = policy._next_noise_shift()
            samples[t] = shift  # noise_scale=1.0, so shift = x_t

        # Steady-state variance should be ≈ 1 (unit variance by design).
        var = samples[100:].var(axis=0).mean()  # skip burn-in
        assert abs(var - 1.0) < 0.1, f"Steady-state variance {var} != 1.0"

    def test_lag1_autocorrelation(self):
        """Lag-1 autocorrelation should be ≈ exp(-1/tau)."""
        tau = 10.0
        policy = _make_policy(noise_tau_steps=tau, noise_scale=1.0)
        policy.reset(seed=42)

        n_steps = 5000
        samples = np.zeros((n_steps, ACTION_DIM), dtype=np.float32)
        for t in range(n_steps):
            shift = policy._next_noise_shift()
            samples[t] = shift

        expected_ac = np.exp(-1.0 / tau)
        # Compute lag-1 autocorrelation averaged over dims.
        s = samples[100:]  # skip burn-in
        ac = np.array([
            np.corrcoef(s[:-1, d], s[1:, d])[0, 1]
            for d in range(ACTION_DIM)
        ]).mean()
        assert abs(ac - expected_ac) < 0.05, (
            f"Lag-1 autocorrelation {ac:.4f} != expected {expected_ac:.4f}"
        )

    def test_tau_zero_is_white_noise(self):
        """tau=0 → a=0 → each step is independent (white noise)."""
        policy = _make_policy(noise_tau_steps=0.0, noise_scale=1.0)
        policy.reset(seed=42)
        assert policy._ou_a == 0.0
        assert policy._ou_innov == 1.0

        # Generate samples and check near-zero autocorrelation.
        n = 2000
        samples = np.zeros((n, ACTION_DIM), dtype=np.float32)
        for t in range(n):
            samples[t] = policy._next_noise_shift()

        s = samples[100:]
        ac = np.array([
            np.corrcoef(s[:-1, d], s[1:, d])[0, 1]
            for d in range(ACTION_DIM)
        ]).mean()
        assert abs(ac) < 0.1, f"White noise autocorrelation {ac} should be ≈ 0"


# ---------------------------------------------------------------------------
# 4. Reset determinism
# ---------------------------------------------------------------------------

class TestResetDeterminism:

    def test_same_seed_same_sequence(self):
        """Same seed → same OU sequence."""
        p1 = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        p2 = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        p1.reset(seed=99)
        p2.reset(seed=99)

        seq1 = [p1._next_noise_shift() for _ in range(20)]
        seq2 = [p2._next_noise_shift() for _ in range(20)]

        for s1, s2 in zip(seq1, seq2):
            assert np.allclose(s1, s2, atol=1e-6)

    def test_different_seed_different_sequence(self):
        """Different seeds → different sequences."""
        p1 = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        p2 = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        p1.reset(seed=1)
        p2.reset(seed=2)

        s1 = p1._next_noise_shift()
        s2 = p2._next_noise_shift()
        assert not np.allclose(s1, s2, atol=1e-4)

    def test_reset_zeroes_state(self):
        """After reset, the first shift should be small (x_0=0, one step)."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=1.0)
        # Run for a while to build up state.
        policy.reset(seed=42)
        for _ in range(100):
            policy._next_noise_shift()
        # Reset and check the first step is just the innovation.
        policy.reset(seed=42)
        first = policy._next_noise_shift()
        # x_1 = a * 0 + sqrt(1-a^2) * xi = sqrt(1-a^2) * xi
        # |first| should be ≈ noise_scale * sqrt(1-a^2) * |xi|
        # which is much smaller than the steady-state std.
        assert policy._ou_x is not None
        # After one step from x=0, |x| should be small relative to steady state.
        # The innovation std is sqrt(1-a^2) ≈ 0.316 for tau=10.
        assert np.all(np.abs(policy._ou_x) < 1.0)  # well within unit variance

    def test_reset_returns_none(self):
        """reset should return None (matching Policy ABC)."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        result = policy.reset(seed=42)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Export round-trip
# ---------------------------------------------------------------------------

class TestExportRoundTrip:

    def test_ou_params_survive_export(self, tmp_path):
        """Export a policy with OU enabled, reload, check params match."""
        policy = _make_policy(
            noise_tau_steps=10.0, noise_scale=0.3,
            entropy_coef=1e-3,
        )
        bp = policy.to_blueprint(dest_path=str(tmp_path), stochastic=True)

        # Load the exported policy.
        from envs.framework.policy import PolicyBlueprint
        exported = bp.build()
        assert exported.stochastic is True

        # Check OU params were passed through.
        inner = exported._policy
        assert inner._noise_tau_steps == 10.0
        assert inner._noise_scale == 0.3

    def test_act_returns_noise_shift_in_extras(self, tmp_path):
        """Exported policy's act(want_extra=True) must include noise_shift."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        bp = policy.to_blueprint(dest_path=str(tmp_path), stochastic=True)
        exported = bp.build()
        exported.reset(seed=42)

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, extras = exported.act(obs, want_extra=True)

        assert extras is not None
        assert "log_prob" in extras
        assert "noise_shift" in extras
        assert np.asarray(extras["noise_shift"]).shape == (ACTION_DIM,)

    def test_reset_forwards_to_inner(self, tmp_path):
        """Exported policy's reset must forward to the inner policy,
        zeroing OU state."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        bp = policy.to_blueprint(dest_path=str(tmp_path), stochastic=True)
        exported = bp.build()

        # Step the OU process a few times.
        exported.reset(seed=1)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for _ in range(10):
            exported.act(obs, want_extra=True)

        # Reset and verify OU state was zeroed.
        exported.reset(seed=2)
        inner = exported._policy
        assert np.all(inner._ou_x == 0.0)

    def test_ctrl_export_has_no_noise_shift(self, tmp_path):
        """With noise_scale=0, act(want_extra=True) should NOT include
        noise_shift (it's disabled)."""
        policy = _make_policy(noise_scale=0.0)
        bp = policy.to_blueprint(dest_path=str(tmp_path), stochastic=True)
        exported = bp.build()
        exported.reset(seed=42)

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, extras = exported.act(obs, want_extra=True)

        assert extras is not None
        assert "log_prob" in extras
        assert "noise_shift" not in extras


# ---------------------------------------------------------------------------
# 6. End-to-end log_prob consistency via PPOBuffer
# ---------------------------------------------------------------------------

class TestPPOBufferConsistency:
    """Construct a trajectory with noise_shift, build a PPOBuffer, and
    verify the recomputed log_probs match the sampling-time log_probs.
    """

    def test_buffer_log_probs_match_sampling(self):
        from baseline.framework.ppo.trajectory import Trajectory, ChannelData, RewardChannel
        from baseline.framework.ppo.trainer import PPOBuffer

        policy = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        policy.reset(seed=42)

        T = 32
        obs = np.random.randn(T, OBS_DIM).astype(np.float32)
        actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
        shifts = np.zeros((T, ACTION_DIM), dtype=np.float32)
        log_probs = np.zeros(T, dtype=np.float32)

        for t in range(T):
            o = torch.from_numpy(obs[t:t+1])
            s = torch.from_numpy(policy._next_noise_shift()).unsqueeze(0)
            with torch.no_grad():
                a, lp = policy.sample_action(o, noise_shift=s)
            actions[t] = a.squeeze(0).numpy()
            shifts[t] = s.squeeze(0).numpy()
            log_probs[t] = lp.item()

        # Build a trajectory with noise_shift.
        traj = Trajectory(
            obs=obs,
            actions=actions,
            last_obs=obs[-1],
            channels={"r_test": ChannelData(
                reward=np.zeros(T, dtype=np.float32),
                is_terminated=False,
                actor_weight=1.0,
            )},
            noise_shift=shifts,
        )

        buf = PPOBuffer(
            [traj], policy, torch.device("cpu"),
            reward_keys=("r_test",),
        )

        # The buffer recomputes log_probs via evaluate_actions with noise_shift.
        # They should match the sampling-time log_probs.
        assert np.allclose(buf.log_probs, log_probs, atol=1e-4), (
            f"PPOBuffer log_probs != sampling log_probs. "
            f"max diff: {np.abs(buf.log_probs - log_probs).max()}"
        )


# ---------------------------------------------------------------------------
# 7. Inconsistent input rejection
# ---------------------------------------------------------------------------

class TestInconsistentInputRejection:

    def test_mixed_shift_presence_raises(self):
        """Mixing trajectories with and without noise_shift must raise."""
        from baseline.framework.ppo.trajectory import Trajectory, ChannelData
        from baseline.framework.ppo.trainer import PPOBuffer

        policy = _make_policy()
        T = 8
        obs = np.random.randn(T, OBS_DIM).astype(np.float32)
        acts = np.tanh(np.random.randn(T, ACTION_DIM)).astype(np.float32)
        cd = ChannelData(
            reward=np.zeros(T, dtype=np.float32),
            is_terminated=False, actor_weight=1.0,
        )

        traj_with = Trajectory(
            obs=obs, actions=acts, last_obs=obs[-1],
            channels={"r": cd},
            noise_shift=np.random.randn(T, ACTION_DIM).astype(np.float32),
        )
        traj_without = Trajectory(
            obs=obs, actions=acts, last_obs=obs[-1],
            channels={"r": cd},
            noise_shift=None,
        )

        with pytest.raises(ValueError, match="mixed noise_shift"):
            PPOBuffer(
                [traj_with, traj_without], policy, torch.device("cpu"),
                reward_keys=("r",),
            )

    def test_shift_shape_mismatch_raises(self):
        """noise_shift with wrong shape must raise."""
        from baseline.framework.ppo.trajectory import Trajectory, ChannelData
        from baseline.framework.ppo.trainer import PPOBuffer

        policy = _make_policy()
        T = 8
        obs = np.random.randn(T, OBS_DIM).astype(np.float32)
        acts = np.tanh(np.random.randn(T, ACTION_DIM)).astype(np.float32)
        cd = ChannelData(
            reward=np.zeros(T, dtype=np.float32),
            is_terminated=False, actor_weight=1.0,
        )

        traj = Trajectory(
            obs=obs, actions=acts, last_obs=obs[-1],
            channels={"r": cd},
            noise_shift=np.random.randn(T, ACTION_DIM + 1).astype(np.float32),
        )

        with pytest.raises(ValueError, match="noise_shift shape"):
            PPOBuffer(
                [traj], policy, torch.device("cpu"),
                reward_keys=("r",),
            )


# ---------------------------------------------------------------------------
# 8. OU parameter management
# ---------------------------------------------------------------------------

class TestOUParamManagement:

    def test_ou_params_set_directly(self):
        """Setting OU params directly updates the policy's configuration."""
        policy = _make_policy()
        assert policy._noise_scale == 0.0
        assert policy._noise_tau_steps == 0.0

        # OU params are set directly (old ExplorationSpec fields removed).
        policy._noise_tau_steps = 15.0
        policy._noise_scale = 0.5
        policy._update_ou_params()

        assert policy._noise_tau_steps == 15.0
        assert policy._noise_scale == 0.5

    def test_ou_params_persist_when_unchanged(self):
        """OU params should remain as configured when not explicitly changed."""
        policy = _make_policy(noise_tau_steps=10.0, noise_scale=0.3)
        assert policy._noise_tau_steps == 10.0
        assert policy._noise_scale == 0.3

    def test_ou_enabled_after_direct_param_set(self):
        """After enabling OU directly, _next_noise_shift returns non-None."""
        policy = _make_policy()
        assert policy._next_noise_shift() is None

        policy._noise_tau_steps = 10.0
        policy._noise_scale = 0.3
        policy._update_ou_params()
        policy.reset(seed=42)
        shift = policy._next_noise_shift()
        assert shift is not None
        assert shift.shape == (ACTION_DIM,)
