"""Tests for ``_apply_discounted_damage_shaping``.

Pins the algebraic contract of the per-step net-damage credit
back-propagation: each step's reward is augmented by ``coef * (shaped[t]
- raw[t])`` where ``shaped[t] = sum_{k>=0} gamma^k * raw[t+k]`` and
``coef = r3_weight * r3_scale``. The original sparse-r3 signal is
preserved at hit frames; gradient appears only in the lead-up window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.rollout import RolloutBatch
from baseline.humanoid21.curriculum import _apply_discounted_damage_shaping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_traj(*, raw_r3, base_reward=None, info_extra=None):
    """Build a minimal RolloutBatch carrying per-step net damage in info."""
    raw_r3 = np.asarray(raw_r3, dtype=np.float32)
    T = raw_r3.shape[0]
    obs_dim = 1
    action_dim = 1
    if base_reward is None:
        base_reward = np.zeros(T, dtype=np.float32)
    base_reward = np.asarray(base_reward, dtype=np.float32)
    info = {"r3_per_step": raw_r3}
    if info_extra:
        info.update(info_extra)
    return RolloutBatch(
        agent_id="robot_a",
        obs=np.zeros((T + 1, obs_dim), dtype=np.float32),
        actions=np.zeros((T, action_dim), dtype=np.float32),
        rewards=base_reward.copy(),
        terminated=False,
        truncated=True,
        info=info,
    )


def _expected_shaped(raw, gamma):
    raw = np.asarray(raw, dtype=np.float64)
    T = raw.shape[0]
    shaped = np.empty(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = raw[t] + gamma * running
        shaped[t] = running
    return shaped


# ---------------------------------------------------------------------------
# Basic correctness — no shaping when disabled.
# ---------------------------------------------------------------------------
class TestDisabled:
    def test_no_op_when_gamma_zero(self):
        traj = _make_traj(raw_r3=[1.0, 0.0, 5.0])
        before = traj.rewards.copy()
        _apply_discounted_damage_shaping(
            [traj], gamma=0.0, r3_scale=0.05, r3_weight=1.0,
        )
        np.testing.assert_array_equal(traj.rewards, before)

    def test_no_op_when_weight_zero(self):
        traj = _make_traj(raw_r3=[1.0, 0.0, 5.0])
        before = traj.rewards.copy()
        _apply_discounted_damage_shaping(
            [traj], gamma=0.95, r3_scale=0.05, r3_weight=0.0,
        )
        np.testing.assert_array_equal(traj.rewards, before)

    def test_no_op_when_scale_zero(self):
        traj = _make_traj(raw_r3=[1.0, 0.0, 5.0])
        before = traj.rewards.copy()
        _apply_discounted_damage_shaping(
            [traj], gamma=0.95, r3_scale=0.0, r3_weight=1.0,
        )
        np.testing.assert_array_equal(traj.rewards, before)

    def test_no_op_when_no_r3_info(self):
        traj = _make_traj(raw_r3=[1.0, 2.0])
        # Strip the info key.
        traj.info = {}
        before = traj.rewards.copy()
        _apply_discounted_damage_shaping(
            [traj], gamma=0.95, r3_scale=0.05, r3_weight=1.0,
        )
        np.testing.assert_array_equal(traj.rewards, before)


# ---------------------------------------------------------------------------
# Algebraic correctness — discounted future sum.
# ---------------------------------------------------------------------------
class TestShapingAlgebra:
    def test_single_terminal_hit_propagates_geometrically(self):
        """One hit at the LAST step. Each preceding step gets gamma^k
        times the hit (k = distance from hit) added to its reward,
        scaled by ``r3_weight * r3_scale``."""
        T = 5
        raw = np.zeros(T, dtype=np.float32)
        raw[-1] = 10.0  # one hit at step T-1
        traj = _make_traj(raw_r3=raw)

        gamma, scale, weight = 0.9, 0.05, 1.0
        coef = weight * scale
        _apply_discounted_damage_shaping(
            [traj], gamma=gamma, r3_scale=scale, r3_weight=weight,
        )

        shaped = _expected_shaped(raw, gamma)
        # Delta added at each step t is coef * (shaped[t] - raw[t]).
        expected_delta = coef * (shaped - raw)
        # Base rewards were zero; resulting reward equals the delta.
        np.testing.assert_allclose(traj.rewards, expected_delta.astype(np.float32),
                                   rtol=1e-5, atol=1e-7)
        # Hit-frame reward delta is zero (raw == shaped at t=T-1).
        assert traj.rewards[-1] == pytest.approx(0.0, abs=1e-7)
        # Earlier steps get strictly increasing magnitudes as we approach
        # the hit (gamma^(T-1-t) * 10).
        for t in range(T - 1):
            k = (T - 1) - t
            assert traj.rewards[t] == pytest.approx(
                coef * (gamma ** k) * 10.0, rel=1e-5,
            )

    def test_two_hits_sum_linearly(self):
        """Linearity: shaped[t] is linear in raw, so two hits' contributions
        add. We construct a sequence with hits at steps 2 and 4 and check
        the shaped contribution at step 0 matches gamma^2 * d2 + gamma^4 * d4."""
        raw = np.array([0.0, 0.0, 7.0, 0.0, 3.0], dtype=np.float32)
        traj = _make_traj(raw_r3=raw)

        gamma, scale, weight = 0.8, 1.0, 1.0
        coef = weight * scale
        _apply_discounted_damage_shaping(
            [traj], gamma=gamma, r3_scale=scale, r3_weight=weight,
        )
        # Reward at step 0: shaped[0] - raw[0] = gamma^2 * 7 + gamma^4 * 3.
        expected_r0 = coef * (gamma ** 2 * 7.0 + gamma ** 4 * 3.0)
        assert traj.rewards[0] == pytest.approx(expected_r0, rel=1e-5)
        # Reward at step 4 (hit frame): delta is zero.
        assert traj.rewards[4] == pytest.approx(0.0, abs=1e-7)
        # Reward at step 3 (one step before hit): gamma * 3 * coef.
        assert traj.rewards[3] == pytest.approx(coef * gamma * 3.0, rel=1e-5)

    def test_base_reward_is_preserved(self):
        """Shaping ADDS to existing rewards, doesn't replace them."""
        T = 4
        raw = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)
        base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        traj = _make_traj(raw_r3=raw, base_reward=base)
        gamma, scale, weight = 0.9, 0.1, 1.0
        coef = weight * scale

        _apply_discounted_damage_shaping(
            [traj], gamma=gamma, r3_scale=scale, r3_weight=weight,
        )
        shaped = _expected_shaped(raw, gamma)
        expected = base + coef * (shaped - raw).astype(np.float32)
        np.testing.assert_allclose(traj.rewards, expected, rtol=1e-5, atol=1e-7)

    def test_weight_and_scale_multiply_into_delta(self):
        raw = np.array([0.0, 4.0], dtype=np.float32)
        traj = _make_traj(raw_r3=raw)
        _apply_discounted_damage_shaping(
            [traj], gamma=0.5, r3_scale=0.1, r3_weight=2.0,
        )
        # shaped[0] = 0 + 0.5 * 4 = 2; delta[0] = (2 - 0) * 0.1 * 2 = 0.4.
        # shaped[1] = 4;             delta[1] = 0
        np.testing.assert_allclose(traj.rewards, [0.4, 0.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Returned diagnostics.
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_diagnostics_match_shaped_sums(self):
        raw_a = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        raw_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        traj_a = _make_traj(raw_r3=raw_a)
        traj_b = _make_traj(raw_r3=raw_b)
        gamma, scale, weight = 0.9, 0.05, 1.0

        diag = _apply_discounted_damage_shaping(
            [traj_a, traj_b], gamma=gamma, r3_scale=scale, r3_weight=weight,
        )
        sa = _expected_shaped(raw_a, gamma)
        sb = _expected_shaped(raw_b, gamma)
        assert diag["raw_r3_mean"] == pytest.approx(
            np.mean([raw_a.sum(), raw_b.sum()]), rel=1e-5,
        )
        assert diag["shaped_r3_mean"] == pytest.approx(
            np.mean([sa.sum(), sb.sum()]), rel=1e-5,
        )
        assert diag["delta_sum_mean"] == pytest.approx(
            weight * scale * np.mean([(sa - raw_a).sum(), (sb - raw_b).sum()]),
            rel=1e-5,
        )

    def test_empty_input_returns_zeros(self):
        diag = _apply_discounted_damage_shaping(
            [], gamma=0.9, r3_scale=0.05, r3_weight=1.0,
        )
        assert diag == {"raw_r3_mean": 0.0, "shaped_r3_mean": 0.0,
                        "delta_sum_mean": 0.0}
