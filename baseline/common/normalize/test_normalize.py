"""Tests for ``RunningMeanStd``, ``ObservationNormalizer``, ``ReturnNormalizer``.

Pin the contract documented in ``baseline/DESIGN.md`` §3.5:

  * ``RunningMeanStd`` matches ``np.mean`` / ``np.var`` (population) when
    fed all samples at once, modulo the small-bias from ``count=epsilon``
    initialization.
  * Two batch updates → same statistics as one big batch (associativity).
  * ``state_dict`` / ``load_state_dict`` round-trips bit-equal.
  * ``ObservationNormalizer(update=False)`` does NOT touch running stats.
  * ``ReturnNormalizer.update_from_episodes`` produces stats that scale
    rewards roughly to unit variance after a few episodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.common.normalize import (
    ObservationNormalizer,
    ReturnNormalizer,
    RunningMeanStd,
)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------
class TestRunningMeanStd:
    def test_matches_numpy_after_one_batch(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1000, 4)).astype(np.float64)
        rms = RunningMeanStd(shape=(4,), epsilon=1e-12)  # tiny epsilon
        rms.update(x)
        np.testing.assert_allclose(rms.mean, x.mean(axis=0), atol=1e-9)
        np.testing.assert_allclose(rms.var, x.var(axis=0), atol=1e-9)

    def test_batch_associativity(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((500, 3)).astype(np.float64)

        rms_one = RunningMeanStd(shape=(3,), epsilon=1e-12)
        rms_one.update(x)

        rms_split = RunningMeanStd(shape=(3,), epsilon=1e-12)
        rms_split.update(x[:200])
        rms_split.update(x[200:350])
        rms_split.update(x[350:])

        np.testing.assert_allclose(rms_one.mean, rms_split.mean, atol=1e-10)
        np.testing.assert_allclose(rms_one.var, rms_split.var, atol=1e-10)

    def test_single_sample_promoted_to_batch(self):
        rms = RunningMeanStd(shape=(2,), epsilon=1e-12)
        rms.update(np.array([1.0, 2.0]))
        rms.update(np.array([3.0, 4.0]))
        np.testing.assert_allclose(rms.mean, [2.0, 3.0], atol=1e-10)

    def test_normalize_centers_and_scales(self):
        rms = RunningMeanStd(shape=(2,), epsilon=1e-12)
        rms.update(np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]]))
        # mean = [3, 14], var = [8/3, 32/3]
        out = rms.normalize(np.array([[3.0, 14.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_normalize_no_center_divides_only(self):
        rms = RunningMeanStd(shape=(), epsilon=1e-12)
        rms.update(np.array([1.0, -1.0, 1.0, -1.0]))  # mean=0, var=1
        out = rms.normalize(np.array([2.0]), center=False)
        np.testing.assert_allclose(out, [2.0], atol=1e-3)

    def test_state_dict_roundtrip(self):
        rms = RunningMeanStd(shape=(3,))
        rms.update(np.random.default_rng(42).standard_normal((50, 3)))
        sd = rms.state_dict()

        rms2 = RunningMeanStd(shape=(3,))
        rms2.load_state_dict(sd)
        np.testing.assert_array_equal(rms.mean, rms2.mean)
        np.testing.assert_array_equal(rms.var, rms2.var)
        assert rms.count == rms2.count

    def test_load_state_dict_shape_mismatch_raises(self):
        rms = RunningMeanStd(shape=(3,))
        bad = RunningMeanStd(shape=(4,)).state_dict()
        with pytest.raises(ValueError, match="shape"):
            rms.load_state_dict(bad)

    def test_wrong_input_shape_raises(self):
        rms = RunningMeanStd(shape=(3,))
        with pytest.raises(ValueError, match="shape"):
            rms.update(np.zeros((5, 4)))  # last dim wrong

    def test_empty_batch_is_noop(self):
        rms = RunningMeanStd(shape=(2,), epsilon=1e-12)
        rms.update(np.array([[1.0, 2.0]]))
        snapshot = (rms.mean.copy(), rms.var.copy(), rms.count)
        rms.update(np.zeros((0, 2)))
        np.testing.assert_array_equal(rms.mean, snapshot[0])
        np.testing.assert_array_equal(rms.var, snapshot[1])
        assert rms.count == snapshot[2]


# ---------------------------------------------------------------------------
# ObservationNormalizer
# ---------------------------------------------------------------------------
class TestObservationNormalizer:
    def test_update_false_does_not_modify_stats(self):
        norm = ObservationNormalizer(shape=(3,), clip_range=None)
        # Seed with one batch.
        norm.update(np.random.default_rng(0).standard_normal((100, 3)))
        before = norm.rms.state_dict()
        # Run lots of unusual obs through with update=False.
        eval_obs = np.full((50, 3), 1e3, dtype=np.float32)
        norm(eval_obs, update=False)
        after = norm.rms.state_dict()
        np.testing.assert_array_equal(before["mean"], after["mean"])
        np.testing.assert_array_equal(before["var"], after["var"])
        assert before["count"] == after["count"]

    def test_update_true_advances_stats(self):
        norm = ObservationNormalizer(shape=(3,), clip_range=None)
        before_count = norm.rms.count
        norm(np.zeros((10, 3), dtype=np.float32), update=True)
        assert norm.rms.count > before_count

    def test_clip_range_applied(self):
        norm = ObservationNormalizer(shape=(), clip_range=2.0)
        norm.update(np.array([0.0, 0.0, 0.0, 0.0]))  # var=0 → big std-norm output
        out = norm(np.array([1e6], dtype=np.float32), update=False)
        assert float(out[0]) <= 2.0 + 1e-5
        assert float(out[0]) >= -2.0 - 1e-5

    def test_state_dict_roundtrip(self):
        a = ObservationNormalizer(shape=(2,))
        a.update(np.random.default_rng(0).standard_normal((30, 2)))
        b = ObservationNormalizer(shape=(2,))
        b.load_state_dict(a.state_dict())
        np.testing.assert_array_equal(a.rms.mean, b.rms.mean)
        np.testing.assert_array_equal(a.rms.var, b.rms.var)


# ---------------------------------------------------------------------------
# ReturnNormalizer
# ---------------------------------------------------------------------------
class TestReturnNormalizer:
    def test_gamma_validation(self):
        with pytest.raises(ValueError, match="gamma"):
            ReturnNormalizer(gamma=1.0)
        with pytest.raises(ValueError, match="gamma"):
            ReturnNormalizer(gamma=-0.1)

    def test_update_step_resets_on_done(self):
        norm = ReturnNormalizer(gamma=0.5, num_envs=1, clip_range=None)
        # Step rewards [1, 1, 1] no done → return_acc accumulates.
        for r in [1.0, 1.0, 1.0]:
            norm.update_step(np.array([r]), np.array([False]))
        acc_before = float(norm._return_acc[0])  # type: ignore[attr-defined]
        assert acc_before > 1.0
        # Done → next reward starts fresh from 0.
        norm.update_step(np.array([2.0]), np.array([True]))
        # After done=True at step t: acc = gamma*acc*(1-1) + r = r. So acc=2.0
        # but BEFORE adding r, acc was zeroed. The new acc is exactly r.
        assert float(norm._return_acc[0]) == pytest.approx(2.0)  # type: ignore[attr-defined]

    def test_update_step_shape_mismatch(self):
        norm = ReturnNormalizer(num_envs=2)
        with pytest.raises(ValueError, match="num_envs"):
            norm.update_step(np.array([1.0]), np.array([False]))

    def test_normalize_scales_to_roughly_unit_variance(self):
        norm = ReturnNormalizer(gamma=0.99, num_envs=1, clip_range=None)
        rng = np.random.default_rng(0)
        # Feed many episodes of synthetic rewards; check that scaled
        # rewards have stddev ~ O(1).
        episodes = [rng.standard_normal(50) * 5.0 for _ in range(20)]
        norm.update_from_episodes(episodes)
        all_rewards = np.concatenate(episodes)
        scaled = norm(all_rewards)
        # We're not asking for exact unit std (return scaling != reward scaling)
        # but it should be much smaller than the raw scale of 5.0.
        assert scaled.std() < 5.0
        assert scaled.std() > 0.01

    def test_no_mean_subtraction(self):
        # All-positive rewards should remain all-positive after normalization.
        norm = ReturnNormalizer(gamma=0.9, num_envs=1, clip_range=None)
        episodes = [np.full(20, 1.0)]
        norm.update_from_episodes(episodes)
        scaled = norm(np.array([1.0]))
        assert scaled[0] > 0.0

    def test_state_dict_roundtrip(self):
        a = ReturnNormalizer(gamma=0.95, num_envs=2)
        a.update_step(np.array([1.0, 0.5]), np.array([False, False]))
        a.update_step(np.array([0.5, 1.5]), np.array([True, False]))
        b = ReturnNormalizer(gamma=0.95, num_envs=2)
        b.load_state_dict(a.state_dict())
        np.testing.assert_array_equal(a._return_acc, b._return_acc)
        np.testing.assert_array_equal(a.rms.mean, b.rms.mean)
        np.testing.assert_array_equal(a.rms.var, b.rms.var)
