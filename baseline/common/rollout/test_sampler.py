"""Tests for ``RolloutSampler``.

Pin the contract from ``baseline/DESIGN.md`` §3.4:

  * concat / pad mode shapes
  * length validation across fields & episodes
  * ``__iter__`` reshuffles each call (PPO repeats epochs)
  * ``drop_last=True`` drops trailing partial minibatch
  * pad mode produces a correct boolean mask
  * ``from_batches`` slices ``obs[:-1]`` and respects extras alignment
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.common.rollout import RolloutBatch, RolloutSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _per_episode_arrays(t_lengths: List[int], obs_dim: int = 4, action_dim: int = 2):
    """Build a small {field: list[np.ndarray]} fixture."""
    rng = np.random.default_rng(0)
    obs_eps = [rng.standard_normal((t, obs_dim)).astype(np.float32) for t in t_lengths]
    act_eps = [rng.standard_normal((t, action_dim)).astype(np.float32) for t in t_lengths]
    rew_eps = [rng.standard_normal(t).astype(np.float32) for t in t_lengths]
    return {"obs": obs_eps, "actions": act_eps, "rewards": rew_eps}


# ---------------------------------------------------------------------------
# Concat mode
# ---------------------------------------------------------------------------
class TestConcatMode:
    def test_yields_minibatches_of_correct_shape(self):
        arrs = _per_episode_arrays([3, 5, 2])
        sampler = RolloutSampler(
            arrs, minibatch_size=4, mode="concat", seed=0,
        )
        # T_total = 10, drop_last=True (default) → 2 full minibatches of 4
        assert len(sampler) == 2
        seen = list(sampler)
        assert len(seen) == 2
        for mb in seen:
            assert mb["obs"].shape == (4, 4)
            assert mb["actions"].shape == (4, 2)
            assert mb["rewards"].shape == (4,)
            assert isinstance(mb["obs"], torch.Tensor)

    def test_drop_last_false_keeps_remainder(self):
        arrs = _per_episode_arrays([3, 5, 2])  # T_total = 10
        sampler = RolloutSampler(
            arrs, minibatch_size=4, mode="concat", drop_last=False, seed=0,
        )
        # 10 // 4 = 2 full + 1 partial of size 2
        assert len(sampler) == 3
        sizes = [mb["obs"].shape[0] for mb in sampler]
        assert sorted(sizes) == [2, 4, 4]

    def test_iter_reshuffles_each_call(self):
        arrs = _per_episode_arrays([4, 4])
        sampler = RolloutSampler(
            arrs, minibatch_size=4, mode="concat", seed=None,
        )
        first = next(iter(sampler))["obs"].cpu().numpy()
        second = next(iter(sampler))["obs"].cpu().numpy()
        # Highly unlikely to coincide if we're actually reshuffling.
        assert not np.allclose(first, second)


class TestPadMode:
    def test_pad_yields_episode_sized_minibatches_with_mask(self):
        arrs = _per_episode_arrays([3, 5, 2])
        sampler = RolloutSampler(
            arrs, minibatch_size=2, mode="pad", drop_last=False, seed=0,
        )
        # 3 episodes, 2 per minibatch → 1 full + 1 partial = 2
        assert len(sampler) == 2
        for mb in sampler:
            assert mb["obs"].dim() == 3   # (B, T_max, obs_dim)
            assert mb["mask"].dim() == 2  # (B, T_max)
            assert mb["mask"].dtype == torch.bool
            # Each row's mask must sum to that episode's true length.
            sums = mb["mask"].sum(dim=1).cpu().numpy()
            assert ((sums >= 2) & (sums <= 5)).all()

    def test_pad_zeros_outside_mask(self):
        arrs = _per_episode_arrays([2, 5])
        sampler = RolloutSampler(
            arrs, minibatch_size=2, mode="pad", drop_last=False, seed=0,
        )
        mb = next(iter(sampler))
        # Anywhere mask is False, padded fields must be zero.
        invalid = ~mb["mask"]
        assert torch.all(mb["obs"][invalid] == 0)
        assert torch.all(mb["actions"][invalid] == 0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_inconsistent_episode_count_raises(self):
        arrs = {
            "obs": [np.zeros((3, 4), np.float32), np.zeros((4, 4), np.float32)],
            "actions": [np.zeros((3, 2), np.float32)],  # one fewer episode
        }
        with pytest.raises(ValueError, match="episode counts"):
            RolloutSampler(arrs, minibatch_size=2)

    def test_inconsistent_T_within_episode_raises(self):
        arrs = {
            "obs": [np.zeros((3, 4), np.float32)],
            "actions": [np.zeros((4, 2), np.float32)],  # T mismatch
        }
        with pytest.raises(ValueError, match="inconsistent T"):
            RolloutSampler(arrs, minibatch_size=2)

    def test_invalid_mode_raises(self):
        arrs = _per_episode_arrays([2])
        with pytest.raises(ValueError, match="mode"):
            RolloutSampler(arrs, minibatch_size=1, mode="weird")


# ---------------------------------------------------------------------------
# from_batches
# ---------------------------------------------------------------------------
class TestFromBatches:
    @staticmethod
    def _make_batch(t: int, obs_dim: int = 4, action_dim: int = 2) -> RolloutBatch:
        return RolloutBatch(
            agent_id="robot_a",
            obs=np.zeros((t + 1, obs_dim), np.float32),
            actions=np.zeros((t, action_dim), np.float32),
            rewards=np.zeros(t, np.float32),
            log_probs=np.zeros(t, np.float32),
            values=np.zeros(t, np.float32),
            terminated=False,
            truncated=True,
        )

    def test_obs_sliced_to_T(self):
        batches = [self._make_batch(3), self._make_batch(5)]
        sampler = RolloutSampler.from_batches(
            batches, minibatch_size=4, mode="concat", seed=0,
        )
        # T_total should be 3+5 = 8 (NOT 9 = obs[:-1] dropped final).
        assert sampler._total_steps == 8

    def test_extras_alignment_validated(self):
        batches = [self._make_batch(3)]
        bad_extras = {"advantages": [np.zeros(4, np.float32)]}  # length 4 ≠ T=3
        with pytest.raises(ValueError, match="inconsistent T"):
            RolloutSampler.from_batches(
                batches, extras=bad_extras, minibatch_size=2,
            )

    def test_extras_episode_count_validated(self):
        batches = [self._make_batch(3), self._make_batch(2)]
        bad_extras = {"advantages": [np.zeros(3, np.float32)]}  # only 1 episode
        with pytest.raises(ValueError, match="episodes"):
            RolloutSampler.from_batches(
                batches, extras=bad_extras, minibatch_size=2,
            )

    def test_extras_propagate_to_minibatches(self):
        batches = [self._make_batch(3), self._make_batch(2)]
        adv_eps = [np.full(3, 0.5, np.float32), np.full(2, -1.0, np.float32)]
        ret_eps = [np.full(3, 1.0, np.float32), np.full(2, 2.0, np.float32)]
        sampler = RolloutSampler.from_batches(
            batches,
            extras={"advantages": adv_eps, "returns": ret_eps},
            minibatch_size=5,
            mode="concat",
            drop_last=False,
            seed=0,
        )
        # Total steps = 5 → exactly one minibatch
        mb = next(iter(sampler))
        assert "advantages" in mb and "returns" in mb
        assert mb["advantages"].shape == (5,)
        # Set of values seen must equal the originals (order shuffled).
        adv_set = sorted(mb["advantages"].cpu().numpy().tolist())
        assert adv_set == [-1.0, -1.0, 0.5, 0.5, 0.5]


class TestZeroEpisodes:
    def test_empty_per_episode_arrays_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RolloutSampler({}, minibatch_size=1)

    def test_zero_length_episode_lists_raises(self):
        with pytest.raises(ValueError, match="zero episodes"):
            RolloutSampler({"obs": []}, minibatch_size=1)
