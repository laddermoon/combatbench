"""Unit tests for ``RolloutBatch``.

These pin the data contract documented in ``baseline/DESIGN.md`` §3.3:

  * ``len(obs) == len(actions) + 1 == len(rewards) + 1``
  * ``initial_obs == obs[0]``, ``final_obs == obs[-1]``
  * ``log_probs`` / ``values`` either ``None`` or length ``T``
  * ``terminated`` and ``truncated`` cannot both be True
"""
from __future__ import annotations

import numpy as np
import pytest

from baseline.common.rollout import RolloutBatch


def _make_batch(t: int = 3, obs_dim: int = 4, action_dim: int = 2, **overrides) -> RolloutBatch:
    defaults = dict(
        agent_id="robot_a",
        obs=np.arange((t + 1) * obs_dim, dtype=np.float32).reshape(t + 1, obs_dim),
        actions=np.zeros((t, action_dim), dtype=np.float32),
        rewards=np.ones(t, dtype=np.float32),
        terminated=False,
        truncated=True,
    )
    defaults.update(overrides)
    return RolloutBatch(**defaults)


class TestShapeInvariants:
    def test_well_formed_batch_passes_validation(self):
        b = _make_batch()
        b.validate()
        assert b.num_steps == 3
        np.testing.assert_array_equal(b.initial_obs, b.obs[0])
        np.testing.assert_array_equal(b.final_obs, b.obs[-1])

    def test_obs_must_be_t_plus_1(self):
        b = _make_batch()
        b.obs = b.obs[:-1]  # length T instead of T+1
        with pytest.raises(ValueError, match="actions length \\+ 1"):
            b.validate()

    def test_rewards_must_match_actions(self):
        b = _make_batch()
        b.rewards = np.ones(b.num_steps + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="rewards length"):
            b.validate()


class TestOptionalFields:
    def test_log_probs_length_must_match_actions(self):
        b = _make_batch()
        b.log_probs = np.zeros(b.num_steps + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="log_probs length"):
            b.validate()

    def test_values_length_must_match_actions(self):
        b = _make_batch()
        b.values = np.zeros(b.num_steps - 1, dtype=np.float32)
        with pytest.raises(ValueError, match="values length"):
            b.validate()

    def test_log_probs_and_values_optional_none(self):
        b = _make_batch()
        b.log_probs = None
        b.values = None
        b.validate()  # no error


class TestTerminationFlags:
    def test_terminated_xor_truncated(self):
        b = _make_batch(terminated=True, truncated=True)
        with pytest.raises(ValueError, match="cannot both be True"):
            b.validate()

    def test_neither_terminated_nor_truncated_is_allowed(self):
        # A mid-stream chunk (e.g. on-policy collection that hasn't ended)
        # should still be expressible. We don't forbid this case.
        b = _make_batch(terminated=False, truncated=False)
        b.validate()


class TestFinalObsAlignment:
    def test_final_obs_is_obs_after_last_action(self):
        # Construct an explicit small example so we can read off final_obs.
        obs = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)  # T=2 → T+1=3
        actions = np.zeros((2, 1), dtype=np.float32)
        rewards = np.zeros(2, dtype=np.float32)
        b = RolloutBatch(
            agent_id="a",
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=False,
            truncated=True,
        )
        b.validate()
        assert float(b.final_obs[0]) == 2.0
        assert float(b.initial_obs[0]) == 0.0
