"""Tests for the canonical Policy contract in envs.framework.policy."""
from __future__ import annotations

import numpy as np
import pytest

from envs.framework.policy import Policy, call_policy, coerce_action


class _MinimalPolicy:
    """Minimal duck-typed Policy: just an ``act`` method."""

    def act(self, observation):
        return np.ones(3, dtype=np.float32)


class _RichPolicy:
    """Policy with all optional hooks implemented."""

    def __init__(self):
        self.reset_seeds = []
        self.closed = False

    def act(self, observation):
        return np.array([1.0, 2.0, 3.0])

    def act_with_extras(self, observation):
        return self.act(observation), {"logprob": -0.5, "value": 1.2}

    def reset(self, seed=None):
        self.reset_seeds.append(seed)

    def close(self):
        self.closed = True


class TestPolicyProtocol:
    def test_minimal_policy_is_protocol_instance(self):
        """Anything with an ``act`` method satisfies the Protocol."""
        assert isinstance(_MinimalPolicy(), Policy)

    def test_rich_policy_is_protocol_instance(self):
        assert isinstance(_RichPolicy(), Policy)

    def test_no_act_method_fails_isinstance(self):
        class NotAPolicy:
            def step(self, obs):  # wrong name
                return np.zeros(3)

        assert not isinstance(NotAPolicy(), Policy)


class TestCoerceAction:
    def test_float32_ndarray_no_copy(self):
        a = np.ones(4, dtype=np.float32)
        out = coerce_action(a)
        assert out is a  # astype(copy=False) returns same object

    def test_float64_ndarray_converted(self):
        a = np.ones(4, dtype=np.float64)
        out = coerce_action(a)
        assert out.dtype == np.float32

    def test_list_converted(self):
        out = coerce_action([1.0, 2.0, 3.0])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_none_rejected(self):
        with pytest.raises(TypeError, match="None"):
            coerce_action(None)


class TestCallPolicy:
    def test_minimal_policy_no_extras(self):
        action, extras = call_policy(_MinimalPolicy(), None, want_extras=False)
        assert action.dtype == np.float32
        assert extras == {}

    def test_minimal_policy_asked_for_extras_falls_back(self):
        """want_extras=True on a policy without act_with_extras just uses
        plain act and returns an empty extras dict — not an error."""
        action, extras = call_policy(_MinimalPolicy(), None, want_extras=True)
        assert action.dtype == np.float32
        assert extras == {}

    def test_rich_policy_with_extras(self):
        action, extras = call_policy(_RichPolicy(), None, want_extras=True)
        assert action.dtype == np.float32
        assert extras == {"logprob": -0.5, "value": 1.2}

    def test_rich_policy_without_extras_skips_act_with_extras(self):
        action, extras = call_policy(_RichPolicy(), None, want_extras=False)
        assert extras == {}

    def test_act_with_extras_bad_return_type_rejected(self):
        class Bad:
            def act(self, obs):
                return np.zeros(3)

            def act_with_extras(self, obs):
                return np.zeros(3)  # not a tuple

        with pytest.raises(TypeError, match="must return .action, extras_dict."):
            call_policy(Bad(), None, want_extras=True)

    def test_act_with_extras_bad_extras_type_rejected(self):
        class Bad:
            def act(self, obs):
                return np.zeros(3)

            def act_with_extras(self, obs):
                return np.zeros(3), [1, 2]  # list, not dict

        with pytest.raises(TypeError, match="extras must be a dict"):
            call_policy(Bad(), None, want_extras=True)


class TestBaseCombatPolicyConformance:
    """The ABC in :mod:`combatbench.policy` must satisfy the canonical
    Protocol. Regression guard against the two classes drifting apart."""

    def test_basecombatpolicy_subclass_satisfies_protocol(self):
        import sys
        from pathlib import Path

        # Ensure ``policy`` package resolves (repo-root import style).
        combatbench_root = Path(__file__).resolve().parents[3]
        if str(combatbench_root) not in sys.path:
            sys.path.insert(0, str(combatbench_root))
        from policy.base import BaseCombatPolicy

        class _Concrete(BaseCombatPolicy):
            def act(self, observation):
                return np.zeros(self.ACTION_DIM, dtype=np.float32)

        p = _Concrete()
        assert isinstance(p, Policy)
        # reset takes a seed kwarg — framework calls reset(seed).
        p.reset(123)
        p.reset()
        p.reset(seed=42)

    def test_noopaction_policy_conforms(self):
        import sys
        from pathlib import Path

        combatbench_root = Path(__file__).resolve().parents[3]
        if str(combatbench_root) not in sys.path:
            sys.path.insert(0, str(combatbench_root))
        from policy.noopaction.policy import NoOpActionPolicy

        p = NoOpActionPolicy()
        assert isinstance(p, Policy)
        action = p.act(None)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        # NoOp must be an explicit zero action — no None sentinel.
        np.testing.assert_array_equal(action, np.zeros(21, dtype=np.float32))

    def test_random_policy_conforms_and_reset_reseeds(self):
        import sys
        from pathlib import Path

        combatbench_root = Path(__file__).resolve().parents[3]
        if str(combatbench_root) not in sys.path:
            sys.path.insert(0, str(combatbench_root))
        from policy.random.policy import RandomCombatPolicy

        p = RandomCombatPolicy(seed=1)
        assert isinstance(p, Policy)
        a0 = p.act(None)
        # Reset with a fresh seed produces a different action sequence.
        p.reset(seed=2)
        a1 = p.act(None)
        # Reset back to the same seed must reproduce the same action.
        p.reset(seed=2)
        a2 = p.act(None)
        np.testing.assert_array_equal(a1, a2)
        assert not np.array_equal(a0, a1)
