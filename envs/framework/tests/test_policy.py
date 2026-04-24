"""Tests for the canonical Policy ABC in :mod:`envs.framework.policy`."""
from __future__ import annotations

import numpy as np
import pytest

from envs.framework.policy import Policy, call_policy, coerce_action


class _MinimalPolicy(Policy):
    """Minimal Policy subclass: just overrides ``act``."""

    def act(self, observation):
        return np.ones(3, dtype=np.float32)


class _RichPolicy(Policy):
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


class TestPolicyABC:
    def test_cannot_instantiate_without_overriding_act(self):
        """Policy is an ABC with ``act`` marked abstract; instantiating a
        subclass that does not override ``act`` must fail."""

        class Incomplete(Policy):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_subclass_with_act_is_instance(self):
        assert isinstance(_MinimalPolicy(), Policy)

    def test_non_subclass_is_not_instance(self):
        """Ducks no longer quack — scheme B is nominal, not structural."""

        class LooksLikePolicy:
            def act(self, obs):
                return np.zeros(3, dtype=np.float32)

        assert not isinstance(LooksLikePolicy(), Policy)

    def test_default_reset_accepts_seed_and_returns_none(self):
        """The ABC provides a default no-op ``reset(seed=None)``; subclasses
        that don't hold state can rely on it."""
        p = _MinimalPolicy()
        assert p.reset() is None
        assert p.reset(123) is None
        assert p.reset(seed=42) is None

    def test_no_init_contract(self):
        """The ABC intentionally does not define ``__init__``. Subclasses
        can take whatever constructor args they want."""

        class CustomInit(Policy):
            def __init__(self, scale, *, seed):
                self.scale = scale
                self.seed = seed

            def act(self, observation):
                return np.full(2, self.scale, dtype=np.float32)

        p = CustomInit(1.5, seed=7)
        assert p.scale == 1.5
        assert p.seed == 7
        np.testing.assert_array_equal(p.act(None), [1.5, 1.5])


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
    """``call_policy`` is intentionally duck-typed — it only cares that the
    object has ``act`` / ``act_with_extras``. The runner boundary
    (``EpisodeRunner._validate_policies``) enforces the nominal Policy ABC."""

    def test_minimal_policy_no_extras(self):
        action, extras = call_policy(_MinimalPolicy(), None, want_extras=False)
        assert action.dtype == np.float32
        assert extras == {}

    def test_minimal_policy_asked_for_extras_falls_back(self):
        """``want_extras=True`` on a policy without ``act_with_extras`` just
        uses plain ``act`` and returns an empty extras dict — not an error."""
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
        class Bad(Policy):
            def act(self, obs):
                return np.zeros(3)

            def act_with_extras(self, obs):
                return np.zeros(3)  # not a tuple

        with pytest.raises(TypeError, match="must return .action, extras_dict."):
            call_policy(Bad(), None, want_extras=True)

    def test_act_with_extras_bad_extras_type_rejected(self):
        class Bad(Policy):
            def act(self, obs):
                return np.zeros(3)

            def act_with_extras(self, obs):
                return np.zeros(3), [1, 2]  # list, not dict

        with pytest.raises(TypeError, match="extras must be a dict"):
            call_policy(Bad(), None, want_extras=True)


class TestBuiltinPolicies:
    """Sanity checks that the shipped reference policies obey the ABC."""

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
        assert a0.dtype == np.float32
        assert a0.shape == (21,)

        # Reset with a fresh seed produces a different action sequence.
        p.reset(seed=2)
        a1 = p.act(None)
        # Reset back to the same seed must reproduce the same action.
        p.reset(seed=2)
        a2 = p.act(None)
        np.testing.assert_array_equal(a1, a2)
        assert not np.array_equal(a0, a1)

    def test_random_policy_accepts_unknown_kwargs(self):
        """Subclasses that accept ``**kwargs`` stay forgiving against
        load_policy's query-string parameters that don't apply."""
        import sys
        from pathlib import Path

        combatbench_root = Path(__file__).resolve().parents[3]
        if str(combatbench_root) not in sys.path:
            sys.path.insert(0, str(combatbench_root))
        from policy.random.policy import RandomCombatPolicy

        # Extra junk kwargs must not crash construction.
        p = RandomCombatPolicy(scale=0.5, seed=1, model_path="/tmp/irrelevant")
        assert p.scale == 0.5
