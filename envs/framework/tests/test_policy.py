"""Tests for the canonical Policy ABC in :mod:`envs.framework.policy`."""
from __future__ import annotations

import numpy as np
import pytest

from envs.framework.policy import Policy


class _MinimalPolicy(Policy):
    """Minimal Policy subclass: just overrides ``act``."""

    def act(self, observation, *, want_extra: bool = False):
        return np.ones(3, dtype=np.float32), None


class _RichPolicy(Policy):
    """Policy with all optional hooks implemented."""

    def __init__(self):
        self.reset_seeds = []
        self.closed = False

    def act(self, observation, *, want_extra: bool = False):
        action = np.array([1.0, 2.0, 3.0])
        if want_extra:
            return action, {"logprob": -0.5, "value": 1.2}
        return action, None

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
            def act(self, obs, *, want_extra=False):
                return np.zeros(3, dtype=np.float32), None

        assert not isinstance(LooksLikePolicy(), Policy)

    def test_default_reset_accepts_seed_and_returns_none(self):
        """The ABC provides a default no-op ``reset(seed=None)``; subclasses
        that don't hold state can rely on it."""
        p = _MinimalPolicy()
        assert p.reset() is None
        assert p.reset(123) is None
        assert p.reset(seed=42) is None

    def test_act_returns_action_extra_tuple(self):
        """The canonical contract: ``act`` always returns ``(action, extra)``."""
        action, extra = _MinimalPolicy().act(None)
        assert isinstance(action, np.ndarray)
        assert extra is None

    def test_want_extra_forwarded(self):
        """``want_extra=True`` lets the policy attach its side-channel payload."""
        action, extra = _RichPolicy().act(None, want_extra=True)
        assert extra == {"logprob": -0.5, "value": 1.2}
        _, extra = _RichPolicy().act(None, want_extra=False)
        assert extra is None

    def test_no_init_contract(self):
        """The ABC intentionally does not define ``__init__``. Subclasses
        can take whatever constructor args they want."""

        class CustomInit(Policy):
            def __init__(self, scale, *, seed):
                self.scale = scale
                self.seed = seed

            def act(self, observation, *, want_extra: bool = False):
                return np.full(2, self.scale, dtype=np.float32), None

        p = CustomInit(1.5, seed=7)
        assert p.scale == 1.5
        assert p.seed == 7
        action, _ = p.act(None)
        np.testing.assert_array_equal(action, [1.5, 1.5])


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

        a0, _ = p.act(None)
        assert a0.dtype == np.float32
        assert a0.shape == (21,)

        # Reset with a fresh seed produces a different action sequence.
        p.reset(seed=2)
        a1, _ = p.act(None)
        # Reset back to the same seed must reproduce the same action.
        p.reset(seed=2)
        a2, _ = p.act(None)
        np.testing.assert_array_equal(a1, a2)
        assert not np.array_equal(a0, a1)

    def test_random_policy_accepts_unknown_kwargs(self):
        """Subclasses that accept ``**kwargs`` stay forgiving against
        blueprint config keys that don't apply."""
        import sys
        from pathlib import Path

        combatbench_root = Path(__file__).resolve().parents[3]
        if str(combatbench_root) not in sys.path:
            sys.path.insert(0, str(combatbench_root))
        from policy.random.policy import RandomCombatPolicy

        # Extra junk kwargs must not crash construction.
        p = RandomCombatPolicy(scale=0.5, seed=1, model_path="/tmp/irrelevant")
        assert p.scale == 0.5
