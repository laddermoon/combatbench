"""Canonical Policy ABC for the combatbench framework.

This module is the **single source of truth** for what counts as a
"policy" in this project. Anything plugged into :class:`EpisodeRunner`,
:class:`RoundRunner`, or :class:`ParallelRunner` must subclass
:class:`Policy` defined here.

Design note
-----------
An earlier iteration split the contract across a :mod:`typing.Protocol`
(structural duck-typed interface) *and* a separate combat-specific ABC
(:class:`BaseCombatPolicy`) living under ``policy/``. That split added
maintenance cost (two docstrings to keep in sync, subtle drift bugs) for
a flexibility the codebase did not actually exercise — every real policy
inherited the ABC anyway. The current design collapses both into this
single nominal ABC: simpler to reason about, easier to ``issubclass``
against in :func:`combatbench.policy.load_policy`, and a narrower
contract surface.

Contract
--------
Required:
    ``act(observation) -> action``
        Synchronous. ``observation`` is whatever the bound observer
        plugin's ``get_output()`` returned; ``action`` must be coercible
        to ``np.ndarray(dtype=float32)``. Returning ``None`` is NOT
        allowed — return an explicit action.

Optional (runners detect via ``hasattr``):
    ``reset(seed: Optional[int] = None) -> None``
        Called once per episode before the first ``act``. ``seed`` is a
        deterministic per-policy child seed derived from the runner's
        ``base_seed`` via :class:`numpy.random.SeedSequence` so policies
        with their own RNGs can stay reproducible. Default: no-op.
    ``act_with_extras(observation) -> (action, extras: dict)``
        Used when :attr:`RolloutConfig.store_extras` is True — lets
        on-policy RL persist log-probs / value estimates per step.
    ``close() -> None``
        Release resources. Runners never call this automatically; caller
        owns policy lifecycle. :meth:`EpisodeRunner.close` invokes it as
        a convenience.

No ``__init__`` contract
------------------------
This ABC intentionally does **not** define ``__init__``. Subclasses are
free to design their constructors however they want (load checkpoints,
take hyperparameters, wire spaces — whatever). The :func:`load_policy`
loader just calls ``cls(**kwargs)`` with parsed query-string arguments;
subclasses that want to participate should accept ``**kwargs`` so
unknown parameters don't crash construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


__all__ = ["Policy", "coerce_action", "call_policy"]


class Policy(ABC):
    """Abstract base class for all combatbench policies. See module docstring."""

    @abstractmethod
    def act(self, observation: Any) -> np.ndarray:
        """Compute an action for the given observation.

        Parameters
        ----------
        observation:
            Whatever the bound observer plugin's ``get_output()`` returned.
            Usually a 1D float32 feature vector; policies needing richer
            inputs should declare a custom observer plugin and bind it via
            :class:`ObserverBinding`.

        Returns
        -------
        action:
            Array-like coercible to ``np.ndarray(dtype=float32)`` matching
            the environment's expected action dimension. Returning ``None``
            is NOT allowed.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Per-episode reset hook. Default: no-op.

        Override when the policy holds RNG or recurrent state. ``seed`` is
        a deterministic per-policy child seed; use it to reseed stochastic
        components so rollouts are reproducible from the runner's
        ``base_seed``.
        """
        return None


# ---------------------------------------------------------------------------
# Helpers used by runners to invoke policies according to the contract.
# ---------------------------------------------------------------------------
def coerce_action(action: Any) -> np.ndarray:
    """Normalize a policy's action to a ``float32`` ndarray.

    Accepted inputs:
      * :class:`numpy.ndarray` — returned as ``float32`` without copy
        when it's already ``float32``.
      * Anything ``np.asarray`` can convert (Python lists, tuples, torch
        tensors via ``__array__``, etc.).

    Raises
    ------
    TypeError
        If ``action`` is ``None`` — disallowed at the contract layer to
        avoid the "hold previous action" ambiguity. Policies that want a
        stand-still / passthrough behaviour must implement it explicitly.
    """
    if action is None:
        raise TypeError(
            "Policy.act returned None. Returning None is not part of the "
            "Policy contract; return an explicit np.ndarray action."
        )
    if isinstance(action, np.ndarray):
        return action.astype(np.float32, copy=False)
    return np.asarray(action, dtype=np.float32)


def call_policy(
    policy: Policy,
    observation: Any,
    *,
    want_extras: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Invoke ``policy`` and return ``(action_ndarray, extras_dict)``.

    When ``want_extras`` is ``True`` and the policy implements
    :meth:`Policy.act_with_extras`, that method is used so on-policy RL
    can persist log-probs / values per step. Otherwise the plain ``act``
    path is taken and ``extras`` is an empty dict.

    Raises
    ------
    TypeError
        If ``act_with_extras`` returns something that isn't a
        ``(action, dict)`` 2-tuple. Programmer error in the policy
        implementation — surfaced loudly rather than papered over.
    """
    action: Any
    extras: Dict[str, Any] = {}
    if want_extras and hasattr(policy, "act_with_extras"):
        result = policy.act_with_extras(observation)  # type: ignore[attr-defined]
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError(
                f"Policy.act_with_extras must return (action, extras_dict); "
                f"got {type(result).__name__}"
            )
        action, extras = result
        if not isinstance(extras, dict):
            raise TypeError(
                f"Policy.act_with_extras extras must be a dict; "
                f"got {type(extras).__name__}"
            )
    else:
        action = policy.act(observation)
    return coerce_action(action), extras
