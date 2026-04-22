"""Canonical Policy contract for the combatbench framework.

This module is the **single source of truth** for what counts as a
"policy" in this project. Anything that plugs into :class:`EpisodeRunner`,
:class:`CombatRoundRunner`, or :class:`ParallelRunner` is expected to
satisfy the :class:`Policy` Protocol defined here.

The ABC base class :class:`combatbench.policy.BaseCombatPolicy` in the
sibling ``policy/`` package is a concrete implementation of this protocol
with extra ergonomics (observation/action spaces, default ``__repr__``,
kwargs passthrough); loaded policies are expected to inherit from it.
New callers that do NOT need the ABC ergonomics can implement the
Protocol directly — duck typing is supported.

Contract
--------
Required:
    ``act(observation) -> action``
        Synchronous. ``observation`` is whatever the bound observer
        plugin's ``get_output()`` returned; ``action`` must be coercible
        to ``np.ndarray(dtype=float32)``. Returning ``None`` is NOT
        allowed — the framework has no "hold previous action" semantic
        at this layer (write one explicitly if you need it, e.g. cache
        the last action in the policy instance).

Optional (duck-typed via ``hasattr``):
    ``reset(seed: Optional[int]) -> None``
        Called once per episode before the first ``act``. ``seed`` is a
        deterministic per-policy child seed derived from the runner's
        ``base_seed`` via :class:`numpy.random.SeedSequence` so policies
        with their own RNGs can stay reproducible.
    ``act_with_extras(observation) -> (action, extras: dict)``
        Used when :attr:`RolloutConfig.store_extras` is True — lets
        on-policy RL persist log-probs / value estimates per step.
    ``close() -> None``
        Release resources. Runners never call this automatically; caller
        owns policy lifecycle. :meth:`EpisodeRunner.close` does call it
        as a convenience.

Why a Protocol + a separate ABC, instead of one class?
------------------------------------------------------
The Protocol lets downstream research code (trivial lambdas, torch
modules, policies loaded from external packages) plug in without
inheriting anything. The ABC in ``policy/base.py`` is where we pile up
the shared boilerplate (space wiring, kwargs, repr, load_util
discovery). Both point at the same contract.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


__all__ = ["Policy", "coerce_action", "call_policy"]


@runtime_checkable
class Policy(Protocol):
    """Minimal duck-typed policy interface.

    See module docstring for the full contract. Structural typing via
    :func:`typing.runtime_checkable` means ``isinstance(obj, Policy)``
    succeeds iff ``obj`` has an ``act`` method — the optional hooks are
    detected by the runners via ``hasattr`` at call time.
    """

    def act(self, observation: Any) -> Any: ...


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
        If ``action`` is ``None`` — ``None`` is explicitly disallowed at
        this contract layer to avoid the "hold previous action" ambiguity
        that plagued the historical RoundRunner. Policies that want to
        no-op should return an explicit zero action.
    """
    if action is None:
        raise TypeError(
            "Policy.act returned None. The framework does not support a "
            "'hold previous action' sentinel; return an explicit action "
            "(e.g. np.zeros(action_dim, dtype=np.float32))."
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
        ``(action, dict)`` 2-tuple. This is a programmer error in the
        policy implementation and is surfaced loudly rather than papered
        over.
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
