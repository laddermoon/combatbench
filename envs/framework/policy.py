"""Canonical Policy ABC for the combatbench framework.

This module is the **single source of truth** for what counts as a
"policy" in this project. Anything plugged into :class:`EpisodeRunner`,
:class:`RoundRunner`, or :class:`ParallelRunner` must subclass
:class:`Policy` defined here.

Contract
--------
Required:
    ``act(observation, want_extra: bool = False) -> (action, extra)``
        Synchronous. Always returns a 2-tuple ``(action, extra)``. When
        ``want_extra`` is False the policy may return ``extra=None`` and
        skip any work needed only for that payload. Returning ``None``
        for ``action`` is NOT allowed — return an explicit action.

Optional (runners detect via ``hasattr``):
    ``reset(seed: Optional[int] = None) -> None``
        Called once per episode before the first ``act``. ``seed`` is a
        deterministic per-policy child seed derived from the runner's
        ``base_seed`` via :class:`numpy.random.SeedSequence`. Policies
        that hold their own RNGs SHOULD use it for reproducibility, but
        the framework does not enforce this — see "Determinism" below.
        Default: no-op.
    ``close() -> None``
        Release resources. Runners never call this automatically; caller
        owns policy lifecycle. :meth:`EpisodeRunner.close` invokes it as
        a convenience.

Observation / action / extra types
----------------------------------
The framework places **no constraints** on the Python types flowing in
and out of :meth:`Policy.act`:

* ``observation`` is whatever the bound observer plugin's
  ``get_output()`` returned — a dict, a numpy array, a custom
  dataclass, anything. The policy and the observer agree on the schema;
  the runner is just a pipe.
* ``action`` is whatever the simulator's ``BaseSimulator.step`` accepts
  for that agent. Some sims want ``np.ndarray(float32)`` joint torques,
  others want dicts of high-level commands. Match the sim; the
  framework does not coerce.
* ``extra`` is fully policy-defined. Typical contents include log-prob
  / value estimates for on-policy RL, attention maps for debugging, or
  raw policy-network outputs. May be ``None`` when ``want_extra`` is
  False or when the policy has nothing to report.

Determinism vs. stochasticity
-----------------------------
Whether a policy is deterministic or stochastic — and, if stochastic,
how it is seeded — is **the policy's own responsibility**, managed
inside ``__init__`` (and optionally re-seeded inside :meth:`reset`).
The framework does not introspect or alter this. Concretely:

* A deterministic policy ignores ``reset(seed=...)`` (or accepts it and
  no-ops). Same observation in -> same action out.
* A stochastic policy owns its RNG (``np.random.Generator``, a torch
  ``Generator``, etc.), constructs it in ``__init__``, and re-seeds in
  :meth:`reset` when the runner-provided ``seed`` is not ``None``.
* Hybrid policies (e.g. exploration noise toggled by an ``eval`` flag)
  expose that toggle via constructor args — there is no framework-level
  ``eval()`` switch.

No ``__init__`` contract
------------------------
This ABC intentionally does **not** define ``__init__``. Subclasses are
free to design their constructors however they want (load checkpoints,
take hyperparameters, wire RNGs, decide deterministic-vs-stochastic
behaviour — whatever). The :func:`load_policy` loader just calls
``cls(**kwargs)`` with parsed query-string arguments; subclasses that
want to participate should accept ``**kwargs`` so unknown parameters
don't crash construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


__all__ = ["Policy"]


class Policy(ABC):
    """Abstract base class for all combatbench policies. See module docstring."""

    @abstractmethod
    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        """Compute an action for the given observation.

        Parameters
        ----------
        observation:
            Whatever the simulator's ``get_observation()``
            returned. The framework imposes no type constraint; the
            policy and the observer agree on the schema.
        want_extra:
            If True the runner wants the optional ``extra`` payload
            (e.g. log-prob / value estimates for on-policy RL). When
            False the policy may return ``extra=None`` and skip any
            work needed only for that payload.

        Returns
        -------
        action:
            Whatever the simulator's ``step`` accepts for this agent.
            Type/shape are simulator-defined; the framework does not
            coerce. Must not be ``None``.
        extra:
            Policy-defined auxiliary payload, or ``None``. Common
            choices: a dict of log-prob / value / entropy tensors for
            on-policy RL trainers.

        Stochasticity is the policy's responsibility — see the module
        docstring's "Determinism vs. stochasticity" section.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Per-episode reset hook. Default: no-op.

        Override when the policy holds RNG or recurrent state. ``seed``
        is a deterministic per-policy child seed derived from the
        runner's ``base_seed``; stochastic policies SHOULD reseed their
        internal RNG with it so rollouts stay reproducible. Determini-
        stic policies can ignore it. The framework neither inspects nor
        enforces what the policy does with this value.
        """
        return None