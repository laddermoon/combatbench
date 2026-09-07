"""Job — one rollout episode specification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Union

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint

#: Per-frame explore_factor: a constant float, or a callable
#: ``(obs, step) -> float`` that returns the value for each step.
#: Callables must be top-level functions to be picklable across
#: multiprocessing workers.
EfSpec = Union[float, Callable[[np.ndarray, int], float]]


@dataclass(frozen=True)
class Job:
    """One rollout episode: two policies + one env + per-policy exploration.

    Fields
    ------
    policy_a_bp / policy_b_bp:
        Deployable policy blueprints for robot_a and robot_b.
    env_bp:
        Environment blueprint (simulator + plugins + observers).
    seed:
        Episode base seed.
    episode_options:
        **Environment-only** configuration forwarded to
        ``simulator.reset(options=...)``.  This must be a plain,
        JSON-serializable dict — it is persisted in episode manifests.
        Do NOT put policy-related fields here.
    explore_factor_a / explore_factor_b:
        Exploration intensity for each policy.  Either a constant
        ``float`` (same value every step) or a callable
        ``(obs, step) -> float`` (per-frame).  Consumed by
        :class:`ExploratoryPolicy` which wraps the raw policy before
        passing it to :class:`EpisodeRunner`.
        Defaults to ``0.0`` (neutral).  Only used when ``stochastic=True``.
    stochastic:
        If True (default), policies are wrapped in
        :class:`ExploratoryPolicy` and ``sample()`` is called for
        stochastic rollout.  If False, policies are used directly as
        ``Policy`` and ``act()`` is called for deterministic evaluation.
    """

    policy_a_bp: PolicyBlueprint
    policy_b_bp: PolicyBlueprint
    env_bp: EnvBlueprint
    seed: int
    episode_options: Dict[str, Any] = field(default_factory=dict)
    explore_factor_a: EfSpec = 0.0
    explore_factor_b: EfSpec = 0.0
    stochastic: bool = True
