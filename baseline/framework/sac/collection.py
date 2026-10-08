"""SAC-owned collection contracts.

These types deliberately do not import the shared ``rollout`` package:
its package initializer currently loads PPO-specific sampling machinery.
``P2-COLL-1`` expands this module into the full runner/recorder layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


@dataclass(frozen=True)
class SACBehaviorSpec:
    """Per-agent behavior-policy specification for one collection job."""

    stochastic: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SACJob:
    """One SAC collection job.

    ``behavior_*`` fields are SAC-native placeholders for exploration and
    provenance. They intentionally do not reuse PPO ``SamplingSpec``.
    """

    policy_a_bp: PolicyBlueprint
    policy_b_bp: PolicyBlueprint
    env_bp: EnvBlueprint
    seed: int
    episode_options: Dict[str, Any] = field(default_factory=dict)
    behavior_a: SACBehaviorSpec = field(default_factory=SACBehaviorSpec)
    behavior_b: SACBehaviorSpec = field(default_factory=SACBehaviorSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SACRollouterNotImplemented(RuntimeError):
    """Raised until P2-COLL-1 installs the SAC-native rollouter."""


def create_rollouter(*args: Any, **kwargs: Any) -> Any:
    """Fail explicitly instead of silently importing PPO rollout code."""
    raise SACRollouterNotImplemented(
        "SAC collection is pending P2-COLL-1; the legacy rollout path "
        "transitively imports PPO and is not part of the SAC contract"
    )


__all__ = [
    "SACBehaviorSpec",
    "SACJob",
    "SACRollouterNotImplemented",
    "create_rollouter",
]
