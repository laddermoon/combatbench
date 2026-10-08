"""SAC-owned collection contracts.

These types deliberately do not import the shared ``rollout`` package:
its package initializer currently loads PPO-specific sampling machinery.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


@dataclass(frozen=True)
class SACBehaviorSpec:
    """Per-agent behavior-policy specification for one collection job."""

    mode: str = "stochastic"
    explore_factor: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    require_extras: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("stochastic", "deterministic"):
            raise ValueError(
                f"SACBehaviorSpec.mode must be 'stochastic' or "
                f"'deterministic', got {self.mode!r}"
            )
        if not np.isfinite(float(self.explore_factor)):
            raise ValueError(
                f"SACBehaviorSpec.explore_factor must be finite, "
                f"got {self.explore_factor}"
            )

    @property
    def stochastic(self) -> bool:
        return self.mode == "stochastic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "explore_factor": float(self.explore_factor),
            "parameters": dict(self.parameters),
            "require_extras": bool(self.require_extras),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SACBehaviorSpec":
        return cls(
            mode=str(data.get("mode", "stochastic")),
            explore_factor=float(data.get("explore_factor", 0.0)),
            parameters=dict(data.get("parameters") or {}),
            require_extras=bool(data.get("require_extras", False)),
        )


@dataclass(frozen=True)
class SACFactSpec:
    """A named fact captured immediately before ``runtime.step()``.

    ``provider`` is a qualified class path. The provider is constructed as
    ``provider(**config)`` and called as ``provider.compute(accessor,
    agent_id)`` before each action step.
    """

    name: str
    agent_id: str
    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SACFactSpec":
        return cls(
            name=str(data["name"]),
            agent_id=str(data["agent_id"]),
            provider=str(data["provider"]),
            config=dict(data.get("config") or {}),
        )


@dataclass(frozen=True)
class SACJob:
    """One SAC collection job.

    ``behavior_*`` fields are SAC-native provenance/exploration controls;
    they intentionally do not reuse PPO ``SamplingSpec`` or
    ``SamplingContext``.
    """

    policy_a_bp: PolicyBlueprint
    policy_b_bp: PolicyBlueprint
    env_bp: EnvBlueprint
    seed: int
    episode_options: Dict[str, Any] = field(default_factory=dict)
    behavior_a: SACBehaviorSpec = field(default_factory=SACBehaviorSpec)
    behavior_b: SACBehaviorSpec = field(default_factory=SACBehaviorSpec)
    fact_specs: Tuple[SACFactSpec, ...] = ()
    run_id: str = ""
    collection_round: int = 0
    job_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def job_key(self) -> str:
        payload = {
            "policy_a": self.policy_a_bp.to_dict(),
            "policy_b": self.policy_b_bp.to_dict(),
            "env": self.env_bp.to_dict(),
            "seed": int(self.seed),
            "episode_options": dict(self.episode_options),
            "behavior_a": self.behavior_a.to_dict(),
            "behavior_b": self.behavior_b.to_dict(),
            "fact_specs": [spec.to_dict() for spec in self.fact_specs],
            "run_id": self.run_id,
            "collection_round": int(self.collection_round),
            "job_index": int(self.job_index),
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def create_rollouter(*args: Any, **kwargs: Any) -> Any:
    """Create the SAC-owned synchronous rollouter."""
    from .collection_rollouter import SACParallelRollouter

    return SACParallelRollouter(*args, **kwargs)


__all__ = [
    "SACBehaviorSpec",
    "SACFactSpec",
    "SACJob",
    "create_rollouter",
]
