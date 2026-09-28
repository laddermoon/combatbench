"""Job — one rollout episode specification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint

#: Per-frame explore_factor: a constant float, or a callable
#: ``(obs, step) -> float`` that returns the value for each step.
#: Callables must be top-level functions to be picklable across
#: multiprocessing workers.
EfSpec = Union[float, Callable[[np.ndarray, int], float]]


@dataclass(frozen=True)
class ReferenceSpec:
    """历史策略动作加权参考 —— 动作空间集成，非参数 EMA。

    ``policies[i]`` 是第 ``i`` 个历史代的 policy blueprint
    （如 ``policy_exports/uNNNNN``），``weights[i]`` 是其权重。
    运行时由 rollout wrapper 对**当前 observation** 逐个确定性
    ``act()`` 求值并加权，得到逐帧 ``reference_action``。

    框架不维护跨 update 的 EMA 模型——哪个/哪些历史策略、权重多少，
    完全由实验侧在构造 Job 时决定。
    """

    policies: Tuple[PolicyBlueprint, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.policies) != len(self.weights):
            raise ValueError(
                f"ReferenceSpec: {len(self.policies)} policies but "
                f"{len(self.weights)} weights"
            )
        if not self.policies:
            raise ValueError("ReferenceSpec: policies must be non-empty")
        w = np.asarray(self.weights, dtype=np.float64)
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError(f"ReferenceSpec: invalid weights {self.weights}")
        s = float(w.sum())
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(
                f"ReferenceSpec: weights must sum to 1, got {s}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policies": [bp.to_dict() for bp in self.policies],
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceSpec":
        return cls(
            policies=tuple(
                PolicyBlueprint.from_dict(p) for p in d["policies"]
            ),
            weights=tuple(float(w) for w in d["weights"]),
        )


@dataclass(frozen=True)
class SamplingSpec:
    """一个 agent 在一个 episode 中的采样意图（spec 级，非逐帧值）。

    由实验构造放入 :class:`Job`，rollout wrapper 用它逐帧构造
    :class:`SamplingContext`。仅在 ``stochastic=True`` 时消费。

    Attributes:
        explore_factor: 标量或 ``(obs, step) -> float`` 逐帧调度器。
        reference: 历史策略加权参考；``None`` = 无参考机制。
        delta_factor: Δ→σ 标定系数 c（静态，随 ctx 逐帧记录）。
        delta_mix: 原尺度/Δ 尺度混合权重 λ ∈ [0,1]（静态）。
    """

    explore_factor: EfSpec = 0.0
    reference: Optional[ReferenceSpec] = None
    delta_factor: float = 0.0
    delta_mix: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.delta_mix) <= 1.0):
            raise ValueError(
                f"SamplingSpec: delta_mix must be in [0,1], got {self.delta_mix}"
            )
        if not np.isfinite(float(self.delta_factor)):
            raise ValueError(
                f"SamplingSpec: delta_factor must be finite, "
                f"got {self.delta_factor}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            # EfSpec callable 不能 JSON 序列化 —— 任务经 pickle 进 worker，
            # 这里仅做结构化打包，callable 原样携带。
            "explore_factor": self.explore_factor,
            "reference": self.reference.to_dict() if self.reference else None,
            "delta_factor": float(self.delta_factor),
            "delta_mix": float(self.delta_mix),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingSpec":
        ref = d.get("reference")
        return cls(
            explore_factor=d.get("explore_factor", 0.0),
            reference=ReferenceSpec.from_dict(ref) if ref else None,
            delta_factor=float(d.get("delta_factor", 0.0)),
            delta_mix=float(d.get("delta_mix", 0.0)),
        )


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
    sampling_a / sampling_b:
        Full per-agent sampling spec (explore_factor + reference +
        delta config).  Consumed by the sampling wrapper which wraps
        the raw policy before passing it to :class:`EpisodeRunner`.
        Defaults to a neutral ``SamplingSpec()``.  Only used when
        ``stochastic=True``.
    stochastic:
        If True (default), policies are wrapped in a sampling wrapper
        and ``sample()`` is called for stochastic rollout.  If False,
        policies are used directly as ``Policy`` and ``act()`` is
        called for deterministic evaluation.
    """

    policy_a_bp: PolicyBlueprint
    policy_b_bp: PolicyBlueprint
    env_bp: EnvBlueprint
    seed: int
    episode_options: Dict[str, Any] = field(default_factory=dict)
    sampling_a: SamplingSpec = field(default_factory=SamplingSpec)
    sampling_b: SamplingSpec = field(default_factory=SamplingSpec)
    stochastic: bool = True
