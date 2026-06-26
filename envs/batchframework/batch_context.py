"""Batch simulation context — blackboard for vectorized plugin dispatch.

与旧 framework 的 context.py 对应，但面向批量（MJX）仿真：

核心区别：
    旧: SimContext 持有标量 episode_step / physics_step，标量 base_seed
    新: BatchSimContext 持有 (B,) env_episode_steps，(B,) base_seeds

    旧: ctx.request_termination(reason) → 整个 episode 停止
    新: ctx.request_termination(env_id, reason) → 单个 env 终止，其余继续

    旧: ctx.mutator grant/revoke 基于 hook 时机
    新: 同样的 grant/revoke，但 mutator 操作的是 (B, ...) 批量数据

    旧: accessor 始终可用，暴露 IDataAccessor 方法（标量返回）
    新: accessor 始终可用，暴露 IBatchDataAccessor 方法（(B, ...) 返回）

设计原则：
1. 所有数组第一维是 batch dim (B,)。
2. JAX 内部细节完全封装，插件写者只面对 numpy。
3. 终止是 per-env 的，不影响其他 env。
4. 权限模型与旧 framework 一致：mutator 在可写 hook 中授予，否则为 None。
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .backend import BaseBatchSimulator, IBatchDataAccessor, IBatchDataMutator


# ---------------------------------------------------------------------------
# Termination reasons (shared with single-env framework)
# ---------------------------------------------------------------------------
class TerminationReason:
    """常见的终止原因（与旧 framework 的 TerminationReason 保持一致）。"""
    TIMEOUT = "timeout"
    KO = "ko"
    FOUL = "foul"
    OUT_OF_BOUNDS = "out_of_bounds"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Accessor / Mutator strict sandbox (batched)
# ---------------------------------------------------------------------------
# 与旧 framework 的 _AccessorView / _MutatorView 设计一致：
# - 只暴露契约中声明的方法，其余属性不可达。
# - 不可变代理（__setattr__ 被阻止）。
# - 名字混淆存储底层 simulator，防止 ctx.accessor._simulator 等直达。

_BATCH_ACCESSOR_ALLOWED: frozenset[str] = frozenset({
    "get_batch_size",
    "get_static_data",
    "get_core_state",
    "get_derived_state",
    "get_sensor_data",
    "get_action",
    "get_observation",
    "get_broadcastview_image",
    "get_physical_frequency",
})

_BATCH_MUTATOR_ALLOWED: frozenset[str] = frozenset({
    "set_action",
    "set_core_state",
    "apply_external_force",
})


class _BatchAccessorView(IBatchDataAccessor):
    """Read-only sandbox proxy over a ``BaseBatchSimulator``.

    与旧 ``_AccessorView`` 的区别：转发的方法返回 (B, ...) 批量数组。
    """

    __slots__ = ("__sim",)

    def __init__(self, simulator: BaseBatchSimulator) -> None:
        object.__setattr__(self, "_BatchAccessorView__sim", simulator)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable (attempted to set {name!r})"
        )

    def __getattr__(self, name: str) -> Any:
        if name in _BATCH_ACCESSOR_ALLOWED:
            return getattr(self.__sim, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}. "
            f"Only {sorted(_BATCH_ACCESSOR_ALLOWED)} are reachable through ctx.accessor."
        )

    # --- 显式转发以满足 IBatchDataAccessor 抽象方法 ---

    def get_batch_size(self) -> int:
        return self.__sim.batch_size

    def get_static_data(self) -> Dict[str, Any]:
        return self.__sim.get_static_data()

    def get_core_state(self, history: bool = False) -> Dict[str, Any]:
        return self.__sim.get_core_state(history=history)

    def get_derived_state(
        self,
        fields: Optional[Sequence[str]] = None,
        history: bool = False,
    ) -> Dict[str, Any]:
        return self.__sim.get_derived_state(fields=fields, history=history)

    def get_sensor_data(self) -> Dict[str, Any]:
        return self.__sim.get_sensor_data()

    def get_action(self) -> Dict[str, Any]:
        return self.__sim.get_action()

    def get_observation(self) -> Dict[str, Any]:
        return self.__sim.get_observation()

    def get_broadcastview_image(
        self, env_ids: Optional[Sequence[int]] = None
    ) -> Any:
        return self.__sim.get_broadcastview_image(env_ids=env_ids)

    def get_physical_frequency(self) -> float:
        return self.__sim.get_physical_frequency()


class _BatchMutatorView(IBatchDataMutator):
    """Write-only sandbox proxy over a ``BaseBatchSimulator``.

    与旧 ``_MutatorView`` 的区别：转发的方法接受 (B, ...) 批量数组。
    """

    __slots__ = ("__sim",)

    def __init__(self, simulator: BaseBatchSimulator) -> None:
        object.__setattr__(self, "_BatchMutatorView__sim", simulator)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable (attempted to set {name!r})"
        )

    def __getattr__(self, name: str) -> Any:
        if name in _BATCH_MUTATOR_ALLOWED:
            return getattr(self.__sim, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}. "
            f"Only {sorted(_BATCH_MUTATOR_ALLOWED)} are reachable through ctx.mutator."
        )

    def set_action(self, action: Dict[str, Any]) -> None:
        self.__sim.set_action(action)

    def set_core_state(
        self,
        state: Dict[str, Any],
        env_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.__sim.set_core_state(state, env_ids=env_ids)

    def apply_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: Optional[np.ndarray] = None,
        robot_id: str = "robot_a",
    ) -> None:
        self.__sim.apply_external_force(
            body_name, force, torque=torque, robot_id=robot_id
        )


# ---------------------------------------------------------------------------
# Batch simulation context (blackboard)
# ---------------------------------------------------------------------------
class BatchSimContext:
    """批量仿真引擎的统一上下文（黑板模式）。

    与旧 ``SimContext`` 的对应关系：
        SimContext.episode_step (int)       → BatchSimContext.env_episode_steps ((B,) int)
        SimContext.physics_step (int)       → BatchSimContext.action_step (int, 全局)
        SimContext.base_seed (int)          → BatchSimContext.base_seeds ((B,) int)
        SimContext.request_termination(r)   → BatchSimContext.request_termination(env_id, r)
        SimContext.is_terminated (bool)     → BatchSimContext.is_any_terminated (bool)
        SimContext.metrics (Dict)           → BatchSimContext.metrics (Dict, 值可为 (B,) 数组)
        SimContext.events (List)            → BatchSimContext.events (List[(env_id, type, data)])

    新增字段：
        batch_size: int — batch 维度大小
        active_mask: (B,) bool — 哪些 env 仍在运行
        terminated_env_ids: List[int] — 本步终止的 env 索引
        termination_reasons: Dict[int, str] — per-env 终止原因
        reset_env_ids: List[int] — 本步重置的 env 索引
    """

    def __init__(self, simulator: BaseBatchSimulator):
        self._simulator = simulator
        self._batch_size: int = simulator.batch_size

        # --- 时序状态 ---
        # 全局 action step 计数器（所有 env 共享，因为 physical_step 同时推进所有 env）
        self.action_step: int = 0
        # per-env episode step 计数器（不同 env 可能在不同步终止/重置）
        self.env_episode_steps: np.ndarray = np.zeros(self._batch_size, dtype=np.int64)

        # --- per-env 运行状态 ---
        self.active_mask: np.ndarray = np.ones(self._batch_size, dtype=bool)

        # --- 本步终止/重置追踪 ---
        self.terminated_env_ids: List[int] = []
        self.termination_reasons: Dict[int, str] = {}
        self.reset_env_ids: List[int] = []

        # --- Episode 参数 ---
        self.base_seeds: Optional[np.ndarray] = None  # (B,) int
        self.episode_options: Dict[str, Any] = {}

        # --- 派生黑板 ---
        # metrics 值可为标量（batch 级）或 (B,) 数组（per-env）
        self.metrics: Dict[str, Any] = {}
        # events 为 (env_id, event_type, data) 三元组列表
        self.events: List[Tuple[int, str, Any]] = []

        # --- 沙盒视图 ---
        self.accessor: IBatchDataAccessor = _BatchAccessorView(simulator)
        self._mutator_view: _BatchMutatorView = _BatchMutatorView(simulator)
        self.mutator: Optional[IBatchDataMutator] = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # --- 终止控制 ---

    def request_termination(
        self, env_id: int, reason: str = TerminationReason.CUSTOM
    ) -> None:
        """请求终止指定 env。

        与旧 framework 的区别：终止是 per-env 的，不影响其他 env。
        多次对同一 env_id 调用不会重复添加。
        """
        if env_id not in self.terminated_env_ids:
            self.terminated_env_ids.append(env_id)
            self.termination_reasons[env_id] = reason
            self.active_mask[env_id] = False

    @property
    def is_any_terminated(self) -> bool:
        """是否有任何 env 在本步终止。"""
        return len(self.terminated_env_ids) > 0

    def is_env_terminated(self, env_id: int) -> bool:
        """指定 env 是否已终止。"""
        return env_id in self.terminated_env_ids

    @property
    def active_env_ids(self) -> Tuple[int, ...]:
        """当前仍在运行的 env 索引。"""
        return tuple(int(i) for i in np.where(self.active_mask)[0])

    # --- 事件记录 ---

    def add_event(
        self, env_id: int, event_type: str, data: Any = None
    ) -> None:
        """记录一个 per-env 事件。

        与旧 framework 的 ctx.events.append(event) 对应，
        但批量版需要标明 env_id。
        """
        self.events.append((env_id, event_type, data))

    # --- 清理 ---

    def clear_episode_state(self) -> None:
        """在 episode 开始前清理历史状态。

        所有权约定：``base_seeds`` 与 ``episode_options`` 的写入由
        ``BatchRuntime.reset`` 的 caller 负责；本方法把它们清回 None / 空 dict。
        """
        self.action_step = 0
        self.env_episode_steps[:] = 0
        self.active_mask[:] = True
        self.terminated_env_ids.clear()
        self.termination_reasons.clear()
        self.reset_env_ids.clear()
        self.metrics.clear()
        self.events.clear()
        self.episode_options.clear()
        self.base_seeds = None

    def clear_step_state(self) -> None:
        """清理 per-step 状态（每个 action step 开始前调用）。"""
        self.terminated_env_ids.clear()
        self.termination_reasons.clear()
        self.reset_env_ids.clear()
        self.events.clear()

    # --- 引擎控制权限 ---

    def _grant_mutator(self) -> None:
        self.mutator = self._mutator_view

    def _revoke_mutator(self) -> None:
        self.mutator = None


# ---------------------------------------------------------------------------
# Read-only snapshot (for observers / recorders)
# ---------------------------------------------------------------------------
class ReadOnlyBatchSimContext:
    """``BatchSimContext`` 的不可变快照，传给 observer / recorder。

    与旧 ``ReadOnlySimContext`` 对应，但字段适配批量语义。
    使用 dataclass(frozen=True) 保证不可变。
    """
    __slots__ = (
        "accessor", "batch_size", "action_step", "env_episode_steps",
        "active_mask", "terminated_env_ids", "termination_reasons",
        "reset_env_ids", "metrics", "events", "is_any_terminated",
        "base_seeds", "episode_options",
    )

    def __init__(self, ctx: BatchSimContext):
        self.accessor = ctx.accessor
        self.batch_size = ctx.batch_size
        self.action_step = ctx.action_step
        self.env_episode_steps = ctx.env_episode_steps.copy()
        self.active_mask = ctx.active_mask.copy()
        self.terminated_env_ids = tuple(ctx.terminated_env_ids)
        self.termination_reasons = dict(ctx.termination_reasons)
        self.reset_env_ids = tuple(ctx.reset_env_ids)
        self.metrics = MappingProxyType(dict(ctx.metrics))
        self.events = tuple(ctx.events)
        self.is_any_terminated = ctx.is_any_terminated
        self.base_seeds = (
            ctx.base_seeds.copy() if ctx.base_seeds is not None else None
        )
        self.episode_options = MappingProxyType(dict(ctx.episode_options))

    def is_env_terminated(self, env_id: int) -> bool:
        return env_id in self.terminated_env_ids

    @property
    def active_env_ids(self) -> Tuple[int, ...]:
        return tuple(int(i) for i in np.where(self.active_mask)[0])
