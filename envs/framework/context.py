from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple
from .backend import BaseSimulator, IDataAccessor, IDataMutator


class TerminationReason:
    """定义常见的终止原因"""
    TIMEOUT = "timeout"
    KO = "ko"
    FOUL = "foul"
    OUT_OF_BOUNDS = "out_of_bounds"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Accessor / Mutator strict sandbox
# ---------------------------------------------------------------------------
# Plugins are expected to consume ONLY what these proxies expose. Unlike the
# raw simulator (which inherits both IDataAccessor and IDataMutator and also
# carries backend-specific fields such as MuJoCo's ``model`` / ``data``),
# the proxies below enforce a strict allowlist:
#
# - ``_AccessorView`` forwards only read methods from the IDataAccessor
#   contract (plus ``get_physical_frequency``, which is a read-only query
#   belonging to BaseSimulator).
# - ``_MutatorView`` forwards only write methods from the IDataMutator
#   contract.
#
# Any other attribute access (including accidental typos, reaches for
# ``model`` / ``data`` / ``_robot_cache``, or attempts to call
# ``set_core_state`` via the accessor) raises ``AttributeError``. The proxies
# are also immutable—``__setattr__`` is blocked so a plugin cannot stash
# state on them.
#
# If an observer genuinely needs low-level physics data, the backend must
# expose it through ``get_static_data`` / ``get_derived_state`` per DATASPEC.
# This keeps the plugin API backend-agnostic and makes the sandbox real.

# Names forwarded by _AccessorView (allowlist).
_ACCESSOR_ALLOWED: frozenset[str] = frozenset({
    "get_static_data",
    "get_core_state",
    "get_derived_state",
    "get_sensor_data",
    "get_action",
    "get_broadcastview_image",
    "get_physical_frequency",
    "get_observation",
})

# Names forwarded by _MutatorView (allowlist).
_MUTATOR_ALLOWED: frozenset[str] = frozenset({
    "set_core_state",
    "set_action",
    "apply_external_force",
})


class _AccessorView(IDataAccessor):
    """Read-only sandbox proxy over a ``BaseSimulator``.

    Exposes only the methods listed in ``_ACCESSOR_ALLOWED``. Any other
    attribute access raises ``AttributeError``.

    Note: the underlying simulator is stored under a **name-mangled** slot
    (``__sim`` → ``_AccessorView__sim``) so that ``ctx.accessor._simulator``
    and similar reaches do NOT resolve. A determined developer can still
    retrieve it via the mangled name, but that is an explicit "I know I am
    breaking the sandbox" signal matching Python convention.
    """

    __slots__ = ("__sim",)

    def __init__(self, simulator: BaseSimulator) -> None:
        object.__setattr__(self, "_AccessorView__sim", simulator)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable (attempted to set {name!r})"
        )

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails. Forward
        # allowlisted method names to the wrapped simulator; reject anything
        # else (including attempts to reach ``_simulator`` / ``_sim``).
        if name in _ACCESSOR_ALLOWED:
            return getattr(self.__sim, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}. "
            f"Only {sorted(_ACCESSOR_ALLOWED)} are reachable through ctx.accessor."
        )

    # Explicit typed forwards so IDataAccessor's abstract API is satisfied
    # (and IDE / mypy see the proper signatures). Implementation simply
    # delegates to the wrapped simulator.
    def get_static_data(self) -> Dict[str, Any]:
        return self.__sim.get_static_data()

    def get_core_state(self) -> Dict[str, Any]:
        return self.__sim.get_core_state()

    def get_derived_state(self, fields=None) -> Dict[str, Any]:
        return self.__sim.get_derived_state(fields)

    def get_sensor_data(self) -> Dict[str, Any]:
        return self.__sim.get_sensor_data()

    def get_action(self) -> Dict[str, Any]:
        return self.__sim.get_action()

    def get_broadcastview_image(self) -> Any:
        return self.__sim.get_broadcastview_image()

    def get_physical_frequency(self) -> float:
        return self.__sim.get_physical_frequency()

    def get_observation(self) -> Dict[str, Any]:
        return self.__sim.get_observation()


class _MutatorView(IDataMutator):
    """Write-only sandbox proxy over a ``BaseSimulator``.

    Exposes only the methods listed in ``_MUTATOR_ALLOWED``. Non-mutator
    methods (reads, lifecycle) are unreachable. See ``_AccessorView`` for
    the name-mangling rationale.
    """

    __slots__ = ("__sim",)

    def __init__(self, simulator: BaseSimulator) -> None:
        object.__setattr__(self, "_MutatorView__sim", simulator)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable (attempted to set {name!r})"
        )

    def __getattr__(self, name: str) -> Any:
        if name in _MUTATOR_ALLOWED:
            return getattr(self.__sim, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}. "
            f"Only {sorted(_MUTATOR_ALLOWED)} are reachable through ctx.mutator."
        )

    def set_core_state(self, state: Dict[str, Any]) -> None:
        self.__sim.set_core_state(state)

    def set_action(self, action: Dict[str, Any]) -> None:
        self.__sim.set_action(action)

    def apply_external_force(self, *args: Any, **kwargs: Any) -> None:
        self.__sim.apply_external_force(*args, **kwargs)


class SimContext:
    """
    仿真引擎的统一上下文（黑板模式 Blackboard）。

    职责：
    1. 通过受控代理对外暴露数据访问器 (``accessor``) 与数据操作器 (``mutator``)。
       - ``accessor`` 始终可用，但只暴露 :class:`IDataAccessor` 契约中声明的读方法
         （加上 ``get_physical_frequency``）。任何 backend 特有字段（如 MuJoCo 的
         ``model`` / ``data`` / ``_robot_cache``）都**不可达**——观察者必须通过
         ``get_static_data()`` / ``get_derived_state()`` 获取需要的数据，遵循
         ``envs/humanoid21/DATASPEC.md``。
       - ``mutator`` 默认为 ``None``。只有运行时在"可写"钩子（``on_pre_action_step``
         / ``on_pre_phy_step`` / ``on_post_phy_step``）前后会临时通过
         ``_grant_mutator`` 授予代理，钩子退出后立刻 ``_revoke_mutator``。
    2. 承载跨插件流转的派生指标 (metrics)、事件 (events) 和控制流信号。
    """

    def __init__(self, simulator: BaseSimulator):
        self._simulator = simulator

        # 内部时序状态
        self.episode_step: int = 0
        self.physics_step: int = 0

        # 本 episode 的 base seed（int）。由 ``EnvRuntime.reset`` 在调用
        # ``clear_episode_state`` 之后、``simulator.reset`` 之前写入，
        # 供 plugin / observer / recorder 的 on_pre_episode 读取并落到 manifest。
        # 详见 envs/framework/SEED.md 与 envs/framework/RESET.md。
        self.base_seed: Optional[int] = None

        # 本 episode 的 options（per-episode 可变参数：HP 延续、课程化扰动
        # 强度、对手快照 ID、初始姿态等）。由 ``EnvRuntime.reset`` 在
        # ``clear_episode_state`` 之后写入，对所有 plugin / observer / recorder
        # 的 on_pre_episode 与所有 on_post_* 钩子可见。详见 RESET.md §4。
        self.episode_options: Dict[str, Any] = {}

        # 派生黑板
        self.metrics: Dict[str, Any] = {}
        self.events: List[Any] = []
        self.termination_proposals: List[str] = []

        # 读视图：始终可用，但仅暴露 IDataAccessor 许可的方法。
        self.accessor: IDataAccessor = _AccessorView(simulator)

        # 写视图：预创建但默认不挂到 ctx.mutator；由运行时按钩子时机授予/撤销。
        self._mutator_view: _MutatorView = _MutatorView(simulator)
        self.mutator: Optional[IDataMutator] = None

    def request_termination(self, reason: str = TerminationReason.CUSTOM) -> None:
        """提出终止请求"""
        self.termination_proposals.append(reason)

    @property
    def is_terminated(self) -> bool:
        """判断是否已经收到终止请求（控制流语义，非 MDP 终止语义）。

        ⚠️  WARNING: This returns True for ANY termination proposal,
        including ``TIMEOUT`` (truncation).  This is correct for
        *control-flow* ("should the episode loop stop?"), but NOT
        for *RL semantics* ("is this a true MDP terminal state?").

        For RL training decisions (bootstrap, done flags, etc.) use
        ``EnvRuntime.get_termination_flags()`` or filter manually:
            terminated = any(r != TerminationReason.TIMEOUT
                             for r in ctx.termination_proposals)
        See ``EpisodeRecorder.on_post_episode`` for the correct pattern.
        """
        return len(self.termination_proposals) > 0

    def clear_episode_state(self) -> None:
        """在 Episode 开始前清理历史状态。

        所有权约定：``base_seed`` 与 ``episode_options`` 的写入由
        ``EnvRuntime.reset`` 的 caller 负责；本方法把它们清回 None / 空 dict
        以避免上一个 episode 的值意外泄漏。详见 RESET.md §3 / §7-G5。
        """
        self.episode_step = 0
        self.physics_step = 0
        self.metrics.clear()
        self.events.clear()
        self.termination_proposals.clear()
        self.episode_options.clear()
        self.base_seed = None

    # --- 引擎控制权限的辅助方法 ---
    def _grant_mutator(self) -> None:
        self.mutator = self._mutator_view

    def _revoke_mutator(self) -> None:
        self.mutator = None


@dataclass(frozen=True)
class ReadOnlySimContext:
    accessor: IDataAccessor
    episode_step: int
    physics_step: int
    metrics: Mapping[str, Any]
    events: Tuple[Any, ...]
    termination_proposals: Tuple[str, ...]
    is_terminated: bool
    base_seed: Optional[int] = None
    episode_options: Mapping[str, Any] = MappingProxyType({})

    @classmethod
    def from_sim_context(cls, ctx: SimContext) -> "ReadOnlySimContext":
        return cls(
            accessor=ctx.accessor,
            episode_step=ctx.episode_step,
            physics_step=ctx.physics_step,
            metrics=MappingProxyType(dict(ctx.metrics)),
            events=tuple(ctx.events),
            termination_proposals=tuple(ctx.termination_proposals),
            is_terminated=ctx.is_terminated,
            base_seed=ctx.base_seed,
            episode_options=MappingProxyType(dict(ctx.episode_options)),
        )
