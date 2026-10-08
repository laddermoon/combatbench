from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple
from .backend import BaseSimulator, IDataAccessor, IDataMutator

AGENT_IDS: Tuple[str, ...] = ("robot_a", "robot_b")


class EventJournal(Sequence):
    """Append-only 事件日志 —— 一个 episode 内**只增不减**。

    生产者只能 ``append``；公开 API 不提供 pop/remove/clear——
    事件一经写入不可撤回、不可改写。回合边界的清空由框架统一执行：
    ``SimContext.clear_episode_state`` 调 ``_reset()``，插件与
    observer 不拥有清空权（也不应尝试——不存在这些方法，
    误删会在调用点立即抛 ``AttributeError``）。

    消费范式：journal 是 episode 级累计记录。"本步/自某时点以来的
    事件"由消费者游标差分得出——``since(mark)`` 返回 ``mark``
    之后追加的段。``epoch`` 每次 ``_reset`` +1：游标消费者用
    ``(epoch, len)`` 作游标即可区分"没有新事件"与"journal 被重置
    后重新增长"（后者应整段重取，而非从旧游标续读）。

    逃逸面说明：``__items`` 经名字混淆隐藏——与 ``_AccessorView``/
    ``_MutatorView`` 同款"显式 opt-out"约定，防误用而非防蓄意绕过。
    """

    __slots__ = ("__items", "_epoch")

    def __init__(self) -> None:
        self.__items: List[Any] = []
        self._epoch: int = 0

    # ---- 生产者 API：仅追加 ----
    def append(self, item: Any) -> None:
        self.__items.append(item)

    # ---- 消费者 API：读 + 游标 ----
    def since(self, mark: int) -> List[Any]:
        """``mark``（此前的 ``len()`` 快照）之后追加的事件段。"""
        return self.__items[mark:]

    @property
    def epoch(self) -> int:
        """Journal 代数：每次框架级 ``_reset`` 递增。"""
        return self._epoch

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self):
        return iter(self.__items)

    def __getitem__(self, index):
        return self.__items[index]

    def __repr__(self) -> str:
        return f"EventJournal(len={len(self.__items)}, epoch={self._epoch})"

    # ---- 框架私有：仅 SimContext.clear_episode_state 调用 ----
    def _reset(self) -> None:
        self.__items.clear()
        self._epoch += 1


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

    Lifecycle: the view is valid only while granted by
    ``SimContext._grant_mutator``; ``_revoke_mutator`` invalidates it, so
    a plugin that stashes ``ctx.mutator`` in a writable hook cannot use
    the stash later in a read-only hook (P-FW-9).
    """

    __slots__ = ("__sim", "__valid")

    def __init__(self, simulator: BaseSimulator) -> None:
        object.__setattr__(self, "_MutatorView__sim", simulator)
        object.__setattr__(self, "_MutatorView__valid", False)

    def _set_valid(self, valid: bool) -> None:
        object.__setattr__(self, "_MutatorView__valid", valid)

    def _assert_valid(self) -> None:
        if not self.__valid:
            raise RuntimeError(
                "_MutatorView revoked: mutator is only valid inside the "
                "hook that granted it — do not stash ctx.mutator"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable (attempted to set {name!r})"
        )

    def __getattr__(self, name: str) -> Any:
        if name in _MUTATOR_ALLOWED:
            self._assert_valid()
            return getattr(self.__sim, name)
        raise AttributeError(
            f"{type(self).__name__} does not expose {name!r}. "
            f"Only {sorted(_MUTATOR_ALLOWED)} are reachable through ctx.mutator."
        )

    def set_core_state(self, state: Dict[str, Any]) -> None:
        self._assert_valid()
        self.__sim.set_core_state(state)

    def set_action(self, action: Dict[str, Any]) -> None:
        self._assert_valid()
        self.__sim.set_action(action)

    def apply_external_force(self, *args: Any, **kwargs: Any) -> None:
        self._assert_valid()
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
        # 注意：这里刻意**不**把裸 simulator 挂到 ctx 上——沙箱边界要求
        # 插件/观察者只能经 ``accessor``（读白名单）与 ``mutator``（授予制）
        # 接触后端。诊断类工具需要后端内部状态时，由构建 runtime 的
        # harness 层持有引用（如 ``runtime.simulator``），不从 ctx 走。

        # 内部时序状态
        # episode_step: step() 调用计数——每个进入的 step 无条件 +1，
        # 不隐含"完整物理步"；终止帧（物理循环中途结束）也会递增。
        # physics_step: 实际执行的物理子步计数；某帧的 action 是否物理
        # 生效由该帧的 physics_step 增量判定（0 = 未生效的退化帧）。
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
        # 事件日志：append-only，一个 episode 内只增不减（见 EventJournal）。
        # 插件只能 append；"本步事件"由消费者游标差分得出，不由容器区分。
        self.events: EventJournal = EventJournal()
        self.agent_termination_proposals: Dict[str, List[str]] = {
            aid: [] for aid in AGENT_IDS
        }
        self.agent_terminated: Dict[str, bool] = {aid: False for aid in AGENT_IDS}

        # 读视图：始终可用，但仅暴露 IDataAccessor 许可的方法。
        self.accessor: IDataAccessor = _AccessorView(simulator)

        # 写视图：预创建但默认不挂到 ctx.mutator；由运行时按钩子时机授予/撤销。
        self._mutator_view: _MutatorView = _MutatorView(simulator)
        self.mutator: Optional[IDataMutator] = None

    def request_termination(
        self,
        reason: str = TerminationReason.CUSTOM,
        agent_id: Optional[str] = None,
    ) -> None:
        """提出终止请求。

        - ``agent_id=None`` (default): issues ``reason`` to ALL agents
          (robot_a AND robot_b). This is how timeout, out-of-bounds, etc.
          work — they terminate everyone at once.
        - ``agent_id="robot_a"``: only terminates robot_a. The episode
          continues for robot_b until it also terminates or a global
          termination is issued.
        """
        if agent_id is None:
            for aid in AGENT_IDS:
                self.agent_termination_proposals[aid].append(reason)
                self.agent_terminated[aid] = True
        else:
            if agent_id not in self.agent_termination_proposals:
                raise ValueError(
                    f"Unknown agent_id {agent_id!r}; expected one of {AGENT_IDS}"
                )
            self.agent_termination_proposals[agent_id].append(reason)
            self.agent_terminated[agent_id] = True

    def is_agent_terminated(self, agent_id: str) -> bool:
        """True if ``agent_id`` has any termination proposal."""
        return self.agent_terminated.get(agent_id, False)

    @property
    def all_agents_terminated(self) -> bool:
        """True if all agents have terminated. This is the episode-end condition."""
        return all(self.agent_terminated.values())

    def clear_episode_state(self) -> None:
        """在 Episode 开始前清理历史状态。

        所有权约定：``base_seed`` 与 ``episode_options`` 的写入由
        ``EnvRuntime.reset`` 的 caller 负责；本方法把它们清回 None / 空 dict
        以避免上一个 episode 的值意外泄漏。详见 RESET.md §3 / §7-G5。
        """
        self.episode_step = 0
        self.physics_step = 0
        self.metrics.clear()
        self.events._reset()
        for aid in AGENT_IDS:
            self.agent_termination_proposals[aid].clear()
            self.agent_terminated[aid] = False
        self.episode_options.clear()
        self.base_seed = None

    # --- 引擎控制权限的辅助方法 ---
    def _grant_mutator(self) -> None:
        self._mutator_view._set_valid(True)
        self.mutator = self._mutator_view

    def _revoke_mutator(self) -> None:
        self._mutator_view._set_valid(False)
        self.mutator = None


@dataclass(frozen=True)
class ReadOnlySimContext:
    accessor: IDataAccessor
    episode_step: int
    physics_step: int
    metrics: Mapping[str, Any]
    events: Tuple[Any, ...]
    agent_termination_proposals: Mapping[str, Tuple[str, ...]]
    agent_terminated: Mapping[str, bool]
    all_agents_terminated: bool
    base_seed: Optional[int] = None
    episode_options: Mapping[str, Any] = MappingProxyType({})
    #: ``ctx.events.epoch`` 快照：游标差分消费者用它与 ``len(events)``
    #: 组成 (epoch, count) 游标；epoch 变化 = journal 被框架重置，
    #: 应整段重取而非从旧位置续读。
    events_epoch: int = 0

    @property
    def is_agent_terminated(self) -> Dict[str, bool]:
        """Read-only mapping of agent_id -> terminated bool."""
        return dict(self.agent_terminated)

    @classmethod
    def from_sim_context(cls, ctx: SimContext) -> "ReadOnlySimContext":
        return cls(
            accessor=ctx.accessor,
            episode_step=ctx.episode_step,
            physics_step=ctx.physics_step,
            metrics=MappingProxyType(dict(ctx.metrics)),
            events=tuple(ctx.events),
            events_epoch=ctx.events.epoch,
            agent_termination_proposals={
                aid: tuple(ctx.agent_termination_proposals[aid])
                for aid in AGENT_IDS
            },
            agent_terminated=dict(ctx.agent_terminated),
            all_agents_terminated=ctx.all_agents_terminated,
            base_seed=ctx.base_seed,
            episode_options=MappingProxyType(dict(ctx.episode_options)),
        )
