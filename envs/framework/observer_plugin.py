from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

from .context import AGENT_IDS, ReadOnlySimContext, SimContext
from .plugin import BasePlugin


# ---------------------------------------------------------------------------
# Observer dispatcher priority constant
# ---------------------------------------------------------------------------
# ``_PluginManager`` 按 ``priority`` 降序执行 plugin（``reverse=True``，详见
# ``env_runtime.py``）。Observer dispatcher 默认抢占所有用户 plugin 之前刷新，
# 这样下游的奖励 / 终止 / 指标 plugin 在同一钩子里读到的 observer 输出总是
# 对应当前步的状态。该默认值通过下面这个常量集中暴露。
#
# 使用约定
# --------
# * **默认 plugin (priority == 0)** —— 在 observer 刷新之后执行。这是 99% 的
#   场景，比如奖励器 / 终止判定 / 数据采集。它们读 observer 输出和 ``ctx``
#   都拿到的是当前步的最新值。
# * **需要在 observer 之前执行的 plugin** —— 把自己的 ``priority`` 设置成
#   **严格大于** ``OBSERVER_DISPATCHER_PRIORITY``（例如
#   ``OBSERVER_DISPATCHER_PRIORITY + 1``）。典型用例：
#     - 计分 / 伤害判定（写入 ``ctx.metrics`` / ``ctx.events``），让
#       observer 在同一步就能读到本步的击打结果，而不是滞后一步。
#     - 任何会修改 ``ctx`` 黑板字段、且这些字段会被 observer 读取的 plugin。
#
# 把它做成常量而不是埋在 ``_ObserverDispatcherPlugin.priority`` 里，是为了
# 让上游 plugin 写者不用反向工程框架内部值。需要"插队到 observer 前"的
# plugin 应当显式 import 这个常量并基于它命名自己的优先级。
OBSERVER_DISPATCHER_PRIORITY: int = 1_000_000


class BaseRuntimeUnit(ABC):
    """Observer-side unit invoked by ``_ObserverDispatcherPlugin``.

    Design intent
    -------------
    An Observer computes some per-step derived quantity (typically a
    reward, but also debug / diagnostic signals) and exposes it through
    :meth:`get_output`. The canonical lifecycle is:

    * :meth:`on_pre_episode` — **preparation** at the start of an episode
      (reset internal accumulators, cache static data, allocate buffers,
      ...). It is *not* required to compute the same thing as
      ``on_post_action_step``; in particular the typical Rewarder leaves
      ``get_output()`` returning an initial / zero value here.
    * :meth:`on_post_action_step` — refresh the unit's internal state from
      the current ``ctx``. The standard call pattern downstream is
      "step the env → call ``on_post_action_step`` (done by the
      dispatcher) → read :meth:`get_output`".
    * :meth:`on_post_episode` — optional end-of-episode finalisation
      (final metrics, summary statistics, ...).

    Nothing in the framework enforces this split — subclasses are free to
    override any hook however they like.

    Primary use cases
    -----------------
    * **Rewarders.** Encapsulate a reward term (or a composite of terms);
      the trainer / recorder reads ``get_output()`` each step.
    * **Debug / dev probes paired with a Recorder.** Compute something
      expensive once per step and let a
      :class:`envs.framework.recorder.PostActionRecorder` snapshot it,
      keeping the inner physics / plugin code clean.

    ``get_output()`` return-shape guidance
    ---------------------------------------
    No shape is enforced at runtime, but consumers are easier to write
    when units follow these conventions:

    * **Reward units** — return either (a) a Python / numpy scalar, or
      (b) a ``dict`` carrying a ``"reward"`` / ``"total_reward"`` / ``"r"``
      key with the scalar (extra keys are fine for per-term breakdowns).
    * **Debug / metric units** — return any JSON-safe value.

    Image / debug dumping used to live here via ``save_debug_image``; it
    now belongs to :class:`envs.framework.recorder.PostActionRecorder`.
    """

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """Episode-start preparation. Default: no-op."""
        return None

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """Per-step refresh. Default: no-op — subclasses override."""
        return None

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        """Episode-end finalisation. Default: no-op."""
        return None

    def on_manual_refresh(self, ctx: ReadOnlySimContext) -> None:
        """Out-of-band refresh (e.g. after a reset chain). Default:
        delegate to :meth:`on_post_action_step`."""
        self.on_post_action_step(ctx)

    @abstractmethod
    def get_output(self) -> Any:
        pass


# ``BaseObserverPlugin`` used to carry observer-specific behaviour
# (save_debug_image). After recorder extraction it became an empty subclass.
# Kept as an alias for backward compatibility with user code that imports the
# name; new code should subclass ``BaseRuntimeUnit`` directly.
BaseObserverPlugin = BaseRuntimeUnit


class CompositeObserver(BaseRuntimeUnit):
    """Bundle multiple :class:`BaseRuntimeUnit` observers behind a single slot.

    Each child observer is registered under a string name. Lifecycle hooks
    (``on_pre_episode`` / ``on_post_action_step`` / ``on_post_episode`` /
    ``on_manual_refresh``) fan out to every child in registration order, and
    :meth:`get_output` returns a ``{name: child.get_output()}`` dict — i.e.
    this composite **does not** combine / weight / reduce the child outputs;
    it just presents them side-by-side.

    Why a composite?
    ----------------
    The runtime's ``observer_plugins`` dict is keyed by name and each entry
    holds exactly one observer. When you want to attach several small
    diagnostic observers (per-agent posture probes, per-component reward
    breakdowns, per-step debug snapshots, ...) without polluting the
    top-level dict, group them under one composite key. Downstream consumers
    that only care about a sub-output index into the dict by name.

    Composes with itself: nested ``CompositeObserver`` instances produce
    nested dicts, so hierarchical observation layouts are fine.

    Notes
    -----
    * Children are stored in an ``OrderedDict``; iteration order matches the
      construction / :meth:`add` order. Hooks are invoked in this order.
    * If a hook on one child raises, **the exception propagates** and the
      remaining children for that hook are skipped. This mirrors the runtime's
      ``strict=True`` default — error isolation belongs at the runtime layer
      (``_safe_call``), not here. Wrap individual children in try/except if
      you need best-effort fan-out.
    * Each child keeps its own state. The composite holds no state of its
      own beyond the registry, so re-attaching the same composite across
      episodes is safe (the children's ``on_pre_episode`` will reset them).
    """

    def __init__(
        self,
        sub_observers: Optional[Dict[str, "BaseRuntimeUnit"]] = None,
    ) -> None:
        # ``OrderedDict`` is explicit even though ``dict`` is insertion-ordered
        # since 3.7 — readers should not have to remember that detail to be
        # confident the fan-out order is stable.
        from collections import OrderedDict

        self._subs: "OrderedDict[str, BaseRuntimeUnit]" = OrderedDict()
        if sub_observers:
            for name, observer in sub_observers.items():
                self.add(str(name), observer)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def add(self, name: str, sub_observer: "BaseRuntimeUnit") -> None:
        """Register ``sub_observer`` under ``name``. Raises if name is taken."""
        key = str(name)
        if key in self._subs:
            raise ValueError(
                f"CompositeObserver: sub-observer name {key!r} already registered"
            )
        if not isinstance(sub_observer, BaseRuntimeUnit):
            raise TypeError(
                f"CompositeObserver: sub-observer {key!r} must be a "
                f"BaseRuntimeUnit instance, got {type(sub_observer).__name__}"
            )
        self._subs[key] = sub_observer

    def remove(self, name: str) -> None:
        self._subs.pop(str(name), None)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._subs.keys())

    def get(self, name: str) -> "BaseRuntimeUnit":
        return self._subs[str(name)]

    # ------------------------------------------------------------------
    # Lifecycle fan-out
    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        for sub in self._subs.values():
            sub.on_pre_episode(ctx)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        for sub in self._subs.values():
            sub.on_post_action_step(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        for sub in self._subs.values():
            sub.on_post_episode(ctx)

    def on_manual_refresh(self, ctx: ReadOnlySimContext) -> None:
        for sub in self._subs.values():
            sub.on_manual_refresh(ctx)

    # ------------------------------------------------------------------
    # Output aggregation (NOT a reduction — just side-by-side dict)
    # ------------------------------------------------------------------
    def get_output(self) -> Dict[str, Any]:
        return {name: sub.get_output() for name, sub in self._subs.items()}


class _ObserverDispatcherPlugin(BasePlugin):
    def __init__(self):
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self._last_process_token: Optional[Tuple[str, int, int, Tuple[str, ...], bool]] = None

    @property
    def name(self) -> str:
        return "observer_dispatcher"

    @property
    def priority(self) -> int:
        # 观察者刷新默认在同一钩子的其它 plugin **之前**执行，这样下游的
        # 终止判定 / 奖励 / 指标插件读到的 observer 输出总是对应当前步的状态。
        # 在 ``on_pre_episode`` 这种可写钩子里，观察者自己靠
        # ``require_mutator=False`` 保证不会获得 ``ctx.mutator``——无论优先级
        # 高低它都只能读。
        # 数值由 :data:`OBSERVER_DISPATCHER_PRIORITY` 集中暴露；需要在 observer
        # 之前执行的 plugin 应当把 ``priority`` 设为严格大于该常量。
        return OBSERVER_DISPATCHER_PRIORITY

    @property
    def require_mutator(self) -> bool:
        # 观察者**永远**是只读的。即便运行在可写钩子上也不会拿到
        # ``ctx.mutator``。这是 Observer 语义的硬约束。
        return False

    def set_observer_plugin(self, name: str, observer_plugin: Optional[BaseObserverPlugin]) -> None:
        self.observer_plugins[name] = observer_plugin
        self._last_process_token = None

    def remove_observer_plugin(self, name: str) -> None:
        self.observer_plugins.pop(name, None)
        self._last_process_token = None

    def get_output(self, name: str) -> Any:
        observer_plugin = self.observer_plugins.get(name)
        return observer_plugin.get_output() if observer_plugin is not None else None

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_pre_episode")

    def on_pre_action_step(self, ctx: SimContext) -> None:
        return None

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_post_action_step")

    def on_post_episode(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_post_episode")

    def on_attach(self) -> None:
        self._last_process_token = None

    def on_detach(self) -> None:
        self._last_process_token = None

    def refresh(self, ctx: SimContext, force: bool = False) -> None:
        self._process_ctx(ctx, trigger_name="on_manual_refresh", force=force)

    def _process_ctx(self, ctx: SimContext, trigger_name: str, force: bool = False) -> None:
        readonly_ctx = ReadOnlySimContext.from_sim_context(ctx)
        # Note(framework/C2): this token intentionally skips ``metrics`` /
        # ``events`` because they are not hashable in general. That means if a
        # plugin mutates metrics without advancing episode_step / physics_step
        # and then requests a refresh WITHOUT ``force=True``, the dispatcher
        # will skip re-processing. All known call sites either advance the
        # step counter or pass ``force=True``; treat this as a contract.
        process_token = (
            trigger_name,
            readonly_ctx.episode_step,
            readonly_ctx.physics_step,
            tuple(readonly_ctx.agent_termination_proposals[aid] for aid in AGENT_IDS),
            readonly_ctx.all_agents_terminated,
        )
        if not force and process_token == self._last_process_token:
            return
        self._last_process_token = process_token
        for runtime_unit in self._iter_runtime_units():
            getattr(runtime_unit, trigger_name)(readonly_ctx)

    def _iter_runtime_units(self):
        seen: Set[int] = set()
        for runtime_unit in list(self.observer_plugins.values()):
            if runtime_unit is None:
                continue
            unit_id = id(runtime_unit)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            yield runtime_unit
