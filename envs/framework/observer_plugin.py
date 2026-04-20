from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

from .context import ReadOnlySimContext, SimContext
from .plugin import BasePlugin


class BaseRuntimeUnit(ABC):
    """Observer-side unit invoked by ``_ObserverDispatcherPlugin``.

    Subclasses produce a per-step output (observation vector, reward scalar,
    metric dict, ...) via ``get_output()``. Image/debug dumping used to live
    here via ``save_debug_image``; it now belongs to
    :class:`envs.framework.recorder.PostActionRecorder`.

    ``get_output()`` return-shape guidance
    ---------------------------------------
    These conventions let :class:`envs.framework.episode_runner.EpisodeRunner`
    and other consumers read observer outputs **without shape-guessing**.
    Existing plugins that deviate still work (nothing is enforced at runtime),
    but new plugins should follow them.

    * **Observation plugins** — return a value that is directly consumable
      by the policy. Typically ``np.ndarray(dtype=float32)`` for flat obs,
      or ``Dict[str, np.ndarray]`` for structured obs (policy handles the
      structure). **Do NOT wrap** in ``(payload, info)`` tuples or
      ``{"obs": ..., "info": ...}`` envelopes — keep ``get_output`` pure.
    * **Reward plugins** — return either (a) a Python / numpy scalar, or
      (b) a ``dict`` with a ``"reward"`` / ``"total_reward"`` / ``"r"`` key
      holding the scalar. Any other keys in the dict are ignored by the
      default reward extractor but are fine to carry (e.g. per-term
      breakdowns for logging). Consumers that want full breakdowns can
      override ``ObserverBinding.reward_extractor``.
    * **Metric plugins** — return any JSON-safe value. No fixed contract.
    """

    def process_data(self, ctx: ReadOnlySimContext) -> None:
        return None

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self.process_data(ctx)

    def on_manual_refresh(self, ctx: ReadOnlySimContext) -> None:
        self.on_post_step(ctx)

    @abstractmethod
    def get_output(self) -> Any:
        pass


# ``BaseObserverPlugin`` used to carry observer-specific behaviour
# (save_debug_image). After recorder extraction it became an empty subclass.
# Kept as an alias for backward compatibility with user code that imports the
# name; new code should subclass ``BaseRuntimeUnit`` directly.
BaseObserverPlugin = BaseRuntimeUnit


class _ObserverDispatcherPlugin(BasePlugin):
    def __init__(self):
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self._last_process_token: Optional[Tuple[str, int, int, Tuple[str, ...], bool]] = None

    @property
    def name(self) -> str:
        return "observer_dispatcher"

    @property
    def priority(self) -> int:
        # 观察者刷新必须在同一钩子的其它 plugin **之前**执行，这样下游的
        # 终止判定 / 奖励 / 指标插件读到的 observer 输出总是对应当前步的状态。
        # 在 ``on_pre_episode`` 这种可写钩子里，观察者自己靠
        # ``require_mutator=False`` 保证不会获得 ``ctx.mutator``——无论优先级
        # 高低它都只能读。因此安全地把优先级抬到最前。
        return 1_000_000

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
        self._process_ctx(ctx, trigger_name="on_reset")

    def on_pre_action_step(self, ctx: SimContext) -> None:
        return None

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        self._process_ctx(ctx, trigger_name="on_post_step")

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
            readonly_ctx.termination_proposals,
            readonly_ctx.is_terminated,
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
