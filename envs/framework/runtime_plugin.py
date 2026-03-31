from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

from .context import ReadOnlySimContext, SimContext
from .plugin import BasePlugin


class BaseRuntimeUnit(ABC):
    @abstractmethod
    def process_data(self, ctx: ReadOnlySimContext) -> None:
        pass

    @abstractmethod
    def get_output(self) -> Any:
        pass


class BaseObserverPlugin(BaseRuntimeUnit, ABC):
    pass


class _ObserverDispatcherPlugin(BasePlugin):
    def __init__(self):
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self._last_process_token: Optional[Tuple[int, int, Tuple[str, ...], bool]] = None

    @property
    def name(self) -> str:
        return "observer_dispatcher"

    @property
    def priority(self) -> int:
        return -1_000_000

    @property
    def require_mutator(self) -> bool:
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
        self._process_ctx(ctx)

    def on_pre_action_step(self, ctx: SimContext) -> None:
        return None

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_phy_step(self, ctx: SimContext) -> None:
        return None

    def on_post_action_step(self, ctx: SimContext) -> None:
        self._process_ctx(ctx)

    def on_post_episode(self, ctx: SimContext) -> None:
        self._process_ctx(ctx)

    def on_attach(self) -> None:
        self._last_process_token = None

    def on_detach(self) -> None:
        self._last_process_token = None

    def refresh(self, ctx: SimContext, force: bool = False) -> None:
        self._process_ctx(ctx, force=force)

    def _process_ctx(self, ctx: SimContext, force: bool = False) -> None:
        readonly_ctx = ReadOnlySimContext.from_sim_context(ctx)
        process_token = (
            readonly_ctx.episode_step,
            readonly_ctx.physics_step,
            readonly_ctx.termination_proposals,
            readonly_ctx.is_terminated,
        )
        if not force and process_token == self._last_process_token:
            return
        self._last_process_token = process_token
        for runtime_unit in self._iter_runtime_units():
            runtime_unit.process_data(readonly_ctx)

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
