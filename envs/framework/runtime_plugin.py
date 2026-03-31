from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from .context import ReadOnlySimContext, SimContext
from .plugin import BasePlugin


class _BaseReadonlyRuntimePlugin(BasePlugin, ABC):
    def __init__(self):
        self._last_process_token: Optional[Tuple[int, int, Tuple[str, ...], bool]] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def priority(self) -> int:
        return -1_000_000

    @property
    def require_mutator(self) -> bool:
        return False

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

    def _process_ctx(self, ctx: SimContext) -> None:
        readonly_ctx = ReadOnlySimContext.from_sim_context(ctx)
        process_token = (
            readonly_ctx.episode_step,
            readonly_ctx.physics_step,
            readonly_ctx.termination_proposals,
            readonly_ctx.is_terminated,
        )
        if process_token == self._last_process_token:
            return
        self._last_process_token = process_token
        self.process_data(readonly_ctx)

    @abstractmethod
    def process_data(self, ctx: ReadOnlySimContext) -> None:
        pass

    @abstractmethod
    def get_output(self) -> Any:
        pass


class BaseObserver(_BaseReadonlyRuntimePlugin, ABC):
    pass


class BaseRewarder(_BaseReadonlyRuntimePlugin, ABC):
    pass
