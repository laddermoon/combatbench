from typing import Any, Dict, Iterable, List, Optional, Tuple
import warnings

import numpy as np

from .backend import BaseSimulator
from .common_plugins import TimeoutPlugin
from .context import ReadOnlySimContext, SimContext, TerminationReason
from .plugin import BasePlugin
from .runtime_plugin import BaseObserverPlugin, _ObserverDispatcherPlugin


class _PluginManager:
    def __init__(self):
        self._plugins: List[BasePlugin] = []

    def attach(self, plugin: BasePlugin) -> None:
        if plugin in self._plugins:
            return
        self._plugins.append(plugin)
        self._plugins.sort(key=lambda current_plugin: current_plugin.priority, reverse=True)
        plugin.on_attach()

    def detach(self, plugin: BasePlugin) -> None:
        if plugin in self._plugins:
            self._plugins.remove(plugin)
            plugin.on_detach()

    def clear(self) -> None:
        for plugin in list(self._plugins):
            self.detach(plugin)

    def invoke(self, hook_name: str, ctx: SimContext, allow_mutator: bool = False) -> None:
        for plugin in self._plugins:
            try:
                method = getattr(plugin, hook_name, None)
                if method is None:
                    continue
                if allow_mutator and plugin.require_mutator:
                    ctx._grant_mutator()
                else:
                    ctx._revoke_mutator()
                method(ctx)
            except Exception as exc:
                warnings.warn(f"Plugin '{plugin.name}' failed at {hook_name}: {exc}")
        ctx._revoke_mutator()


class _RuntimeCore:
    def __init__(self, simulator: BaseSimulator, phy_steps_per_action: int = 1):
        self.simulator = simulator
        self.phy_steps_per_action = phy_steps_per_action
        self.ctx = SimContext(simulator)
        self.plugin_manager = _PluginManager()
        self._is_episode_active = False

    @property
    def is_episode_active(self) -> bool:
        return self._is_episode_active

    def attach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.attach(plugin)

    def detach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.detach(plugin)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        self.ctx.clear_episode_state()
        self._is_episode_active = True
        self.simulator.reset(seed=seed, options=options)
        self.plugin_manager.invoke("on_pre_episode", self.ctx, allow_mutator=True)
        if self.ctx.is_terminated:
            self._handle_termination()

    def step(self, action: Dict[str, Any]) -> None:
        if not self._is_episode_active:
            return
        self.ctx._grant_mutator()
        self.ctx.mutator.set_action(action)
        self.ctx._revoke_mutator()
        self.plugin_manager.invoke("on_pre_action_step", self.ctx, allow_mutator=True)
        if self._check_and_handle_termination():
            return
        for _ in range(self.phy_steps_per_action):
            self.plugin_manager.invoke("on_pre_phy_step", self.ctx, allow_mutator=True)
            if self._check_and_handle_termination():
                return
            self.simulator.physical_step()
            self.ctx.physics_step += 1
            self.plugin_manager.invoke("on_post_phy_step", self.ctx, allow_mutator=True)
            if self._check_and_handle_termination():
                return
        self.ctx.episode_step += 1
        self.plugin_manager.invoke("on_post_action_step", self.ctx, allow_mutator=False)
        self._check_and_handle_termination()

    def close(self) -> None:
        self.plugin_manager.clear()
        self._is_episode_active = False
        self.simulator.close()

    def _check_and_handle_termination(self) -> bool:
        if self.ctx.is_terminated:
            self._handle_termination()
            return True
        return False

    def _handle_termination(self) -> None:
        self._is_episode_active = False
        self.plugin_manager.invoke("on_post_episode", self.ctx, allow_mutator=False)


class EnvRuntime:
    AGENT_IDS = ("robot_a", "robot_b")

    def __init__(
        self,
        simulator: BaseSimulator,
        observer_plugins: Optional[Dict[str, BaseObserverPlugin]] = None,
        plugins: Optional[List[BasePlugin]] = None,
        phy_steps_per_action: int = 1,
        max_steps: Optional[int] = None,
    ):
        self._core = _RuntimeCore(simulator, phy_steps_per_action)
        self._observer_dispatcher = _ObserverDispatcherPlugin()
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self.shared_info_builder = None

        self._core.attach_plugin(self._observer_dispatcher)

        if max_steps is not None:
            self._core.attach_plugin(TimeoutPlugin(max_steps))

        for plugin in plugins or []:
            self._core.attach_plugin(plugin)

        for name, observer_plugin in (observer_plugins or {}).items():
            self.attach_observer_plugin(name, observer_plugin)

    @property
    def simulator(self) -> BaseSimulator:
        return self._core.simulator

    @property
    def ctx(self) -> SimContext:
        return self._core.ctx

    @property
    def is_episode_active(self) -> bool:
        return self._core.is_episode_active

    def attach_plugin(self, plugin: BasePlugin) -> None:
        self._core.attach_plugin(plugin)

    def detach_plugin(self, plugin: BasePlugin) -> None:
        if plugin is self._observer_dispatcher:
            raise ValueError("Observer dispatcher is managed internally by EnvRuntime and cannot be detached.")
        self._core.detach_plugin(plugin)

    def attach_observer_plugin(self, name: str, observer_plugin: Optional[BaseObserverPlugin]) -> None:
        current = self.observer_plugins.get(name)
        if current is observer_plugin:
            return
        self.observer_plugins[name] = observer_plugin
        if observer_plugin is None:
            self._observer_dispatcher.remove_observer_plugin(name)
        else:
            self._observer_dispatcher.set_observer_plugin(name, observer_plugin)
        if self._core.is_episode_active:
            self._observer_dispatcher.refresh(self._core.ctx, force=True)

    def detach_observer_plugin(self, name: str) -> None:
        self.attach_observer_plugin(name, None)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        self._core.reset(seed=seed, options=options)

    def step(self, action_a: Any, action_b: Any) -> None:
        if not self._core.is_episode_active:
            raise RuntimeError("EnvRuntime.step() called before reset() or after episode termination.")
        self._core.step({"robot_a": action_a, "robot_b": action_b})

    def get_observer_output(self, name: str) -> Any:
        return self._observer_dispatcher.get_output(name)

    def get_observer_outputs(self, names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        target_names = list(names) if names is not None else list(self.observer_plugins.keys())
        return {name: self.get_observer_output(name) for name in target_names}

    def refresh_observers(self, force: bool = False) -> None:
        self._observer_dispatcher.refresh(self._core.ctx, force=force)

    def get_shared_info(self) -> Dict[str, Any]:
        ctx = self._core.ctx
        shared_info = {
            "metrics": dict(ctx.metrics),
            "events": list(ctx.events),
            "termination_reasons": list(ctx.termination_proposals),
            "episode_step": ctx.episode_step,
            "physics_step": ctx.physics_step,
            "is_terminated": ctx.is_terminated,
        }
        if callable(self.shared_info_builder):
            extra_shared_info = self.shared_info_builder(ReadOnlySimContext.from_sim_context(ctx))
            if isinstance(extra_shared_info, dict):
                shared_info.update(extra_shared_info)
        return shared_info

    def get_termination_flags(self) -> Tuple[bool, bool]:
        proposals = self._core.ctx.termination_proposals
        if not proposals:
            return False, False
        if TerminationReason.TIMEOUT in proposals:
            has_non_timeout_reason = any(reason != TerminationReason.TIMEOUT for reason in proposals)
            if has_non_timeout_reason:
                return True, False
            return False, True
        return True, False

    def render(self) -> Optional[np.ndarray]:
        return self._core.simulator.get_broadcastview_image()

    def close(self) -> None:
        self._core.close()
