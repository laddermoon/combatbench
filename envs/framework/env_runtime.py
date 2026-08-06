from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar
import logging

import numpy as np

from .backend import BaseSimulator
from .common_plugins import TimeoutPlugin
from .context import ReadOnlySimContext, SimContext
from .plugin import BasePlugin
from .recorder import PostActionRecorder
from .observer_plugin import BaseObserverPlugin, _ObserverDispatcherPlugin


# TODO(framework/B2): introduce a VectorizedSimulator interface and a
# batched variant of EnvRuntime. Current design is single-env only; trainers
# rely on worker processes (RolloutCollector) which pay per-env reset cost
# and cannot exploit GPU-resident simulators. A future migration path:
# - add BaseVectorizedSimulator(batch_size, batched_step, batched_reset)
# - add EnvRuntimeBatched mirroring EnvRuntime with batched ctx views
# - keep EnvRuntime as the B=1 specialization.


_logger = logging.getLogger("combatbench.envs.framework")

_PluginT = TypeVar("_PluginT", bound=BasePlugin)


def _safe_call(
    target: Any,
    hook_name: str,
    strict: bool,
    label: str,
    *args: Any,
) -> None:
    """Invoke ``target.<hook_name>(*args)`` with uniform error handling.

    On exception:
      * ``strict=True`` (default) re-raises, preserving the original traceback.
      * ``strict=False`` logs the full traceback via ``logging.exception`` so
        it is visible in logs (unlike ``warnings.warn`` which is rate-limited).
    ``label`` is included in the log message to identify the failing unit.
    """
    method = getattr(target, hook_name, None)
    if method is None:
        return
    try:
        method(*args)
    except Exception:
        if strict:
            raise
        _logger.exception("%s '%s' failed at %s", label, hook_name, label)


class _PluginManager:
    def __init__(self, strict: bool = True):
        self._plugins: List[BasePlugin] = []
        self._strict = bool(strict)

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

    def iter_plugins(self) -> Tuple[BasePlugin, ...]:
        return tuple(self._plugins)

    def invoke(self, hook_name: str, ctx: SimContext, allow_mutator: bool = False) -> None:
        for plugin in self._plugins:
            if getattr(plugin, hook_name, None) is None:
                continue
            if allow_mutator and plugin.require_mutator:
                ctx._grant_mutator()
            else:
                ctx._revoke_mutator()
            _safe_call(
                plugin, hook_name, self._strict,
                f"Plugin '{plugin.name}'",
                ctx,
            )
        ctx._revoke_mutator()


class _RuntimeCore:
    def __init__(self, simulator: BaseSimulator, phy_steps_per_action: int = 1, strict: bool = True):
        self.simulator = simulator
        self.phy_steps_per_action = phy_steps_per_action
        self.ctx = SimContext(simulator)
        self.plugin_manager = _PluginManager(strict=strict)
        self._is_episode_active = False

    @property
    def is_episode_active(self) -> bool:
        return self._is_episode_active

    def attach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.attach(plugin)

    def detach_plugin(self, plugin: BasePlugin) -> None:
        self.plugin_manager.detach(plugin)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        base_seed: Optional[int] = None,
    ) -> None:
        """Begin a new episode. See ``envs/framework/RESET.md`` for the
        full reset chain & invariants.

        ``seed``: simulator-level RNG seed (one of the children derived from
        ``base_seed`` by :class:`EpisodeRunner`).
        ``options``: per-episode parameter dict; forwarded to
        ``simulator.reset(options=...)`` AND published on
        ``ctx.episode_options`` for plugins / observers / recorders.
        ``base_seed``: the user-facing batch seed; published on
        ``ctx.base_seed`` so recorders can persist it to manifest. ``None``
        is allowed for direct EnvRuntime usage outside of EpisodeRunner.

        If a previous episode is still active when ``reset`` is called,
        it is gracefully terminated first (``on_post_episode`` fires with
        reason ``"abandoned"``) — see RESET.md §7-G4.
        """
        if self._is_episode_active:
            # Abandon the in-flight episode cleanly so recorder manifests
            # and observer state are flushed before we wipe the context.
            self.ctx.request_termination("abandoned")
            self._handle_termination()
        self.ctx.clear_episode_state()
        self.ctx.base_seed = base_seed
        self.ctx.episode_options = dict(options or {})
        self._is_episode_active = True
        self.simulator.reset(seed=seed, options=options)
        self.plugin_manager.invoke("on_pre_episode", self.ctx, allow_mutator=True)
        if self.ctx.all_agents_terminated:
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
        if self.ctx.all_agents_terminated:
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
        recorders: Optional[List[PostActionRecorder]] = None,
        phy_steps_per_action: int = 1,
        max_steps: Optional[int] = None,
        strict: bool = True,
    ):
        """``strict``: if True (default) any exception raised by a plugin,
        observer or recorder hook propagates and stops the runtime. Set to
        False only for smoke-test / best-effort scripts where you want the
        error logged (with traceback) but the episode to continue.
        """
        self._strict = bool(strict)
        self._core = _RuntimeCore(simulator, phy_steps_per_action, strict=self._strict)
        self._observer_dispatcher = _ObserverDispatcherPlugin()
        self.observer_plugins: Dict[str, Optional[BaseObserverPlugin]] = {}
        self._recorders: List[PostActionRecorder] = []

        self._core.attach_plugin(self._observer_dispatcher)

        if max_steps is not None:
            self._core.attach_plugin(TimeoutPlugin(max_steps))

        for plugin in plugins or []:
            self._core.attach_plugin(plugin)

        for name, observer_plugin in (observer_plugins or {}).items():
            self.attach_observer_plugin(name, observer_plugin)

        for recorder in recorders or []:
            self.attach_recorder(recorder)

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

    @property
    def plugins(self) -> Tuple[BasePlugin, ...]:
        """Read-only snapshot of attached plugins (including the internal
        observer dispatcher). Useful for introspection / bulk config; do not
        mutate directly - use attach_plugin / detach_plugin instead.
        """
        return self._core.plugin_manager.iter_plugins()

    def find_plugins(self, plugin_type: Type[_PluginT]) -> Tuple[_PluginT, ...]:
        """Return all attached plugins matching ``plugin_type`` (``isinstance``)."""
        return tuple(plugin for plugin in self.plugins if isinstance(plugin, plugin_type))

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

    # ------------------------------------------------------------------
    # Post-action recorders
    # ------------------------------------------------------------------
    def attach_recorder(self, recorder: PostActionRecorder) -> None:
        if recorder in self._recorders:
            return
        self._recorders.append(recorder)
        recorder.on_attach()

    def detach_recorder(self, recorder: PostActionRecorder) -> None:
        if recorder in self._recorders:
            self._recorders.remove(recorder)
            recorder.on_detach()

    @property
    def recorders(self) -> Tuple[PostActionRecorder, ...]:
        return tuple(self._recorders)

    def _invoke_recorders(self, hook_name: str, *extra_args: Any) -> None:
        """Fan out a lifecycle hook to all attached recorders.

        Dispatches hook-specific positional args:
          - ``on_pre_episode`` / ``on_post_episode``: ``(ctx,)``
          - ``on_post_action_step``: ``(ctx, observation, action, observer_outputs, action_extras)``

        ``observation`` is the *pre-action* observation (captured before
        ``core.step()``) so recorders store ``obs_t`` alongside ``action_t``.
        """
        if not self._recorders:
            return
        readonly_ctx = ReadOnlySimContext.from_sim_context(self._core.ctx)
        if hook_name == "on_post_action_step":
            # extra_args = (pre_action_observation, action_extras)
            observation = extra_args[0]
            action = self._core.simulator.get_action()
            observer_outputs = self.get_observer_outputs()
            action_extras = extra_args[1] if len(extra_args) > 1 else None
            args = (readonly_ctx, observation, action, observer_outputs, action_extras)
        else:
            args = (readonly_ctx,)
        for recorder in self._recorders:
            _safe_call(
                recorder, hook_name, self._strict,
                f"Recorder '{type(recorder).__name__}'",
                *args,
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        base_seed: Optional[int] = None,
    ) -> None:
        """See :meth:`_RuntimeCore.reset` for the parameter semantics and
        ``envs/framework/RESET.md`` for the full reset chain."""
        self._core.reset(seed=seed, options=options, base_seed=base_seed)
        self._invoke_recorders("on_pre_episode")
        if not self._core.is_episode_active:
            # Reset triggered an immediate termination (e.g. invalid init state).
            self._invoke_recorders("on_post_episode")

    def step(
        self,
        action_a: Any,
        action_b: Any,
        action_a_extra: Optional[Mapping[str, Any]] = None,
        action_b_extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Advance one action step.

        ``action_a_extra`` / ``action_b_extra`` are *optional* per-agent
        side-channel payloads produced by the policy alongside the action
        (e.g. ``log_prob``, ``value``, sampling noise, exploration tags).
        They are NOT consumed by the simulator — they are passed through to
        recorders' ``on_post_action_step`` hook as a single bundle
        ``{"robot_a": action_a_extra, "robot_b": action_b_extra}``. Each
        entry is ``None`` when the caller did not provide extras for that
        agent (e.g. scripted opponent), so recorders should treat ``None``
        as "no extras this step for this agent".

        This is the canonical way for an RL trainer recorder to capture
        per-step log_prob / value alongside the observation snapshot
        without bolting policy-internals onto :class:`EpisodeRunner` or
        :class:`EnvRuntime`.
        """
        if not self._core.is_episode_active:
            raise RuntimeError("EnvRuntime.step() called before reset() or after episode termination.")
        # Capture the observation *before* the action is applied.  This is
        # ``obs_t`` — the state the policy saw when it selected the action.
        # Recorders receive it as the ``observation`` parameter so each frame
        # stores the pre-action state (the one that produced the action).
        observation = self._core.simulator.get_observation()
        self._core.step({"robot_a": action_a, "robot_b": action_b})
        action_extras: Dict[str, Optional[Mapping[str, Any]]] = {
            "robot_a": action_a_extra,
            "robot_b": action_b_extra,
        }
        self._invoke_recorders("on_post_action_step", observation, action_extras)
        if not self._core.is_episode_active:
            self._invoke_recorders("on_post_episode")

    def get_observer_output(self, name: str) -> Any:
        return self._observer_dispatcher.get_output(name)

    def get_observer_outputs(self, names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        target_names = list(names) if names is not None else list(self.observer_plugins.keys())
        return {name: self.get_observer_output(name) for name in target_names}

    def refresh_observers(self, force: bool = False) -> None:
        self._observer_dispatcher.refresh(self._core.ctx, force=force)

    def get_observation(self) -> Tuple[Any, Any]:
        """Return the current per-agent observations as a tuple.

        Delegates to ``simulator.get_observation()`` which is now part of
        the :class:`IDataAccessor` contract. The result is unpacked from
        the dict ``{"robot_a": obs_a, "robot_b": obs_b}`` into a plain
        ``(obs_a, obs_b)`` tuple so callers (e.g. :class:`EpisodeRunner`)
        can avoid dict indirection in the hot loop.

        Raises ``KeyError`` if the simulator response is missing one of
        the expected keys.
        """
        obs = self._core.simulator.get_observation()
        return obs["robot_a"], obs["robot_b"]

    def is_episode_over(self) -> bool:
        """True when all agents have terminated."""
        return self._core.ctx.all_agents_terminated

    def is_agent_active(self, agent_id: str) -> bool:
        """True if ``agent_id`` has not yet terminated."""
        return not self._core.ctx.agent_terminated.get(agent_id, False)

    def get_agent_termination(self) -> Dict[str, Optional[str]]:
        """Return first termination reason per agent, or None if not terminated."""
        ctx = self._core.ctx
        result: Dict[str, Optional[str]] = {}
        for aid in self.AGENT_IDS:
            proposals = ctx.agent_termination_proposals.get(aid, [])
            result[aid] = proposals[0] if proposals else None
        return result

    def render(self) -> Optional[np.ndarray]:
        return self._core.simulator.get_broadcastview_image()

    def to_blueprint(self) -> "EnvBlueprint":  # noqa: F821
        """Snapshot this runtime as an :class:`EnvBlueprint`.

        Recorders, the internal observer dispatcher, the auto-attached
        :class:`TimeoutPlugin`, and any plugin / observer with
        ``BLUEPRINT_EXCLUDE = True`` are deliberately omitted. See
        :mod:`envs.framework.blueprint` for the protocol.
        """
        # Local import to avoid an import cycle at module load.
        from .blueprint import EnvBlueprint

        return EnvBlueprint.from_runtime(self)

    def close(self) -> None:
        for recorder in list(self._recorders):
            self.detach_recorder(recorder)
        self._core.close()
