from typing import Any, Dict, List, Optional
from .context import SimContext

class BasePlugin:
    """
    Simulation extension plugin.

    Responsibilities:
    1. Inject custom logic at specific lifecycle points of the simulation engine.
    2. Read data via ctx.accessor.
    3. Modify data via ctx.mutator (if the current lifecycle permits; otherwise None).
    """
    
    @property
    def name(self) -> str:
        return "unnamed_plugin"
        
    @property
    def priority(self) -> int:
        """Plugin dispatch priority — larger values run first (``_PluginManager`` sorts descending).

        Conventions:
        * Default ``0``. Most plugins (reward / termination / data collection) can keep the default;
          they run after the observer dispatcher and naturally see the current step's observer output.
        * To run before observers (e.g. scoring / damage resolution so the observer sees the hit event
          within the same step), set ``priority`` strictly greater than
          :data:`envs.framework.observer_plugin.OBSERVER_DISPATCHER_PRIORITY`.
        """
        return 0

    @property
    def require_mutator(self) -> bool:
        """
        Whether to request data-mutation permission.
        If False, ctx.mutator passed to this plugin will be None even during
        mutator-allowed lifecycles (e.g. on_pre_phy_step). Follows the principle of least privilege.
        """
        return False

    # ==========================================
    # Randomness Hook
    # ==========================================

    def set_episode_seed(self, seed: int) -> None:
        """[Timing]: Called by EpisodeRunner before the episode starts and before on_pre_episode.
        [Responsibility]: Plugins that own an RNG should rebuild it immediately here.

        Default no-op: plugins that do not consume randomness need not override.
        Implementers should directly ``self._rng = np.random.RandomState(int(seed))``
        (or equivalent) inside this method; do not defer to on_pre_episode so that
        set_episode_seed remains the single entry point for RNG reconstruction.
        See ``SEED.md`` for details.
        """
        pass

    # ==========================================
    # Lifecycle Hooks
    # ==========================================

    def on_pre_episode(self, ctx: SimContext) -> None:
        """
        [Timing]: Before a new episode begins, right after the environment is reset.
        [Responsibility]: Initialize state (resetters), clear historical statistics.
        [Permission]: Read-write (ctx.mutator available).
        """
        pass

    def on_pre_action_step(self, ctx: SimContext) -> None:
        """
        [Timing]: After receiving an external action but before it is split and sent to the physics engine.
        [Responsibility]: Control mode mapping, action space mapping, action clipping.
        [Permission]: Read-write (ctx.mutator available; action may be modified).
        """
        pass

    def on_pre_phy_step(self, ctx: SimContext) -> None:
        """
        [Timing]: Before each fine-grained physics simulation step.
        [Responsibility]: Inject external disturbances.
        [Permission]: Read-write (ctx.mutator available; external forces may be modified).
        """
        pass

    def on_post_phy_step(self, ctx: SimContext) -> None:
        """
        [Timing]: After each fine-grained physics step, once state has been updated.
        [Responsibility]: Hard state constraints, high-frequency data collection.
        [Permission]: Read-write (ctx.mutator available; state may be overridden for projection).
        """
        pass

    def on_post_action_step(self, ctx: SimContext) -> None:
        """
        [Timing]: After all physics steps corresponding to one action step have finished.
        [Responsibility]: Metric aggregation, termination proposals, reward computation.
        [Permission]: Read-only (ctx.mutator is None).
        """
        pass

    def on_post_episode(self, ctx: SimContext) -> None:
        """
        [Timing]: After the episode has definitively terminated.
        [Responsibility]: Episode-level log aggregation and data reporting.
        [Permission]: Read-only (ctx.mutator is None).
        """
        pass

    # ==========================================
    # Management Hooks
    # ==========================================
    #
    # Design rationale
    # ----------------
    # ``on_pre_episode`` / ``on_post_episode`` are bound to the **episode**
    # lifecycle (fired every episode). ``on_attach`` / ``on_detach`` are bound
    # to the **runtime attachment** lifecycle (fired once across all episodes),
    # managing one-off resources and caches tied to the plugin instance rather
    # than to an episode.
    #
    # Why ``__init__`` / ``__del__`` cannot replace them:
    #   * ``__init__`` runs in user code before the plugin is attached to a runtime,
    #     so it cannot observe the runtime lifecycle (no way to be notified on
    #     ``runtime.close()``).
    #   * ``__del__`` timing in Python is non-deterministic (may never fire); it is
    #     unsuitable for releasing side-effectful resources such as file handles,
    #     sockets, or GPU contexts.
    #
    # Framework dispatch points (do not rely on other call sites):
    #   * ``EnvRuntime.attach_plugin(plugin)`` / ``attach_recorder``
    #     → ``plugin.on_attach()``
    #   * ``EnvRuntime.detach_plugin(plugin)`` / ``detach_recorder``
    #     → ``plugin.on_detach()``
    #   * ``EnvRuntime.close()`` internally ``clear()`` detaches one by one →
    #     every registered plugin's ``on_detach`` is guaranteed to fire once
    #     (graceful-shutdown contract; see tests/test_edge_cases.py::test_close_clears_plugins).

    def on_attach(self) -> None:
        """
        [Timing]: When the plugin is attached to a runtime (``EnvRuntime.attach_plugin``).
                A given plugin instance fires once per runtime; if detached and re-attached,
                it will fire again.
        [Responsibility]: Allocate one-off resources and initialize caches —
                * open log / video file handles;
                * establish sockets, connect to remote services;
                * allocate GPU contexts, pre-compile kernels;
                * clear / initialize caches reused across episodes (e.g.
                  ``_ObserverDispatcherPlugin`` clears deduplication tokens here).
        [Permission]: No ctx exists yet; do not access simulation state.
                Read initial state in ``on_pre_episode`` instead.
        [Default]: no-op. Plugins without one-off resource needs need not override.
        """
        pass

    def on_detach(self) -> None:
        """
        [Timing]: When the plugin is detached from a runtime (``EnvRuntime.detach_plugin``),
                or during ``runtime.close()`` cleanup (every registered plugin's
                ``on_detach`` is guaranteed to fire; see
                ``test_edge_cases.py::test_close_clears_plugins``).
                Corresponds one-to-one with ``on_attach``; detaching without a prior
                attach will never happen on a given runtime.
        [Responsibility]: Release resources acquired in ``on_attach`` —
                * flush and close video / log files;
                * close sockets, disconnect remote services;
                * free GPU memory, clear persistent caches.
        [Permission]: ctx is no longer available and the episode may have ended
                or never started; do not access simulation state.
        [Default]: no-op.
        """
        pass
