"""``ParallelRollouter``: blueprint × policy_factories × seeds → EpisodeCollection.

See ``baseline/common/rollout/DESIGN.md`` §4 for design notes. Highlights:

* Workers are persistent (initialized once per ``num_workers > 1`` run) and
  rebuilt only when the rollouter is closed. Each worker constructs its own
  :class:`EnvRuntime`, policies, and :class:`EpisodeRecorder` from the
  factories supplied at ``__init__`` time.
* ``num_workers <= 1`` short-circuits the multiprocessing path and runs
  in-process, which keeps stack traces clean for pdb.
* Optional ``policy_state_dicts`` per :meth:`collect` call broadcasts fresh
  weights to all workers (so PPO/GRPO can hot-reload between iterations
  without rebuilding the runtime).
* ``deterministic`` is a duck-typed switch: if a policy exposes
  ``set_deterministic(bool)`` the rollouter calls it; otherwise the flag
  is silently ignored. Policies that depend on this for correct rollouts
  must implement that hook.

Worker → main: the produced :class:`Episode` (numpy + Python primitives)
crosses the pickle boundary; the runtime / policies never do.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from envs.framework.blueprint import EnvBlueprint
from envs.framework.episode_runner import EpisodeRunner
from envs.framework.plugin import BasePlugin
from envs.framework.policy import Policy

from .episode import Episode, blueprint_hash
from .episode_collection import EpisodeCollection
from .episode_recorder import EpisodeRecorder

_logger = logging.getLogger(__name__)


PolicyFactory = Callable[[], Policy]
DebugPluginsFactory = Callable[[], Sequence[BasePlugin]]
EpisodeOptionsFn = Callable[[int], Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Worker-side state
# ---------------------------------------------------------------------------
# A single long-lived runner per worker. ``_worker_state`` is module-level
# so the worker can keep it across tasks; tasks themselves are stateless
# function calls that read/mutate this state.
_worker_state: Dict[str, Any] = {}


def _worker_init(
    blueprint_dict: Mapping[str, Any],
    policy_factories: Mapping[str, PolicyFactory],
    observer_names_to_keep: Optional[Sequence[str]],
    deterministic: bool,
    debug_plugins_factory: Optional[DebugPluginsFactory],
) -> None:
    """Construct the per-worker runtime / runner / recorder once."""
    blueprint = EnvBlueprint.from_dict(blueprint_dict)
    bp_hash = blueprint_hash(blueprint)
    recorder = EpisodeRecorder(
        blueprint_hash=bp_hash,
        observer_names_to_keep=observer_names_to_keep,
    )
    debug_plugins: Sequence[BasePlugin] = ()
    if debug_plugins_factory is not None:
        debug_plugins = tuple(debug_plugins_factory())
    runtime = blueprint.build(
        recorders=[recorder],
        debug_plugins=debug_plugins,
    )
    policies: Dict[str, Policy] = {
        agent_id: factory() for agent_id, factory in policy_factories.items()
    }
    if deterministic:
        for policy in policies.values():
            setter = getattr(policy, "set_deterministic", None)
            if callable(setter):
                setter(True)
    runner = EpisodeRunner(runtime=runtime, policies=policies)

    _worker_state.clear()
    _worker_state.update(
        runtime=runtime,
        runner=runner,
        recorder=recorder,
        policies=policies,
        blueprint_hash=bp_hash,
    )


def _worker_load_state_dicts(state_dicts: Mapping[str, Any]) -> None:
    """Apply fresh policy weights to the worker's policy instances.

    Each entry in ``state_dicts`` is forwarded to
    ``policy.load_state_dict(...)`` if available; missing methods are
    a hard error since asking to load weights into a policy that cannot
    accept them is a programming bug, not a runtime condition.
    """
    policies: Mapping[str, Policy] = _worker_state["policies"]
    for agent_id, sd in state_dicts.items():
        if agent_id not in policies:
            raise KeyError(
                f"policy_state_dicts has agent {agent_id!r} but worker has "
                f"no such policy"
            )
        loader = getattr(policies[agent_id], "load_state_dict", None)
        if not callable(loader):
            raise AttributeError(
                f"policy[{agent_id!r}] of type "
                f"{type(policies[agent_id]).__name__} has no load_state_dict; "
                f"cannot apply state_dict broadcast"
            )
        loader(sd)


def _worker_run_one(
    seed: int,
    options: Optional[Mapping[str, Any]],
) -> Episode:
    """Run a single episode in this worker and return its :class:`Episode`."""
    runner: EpisodeRunner = _worker_state["runner"]
    recorder: EpisodeRecorder = _worker_state["recorder"]
    runner.run_episode(seed=int(seed), options=dict(options) if options else None)
    return recorder.get_last_episode()


# ---------------------------------------------------------------------------
# ParallelRollouter
# ---------------------------------------------------------------------------
class ParallelRollouter(AbstractContextManager):
    """Collect :class:`Episode`s in parallel from a blueprint + policies.

    Parameters
    ----------
    blueprint:
        Single source of truth for the environment. Pickled (via
        ``to_dict``) into worker processes; the workers each call
        :meth:`EnvBlueprint.build`.
    policy_factories:
        ``{agent_id: factory}``. Each factory is invoked once per worker
        to build that worker's policy instance. **Factories must be
        top-level importable** when ``num_workers > 1`` (no lambdas /
        closures), otherwise pickling fails.
    num_workers:
        ``<= 1`` runs everything in the calling process. ``>= 2`` spawns
        a process pool that lives until :meth:`close`.
    observer_names_to_keep:
        Forwarded to :class:`EpisodeRecorder` so workers can drop the
        heavyweight observers (e.g. scoring snapshots) before pickling.
    deterministic:
        Calls ``policy.set_deterministic(True)`` on each policy that
        supports it. Silent no-op for policies that don't.
    debug_plugins_factory:
        Optional factory returning extra plugins to attach (each must
        have ``BLUEPRINT_EXCLUDE = True``). Called once per worker.
    mp_context:
        Override the multiprocessing start method (default ``"spawn"``;
        ``"fork"`` may be needed for very heavy GPU / library setups).
    """

    def __init__(
        self,
        blueprint: EnvBlueprint,
        policy_factories: Mapping[str, PolicyFactory],
        num_workers: int = 1,
        observer_names_to_keep: Optional[Sequence[str]] = None,
        deterministic: bool = False,
        debug_plugins_factory: Optional[DebugPluginsFactory] = None,
        mp_context: str = "spawn",
    ) -> None:
        if not policy_factories:
            raise ValueError("policy_factories must be non-empty")
        self._blueprint = blueprint
        self._policy_factories = dict(policy_factories)
        self._num_workers = int(max(1, num_workers))
        self._observer_names_to_keep = (
            list(observer_names_to_keep) if observer_names_to_keep is not None else None
        )
        self._deterministic = bool(deterministic)
        self._debug_plugins_factory = debug_plugins_factory
        self._mp_context = mp_context

        self._pool: Optional[mp.pool.Pool] = None
        self._closed = False

        # In-process state for num_workers <= 1; lazily built so callers
        # that only use parallel mode pay nothing.
        self._inprocess_initialized = False

        if self._num_workers > 1:
            self._spawn_pool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def blueprint(self) -> EnvBlueprint:
        return self._blueprint

    def collect(
        self,
        seeds: Sequence[int],
        policy_state_dicts: Optional[Mapping[str, Any]] = None,
        episode_options_fn: Optional[EpisodeOptionsFn] = None,
    ) -> EpisodeCollection:
        """Run one episode per seed and return a :class:`EpisodeCollection`.

        Episodes appear in the returned collection in the same order as
        ``seeds`` regardless of execution order across workers.
        """
        if self._closed:
            raise RuntimeError("ParallelRollouter is closed")
        seeds_list: List[int] = [int(s) for s in seeds]
        options_per_seed: List[Optional[Mapping[str, Any]]] = [
            (episode_options_fn(idx) if episode_options_fn is not None else None)
            for idx in range(len(seeds_list))
        ]

        if self._num_workers <= 1:
            episodes = self._collect_inprocess(
                seeds_list, options_per_seed, policy_state_dicts,
            )
        else:
            episodes = self._collect_pool(
                seeds_list, options_per_seed, policy_state_dicts,
            )

        collection = EpisodeCollection(self._blueprint)
        collection.extend(episodes)
        return collection

    def close(self) -> None:
        """Release worker pool / in-process runtime. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._inprocess_initialized:
            runtime = _worker_state.get("runtime")
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    _logger.exception("error closing in-process runtime")
            _worker_state.clear()
            self._inprocess_initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _spawn_pool(self) -> None:
        ctx = mp.get_context(self._mp_context)
        init_args = (
            self._blueprint.to_dict(),
            self._policy_factories,
            self._observer_names_to_keep,
            self._deterministic,
            self._debug_plugins_factory,
        )
        self._pool = ctx.Pool(
            processes=self._num_workers,
            initializer=_worker_init,
            initargs=init_args,
        )

    def _ensure_inprocess(self) -> None:
        if self._inprocess_initialized:
            return
        _worker_init(
            self._blueprint.to_dict(),
            self._policy_factories,
            self._observer_names_to_keep,
            self._deterministic,
            self._debug_plugins_factory,
        )
        self._inprocess_initialized = True

    def _collect_inprocess(
        self,
        seeds: Sequence[int],
        options_per_seed: Sequence[Optional[Mapping[str, Any]]],
        state_dicts: Optional[Mapping[str, Any]],
    ) -> List[Episode]:
        self._ensure_inprocess()
        if state_dicts:
            _worker_load_state_dicts(state_dicts)
        return [
            _worker_run_one(seed, options)
            for seed, options in zip(seeds, options_per_seed)
        ]

    def _collect_pool(
        self,
        seeds: Sequence[int],
        options_per_seed: Sequence[Optional[Mapping[str, Any]]],
        state_dicts: Optional[Mapping[str, Any]],
    ) -> List[Episode]:
        assert self._pool is not None
        # Broadcast new weights, if any. We dispatch one task per worker
        # so every worker's local policies are refreshed before the
        # rollout batch starts. ``num_workers`` apply tasks are enough
        # because each worker handles one apply at a time and there is
        # no cross-task ordering requirement.
        if state_dicts:
            apply_handles = [
                self._pool.apply_async(_worker_load_state_dicts, (dict(state_dicts),))
                for _ in range(self._num_workers)
            ]
            for handle in apply_handles:
                handle.get()

        # Fan out the seeds. ``imap`` preserves input order so episodes
        # come back aligned with ``seeds`` without extra bookkeeping.
        tasks = list(zip(seeds, options_per_seed))
        results: List[Episode] = list(
            self._pool.starmap(_worker_run_one, tasks)
        )
        return results


__all__ = ["ParallelRollouter", "PolicyFactory", "DebugPluginsFactory"]
