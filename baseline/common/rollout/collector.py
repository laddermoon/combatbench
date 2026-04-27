"""``RolloutCollector``: thin wrapper over :class:`EpisodeRunner`.

Design (see ``baseline/DESIGN.md`` §3.3):

  * Never reimplements the episode loop. Episode iteration, observer
    routing, ``ctx.episode_options`` plumbing, seeding, and recorder
    lifecycle are all delegated to :class:`EpisodeRunner` /
    :meth:`run_n_episodes`. This module only does:
      1. observer-binding assembly from agent-id templates;
      2. ``EpisodeResult.trajectories[agent] -> RolloutBatch`` translation;
      3. weight hot-reload via :meth:`TorchPolicyAdapter.load_state_dict`
         on each :meth:`collect` call.

  * Multi-controlled-agent from day one. ``policy_factories`` is a
    ``{agent_id: factory}`` mapping; pass two PPO-style adapters and the
    collector returns ``{agent_id: list[RolloutBatch]}`` for self-play.

  * **Parallel rollouts** (``max_workers > 1``). A persistent process
    pool is lazily built on the first :meth:`collect`; each worker
    constructs its own ``EnvRuntime`` + policies via the supplied
    factories and reuses them for every subsequent chunk. Per-iteration
    weight broadcast is carried inside the chunk task payload, so each
    worker :meth:`load_state_dict` s before running its slice of
    episodes — this mirrors the pattern that was known to train a
    stable standing policy in ``humanoid21/standing_grpo_rtg_tune.py``.
    Seeds are derived bit-identically to :meth:`EpisodeRunner.run_n_episodes`
    so sequential (``max_workers=1``) and parallel (``max_workers>1``)
    runs produce the same per-seed trajectories (modulo whatever
    non-determinism lives inside the simulator / policies themselves).

  * Parallel mode requires the factories, state_dicts, and options_fn
    outputs to be **picklable**. In practice that means top-level
    functions / classes for the factories, and CPU-side detached
    tensors for any state_dict you broadcast. :class:`TorchPolicyAdapter`
    already returns CPU-safe state_dicts.

  * ``store_initial_observation=True`` is forced on, so produced
    ``RolloutBatch.obs`` is always ``(T+1, *obs_shape)`` (the framework
    convention; :class:`RolloutBatch` validates this).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    AgentTrajectory,
    EpisodeResult,
    EpisodeRunner,
    ObserverBinding,
    RolloutConfig,
    _derive_batch_seeds,
    _resolve_seed,
)
from envs.framework.policy import Policy

from .batch import RolloutBatch

_logger = logging.getLogger("baseline.common.rollout.collector")


PolicyFactory = Callable[[], Policy]
RuntimeFactory = Callable[[], EnvRuntime]


class RolloutCollector:
    """Single-process rollout collector built on :class:`EpisodeRunner`.

    Parameters
    ----------
    runtime_factory:
        Zero-argument callable producing a fresh :class:`EnvRuntime`.
        Called **once** at first :meth:`collect`; the runtime is then
        reused via ``runtime.reset(...)`` for every subsequent episode
        (the same pattern :class:`MatchRunner` uses post-G3).
    policy_factories:
        Mapping ``{agent_id: () -> Policy}``. Must cover every agent
        the underlying :class:`EpisodeRunner` expects (currently
        ``("robot_a", "robot_b")``).
    capture_agents:
        Which agents have their trajectories converted to
        :class:`RolloutBatch`. ``None`` (default) captures all of them.
        Note that :class:`RolloutConfig.capture` only knows ``robot_a``
        and ``robot_b`` today — passing a different id raises ``KeyError``
        the moment the underlying runner needs to consult that flag.
    obs_observer_template / reward_observer_template:
        Format strings; ``{agent}`` is replaced by the agent id when
        building :class:`ObserverBinding` for each agent. Set
        ``reward_observer_template=None`` to skip the reward observer
        and use ``default_reward`` for every step (eval mode).
    reward_extractor / default_reward:
        Forwarded into each :class:`ObserverBinding`.
    store_extras:
        ``True`` (default) propagates ``act_with_extras`` from each
        policy into ``AgentTrajectory.extras`` — log_prob / value land in
        ``RolloutBatch.log_probs`` / ``.values`` automatically. Turn off
        for pure inference / eval.
    """

    def __init__(
        self,
        runtime_factory: RuntimeFactory,
        policy_factories: Mapping[str, PolicyFactory],
        *,
        capture_agents: Optional[Sequence[str]] = None,
        obs_observer_template: str = "{agent}_obs",
        reward_observer_template: Optional[str] = "{agent}_reward",
        reward_extractor: Optional[Callable[[Any], float]] = None,
        default_reward: float = 0.0,
        store_extras: bool = True,
        max_workers: int = 1,
        mp_start_method: str = "spawn",
    ) -> None:
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable.")
        if not policy_factories:
            raise ValueError("policy_factories must contain at least one agent.")
        for agent, factory in policy_factories.items():
            if not callable(factory):
                raise TypeError(
                    f"policy_factories[{agent!r}] must be callable, got "
                    f"{type(factory).__name__}"
                )

        self._runtime_factory = runtime_factory
        self._policy_factories: Dict[str, PolicyFactory] = dict(policy_factories)
        self._capture_agents: Optional[List[str]] = (
            list(capture_agents) if capture_agents is not None else None
        )
        self._obs_template = str(obs_observer_template)
        self._reward_template = reward_observer_template
        self._reward_extractor = reward_extractor
        self._default_reward = float(default_reward)
        self._store_extras = bool(store_extras)
        self._max_workers = max(1, int(max_workers))
        self._mp_start_method = str(mp_start_method)
        # Built lazily on the first collect() so users can construct a
        # RolloutCollector before MuJoCo / GPU resources are available.
        self._runner: Optional[EpisodeRunner] = None
        self._policies: Dict[str, Policy] = {}
        self._pool: Optional[Any] = None  # mp.pool.Pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(
        self,
        n: Optional[int] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        base_seed: Optional[int] = None,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        deterministic: Optional[bool] = None,
        state_dicts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, List[RolloutBatch]]:
        """Run a batch of episodes and return per-agent ``RolloutBatch`` lists.

        Either supply ``n`` (with optional ``base_seed``) — seeds are then
        derived deterministically by :meth:`EpisodeRunner.run_n_episodes`
        (matches :class:`ParallelRunner` derivation; see ``SEED.md``) —
        or supply an explicit ``seeds`` sequence and the collector loops
        :meth:`EpisodeRunner.run_episode` over it.

        ``options_fn(episode_index) -> options`` is forwarded to
        ``run_n_episodes`` (RESET.md G1) and is the recommended hook for
        per-episode curriculum knobs (see ``examples/03_training_aids.py``).

        ``deterministic`` toggles the deterministic mode of every adapter
        in :attr:`_policies` that exposes ``set_deterministic`` — e.g.
        :class:`TorchPolicyAdapter`. ``None`` (default) leaves the flag
        as it was on the last call (or the adapter's ctor default).

        ``state_dicts`` is ``{agent_id: state_dict}``; when provided each
        addressed policy receives a :meth:`load_state_dict` call before
        rollout starts. This is the training-loop hook for pushing fresh
        PPO weights without rebuilding the runtime.
        """
        self._ensure_built()
        if self._pool is not None:
            return self._collect_parallel(
                n=n,
                seeds=seeds,
                base_seed=base_seed,
                options_fn=options_fn,
                deterministic=deterministic,
                state_dicts=state_dicts,
            )
        self._apply_state_dicts(state_dicts)
        self._apply_deterministic(deterministic)

        results = self._run_episodes(
            n=n, seeds=seeds, base_seed=base_seed, options_fn=options_fn,
        )
        return self._batches_from_results(results)

    def close(self) -> None:
        """Release the underlying runtime and worker pool. Idempotent."""
        if self._runner is not None:
            close_fn = getattr(self._runner.runtime, "close", None)
            if callable(close_fn):
                close_fn()
            self._runner = None
            self._policies = {}
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:  # noqa: BLE001
                with suppress(Exception):
                    self._pool.terminate()
                    self._pool.join()
            self._pool = None

    def __enter__(self) -> "RolloutCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._max_workers > 1:
            if self._pool is not None:
                return
            ctx = mp.get_context(self._mp_start_method)
            blueprint = self._build_blueprint()
            self._pool = ctx.Pool(
                processes=self._max_workers,
                initializer=_init_worker,
                initargs=(blueprint,),
            )
            return
        if self._runner is not None:
            return
        runtime = self._runtime_factory()
        self._policies = {
            agent: factory() for agent, factory in self._policy_factories.items()
        }
        bindings = self._build_bindings()
        capture_a, capture_b = self._capture_flags()
        rollout_cfg = RolloutConfig(
            capture_a=capture_a,
            capture_b=capture_b,
            store_extras=self._store_extras,
            store_initial_observation=True,  # required for RolloutBatch invariant
        )
        self._runner = EpisodeRunner(
            runtime=runtime,
            policies=self._policies,
            observer_bindings=bindings,
            rollout=rollout_cfg,
        )

    def _build_blueprint(self) -> "_WorkerBlueprint":
        capture_a, capture_b = self._capture_flags()
        return _WorkerBlueprint(
            runtime_factory=self._runtime_factory,
            policy_factories=dict(self._policy_factories),
            obs_template=self._obs_template,
            reward_template=self._reward_template,
            reward_extractor=self._reward_extractor,
            default_reward=self._default_reward,
            store_extras=self._store_extras,
            capture_a=capture_a,
            capture_b=capture_b,
            capture_agents=(
                list(self._capture_agents)
                if self._capture_agents is not None
                else None
            ),
        )

    def _build_bindings(self) -> Dict[str, ObserverBinding]:
        bindings: Dict[str, ObserverBinding] = {}
        for agent in self._policy_factories:
            obs_name = self._obs_template.format(agent=agent)
            if self._reward_template is None:
                reward_name = None
            else:
                reward_name = self._reward_template.format(agent=agent)
            kwargs: Dict[str, Any] = {
                "obs_name": obs_name,
                "reward_name": reward_name,
                "default_reward": self._default_reward,
            }
            if self._reward_extractor is not None:
                kwargs["reward_extractor"] = self._reward_extractor
            bindings[agent] = ObserverBinding(**kwargs)
        return bindings

    def _capture_flags(self) -> tuple[bool, bool]:
        if self._capture_agents is None:
            return True, True
        capture_set = set(self._capture_agents)
        return ("robot_a" in capture_set), ("robot_b" in capture_set)

    # ------------------------------------------------------------------
    # Per-iteration hooks
    # ------------------------------------------------------------------
    def _apply_state_dicts(
        self,
        state_dicts: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> None:
        if not state_dicts:
            return
        for agent, sd in state_dicts.items():
            policy = self._policies.get(agent)
            if policy is None:
                raise KeyError(
                    f"state_dicts has key {agent!r} but no policy is registered "
                    f"for that agent_id. Known agents: {sorted(self._policies)}"
                )
            load_fn = getattr(policy, "load_state_dict", None)
            if not callable(load_fn):
                raise TypeError(
                    f"Policy for {agent!r} ({type(policy).__name__}) does not "
                    "implement load_state_dict; cannot hot-reload weights. "
                    "Wrap it in baseline.common.policies.TorchPolicyAdapter."
                )
            load_fn(sd)

    def _apply_deterministic(self, deterministic: Optional[bool]) -> None:
        if deterministic is None:
            return
        for agent, policy in self._policies.items():
            setter = getattr(policy, "set_deterministic", None)
            if callable(setter):
                setter(bool(deterministic))

    # ------------------------------------------------------------------
    # Drive EpisodeRunner
    # ------------------------------------------------------------------
    def _run_episodes(
        self,
        *,
        n: Optional[int],
        seeds: Optional[Sequence[int]],
        base_seed: Optional[int],
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]],
    ) -> List[EpisodeResult]:
        assert self._runner is not None  # _ensure_built ran
        if seeds is not None:
            if n is not None and n != len(seeds):
                raise ValueError(
                    f"Got both n={n} and seeds (len={len(seeds)}); they must "
                    "agree. Pass exactly one of them."
                )
            results: List[EpisodeResult] = []
            for index, seed in enumerate(seeds):
                opts = options_fn(index) if options_fn is not None else None
                results.append(self._runner.run_episode(seed=int(seed), options=opts))
            return results
        if n is None:
            raise ValueError("collect() requires either n=... or seeds=...")
        return self._runner.run_n_episodes(
            n, base_seed=base_seed, options_fn=options_fn,
        )

    # ------------------------------------------------------------------
    # Parallel rollout
    # ------------------------------------------------------------------
    def _collect_parallel(
        self,
        *,
        n: Optional[int],
        seeds: Optional[Sequence[int]],
        base_seed: Optional[int],
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]],
        deterministic: Optional[bool],
        state_dicts: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Dict[str, List[RolloutBatch]]:
        """Parallel path: chunk episodes across workers, merge in seed order.

        Derivation of seeds is bit-identical to
        :meth:`EpisodeRunner.run_n_episodes` (via :func:`_derive_batch_seeds`),
        so switching between ``max_workers=1`` and ``max_workers>1`` does
        not change which seeds are executed.
        """
        assert self._pool is not None

        # 1) Resolve the full (seed, options) list on the main process.
        if seeds is not None:
            if n is not None and n != len(seeds):
                raise ValueError(
                    f"Got both n={n} and seeds (len={len(seeds)}); they must "
                    "agree. Pass exactly one of them."
                )
            seed_list: List[int] = [int(s) for s in seeds]
        else:
            if n is None:
                raise ValueError("collect() requires either n=... or seeds=...")
            seed_list = [
                int(s) for s in _derive_batch_seeds(_resolve_seed(base_seed), n)
            ]
        options_list: List[Optional[Dict[str, Any]]] = [
            options_fn(i) if options_fn is not None else None
            for i in range(len(seed_list))
        ]
        if not seed_list:
            return {}

        # 2) Validate state_dicts keys BEFORE shipping (can't raise from a worker).
        sd_payload: Optional[Dict[str, Dict[str, Any]]] = None
        if state_dicts:
            known = set(self._policy_factories.keys())
            for agent in state_dicts:
                if agent not in known:
                    raise KeyError(
                        f"state_dicts has key {agent!r} but no policy is "
                        f"registered for that agent_id. Known agents: "
                        f"{sorted(known)}"
                    )
            sd_payload = {agent: dict(sd) for agent, sd in state_dicts.items()}

        # 3) Chunk. One task per worker so state_dict broadcast cost is
        #    amortized per-worker, not per-episode.
        n_chunks = min(self._max_workers, len(seed_list))
        chunk_indices = [
            c.tolist()
            for c in np.array_split(np.arange(len(seed_list)), n_chunks)
            if len(c) > 0
        ]
        tasks: List[Dict[str, Any]] = []
        for chunk_idx in chunk_indices:
            items = [(seed_list[i], options_list[i]) for i in chunk_idx]
            tasks.append(
                {
                    "items": items,
                    "state_dicts": sd_payload,
                    "deterministic": (
                        None if deterministic is None else bool(deterministic)
                    ),
                }
            )

        # 4) Submit. ``imap`` preserves task-submission order, which matches
        #    seed-submission order because chunks are contiguous slices.
        merged: Dict[str, List[RolloutBatch]] = {}
        try:
            for chunk_result in self._pool.imap(_worker_run_chunk, iter(tasks)):
                for agent, batches in chunk_result.items():
                    merged.setdefault(agent, []).extend(batches)
        except BaseException:
            # Pool is poisoned — shut it down so __exit__ doesn't hang.
            self._hard_kill_pool()
            raise
        return merged

    def _hard_kill_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            finally:
                self._pool = None

    # ------------------------------------------------------------------
    # Trajectory -> RolloutBatch
    # ------------------------------------------------------------------
    def _batches_from_results(
        self,
        results: Sequence[EpisodeResult],
    ) -> Dict[str, List[RolloutBatch]]:
        captured = self._capture_agents
        batches: Dict[str, List[RolloutBatch]] = {}
        for episode_index, result in enumerate(results):
            for agent, traj in result.trajectories.items():
                if traj is None:
                    continue
                if captured is not None and agent not in captured:
                    continue
                batches.setdefault(agent, []).append(
                    _trajectory_to_batch(traj, episode_result=result)
                )
        return batches


# ---------------------------------------------------------------------------
# Pure helper: AgentTrajectory -> RolloutBatch
# ---------------------------------------------------------------------------
def _trajectory_to_batch(
    traj: AgentTrajectory,
    *,
    episode_result: EpisodeResult,
) -> RolloutBatch:
    if not traj.actions:
        raise ValueError(
            f"AgentTrajectory for {traj.agent_id!r} has zero steps; cannot "
            "convert to RolloutBatch (the episode produced no actions). "
            "If this is a pre-episode termination case, filter it upstream."
        )
    obs_array = np.asarray(traj.observations, dtype=np.float32)
    actions_array = np.asarray(traj.actions, dtype=np.float32)
    rewards_array = np.asarray(traj.rewards, dtype=np.float32)

    log_probs_array = _stack_extras(traj.extras, "log_prob")
    values_array = _stack_extras(traj.extras, "value")

    # Gymnasium convention: exactly one of (terminated, truncated) is
    # true for a terminated episode. The EnvRuntime intentionally allows
    # both to fire simultaneously when an MDP-terminal condition (e.g.
    # KO / fall) and the timeout plugin both trigger on the same step
    # (see ``EnvRuntime.get_termination_flags`` docstring). RolloutBatch
    # -space requires mutual exclusivity: collapse to ``terminated=True``
    # (the MDP-terminal condition would have ended the episode regardless
    # of the timeout). The raw flags + reasons remain available in
    # ``info['termination_reasons']`` for users who need the full story.
    terminated = bool(traj.terminated)
    truncated = bool(traj.truncated) and not terminated

    info: Dict[str, Any] = {
        "seed": episode_result.seed,
        "num_steps": episode_result.num_steps,
        "termination_reasons": list(episode_result.termination_reasons),
    }

    batch = RolloutBatch(
        agent_id=traj.agent_id,
        obs=obs_array,
        actions=actions_array,
        rewards=rewards_array,
        terminated=terminated,
        truncated=truncated,
        log_probs=log_probs_array,
        values=values_array,
        info=info,
    )
    batch.validate()
    return batch


def _stack_extras(
    extras: Sequence[Dict[str, Any]],
    key: str,
) -> Optional[np.ndarray]:
    """Stack one extras key into a (T,) float32 array, or return None."""
    if not extras:
        return None
    if not all(key in e for e in extras):
        return None
    return np.asarray([float(e[key]) for e in extras], dtype=np.float32)


# ---------------------------------------------------------------------------
# Parallel-rollout worker
# ---------------------------------------------------------------------------
@dataclass
class _WorkerBlueprint:
    """Picklable recipe for reconstructing an ``EpisodeRunner`` in a worker.

    Everything a worker needs to build its own runtime + policies +
    bindings, derived from :class:`RolloutCollector`'s ctor args. Crosses
    the process boundary once (at :func:`_init_worker` time) and then
    lives as a module global inside the worker for the rest of its life.
    """

    runtime_factory: RuntimeFactory
    policy_factories: Dict[str, PolicyFactory]
    obs_template: str
    reward_template: Optional[str]
    reward_extractor: Optional[Callable[[Any], float]]
    default_reward: float
    store_extras: bool
    capture_a: bool
    capture_b: bool
    capture_agents: Optional[List[str]] = None


# Module globals live in the *worker* process only. The main process
# never touches them. Each worker process runs ``_init_worker`` exactly
# once (via ``Pool(initializer=...)``) and then many ``_worker_run_chunk``
# calls reuse the cached runner.
_WORKER_BLUEPRINT: Optional[_WorkerBlueprint] = None
_WORKER_RUNNER: Optional[EpisodeRunner] = None
_WORKER_POLICIES: Dict[str, Policy] = {}


def _init_worker(blueprint: _WorkerBlueprint) -> None:  # pragma: no cover - child process
    """Pool initializer. Stashes the blueprint; runtime is built lazily.

    Deferring the runtime construction to the first task keeps pool
    startup cheap (no MuJoCo allocation on a worker that ends up idle).
    We also clamp torch to single-threaded here so N workers don't
    fight each other on BLAS pools — the standard
    ``ProcessPoolExecutor(initializer=...)`` idiom from
    ``humanoid21/standing_grpo_rtg_tune.py``.
    """
    global _WORKER_BLUEPRINT, _WORKER_RUNNER, _WORKER_POLICIES
    _WORKER_BLUEPRINT = blueprint
    _WORKER_RUNNER = None
    _WORKER_POLICIES = {}
    try:
        import torch  # noqa: WPS433 - optional dep

        torch.set_num_threads(1)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _worker_ensure_runner() -> Tuple[EpisodeRunner, Dict[str, Policy]]:
    """Build the cached ``EpisodeRunner`` + policies on first use."""
    global _WORKER_RUNNER, _WORKER_POLICIES
    assert _WORKER_BLUEPRINT is not None, "_init_worker did not run"
    if _WORKER_RUNNER is not None:
        return _WORKER_RUNNER, _WORKER_POLICIES
    bp = _WORKER_BLUEPRINT
    runtime = bp.runtime_factory()
    policies: Dict[str, Policy] = {
        agent: factory() for agent, factory in bp.policy_factories.items()
    }
    bindings: Dict[str, ObserverBinding] = {}
    for agent in bp.policy_factories:
        obs_name = bp.obs_template.format(agent=agent)
        reward_name = (
            None if bp.reward_template is None
            else bp.reward_template.format(agent=agent)
        )
        kwargs: Dict[str, Any] = {
            "obs_name": obs_name,
            "reward_name": reward_name,
            "default_reward": bp.default_reward,
        }
        if bp.reward_extractor is not None:
            kwargs["reward_extractor"] = bp.reward_extractor
        bindings[agent] = ObserverBinding(**kwargs)
    rollout_cfg = RolloutConfig(
        capture_a=bp.capture_a,
        capture_b=bp.capture_b,
        store_extras=bp.store_extras,
        store_initial_observation=True,
    )
    _WORKER_RUNNER = EpisodeRunner(
        runtime=runtime,
        policies=policies,
        observer_bindings=bindings,
        rollout=rollout_cfg,
    )
    _WORKER_POLICIES = policies
    return _WORKER_RUNNER, _WORKER_POLICIES


def _worker_run_chunk(task: Dict[str, Any]) -> Dict[str, List[RolloutBatch]]:  # pragma: no cover - child process
    """Task function: apply state_dicts, then run a chunk of episodes.

    Converts trajectories into :class:`RolloutBatch` inside the worker
    so only lightweight numpy arrays cross the pickle boundary back to
    the main process (``EpisodeResult`` itself is big and deeply nested).
    """
    runner, policies = _worker_ensure_runner()
    assert _WORKER_BLUEPRINT is not None  # set by _init_worker
    bp = _WORKER_BLUEPRINT

    # 1) Hot-reload weights.
    sd_payload = task.get("state_dicts")
    if sd_payload:
        for agent, sd in sd_payload.items():
            policy = policies.get(agent)
            if policy is None:
                # Main-proc validation already filters this; stay defensive.
                continue
            load_fn = getattr(policy, "load_state_dict", None)
            if callable(load_fn):
                load_fn(sd)

    # 2) Deterministic toggle (if the policies support it).
    deterministic = task.get("deterministic")
    if deterministic is not None:
        for policy in policies.values():
            setter = getattr(policy, "set_deterministic", None)
            if callable(setter):
                setter(bool(deterministic))

    # 3) Run this worker's slice of episodes.
    out: Dict[str, List[RolloutBatch]] = {}
    for seed, options in task["items"]:
        result = runner.run_episode(seed=int(seed), options=options)
        for agent, traj in result.trajectories.items():
            if traj is None:
                continue
            if bp.capture_agents is not None and agent not in bp.capture_agents:
                continue
            out.setdefault(agent, []).append(
                _trajectory_to_batch(traj, episode_result=result)
            )
    return out
