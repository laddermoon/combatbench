"""``RolloutCollector``: thin RL-loop adapter on top of :class:`ParallelRunner`.

Design (see ``baseline/DESIGN.md`` §3.3):

  * Never reimplements the episode loop. Episode iteration, observer
    routing, ``ctx.episode_options`` plumbing, seeding, recorder
    lifecycle, **and the persistent worker pool** are all delegated to
    the framework: this module only does what is genuinely RL-loop
    specific —

      1. observer-binding assembly from ``{agent}_obs`` / ``{agent}_reward``
         templates;
      2. per-iteration ``state_dict`` broadcast (``TorchPolicyAdapter``-
         style hot-reload of fresh PPO/GRPO weights without rebuilding
         the runtime);
      3. ``deterministic`` toggle propagation (eval vs. training);
      4. ``EpisodeResult.trajectories[agent].as_rollout_batch(...)``
         translation, performed *inside the worker* so only the small
         frozen :class:`RolloutBatch` view crosses the pickle boundary.

  * Multi-controlled-agent from day one. ``policy_factories`` is a
    ``{agent_id: factory}`` mapping; pass two PPO-style adapters and
    the collector returns ``{agent_id: list[RolloutBatch]}`` for
    self-play.

  * Sequential and parallel modes share **one** code path:
    :meth:`ParallelRunner.map_chunks` dispatches the same chunk
    function over the in-process runner (``max_workers <= 1``) or
    across worker processes (``max_workers > 1``). Seed derivation is
    bit-identical to :meth:`EpisodeRunner.run_n_episodes`, so flipping
    ``max_workers`` does not change which seeds are executed (modulo
    whatever non-determinism lives inside the simulator / policies
    themselves).

  * Parallel mode requires factories, state_dicts, and ``options_fn``
    outputs to be **picklable**. In practice that means top-level
    functions / classes for the factories, and CPU-side detached
    tensors for any state_dict you broadcast. :class:`TorchPolicyAdapter`
    already returns CPU-safe state_dicts.

  * ``store_initial_observation=True`` is forced on, so produced
    ``RolloutBatch.obs`` is always ``(T+1, *obs_shape)`` (the framework
    convention; :class:`RolloutBatch.validate` enforces this).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence,
)

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    EpisodeRunner,
    ObserverBinding,
    RolloutConfig,
)
from envs.framework.parallel_runner import ParallelRunner
from envs.framework.policy import Policy
from envs.framework.rollout_batch import RolloutBatch

_logger = logging.getLogger("baseline.common.rollout.collector")


PolicyFactory = Callable[[], Policy]
RuntimeFactory = Callable[[], EnvRuntime]


# ---------------------------------------------------------------------------
# RolloutCollector
# ---------------------------------------------------------------------------
class RolloutCollector:
    """RL-loop rollout collector built on :class:`ParallelRunner`.

    Parameters
    ----------
    runtime_factory:
        Zero-argument callable producing a fresh :class:`EnvRuntime`.
        Called **once per worker** at first :meth:`collect`; the
        runtime is then reused via ``runtime.reset(...)`` for every
        subsequent episode (same pattern :class:`MatchRunner` uses
        post-G3). For parallel mode (``max_workers > 1``) the factory
        must be top-level / picklable — no lambdas, no closures over
        un-picklable state.
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
        policy into ``AgentTrajectory.extras`` — log_prob / value land
        in ``RolloutBatch.log_probs`` / ``.values`` automatically. Turn
        off for pure inference / eval.
    max_workers:
        ``<= 1`` runs in-process (debugging / unit tests / small N).
        ``> 1`` spins a persistent worker pool via :class:`ParallelRunner`.
    mp_start_method:
        ``"spawn"`` (default — safest with MuJoCo / CUDA / torch),
        ``"forkserver"``, or ``"fork"``. Only honored when
        ``max_workers > 1``.
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

        self._policy_factories: Dict[str, PolicyFactory] = dict(policy_factories)
        self._capture_agents: Optional[List[str]] = (
            list(capture_agents) if capture_agents is not None else None
        )
        capture_a, capture_b = self._capture_flags()
        self._blueprint = _RunnerBlueprint(
            runtime_factory=runtime_factory,
            policy_factories=dict(policy_factories),
            obs_template=str(obs_observer_template),
            reward_template=reward_observer_template,
            reward_extractor=reward_extractor,
            default_reward=float(default_reward),
            store_extras=bool(store_extras),
            capture_a=capture_a,
            capture_b=capture_b,
        )
        self._max_workers = max(1, int(max_workers))
        self._mp_start_method = str(mp_start_method)
        # Built lazily on the first collect() so users can construct a
        # RolloutCollector before MuJoCo / GPU resources are available.
        self._parallel: Optional[ParallelRunner] = None

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

        Either supply ``n`` (with optional ``base_seed``) — seeds are
        then derived deterministically by
        :meth:`ParallelRunner._build_tasks` (which uses the same
        :func:`numpy.random.SeedSequence` derivation as
        :meth:`EpisodeRunner.run_n_episodes`; see ``SEED.md``) — or
        supply an explicit ``seeds`` sequence.

        ``options_fn(episode_index) -> options`` is forwarded as a
        per-episode dict, exactly as in :meth:`ParallelRunner.run`
        (RESET.md G1) and is the recommended hook for per-episode
        curriculum knobs (see ``examples/03_training_aids.py``).

        ``deterministic`` toggles the deterministic mode of every
        adapter that exposes ``set_deterministic`` — e.g.
        :class:`TorchPolicyAdapter`. ``None`` (default) leaves the flag
        as it was on the last call (or the adapter's ctor default).

        ``state_dicts`` is ``{agent_id: state_dict}``; when provided
        each addressed policy receives a :meth:`load_state_dict` call
        before rollout starts. This is the training-loop hook for
        pushing fresh PPO weights without rebuilding the runtime. Keys
        are validated against ``policy_factories`` on the main process
        before shipping to workers.
        """
        # Validate state_dicts keys before paying any worker / pool cost.
        if state_dicts:
            self._validate_state_dict_keys(state_dicts)

        # Resolve seeds + per-episode options on the main process.
        seed_list, options_list = self._resolve_seed_options(
            n=n, seeds=seeds, base_seed=base_seed, options_fn=options_fn,
        )
        if not seed_list:
            return {}

        # Chunk so per-task overhead (state_dict broadcast) is amortized
        # per worker, not per episode. In-process mode collapses to one
        # chunk; multi-worker mode hands one chunk per worker.
        n_chunks = min(self._max_workers, len(seed_list))
        chunks: List[List[int]] = self._chunk_indices(len(seed_list), n_chunks)
        sd_payload: Optional[Dict[str, Dict[str, Any]]] = (
            {agent: dict(sd) for agent, sd in state_dicts.items()}
            if state_dicts else None
        )
        det_payload = None if deterministic is None else bool(deterministic)
        capture_payload = (
            list(self._capture_agents)
            if self._capture_agents is not None else None
        )

        tasks: List[Dict[str, Any]] = []
        for chunk_idx in chunks:
            items = [(seed_list[i], options_list[i]) for i in chunk_idx]
            tasks.append(
                {
                    "items": items,
                    "state_dicts": sd_payload,
                    "deterministic": det_payload,
                    "capture_agents": capture_payload,
                }
            )

        # Drive the framework. Strict mode is what we want here — a
        # crashing rollout should NOT silently halve the on-policy batch.
        parallel = self._ensure_parallel()
        merged: Dict[str, List[RolloutBatch]] = {}
        for chunk_result in parallel.map_chunks(tasks, _rollout_chunk_fn):
            for agent, batches in chunk_result.items():
                merged.setdefault(agent, []).extend(batches)
        return merged

    def close(self) -> None:
        """Release the underlying worker pool. Idempotent."""
        if self._parallel is not None:
            try:
                self._parallel.close()
            finally:
                self._parallel = None

    def __enter__(self) -> "RolloutCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_parallel(self) -> ParallelRunner:
        if self._parallel is not None:
            return self._parallel
        factory = _BlueprintRunnerFactory(self._blueprint)
        self._parallel = ParallelRunner(
            factory,
            num_workers=self._max_workers,
            mp_start_method=self._mp_start_method,
            strict=True,  # on-policy RL must not silently lose episodes
        )
        return self._parallel

    def _capture_flags(self) -> tuple[bool, bool]:
        if self._capture_agents is None:
            return True, True
        capture_set = set(self._capture_agents)
        return ("robot_a" in capture_set), ("robot_b" in capture_set)

    def _validate_state_dict_keys(
        self,
        state_dicts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        known = set(self._policy_factories.keys())
        for agent in state_dicts:
            if agent not in known:
                raise KeyError(
                    f"state_dicts has key {agent!r} but no policy is "
                    f"registered for that agent_id. Known agents: "
                    f"{sorted(known)}"
                )

    def _resolve_seed_options(
        self,
        *,
        n: Optional[int],
        seeds: Optional[Sequence[int]],
        base_seed: Optional[int],
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]],
    ) -> tuple[List[int], List[Optional[Dict[str, Any]]]]:
        """Resolve the (seeds, options) pair on the main process.

        Mirrors :meth:`EpisodeRunner.run_n_episodes` semantics exactly
        so that switching between explicit ``seeds=`` and ``n=`` +
        ``base_seed=`` produces the same set of episodes.
        """
        # Lazy import to keep parallel_runner the single source of truth.
        from envs.framework.parallel_runner import _derive_seeds

        if seeds is not None:
            if n is not None and n != len(seeds):
                raise ValueError(
                    f"Got both n={n} and seeds (len={len(seeds)}); they "
                    "must agree. Pass exactly one of them."
                )
            seed_list: List[int] = [int(s) for s in seeds]
        else:
            if n is None:
                raise ValueError(
                    "collect() requires either n=... or seeds=..."
                )
            seed_list = [int(s) for s in _derive_seeds(base_seed, n)]
        options_list: List[Optional[Dict[str, Any]]] = [
            options_fn(i) if options_fn is not None else None
            for i in range(len(seed_list))
        ]
        return seed_list, options_list

    @staticmethod
    def _chunk_indices(total: int, n_chunks: int) -> List[List[int]]:
        """Slice ``range(total)`` into ``n_chunks`` contiguous index lists."""
        if total <= 0 or n_chunks <= 0:
            return []
        n_chunks = min(n_chunks, total)
        # Equivalent to numpy.array_split but pure-Python so this module
        # does not need to import numpy just for chunking.
        base, extra = divmod(total, n_chunks)
        out: List[List[int]] = []
        start = 0
        for i in range(n_chunks):
            size = base + (1 if i < extra else 0)
            out.append(list(range(start, start + size)))
            start += size
        return out


# ---------------------------------------------------------------------------
# Picklable runner factory (top-level so ParallelRunner can pickle it).
# ---------------------------------------------------------------------------
@dataclass
class _RunnerBlueprint:
    """Picklable recipe for reconstructing an :class:`EpisodeRunner`.

    Carries everything a worker needs to build its own runtime +
    policies + observer bindings + rollout config. Crosses the pickle
    boundary once, when :class:`ParallelRunner` ships the
    :class:`_BlueprintRunnerFactory` to each worker.
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


class _BlueprintRunnerFactory:
    """Top-level callable: ``(worker_id) -> EpisodeRunner``.

    A class (not a closure) so :mod:`pickle` can serialize it under
    ``"spawn"`` start method. The blueprint is held on the instance and
    used to build a fresh runner inside each worker. ``worker_id`` is
    accepted to satisfy :data:`envs.framework.parallel_runner.RunnerFactory`
    but is not used by the default rollout setup.
    """

    def __init__(self, blueprint: _RunnerBlueprint) -> None:
        self._bp = blueprint

    def __call__(self, worker_id: int) -> EpisodeRunner:
        bp = self._bp
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
            store_initial_observation=True,  # required by RolloutBatch
        )
        return EpisodeRunner(
            runtime=runtime,
            policies=policies,
            observer_bindings=bindings,
            rollout=rollout_cfg,
        )


# ---------------------------------------------------------------------------
# Top-level chunk function (handed to ParallelRunner.map_chunks).
# ---------------------------------------------------------------------------
def _rollout_chunk_fn(
    runner: EpisodeRunner,
    task: Dict[str, Any],
) -> Dict[str, List[RolloutBatch]]:
    """Apply per-iteration knobs, run the chunk's episodes, freeze trajectories.

    Lives at module scope so :class:`ParallelRunner.map_chunks` can
    pickle a reference to it under the ``"spawn"`` start method.

    Task shape::

        {
            "items": [(seed: int, options: Optional[dict]), ...],
            "state_dicts": Optional[{agent_id: state_dict}],
            "deterministic": Optional[bool],
            "capture_agents": Optional[List[str]],
        }

    Returns ``{agent_id: List[RolloutBatch]}``. The conversion happens
    *inside the worker*, so only the small frozen :class:`RolloutBatch`
    view (numpy arrays) crosses the pickle boundary back — full
    :class:`EpisodeResult` objects with all their plugin-shared-info
    snapshots stay worker-local.
    """
    # 1) Hot-reload weights (training-loop on-policy update).
    sd_payload = task.get("state_dicts")
    if sd_payload:
        for agent, sd in sd_payload.items():
            policy = runner.policies.get(agent)
            if policy is None:
                # Main-proc validation already filters this; stay defensive.
                continue
            load_fn = getattr(policy, "load_state_dict", None)
            if callable(load_fn):
                load_fn(sd)

    # 2) Deterministic toggle (eval vs. training).
    deterministic = task.get("deterministic")
    if deterministic is not None:
        for policy in runner.policies.values():
            setter = getattr(policy, "set_deterministic", None)
            if callable(setter):
                setter(bool(deterministic))

    # 3) Run the chunk's episodes; freeze each trajectory in-worker.
    capture_agents = task.get("capture_agents")
    out: Dict[str, List[RolloutBatch]] = {}
    for seed, options in task["items"]:
        result = runner.run_episode(seed=int(seed), options=options)
        for agent, traj in result.trajectories.items():
            if traj is None:
                continue
            if capture_agents is not None and agent not in capture_agents:
                continue
            # Per-agent reward observers may expose ``episode_summary()``
            # to publish derived per-episode scalars (e.g. weighted-reward
            # component breakdowns for curriculum gating). When present,
            # the dict is merged into ``RolloutBatch.info``.
            info: Dict[str, Any] = {}
            reward_obs_name = f"{agent}_reward"
            reward_obs = runner.runtime.observer_plugins.get(reward_obs_name)
            summary_fn = getattr(reward_obs, "episode_summary", None) if reward_obs is not None else None
            if callable(summary_fn):
                try:
                    summary = summary_fn()
                except Exception:
                    summary = None
                if isinstance(summary, dict):
                    info.update(summary)
            out.setdefault(agent, []).append(
                traj.as_rollout_batch(result, info=info if info else None)
            )
    return out
