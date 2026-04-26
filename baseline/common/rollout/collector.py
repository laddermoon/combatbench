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

  * MVP scope: ``max_workers=1``. Multi-worker training-time weight
    sync (broadcasting actor state_dicts to a persistent
    :class:`ParallelRunner` pool each iteration) is a separate problem
    — see DESIGN.md §6 follow-ups. Eval-style fixed-weight parallel
    rollouts can be done by the caller via :class:`ParallelRunner`
    directly until that lands.

  * ``store_initial_observation=True`` is forced on, so produced
    ``RolloutBatch.obs`` is always ``(T+1, *obs_shape)`` (the framework
    convention; :class:`RolloutBatch` validates this).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import (
    AgentTrajectory,
    EpisodeResult,
    EpisodeRunner,
    ObserverBinding,
    RolloutConfig,
)
from envs.framework.policy import Policy

from .batch import RolloutBatch


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
    ) -> None:
        if max_workers != 1:
            raise NotImplementedError(
                "RolloutCollector MVP supports max_workers=1 only. For fixed-"
                "weight parallel eval use envs.framework.parallel_runner.ParallelRunner; "
                "training-time multi-worker collection is a follow-up — see "
                "baseline/DESIGN.md §6."
            )
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
        # Built lazily on the first collect() so users can construct a
        # RolloutCollector before MuJoCo / GPU resources are available.
        self._runner: Optional[EpisodeRunner] = None
        self._policies: Dict[str, Policy] = {}

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
        self._apply_state_dicts(state_dicts)
        self._apply_deterministic(deterministic)

        results = self._run_episodes(
            n=n, seeds=seeds, base_seed=base_seed, options_fn=options_fn,
        )
        return self._batches_from_results(results)

    def close(self) -> None:
        """Release the underlying runtime. Idempotent."""
        if self._runner is not None:
            close_fn = getattr(self._runner.runtime, "close", None)
            if callable(close_fn):
                close_fn()
            self._runner = None
            self._policies = {}

    def __enter__(self) -> "RolloutCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------
    def _ensure_built(self) -> None:
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
        terminated=bool(traj.terminated),
        truncated=bool(traj.truncated),
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
