"""Episode-level runner: glue ``EnvRuntime`` + two policies + rollout capture.

``EpisodeRunner`` is the layer above :class:`EnvRuntime` that most training,
evaluation, and data-collection code should talk to. It owns the "run one
episode" loop so individual consumers (RL trainers, eval scripts,
visualization tools, behavioural cloning dataset builders, …) do not each
reimplement the same ``policy.act → runtime.step → pull obs/reward`` glue.

Scope
-----
This runner is **specific to the 1-vs-1 combat project** — it hard-codes two
agents named ``robot_a`` / ``robot_b``. It is not trying to be a generic
multi-agent framework.

Responsibilities
----------------
1. Hold a live :class:`EnvRuntime` plus two :class:`Policy`-protocol objects.
2. On each step: pull each agent's observation from a named observer plugin,
   call ``policy.act(obs)``, forward both actions to the runtime, pull the
   per-agent reward from another named observer plugin.
3. Record everything into a configurable :class:`AgentTrajectory` buffer —
   capture for each side independently toggleable.
4. Manage seeds deterministically via :class:`numpy.random.SeedSequence`:
   one ``base_seed`` → one reproducible episode OR a reproducible batch.
5. Forward :class:`PostActionRecorder` instances to the runtime so disk
   recording is just a kwarg away.

Non-responsibilities
--------------------
- Reward computation. Reward is **read** from observer plugins; it is not
  computed inside the runner. If no reward plugin is bound for an agent,
  that agent's ``trajectory.rewards`` will be filled with zeros (or
  whatever :attr:`ObserverBinding.default_reward` says).
- Combat semantics (winner / HP / damage). Those belong to a subclass
  (see :class:`envs.framework.round_runner.CombatRoundRunner`) or to a
  post-hoc reducer over :attr:`EpisodeResult.shared_info_final`.
- Process-level parallelism. See module docstring of
  ``envs.framework.parallel_runner`` (future) for the outer layer; this
  runner is designed to be constructed inside each worker process via a
  factory, not pickled across processes.

Example
-------
.. code-block:: python

    from envs.framework import EnvRuntime, EpisodeRunner, RolloutConfig

    runtime = EnvRuntime(
        simulator=..., plugins=[...],
        observer_plugins={
            "robot_a_obs":    ObsPlugin(agent="robot_a"),
            "robot_a_reward": RewardPlugin(agent="robot_a"),
            "robot_b_obs":    ObsPlugin(agent="robot_b"),
            "robot_b_reward": RewardPlugin(agent="robot_b"),
        },
    )
    runner = EpisodeRunner(
        runtime=runtime,
        policies={"robot_a": policy_a, "robot_b": policy_b},
        rollout=RolloutConfig(capture_a=True, capture_b=False),
    )
    results = runner.run_n_episodes(100, base_seed=42)
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np

from .env_runtime import EnvRuntime
from .plugin import BasePlugin
from .policy import Policy, call_policy, coerce_action
from .recorder import PostActionRecorder
from .rollout_batch import RolloutBatch

_logger = logging.getLogger(__name__)


# Agent naming is project-scoped (1v1 combat); do NOT generalize without
# breaking downstream consumers that key by these exact strings.
AGENT_IDS: Tuple[str, str] = ("robot_a", "robot_b")

# Back-compat aliases: pre-split code imported these from episode_runner.
# The canonical location is :mod:`envs.framework.policy`.
_call_policy = call_policy
_coerce_action = coerce_action


# ---------------------------------------------------------------------------
# Seed helpers (see envs/framework/SEED.md)
# ---------------------------------------------------------------------------
def _resolve_seed(seed: Optional[int]) -> int:
    """Resolve ``None`` to a concrete ``uint32`` seed.

    Per the framework's seeding contract, ``None`` never propagates past
    the runner boundary — every episode has a concrete, loggable,
    reproducible base seed. When the caller passes ``None`` we draw one
    from :func:`secrets.randbits(32)` so it is independent of any
    per-process ``np.random`` state.
    """
    if seed is None:
        return int(secrets.randbits(32))
    return int(seed)


def _derive_batch_seeds(base_seed: int, n: int) -> np.ndarray:
    """Return ``n`` per-episode ``uint32`` seeds from a resolved ``base_seed``.

    Kept byte-for-byte equivalent across :class:`EpisodeRunner` and
    :class:`ParallelRunner` so switching execution mode does not change
    which seeds are run. The top-level split is ``SeedSequence(base_seed)``
    and each episode slot is drawn via ``generate_state`` — each episode
    thereafter spawns its own per-consumer tree inside the runner.
    """
    return np.random.SeedSequence(int(base_seed)).generate_state(n, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Observer binding
# ---------------------------------------------------------------------------
def default_reward_extractor(raw: Any) -> float:
    """Extract a scalar reward from whatever an observer plugin returned.

    Contract (in order of preference):
      1. ``None`` → raises ``ValueError`` (caller's bug; bind None explicitly
         via :attr:`ObserverBinding.reward_name=None` to skip reward pull).
      2. Python scalar (``int``/``float``/``bool``) → ``float(raw)``.
      3. Numpy scalar / 0-d array / 1-element array → ``float(raw.item())``.
      4. ``dict`` → first hit of ``"reward"`` / ``"total_reward"`` / ``"r"``.
         If none present, raises ``KeyError`` listing available keys.
      5. Anything else → ``TypeError``.

    Users can override per-binding via :attr:`ObserverBinding.reward_extractor`
    when their reward plugin returns something fancier (e.g. per-term
    breakdowns in a dataclass).
    """
    if raw is None:
        raise ValueError(
            "Reward observer returned None. Either bind reward_name=None "
            "to skip reward, or fix the observer plugin to return a value."
        )
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, np.ndarray):
        flat = raw.reshape(-1)
        if flat.size != 1:
            raise TypeError(
                f"Reward ndarray must be scalar (size 1); got shape {raw.shape}."
            )
        return float(flat[0])
    if hasattr(raw, "item") and not isinstance(raw, dict):
        # numpy scalar, torch scalar, etc.
        try:
            return float(raw.item())
        except Exception:
            pass  # fall through to dict / error branches
    if isinstance(raw, dict):
        for key in ("reward", "total_reward", "r"):
            if key in raw:
                return float(raw[key])
        raise KeyError(
            f"Reward dict has no recognized scalar key; available: "
            f"{sorted(raw.keys())}. Provide a custom reward_extractor to "
            f"ObserverBinding if your plugin uses a different key."
        )
    raise TypeError(f"Cannot extract reward from {type(raw).__name__}: {raw!r}")


@dataclass(frozen=True)
class ObserverBinding:
    """Tell :class:`EpisodeRunner` which observer plugins feed an agent.

    Parameters
    ----------
    obs_name: name of the observer plugin whose ``get_output()`` is passed
        verbatim to the policy's ``act``. Must be registered on the runtime.
    reward_name: optional observer plugin name for the reward scalar. If
        ``None``, ``trajectory.rewards`` is filled with ``default_reward``
        on every step (useful for pure evaluation runs).
    reward_extractor: callable applied to the reward plugin's raw output
        to get a ``float``. Defaults to :func:`default_reward_extractor`.
    default_reward: value used when ``reward_name is None``.
    """
    obs_name: str
    reward_name: Optional[str] = None
    reward_extractor: Callable[[Any], float] = default_reward_extractor
    default_reward: float = 0.0


def default_bindings() -> Dict[str, ObserverBinding]:
    """Standard ``{agent_id: ObserverBinding}`` following the ``<agent>_obs`` /
    ``<agent>_reward`` naming convention used by every baseline in this repo.
    """
    return {
        agent: ObserverBinding(
            obs_name=f"{agent}_obs", reward_name=f"{agent}_reward",
        )
        for agent in AGENT_IDS
    }


# ---------------------------------------------------------------------------
# Rollout config
# ---------------------------------------------------------------------------
@dataclass
class RolloutConfig:
    """What the runner persists into :class:`AgentTrajectory` buffers.

    Independently toggle each side. Turning everything off still produces a
    valid :class:`EpisodeResult` (with ``trajectories`` values set to
    ``None``) — useful for pure evaluation runs where only
    ``shared_info_final`` / ``termination_reasons`` matter.
    """
    capture_a: bool = True
    capture_b: bool = True
    store_extras: bool = False
    # The initial observation (pre-first-step) is stored at trajectory[0],
    # so len(observations) == len(actions) + 1. Turn off to save memory
    # when the initial observation is never needed (e.g. bandit-style).
    store_initial_observation: bool = True
    # Snapshot ``runtime.get_shared_info()`` into a per-step list. Off by
    # default because it deep-copies metrics/events every step and most
    # consumers only need the final snapshot.
    store_shared_info_per_step: bool = False

    def capture(self, agent_id: str) -> bool:
        return {"robot_a": self.capture_a, "robot_b": self.capture_b}[agent_id]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class AgentTrajectory:
    """Rollout buffer for a single agent across one episode.

    Length invariants (when :attr:`RolloutConfig.store_initial_observation`
    is ``True``, the default):
        ``len(observations) == len(actions) + 1 == len(rewards) + 1``
        ``len(extras)       == len(actions)``  (only when ``store_extras``)

    When ``store_initial_observation`` is ``False``:
        ``len(observations) == len(actions) == len(rewards)``
    """
    agent_id: str
    observations: List[Any] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    extras: List[Dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    # ------------------------------------------------------------------
    # RL-style views
    # ------------------------------------------------------------------
    def to_gymnasium_style(self) -> Tuple[bool, bool]:
        """Coerce framework flags to Gymnasium ``(terminated, truncated)``.

        :class:`EnvRuntime` intentionally allows both flags to fire on the
        same step (e.g. a KO and a timeout landing on the exact same
        action), because both are objectively true and downstream
        consumers may want to know. Standard RL pipelines (PPO/GAE,
        Gymnasium) on the other hand require **mutual exclusivity**:
        ``terminated=True`` means "MDP-terminal — bootstrap with 0",
        ``truncated=True`` means "time-limit — bootstrap with V(s')".

        Coercion rule: ``terminated`` wins. Rationale: the MDP-terminal
        condition would have ended the episode regardless of the
        timeout, so for value-bootstrap purposes it is the correct
        signal. The raw flags remain available on ``self``; the
        per-episode :attr:`EpisodeResult.termination_reasons` carries
        the full story.
        """
        terminated = bool(self.terminated)
        truncated = bool(self.truncated) and not terminated
        return terminated, truncated

    def as_rollout_batch(
        self,
        episode_result: Optional["EpisodeResult"] = None,
        *,
        extras_keys: Tuple[str, str] = ("log_prob", "value"),
        info: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> RolloutBatch:
        """Freeze this trajectory into a :class:`RolloutBatch`.

        This is the canonical bridge between the live, mutable rollout
        buffer used by :class:`EpisodeRunner` and the numpy-only,
        algorithm-friendly contract consumed by RL algorithms / eval
        pipelines. Algorithm packages (PPO, GRPO, …) should never need
        to write their own translation; they call this method.

        Parameters
        ----------
        episode_result:
            Optional :class:`EpisodeResult` carrying per-episode metadata
            (``seed``, ``num_steps``, ``termination_reasons``). When
            supplied, those fields land in :attr:`RolloutBatch.info`.
            Provide ``None`` if you only have the trajectory itself
            (e.g. a unit test) — ``info`` will then be empty unless
            an explicit dict is passed.
        extras_keys:
            ``(log_prob_key, value_key)``. The two extras keys that get
            stacked into :attr:`RolloutBatch.log_probs` /
            :attr:`RolloutBatch.values`. Either can be ``""`` to skip.
            Default matches what :class:`Policy.act_with_extras` returns
            for PPO-style actors.
        info:
            Extra ``info`` entries merged on top of the
            ``episode_result``-derived defaults (caller wins on key
            collision). Useful for stamping curriculum / opponent ids.
        validate:
            Run :meth:`RolloutBatch.validate` before returning. Default
            on; turn off in hot training loops once the contract is
            known-good.

        Raises
        ------
        ValueError
            If the trajectory has zero steps (``len(actions) == 0``):
            we cannot synthesize a meaningful RL batch from a pre-step
            termination. Filter such episodes upstream.
        """
        if not self.actions:
            raise ValueError(
                f"AgentTrajectory for {self.agent_id!r} has zero steps; "
                "cannot convert to RolloutBatch (the episode produced no "
                "actions). If this is a pre-episode termination case, "
                "filter it upstream."
            )

        obs_array = np.asarray(self.observations, dtype=np.float32)
        actions_array = np.asarray(self.actions, dtype=np.float32)
        rewards_array = np.asarray(self.rewards, dtype=np.float32)

        log_key, value_key = extras_keys
        log_probs_array = (
            _stack_extras(self.extras, log_key) if log_key else None
        )
        values_array = (
            _stack_extras(self.extras, value_key) if value_key else None
        )

        terminated, truncated = self.to_gymnasium_style()

        merged_info: Dict[str, Any] = {}
        if episode_result is not None:
            merged_info["seed"] = episode_result.seed
            merged_info["num_steps"] = episode_result.num_steps
            merged_info["termination_reasons"] = list(
                episode_result.termination_reasons
            )
        if info:
            merged_info.update(info)

        batch = RolloutBatch(
            agent_id=self.agent_id,
            obs=obs_array,
            actions=actions_array,
            rewards=rewards_array,
            terminated=terminated,
            truncated=truncated,
            log_probs=log_probs_array,
            values=values_array,
            info=merged_info,
        )
        if validate:
            batch.validate()
        return batch


def _stack_extras(
    extras: Sequence[Dict[str, Any]],
    key: str,
) -> Optional[np.ndarray]:
    """Stack one extras key into a ``(T,)`` float32 array, or return ``None``.

    Returns ``None`` when ``extras`` is empty or when any step is
    missing the key — the latter signals that the policy did not
    produce that quantity (e.g. eval-only policy with no log_prob),
    in which case the algorithm side is expected to leave the
    corresponding ``RolloutBatch`` slot empty.
    """
    if not extras:
        return None
    if not all(key in e for e in extras):
        return None
    return np.asarray([float(e[key]) for e in extras], dtype=np.float32)


@dataclass(frozen=True)
class EpisodeSeeds:
    """Concrete per-consumer seeds for a single episode.

    See ``envs/framework/SEED.md`` for the derivation rules. This structure
    is *internal* to the runner — it is not persisted. Only ``base`` is
    recorded on :class:`EpisodeResult` / in :class:`Recorder` manifests;
    the rest are recomputed deterministically from ``base`` + the current
    plugin/policy configuration.
    """
    base: int
    runtime: int
    policies: Dict[str, int]           # agent_id -> int
    plugins: Dict[int, int]            # id(plugin) -> int


@dataclass
class EpisodeResult:
    """Structured output of :meth:`EpisodeRunner.run_episode`.

    ``seed`` is the *resolved* base seed used for this episode — always an
    ``int`` (never ``None``). ``None`` inputs are resolved at the runner
    entry via :func:`secrets.randbits(32)`. Re-running with this seed plus
    the same code reproduces the episode byte-for-byte.
    """
    seed: int
    num_steps: int
    wall_time_sec: float
    terminated: bool
    truncated: bool
    termination_reasons: List[str]
    shared_info_final: Dict[str, Any]
    trajectories: Dict[str, Optional[AgentTrajectory]]


@dataclass
class StepContext:
    """Payload handed to an :class:`EpisodeRunner` ``on_step`` hook."""
    step_index: int                       # 1-based; 1 after first step
    observations: Dict[str, Any]          # pre-step obs per agent
    actions: Dict[str, np.ndarray]        # action this step per agent
    rewards: Dict[str, float]             # reward this step per agent
    shared_info: Dict[str, Any]           # live shared_info snapshot
    terminated: bool
    truncated: bool


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class EpisodeRunner:
    """Runs episodes through :class:`EnvRuntime` with two policies attached.

    See module docstring for scope and responsibilities.
    """

    AGENT_IDS: Tuple[str, str] = AGENT_IDS

    def __init__(
        self,
        runtime: EnvRuntime,
        policies: Mapping[str, Policy],
        *,
        observer_bindings: Optional[Mapping[str, ObserverBinding]] = None,
        rollout: Optional[RolloutConfig] = None,
        recorders: Sequence[PostActionRecorder] = (),
        on_step: Optional[Callable[[StepContext], None]] = None,
        on_episode_end: Optional[Callable[[EpisodeResult], None]] = None,
    ) -> None:
        self.runtime = runtime
        self._validate_policies(policies)
        self.policies: Dict[str, Policy] = dict(policies)
        self.rollout = rollout if rollout is not None else RolloutConfig()
        self.observer_bindings: Dict[str, ObserverBinding] = dict(
            observer_bindings if observer_bindings is not None else default_bindings()
        )
        self._validate_bindings()
        self._on_step = on_step
        self._on_episode_end = on_episode_end
        # Attach recorders to the runtime up front. `attach_recorder` is
        # idempotent so re-attaching across runner lifetimes is safe.
        for recorder in recorders:
            self.runtime.attach_recorder(recorder)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_policies(self, policies: Mapping[str, Policy]) -> None:
        missing = set(self.AGENT_IDS) - set(policies)
        extra = set(policies) - set(self.AGENT_IDS)
        if missing or extra:
            raise ValueError(
                f"EpisodeRunner.policies must have exactly keys {self.AGENT_IDS}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        for agent, policy in policies.items():
            if not isinstance(policy, Policy):
                raise TypeError(
                    f"Policy for {agent!r} must subclass "
                    f"envs.framework.policy.Policy; got {type(policy).__name__}"
                )

    def _validate_bindings(self) -> None:
        """Fail loudly at construction if any bound observer plugin is
        missing. Catching this at episode-start saves hours of "why is the
        rollout all zeros" debugging."""
        missing_agents = set(self.AGENT_IDS) - set(self.observer_bindings)
        if missing_agents:
            raise ValueError(
                f"observer_bindings missing entries for {sorted(missing_agents)}"
            )
        registered = set(self.runtime.observer_plugins.keys())
        for agent, binding in self.observer_bindings.items():
            if binding.obs_name not in registered:
                raise KeyError(
                    f"Observer plugin {binding.obs_name!r} (for obs of "
                    f"{agent!r}) is not registered on the runtime. "
                    f"Registered: {sorted(registered)}"
                )
            if binding.reward_name is not None and binding.reward_name not in registered:
                raise KeyError(
                    f"Observer plugin {binding.reward_name!r} (for reward of "
                    f"{agent!r}) is not registered on the runtime. "
                    f"Registered: {sorted(registered)}"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_episode(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> EpisodeResult:
        """Run a single episode. Returns a populated :class:`EpisodeResult`.

        ``seed=None`` is resolved to a concrete ``uint32`` via
        :func:`secrets.randbits(32)` at entry and written back to the
        returned :attr:`EpisodeResult.seed` — the episode is always
        reproducible from that value (see ``framework/SEED.md``).

        ``options`` is forwarded to :meth:`EnvRuntime.reset` and published
        on ``ctx.episode_options`` for plugins / observers / recorders to
        read per-episode parameters (HP carry-over, curriculum knobs,
        opponent snapshot id, …). See ``framework/RESET.md`` §4.
        """
        base_seed = _resolve_seed(seed)
        episode_seeds = self._derive_seeds(base_seed)
        self._reset_all(episode_seeds, options=options)

        trajectories = self._init_trajectories()
        start = time.monotonic()
        step_idx = 0

        # Capture initial observation BEFORE any step so trajectories have
        # T+1 observations / T actions (standard RL convention).
        initial_obs = self._pull_all_observations()
        if self.rollout.store_initial_observation:
            for agent_id, traj in trajectories.items():
                if traj is not None:
                    traj.observations.append(initial_obs[agent_id])

        last_obs = initial_obs
        while self.runtime.is_episode_active:
            actions: Dict[str, np.ndarray] = {}
            extras: Dict[str, Dict[str, Any]] = {}
            for agent_id in self.AGENT_IDS:
                action, agent_extras = _call_policy(
                    self.policies[agent_id],
                    last_obs[agent_id],
                    want_extras=self.rollout.store_extras,
                )
                actions[agent_id] = action
                extras[agent_id] = agent_extras

            self.runtime.step(actions["robot_a"], actions["robot_b"])
            step_idx += 1

            rewards = self._pull_all_rewards()
            next_obs = self._pull_all_observations()
            terminated, truncated = self.runtime.get_termination_flags()

            for agent_id, traj in trajectories.items():
                if traj is None:
                    continue
                traj.actions.append(actions[agent_id])
                traj.rewards.append(rewards[agent_id])
                if self.rollout.store_extras:
                    traj.extras.append(extras[agent_id])
                if self.rollout.store_initial_observation:
                    traj.observations.append(next_obs[agent_id])
                else:
                    # Without initial obs, store pre-step obs so each action
                    # is still aligned with the obs that produced it.
                    traj.observations.append(last_obs[agent_id])

            if self._on_step is not None:
                self._on_step(StepContext(
                    step_index=step_idx,
                    observations=last_obs,
                    actions=actions,
                    rewards=rewards,
                    shared_info=self.runtime.get_shared_info(),
                    terminated=terminated,
                    truncated=truncated,
                ))

            last_obs = next_obs
            if terminated or truncated:
                break

        # Record per-agent terminated/truncated flags (shared for all).
        terminated, truncated = self.runtime.get_termination_flags()
        for traj in trajectories.values():
            if traj is not None:
                traj.terminated = terminated
                traj.truncated = truncated

        shared_info_final = self.runtime.get_shared_info()
        result = EpisodeResult(
            seed=base_seed,
            num_steps=step_idx,
            wall_time_sec=time.monotonic() - start,
            terminated=terminated,
            truncated=truncated,
            termination_reasons=list(shared_info_final.get("termination_reasons", [])),
            shared_info_final=shared_info_final,
            trajectories=trajectories,
        )
        if self._on_episode_end is not None:
            self._on_episode_end(result)
        return result

    def run_n_episodes(
        self,
        n: int,
        *,
        base_seed: Optional[int] = None,
        options_fn: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
    ) -> List[EpisodeResult]:
        """Run ``n`` episodes, deriving a child seed per episode via
        :class:`numpy.random.SeedSequence` so the batch is reproducible from
        ``base_seed`` alone.

        ``base_seed=None`` is resolved at entry (see
        :func:`_resolve_seed`) and logged; the resolved value is what each
        :attr:`EpisodeResult.seed` records for its own derivation, making
        the batch reproducible even when the caller didn't supply a seed.

        ``options_fn(episode_index) -> options_dict`` is the canonical
        per-episode-params channel for curriculum / opponent-pool / HP
        carry-over (see ``framework/RESET.md`` §4). It is called once per
        episode with the 0-based index; return ``None`` for "no options".
        """
        if n < 0:
            raise ValueError(f"n must be non-negative; got {n}")
        if n == 0:
            return []
        batch_seed = _resolve_seed(base_seed)
        _logger.info("run_n_episodes: base_seed=%d, n=%d", batch_seed, n)
        episode_seeds = _derive_batch_seeds(batch_seed, n)
        results: List[EpisodeResult] = []
        for episode_index, episode_seed in enumerate(episode_seeds):
            options = options_fn(episode_index) if options_fn is not None else None
            results.append(self.run_episode(int(episode_seed), options=options))
        return results

    def close(self) -> None:
        """Close attached policies that support it. Runtime lifecycle is
        owned by the caller — we intentionally do NOT close the runtime
        here to keep the runner a thin composition layer."""
        for policy in self.policies.values():
            close_fn = getattr(policy, "close", None)
            if callable(close_fn):
                close_fn()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _seedable_plugins(self) -> Tuple[BasePlugin, ...]:
        """Return plugins that override :meth:`BasePlugin.set_episode_seed`.

        Ordering follows :meth:`EnvRuntime.plugins` (priority-sorted, stable),
        which is what we use to allocate seeds — so as long as the plugin
        set and their priorities don't change, each plugin gets a stable
        slot in the derivation tree.
        """
        return tuple(
            p for p in self.runtime.plugins
            if type(p).set_episode_seed is not BasePlugin.set_episode_seed
        )

    def _derive_seeds(self, base_seed: int) -> EpisodeSeeds:
        """Derive a concrete :class:`EpisodeSeeds` bundle from ``base_seed``.

        Uses :meth:`numpy.random.SeedSequence.spawn` end-to-end so any
        sub-consumer that wants to keep spawning its own children (e.g. a
        plugin with multiple RNGs) can do so off a clean child sequence.
        Leaf ``int`` values for the final consumers are extracted via
        ``generate_state(1, dtype=uint32)[0]``.

        See ``envs/framework/SEED.md``.
        """
        seedable_plugins = self._seedable_plugins()
        n_consumers = 1 + len(self.AGENT_IDS) + len(seedable_plugins)
        children = np.random.SeedSequence(int(base_seed)).spawn(n_consumers)
        runtime_ss, *rest = children
        policy_sss = rest[: len(self.AGENT_IDS)]
        plugin_sss = rest[len(self.AGENT_IDS):]

        def _leaf(ss: np.random.SeedSequence) -> int:
            return int(ss.generate_state(1, dtype=np.uint32)[0])

        return EpisodeSeeds(
            base=int(base_seed),
            runtime=_leaf(runtime_ss),
            policies={agent: _leaf(ss) for agent, ss in zip(self.AGENT_IDS, policy_sss)},
            plugins={id(plugin): _leaf(ss) for plugin, ss in zip(seedable_plugins, plugin_sss)},
        )

    def _reset_all(
        self,
        seeds: EpisodeSeeds,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Apply ``seeds`` and ``options`` to every consumer.

        Order matters:
          1. ``plugin.set_episode_seed`` rebuilds each plugin's RNG
             immediately — **before** ``runtime.reset`` because reset
             triggers ``on_pre_episode`` where seeded plugins consume
             their RNG to sample initial-state deltas, push intervals, etc.
          2. ``runtime.reset(seed, options, base_seed)`` clears ctx,
             publishes ``ctx.base_seed`` and ``ctx.episode_options``,
             drives the simulator (with ``seeds.runtime`` and ``options``),
             then fires the plugin lifecycle hooks. See
             ``envs/framework/RESET.md`` §3 for the full chain.
          3. ``policy.reset`` reseeds each policy's RNG.
        """
        for plugin in self._seedable_plugins():
            plugin.set_episode_seed(seeds.plugins[id(plugin)])
        self.runtime.reset(
            seed=seeds.runtime,
            options=options,
            base_seed=seeds.base,
        )
        for agent_id, policy in self.policies.items():
            reset_fn = getattr(policy, "reset", None)
            if callable(reset_fn):
                reset_fn(seeds.policies[agent_id])

    def _init_trajectories(self) -> Dict[str, Optional[AgentTrajectory]]:
        return {
            agent_id: AgentTrajectory(agent_id=agent_id) if self.rollout.capture(agent_id) else None
            for agent_id in self.AGENT_IDS
        }

    def _pull_all_observations(self) -> Dict[str, Any]:
        return {
            agent: self.runtime.get_observer_output(self.observer_bindings[agent].obs_name)
            for agent in self.AGENT_IDS
        }

    def _pull_all_rewards(self) -> Dict[str, float]:
        rewards: Dict[str, float] = {}
        for agent in self.AGENT_IDS:
            binding = self.observer_bindings[agent]
            if binding.reward_name is None:
                rewards[agent] = binding.default_reward
                continue
            raw = self.runtime.get_observer_output(binding.reward_name)
            rewards[agent] = binding.reward_extractor(raw)
        return rewards
