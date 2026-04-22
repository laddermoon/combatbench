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

import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np

from .env_runtime import EnvRuntime
from .policy import Policy, call_policy, coerce_action
from .recorder import PostActionRecorder


# Agent naming is project-scoped (1v1 combat); do NOT generalize without
# breaking downstream consumers that key by these exact strings.
AGENT_IDS: Tuple[str, str] = ("robot_a", "robot_b")

# Back-compat aliases: pre-split code imported these from episode_runner.
# The canonical location is :mod:`envs.framework.policy`.
_call_policy = call_policy
_coerce_action = coerce_action


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


@dataclass
class EpisodeResult:
    """Structured output of :meth:`EpisodeRunner.run_episode`."""
    seed: Optional[int]
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
            if not hasattr(policy, "act"):
                raise TypeError(
                    f"Policy for {agent!r} lacks required 'act' method "
                    f"(got {type(policy).__name__})"
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
    def run_episode(self, seed: Optional[int] = None) -> EpisodeResult:
        """Run a single episode. Returns a populated :class:`EpisodeResult`."""
        runtime_seed, policy_seeds = self._derive_seeds(seed)
        self._reset_all(runtime_seed, policy_seeds)

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
            seed=seed,
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
    ) -> List[EpisodeResult]:
        """Run ``n`` episodes, deriving a child seed per episode via
        :class:`numpy.random.SeedSequence` so the batch is reproducible from
        ``base_seed`` alone. ``base_seed=None`` produces a fresh random batch.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative; got {n}")
        if n == 0:
            return []
        seed_source = np.random.SeedSequence(base_seed)
        episode_seeds = seed_source.generate_state(n, dtype=np.uint32)
        return [self.run_episode(int(s)) for s in episode_seeds]

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
    def _derive_seeds(
        self, seed: Optional[int],
    ) -> Tuple[Optional[int], Dict[str, int]]:
        """Split a user seed into (runtime_seed, {agent: policy_seed}).

        Using SeedSequence children prevents accidental correlation between
        runtime RNG and policy RNGs (and between the two policies), which
        would otherwise be easy to introduce by sharing the raw seed.
        """
        if seed is None:
            # Explicitly propagate None to runtime/policies so they can
            # pick their own nondeterministic seeds.
            return None, {agent: None for agent in self.AGENT_IDS}  # type: ignore[return-value]
        ss = np.random.SeedSequence(int(seed))
        runtime_seed, *policy_seed_vals = (
            int(x) for x in ss.generate_state(1 + len(self.AGENT_IDS), dtype=np.uint32)
        )
        policy_seeds = dict(zip(self.AGENT_IDS, policy_seed_vals))
        return runtime_seed, policy_seeds

    def _reset_all(
        self,
        runtime_seed: Optional[int],
        policy_seeds: Mapping[str, Optional[int]],
    ) -> None:
        self.runtime.reset(seed=runtime_seed)
        for agent_id, policy in self.policies.items():
            reset_fn = getattr(policy, "reset", None)
            if callable(reset_fn):
                reset_fn(policy_seeds.get(agent_id))

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
