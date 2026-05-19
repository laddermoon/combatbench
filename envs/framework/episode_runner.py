"""Episode-level runner: glue ``EnvRuntime`` + two policies.

``EpisodeRunner`` is the layer above :class:`EnvRuntime` that drives a single
episode forward. It owns the "run one episode" loop so individual consumers
(RL trainers, eval scripts, visualization tools, behavioural cloning dataset
builders, …) do not each reimplement the same ``policy.act → runtime.step``
glue.

Scope
-----
This runner is **specific to the 1-vs-1 combat project** — it hard-codes two
agents named ``robot_a`` / ``robot_b``. It is not trying to be a generic
multi-agent framework.

Responsibilities
----------------
1. Hold a live :class:`EnvRuntime` plus two :class:`Policy`-protocol objects.
2. On each step: pull each agent's observation from a fixed-name observer
   plugin (``robot_a_obs`` / ``robot_b_obs``), call ``policy.act(obs)``,
   forward both actions to ``runtime.step``.
3. Manage seeds deterministically via :class:`numpy.random.SeedSequence`:
   one ``base_seed`` → one reproducible episode (``None`` is resolved at
   entry to a concrete ``uint32`` so every episode is loggable / replayable;
   see ``envs/framework/SEED.md``).

Non-responsibilities
--------------------
- **Data recording / trajectory capture.** That is the job of
  :class:`PostActionRecorder` instances **attached directly to the runtime**
  (``runtime.attach_recorder(...)``). Recorders see the same ``ctx`` and
  observer outputs as everything else, and run regardless of whether the
  loop driver is this :class:`EpisodeRunner` or some other harness. If you
  used to read trajectories off the runner's return value, attach a recorder
  and read its on-disk output (or write a custom in-memory recorder).
- **Reward extraction.** The runner does not pull or interpret rewards. RL
  trainers should attach a recorder that snapshots the reward observer's
  output per step, or read ``runtime.get_observer_output(...)`` themselves
  via their own callback / wrapper.
- **Result aggregation.** :meth:`run_episode` returns ``None``. Anything you
  want to know about the episode lives in the attached recorders.
- **Combat semantics (winner / HP / damage).** Those belong to a subclass
  (e.g. :class:`envs.framework.round_runner.CombatRoundRunner`) or to a
  post-hoc reducer over recorder data.
- **Process-level parallelism.** This runner is constructed inside each
  worker; cross-process orchestration is handled by ``parallel_runner``.
- **Batch / multi-episode driving.** ``run_n_episodes`` is intentionally
  not provided here — outer loops (training updates, eval sweeps, dataset
  collection) own batch semantics and can call :meth:`run_episode` in a loop.

Observer name convention
------------------------
Each agent's observation is read from a runtime observer plugin whose name
follows the framework convention ``"<agent_id>_obs"`` —
``runtime.observer_plugins["robot_a_obs"]`` and
``runtime.observer_plugins["robot_b_obs"]``. Both must be registered before
:meth:`run_episode` is called (the runner validates this lazily on first
use). This is the only contract between :class:`EpisodeRunner` and the
observer system; everything else (rewards, metrics, debug probes) is the
recorder's / consumer's business.

Example
-------
.. code-block:: python

    from envs.framework import EnvRuntime, EpisodeRunner

    runtime = EnvRuntime(
        simulator=...,
        plugins=[...],
        observer_plugins={
            "robot_a_obs": ObsPlugin(agent="robot_a"),
            "robot_b_obs": ObsPlugin(agent="robot_b"),
            # Reward / debug observers are fine to register here too —
            # the runner ignores them; recorders / consumers read them.
        },
    )
    runtime.attach_recorder(MyTrajectoryRecorder())  # data collection lives here

    runner = EpisodeRunner(
        runtime=runtime,
        policies={"robot_a": policy_a, "robot_b": policy_b},
    )
    runner.run_episode(seed=42)
    # Inspect recorder state for results.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .env_runtime import EnvRuntime
from .plugin import BasePlugin
from .policy import Policy, call_policy

_logger = logging.getLogger(__name__)


# Agent naming is project-scoped (1v1 combat); do NOT generalize without
# breaking downstream consumers that key by these exact strings.
AGENT_IDS: Tuple[str, str] = ("robot_a", "robot_b")

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


@dataclass(frozen=True)
class _EpisodeSeeds:
    """Concrete per-consumer seeds for a single episode (internal).

    See ``envs/framework/SEED.md`` for the derivation rules. This structure
    is *internal* to the runner — it is not persisted. Only ``base`` is the
    user-facing handle (``ctx.base_seed`` and recorder manifests record it);
    the rest are recomputed deterministically from ``base`` + the current
    plugin/policy configuration.
    """
    base: int
    runtime: int
    policies: Dict[str, int]           # agent_id -> int
    plugins: Dict[int, int]            # id(plugin) -> int


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class EpisodeRunner:
    """Runs episodes through :class:`EnvRuntime` with two policies attached.

    See module docstring for scope, responsibilities, and the
    ``<agent>_obs`` observer-name convention. **Data recording and reward
    extraction are NOT this class's job** — attach a
    :class:`PostActionRecorder` to the runtime instead.
    """

    AGENT_IDS: Tuple[str, str] = AGENT_IDS

    def __init__(
        self,
        runtime: EnvRuntime,
        policies: Mapping[str, Policy],
    ) -> None:
        self.runtime = runtime
        self._validate_policies(policies)
        self.policies: Dict[str, Policy] = dict(policies)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_episode(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run a single episode end-to-end. Returns ``None``.

        Anything the caller wants to know about the episode is read from
        the attached recorders' state after this call returns. The
        runner intentionally does not aggregate per-step or per-episode
        data — see the module docstring's Non-responsibilities section.

        ``seed=None`` is resolved to a concrete ``uint32`` via
        :func:`secrets.randbits(32)` at entry; the resolved value is
        published on ``ctx.base_seed`` and is what recorders persist into
        their manifests for replay (see ``framework/SEED.md``).

        ``options`` is forwarded to :meth:`EnvRuntime.reset` and published
        on ``ctx.episode_options`` for plugins / observers / recorders to
        read per-episode parameters (HP carry-over, curriculum knobs,
        opponent snapshot id, …). See ``framework/RESET.md`` §4.
        """
        base_seed = _resolve_seed(seed)
        episode_seeds = self._derive_seeds(base_seed)
        self._reset_all(episode_seeds, options=options)

        # Initial observation pulled BEFORE the first step so the policy
        # sees ``ctx`` AS-OF reset (matches standard RL conventions). We do
        # NOT store it — the runner has no trajectory buffer; recorders
        # snapshot whatever they need on their own ``on_pre_episode`` hook.
        obs_a, obs_b = self.runtime.get_observation()
        last_obs = {"robot_a": obs_a, "robot_b": obs_b}

        while self.runtime.is_episode_active:
            actions: Dict[str, np.ndarray] = {}
            extras: Dict[str, Optional[Dict[str, Any]]] = {}
            for agent_id in self.AGENT_IDS:
                # ``want_extras=True`` asks the policy for its side-channel
                # payload (log_prob / value / sampling info / …). The runner
                # itself never inspects ``policy_extras`` — it only forwards
                # the per-agent bundle to ``runtime.step`` so that attached
                # recorders can persist it alongside the action snapshot.
                # Policies that emit no extras return ``{}``; we forward
                # ``None`` in that case to keep recorder schemas tidy
                # (empty-dict vs missing is a meaningless distinction here).
                action, policy_extras = call_policy(
                    self.policies[agent_id],
                    last_obs[agent_id],
                    want_extras=True,
                )
                actions[agent_id] = action
                extras[agent_id] = policy_extras if policy_extras else None

            self.runtime.step(
                actions["robot_a"],
                actions["robot_b"],
                action_a_extra=extras["robot_a"],
                action_b_extra=extras["robot_b"],
            )

            # Termination check uses runtime flags; recorders' post-step
            # hooks have already fired inside ``runtime.step`` so they see
            # the just-applied state.
            terminated, truncated = self.runtime.get_termination_flags()
            if terminated or truncated:
                break

            obs_a, obs_b = self.runtime.get_observation()
            last_obs = {"robot_a": obs_a, "robot_b": obs_b}

    def close(self) -> None:
        """Close attached policies that support it.

        Runtime lifecycle is owned by the caller — we intentionally do
        NOT close the runtime here to keep the runner a thin composition
        layer.
        """
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

    def _derive_seeds(self, base_seed: int) -> _EpisodeSeeds:
        """Derive a concrete :class:`_EpisodeSeeds` bundle from ``base_seed``.

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

        return _EpisodeSeeds(
            base=int(base_seed),
            runtime=_leaf(runtime_ss),
            policies={agent: _leaf(ss) for agent, ss in zip(self.AGENT_IDS, policy_sss)},
            plugins={id(plugin): _leaf(ss) for plugin, ss in zip(seedable_plugins, plugin_sss)},
        )

    def _reset_all(
        self,
        seeds: _EpisodeSeeds,
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

