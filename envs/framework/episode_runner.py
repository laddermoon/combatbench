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
1. Hold a live :class:`EnvRuntime` plus two :class:`Policy` instances.
2. On each step: pull each agent's observation via
   ``runtime.get_observation()`` (which delegates to the simulator),
   call ``policy.act(obs, want_extra=True)``, and forward both
   actions to ``runtime.step``.
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
  (e.g. :class:`envs.framework.round_runner.RoundRunner`) or to a
  post-hoc reducer over recorder data.
- **Process-level parallelism.** This runner is constructed inside each
  worker; cross-process orchestration is handled by ``parallel_runner``.
- **Batch / multi-episode driving.** ``run_n_episodes`` is intentionally
  not provided here — outer loops (training updates, eval sweeps, dataset
  collection) own batch semantics and can call :meth:`run_episode` in a loop.

Observation contract
--------------------
The runner reads per-agent observations via ``runtime.get_observation()``,
which delegates to the simulator's ``get_observation()`` method. The exact
Python type of each observation is simulator-defined; the runner does not
inspect or coerce it. This is the only contract between
:class:`EpisodeRunner` and the observation system.

Observer plugins (rewarders, recorders, debug probes) are still registered
on the runtime, but the runner itself never reads them — it only forwards
the policy's action and optional ``extra`` payload to ``runtime.step`` so
that attached recorders can snapshot whatever they need.

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
        policy_a=policy_a,
        policy_b=policy_b,
    )
    runner.run_episode(seed=42)
    # Inspect recorder state for results.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .env_runtime import EnvRuntime
from .plugin import BasePlugin
from .policy import Policy

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
        policy_a: Policy,
        policy_b: Policy,
        post_termination_action: str = "policy",
    ) -> None:
        self.runtime = runtime
        if not isinstance(policy_a, Policy):
            raise TypeError(
                f"policy_a must subclass envs.framework.policy.Policy; "
                f"got {type(policy_a).__name__}"
            )
        if not isinstance(policy_b, Policy):
            raise TypeError(
                f"policy_b must subclass envs.framework.policy.Policy; "
                f"got {type(policy_b).__name__}"
            )
        self.policy_a = policy_a
        self.policy_b = policy_b
        if post_termination_action not in ("policy", "hold"):
            raise ValueError(
                f"post_termination_action must be 'policy' or 'hold'; "
                f"got {post_termination_action!r}"
            )
        self.post_termination_action = post_termination_action

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_episode(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        want_extras: bool = False,
        explore_intensity_a: Union[float, Callable[[np.ndarray, int], float]] = 0.0,
        explore_intensity_b: Union[float, Callable[[np.ndarray, int], float]] = 0.0,
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
        ``options`` must be **environment-only** — do not put
        policy-related fields here.

        ``want_extras`` controls whether each ``policy.act`` is called
        with ``want_extra=True``. Default is ``False``; set to ``True``
        when you need the policy's side-channel payload (log-prob /
        value estimates for on-policy RL, etc.).

        ``explore_intensity_a`` / ``explore_intensity_b`` are the
        exploration intensities for each policy.  Either a constant
        ``float`` (same value every step) or a callable
        ``(obs, step) -> float`` that is called each step to produce a
        per-frame value.  The runner records the resolved per-frame
        value alongside the observation.  Default ``0.0`` (neutral).
        """
        base_seed = _resolve_seed(seed)
        episode_seeds = self._derive_seeds(base_seed)
        self._reset_all(episode_seeds, options=options)

        obs_a, obs_b = self.runtime.get_observation()
        a_active = True
        b_active = True
        last_action_a: Optional[np.ndarray] = None
        last_action_b: Optional[np.ndarray] = None
        step = 0

        while not self.runtime.is_episode_over():
            ei_a = float(explore_intensity_a(obs_a, step)) if callable(explore_intensity_a) else float(explore_intensity_a)
            ei_b = float(explore_intensity_b(obs_b, step)) if callable(explore_intensity_b) else float(explore_intensity_b)

            if a_active or self.post_termination_action == "policy":
                action_a, extra_a = self.policy_a.act(
                    obs_a,
                    explore_intensity=ei_a,
                    want_extra=want_extras,
                )
                last_action_a = action_a
            else:
                if last_action_a is None:
                    raise RuntimeError("hold strategy requires at least one prior action")
                action_a, extra_a = last_action_a, None

            if b_active or self.post_termination_action == "policy":
                action_b, extra_b = self.policy_b.act(
                    obs_b,
                    explore_intensity=ei_b,
                    want_extra=want_extras,
                )
                last_action_b = action_b
            else:
                if last_action_b is None:
                    raise RuntimeError("hold strategy requires at least one prior action")
                action_b, extra_b = last_action_b, None

            self.runtime.step(
                action_a,
                action_b,
                action_a_extra=extra_a if extra_a else None,
                action_b_extra=extra_b if extra_b else None,
                explore_intensity_a=ei_a,
                explore_intensity_b=ei_b,
            )

            a_active = a_active and self.runtime.is_agent_active("robot_a")
            b_active = b_active and self.runtime.is_agent_active("robot_b")

            obs_a, obs_b = self.runtime.get_observation()
            step += 1

    def set_policy_a(self, policy: Policy) -> None:
        """Replace policy_a in-place (no env rebuild)."""
        if not isinstance(policy, Policy):
            raise TypeError(
                f"policy must subclass envs.framework.policy.Policy; "
                f"got {type(policy).__name__}"
            )
        self.policy_a = policy

    def set_policy_b(self, policy: Policy) -> None:
        """Replace policy_b in-place (no env rebuild)."""
        if not isinstance(policy, Policy):
            raise TypeError(
                f"policy must subclass envs.framework.policy.Policy; "
                f"got {type(policy).__name__}"
            )
        self.policy_b = policy

    def set_runtime(self, runtime: EnvRuntime) -> None:
        """Replace the EnvRuntime in-place (policies are kept)."""
        self.runtime = runtime

    def close(self) -> None:
        """Close attached policies that support it.

        Runtime lifecycle is owned by the caller — we intentionally do
        NOT close the runtime here to keep the runner a thin composition
        layer.
        """
        seen = set()
        for policy in (self.policy_a, self.policy_b):
            if id(policy) in seen:
                continue
            seen.add(id(policy))
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
        for agent_id, policy in (("robot_a", self.policy_a), ("robot_b", self.policy_b)):
            reset_fn = getattr(policy, "reset", None)
            if callable(reset_fn):
                reset_fn(seeds.policies[agent_id])

