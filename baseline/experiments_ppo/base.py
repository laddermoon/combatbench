"""CombatExperimentPPOBase — shared base for humanoid21 V2 PPO experiments.

Provides default values for all framework parameters, shared helpers
(self-play job construction, actor/critic building), and state persistence.
PPO-only — no SAC support.

Subclass and override class attributes + abstract methods.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.ppo import (
    CommonParams,
    ExperimentPPO,
    ExplorationSpec,
    PPOParams,
    TrainablePolicy,
    UpdateArtifacts,
)
from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.rollout.job import Job, ReferenceSpec, SamplingSpec


def _coerce_set_value(raw: Any, current: Any) -> Any:
    """Coerce a ``--set`` string value to the declared attribute's type.

    bool → "1/true/yes/on" (checked before int — bool is an int
    subclass); int/float/str → direct cast; anything else (None,
    containers) → ``ast.literal_eval`` with string fallback.
    """
    if isinstance(current, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


class CombatExperimentPPOBase(ExperimentPPO):
    """Class-attribute style base for humanoid21 combat V2 experiments.

    Subclass and override:
    - Class attributes (name, obs_dim, action_dim, env_blueprint, etc.)
    - ``reward_channels()`` — declare reward channels
    - ``build_trajectories()`` — episode → trajectories
    - ``on_eval()`` — eval processing + best-of-run
    """

    def __init__(self, **kwargs):
        """Accept ``--set KEY=VALUE`` patches from the training CLI.

        KEY must name a declared, non-callable, non-private class
        attribute of the experiment.  The CLI passes strings; each
        value is coerced to the declared attribute's type (bool →
        "1/true/yes/on", int, float, str; other defaults via
        ``ast.literal_eval`` with string fallback).  Unknown keys raise
        TypeError so typos fail loudly at launch.

        Subclasses defining their own ``__init__`` keep their own
        signature (e.g. todo/ experiments taking
        ``policy_blueprint_path``) — this only fills the gap for the
        class-attribute style.
        """
        for key, raw in kwargs.items():
            cur = getattr(type(self), key, None)
            if (
                key.startswith("_")
                or not hasattr(type(self), key)
                or callable(cur)
            ):
                raise TypeError(
                    f"{type(self).__name__}: unknown --set parameter "
                    f"{key!r} (must be a declared class attribute)"
                )
            setattr(self, key, _coerce_set_value(raw, cur))

        if self.delta_factor != 0.0 and self.reference_horizon <= 0:
            raise ValueError(
                f"{type(self).__name__}: delta_factor={self.delta_factor} "
                f"requires reference_horizon > 0 "
                f"(got {self.reference_horizon}) — a nonzero delta_factor "
                f"can never activate without a reference ensemble"
            )
        if self.delta_mode not in ("dynamic", "frozen"):
            raise ValueError(
                f"{type(self).__name__}: delta_mode must be "
                f"'dynamic' or 'frozen', got {self.delta_mode!r}"
            )
        if self.delta_mode == "frozen" and self.delta_factor == 0.0:
            raise ValueError(
                f"{type(self).__name__}: delta_mode='frozen' requires "
                f"delta_factor != 0 — the mode only selects how the delta "
                f"payload is recorded"
            )
        self._ref_history: List[PolicyBlueprint] = []

    # --- Identity ---
    name: str = ""

    # --- Network shape ---
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # --- Exploration ---
    # explore_factor: additive exploration strength ∈ [-1, 1].
    #   0 = neutral (no change to policy distribution).
    #   → +1 = maximum added exploration.
    #   → -1 = maximum exploration suppression.
    #   The specific mapping is policy-defined.
    # uncertainty_floor: training-side uncertainty floor ∈ [0, 1].
    #   The framework computes relu(uncertainty_floor - U) to prevent
    #   policy collapse.  Set to 0 to disable.
    # uncertainty_coef: coefficient for the uncertainty floor loss.
    explore_factor: float = 0.0
    uncertainty_floor: float = 0.3
    uncertainty_coef: float = 0.01

    # --- Reference-delta exploration ---
    # delta_factor (c): Δ→σ floor scale — σ_eff = max(σ, c·|Δ|) per dim,
    #   where Δ is the drift between the current policy's deterministic
    #   action and the reference ensemble's.  0 = mechanism off
    #   (default).  The delta can only widen exploration, never narrow
    #   it — it is a floor on the sampling scale, not a blend.
    # reference_horizon (H): number of policy versions forming the
    #   uniform reference ensemble — strictly BEFORE the current policy
    #   in both modes: the newest history entry is the just-finished
    #   update's export (= the rollout-time behavior policy itself,
    #   Gen0), excluded because a self member pins Δ≈0 mechanically.
    #   H is also the warmup gate: delta activates only once H
    #   strictly-past versions exist (first delta-active update = H+2),
    #   so the ensemble size — and thus the recorded Δ semantics —
    #   stays constant at n=H instead of ramping 1→H.
    # delta_mode: "dynamic" replays a_ref and recomputes Δ = m_θ−a_ref
    #   at train; "frozen" computes Δ = det_action(Gen0) − a_ref at
    #   rollout, records it, and replays it verbatim — σ_eff stays
    #   constant w.r.t. θ inside an update, removing the value-level
    #   σ(m) coupling that destabilizes PPO under the dynamic
    #   parameterization.
    delta_factor: float = 0.0
    reference_horizon: int = 0
    delta_mode: str = "dynamic"

    # --- Shared training ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0

    # --- PPO knobs ---
    clip_eps: float = 0.2
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 8192
    # Minibatches averaged for the KL early-stop decision — a sliding
    # window that persists across epoch boundaries.  1 = instantaneous
    # per-minibatch KL; larger values smooth sampling noise at the cost
    # of stopping a few minibatches later.
    early_stop_kl_window: int = 10
    # Per-channel advantage normalization: "zscore" = (A−μ)/σ
    # (batch-mean centering), "std" = A/σ (scale only, preserves the
    # raw advantage sign of every frame).
    adv_norm: str = "zscore"

    # --- ADV gradient-signal diagnostic (theta_old frame sampling) ---
    # DUMP-ONLY: the diagnostic runs only on updates that carry a dump
    # request; the loop samples a fixed internal frame count and writes
    # the payload into dumps/uNNNNN/gradsig.npz.  No periodic sampling
    # knobs exist — it is targeted investigation instrumentation, not
    # routine telemetry.
    # Histogram resolution: cosine bins over [-1,1] × equal-mass
    # quantile per-frame-norm bins (each row ~1/norm_bins of frames).
    grad_sig_cos_bins: int = 64
    grad_sig_norm_bins: int = 40
    # Explicit norm-bin range (log-spaced); both 0 = auto-derive on the
    # first computed update, then freeze in gradsig/meta.json.
    grad_sig_norm_lo: float = 0.0
    grad_sig_norm_hi: float = 0.0

    # --- Rollout schedule ---
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)

    # Rollout inference placement: "cpu" = in-worker local path (status
    # quo); "gpu" = centralized UDS inference server with GPU-batched
    # forwards (workers ship obs+noise, server returns action+extras).
    rollout_inference: str = "cpu"

    seed: int = 42

    # --- Policy blueprint ---
    # Filename of the initial policy blueprint YAML under
    # humanoid21/blueprints/.  Used by build_actor() to construct the
    # actor.  Default is TruncatedNormalPolicy.
    actor_blueprint: str = "init_policy_truncated_normal.yaml"

    # --- Rollout / env configuration (subclass overrides) ---
    # These parameters control how build_jobs() constructs rollout jobs.
    # Each job is a Job dataclass carrying policy blueprints, env
    # blueprint, seed, episode options, explore_factor, and stochastic
    # flag.  The framework's ParallelRollouter.collect() consumes these
    # jobs to run parallel environment rollouts and produce Episode
    # objects.
    #
    # env_blueprint: YAML filename under humanoid21/blueprints/ that
    #   defines the ParameterizedEnvBlueprint (env plugins, observers,
    #   termination conditions, etc.).  _env_pb() loads it from this path.
    #
    # agent_used: Controls which agent's perspective the rollout observes.
    #   "random"  — each episode randomly selects robot_a or robot_b as
    #               the observed agent (self-play).  The env_bp is
    #               materialized with agent_id set per-episode.
    #   "both"    — both agents are observed in a single env (dual mode).
    #               The env_bp is materialized without agent_id; the env
    #               itself manages both agents via DualImbalanceTerminationPlugin.
    #   "robot_a" — always observe robot_a (fixed single-agent mode).
    #   "robot_b" — always observe robot_b (fixed single-agent mode).
    #
    # max_steps: Maximum number of environment steps per episode.
    #   Passed to env_bp.materialize(max_steps=...).
    #
    # init_distance_min / init_distance_max: Range for the initial
    #   distance between the two robots at episode reset.  A random
    #   value uniformly sampled from [min, max] is placed in
    #   episode_options["initial_distance"] for each job.
    env_blueprint: str = ""
    agent_used: str = "random"
    max_steps: int = 200
    init_distance_min: float = 1.5
    init_distance_max: float = 3.5

    # ------------------------------------------------------------------
    # Parameter access (ExperimentPPO interface)
    # ------------------------------------------------------------------

    def common_params(self) -> CommonParams:
        return CommonParams(
            name=self.name,
            learning_rate=self.learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            episodes_per_update=self.episodes_per_update,
            max_updates=self.max_updates,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=self.video_eval_interval,
            rollout_workers=self.rollout_workers,
            seed=self.seed,
            rollout_inference=self.rollout_inference,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            clip_eps=self.clip_eps,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            early_stop_kl_window=self.early_stop_kl_window,
            adv_norm=self.adv_norm,
            grad_sig_cos_bins=self.grad_sig_cos_bins,
            grad_sig_norm_bins=self.grad_sig_norm_bins,
            grad_sig_norm_lo=self.grad_sig_norm_lo,
            grad_sig_norm_hi=self.grad_sig_norm_hi,
        )

    # ------------------------------------------------------------------
    # Update feedback & Exploration scheduling
    # ------------------------------------------------------------------

    def post_update(self, stats, update: int, *, artifacts=None):
        """Default: track the trained-policy version chain for the
        reference-delta mechanism, then no-op.

        ``artifacts.policy_bp`` is this update's post-update exported
        policy — appended to ``self._ref_history`` (bounded by
        ``reference_horizon``) so ``build_jobs`` can assemble the
        uniform :class:`ReferenceSpec` ensemble on later updates.

        Override to additionally accumulate training stats for
        closed-loop exploration scheduling, and/or to emit
        experiment-defined metrics (``exp.*`` in the viewer) — always
        via ``super().post_update(stats, update, artifacts=artifacts)``
        so the reference chain keeps working, e.g.::

            def post_update(self, stats, update, *, artifacts=None):
                metrics = super().post_update(
                    stats, update, artifacts=artifacts) or {}
                self._kl_history.append(stats.kl_mean)
                metrics["kl_3u_mean"] = sum(self._kl_history[-3:]) / 3
                return metrics

            def exploration(self, update):
                coef = self.uncertainty_coef
                if len(self._kl_history) >= 3 and all(
                    kl < 0.005 for kl in self._kl_history[-3:]
                ):
                    coef *= 4.0  # KL flat for 3 updates, push exploration
                return ExplorationSpec(uncertainty_coef=coef)
        """
        if (
            isinstance(artifacts, UpdateArtifacts)
            and artifacts.policy_bp is not None
        ):
            hist = getattr(self, "_ref_history", None)
            if hist is None:
                hist = self._ref_history = []
            hist.append(artifacts.policy_bp)
            if self.reference_horizon > 0:
                # Keep H+1: the newest entry is the current policy
                # itself — the ensemble takes the H before it.
                del hist[:-(self.reference_horizon + 1)]
        return None

    def exploration(self, update: int) -> ExplorationSpec:
        """Static exploration spec built from the class attributes.

        Returns ``ExplorationSpec`` with:
        - ``uncertainty_floor``: training-side uncertainty floor.  Default 0.3.
        - ``uncertainty_coef``: coefficient for the uncertainty floor loss.

        Note: ``explore_factor`` is NOT part of this spec — it is
        read from ``self.explore_factor`` inside ``build_jobs``.

        Subclasses that want a schedule override ``post_update`` (to absorb
        stats) and this method (to read accumulated state).
        """
        return ExplorationSpec(
            uncertainty_floor=self.uncertainty_floor,
            uncertainty_coef=self.uncertainty_coef,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        actor = bp.build().to(device)
        return actor

    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        return CriticMLP(
            obs_dim=self.obs_dim, hidden_dim=self.critic_hidden_dim,
        ).to(device)

    # ------------------------------------------------------------------
    # Job construction (unified build_jobs)
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        *,
        update: int,
        stochastic: bool = True,
    ) -> List[Job]:
        """Build self-play rollout jobs.

        The sampling spec is assembled by :meth:`_sampling_spec` —
        ``explore_factor`` always, plus the reference-delta ensemble
        (uniform over the last ``reference_horizon`` trained versions)
        when ``delta_factor != 0`` and history exists.  ``stochastic`` is
        placed into each :class:`Job`'s ``stochastic`` field.

        ``update`` is part of the framework contract; the base
        implementation does not need it (history, not call ordering,
        drives the spec).  Subclass can override for non-self-play
        scenarios.
        """
        return self._build_selfplay_jobs(
            self._env_pb(), policy_bp, base_seed, n_episodes, stochastic,
        )

    def _sampling_spec(self) -> SamplingSpec:
        """Assemble this round's :class:`SamplingSpec`.

        With ``delta_factor != 0`` and a **full** strictly-past window,
        the spec carries a uniform :class:`ReferenceSpec` plus
        ``delta_factor`` / ``delta_mode``.  The ensemble
        is the last ``reference_horizon`` versions **before** the
        current policy, in both modes: the newest history entry is the
        just-finished update's export — the rollout-time behavior
        policy itself (= Gen0 in Δ = det_action(Gen0) − a_ref) — and is
        excluded because a self member pins Δ toward 0 mechanically
        (at n=1 the whole ensemble would be self).  Gen0 still
        participates in frozen Δ through the inner policy's own
        deterministic action — it does not need ensemble membership.

        **Warmup gate**: delta does NOT activate on a partial window —
        the mechanism stays off (plain spec) until ``pool`` holds all
        ``reference_horizon`` strictly-past versions, i.e. the first
        delta-active update is ``H + 2`` (H refs + Gen0 + one update
        boundary).  Rationale: a growing n=1..H−1 ensemble would blend
        "adjacent-generation drift" and "drift vs window centroid" —
        two different quantities under one constant c — whereas a full
        window keeps the recorded ``sctx__delta`` semantically uniform
        for analysis.

        Otherwise it is the plain ``explore_factor`` spec — including
        warmup updates where no eligible version exists.
        """
        history = getattr(self, "_ref_history", None) or []
        pool = history[:-1]
        if (
            self.delta_factor != 0.0
            and len(pool) >= self.reference_horizon > 0
        ):
            window = pool[-self.reference_horizon:]
            n = len(window)
            return SamplingSpec(
                explore_factor=self.explore_factor,
                reference=ReferenceSpec(
                    policies=tuple(window),
                    weights=tuple([1.0 / n] * n),
                ),
                delta_factor=self.delta_factor,
                delta_mode=self.delta_mode,
            )
        return SamplingSpec(explore_factor=self.explore_factor)

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        """Load the ParameterizedEnvBlueprint from ``env_blueprint`` filename.

        Subclass sets ``env_blueprint`` to the yaml filename (relative
        to ``humanoid21/blueprints/``).  Override only for non-standard
        blueprint loading logic.
        """
        if not self.env_blueprint:
            raise ValueError(
                f"{self.__class__.__name__} must set env_blueprint "
                "to a blueprint filename"
            )
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints" / self.env_blueprint
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_sampling_ctx(episode, agent_id: str, T: int) -> Optional[Dict[str, np.ndarray]]:
        """Extract all per-frame SamplingContext fields for one agent.

        Reads ``episode.sampling_contexts[agent_id]`` — the grouped view
        of every ctx field passed to ``policy.sample()`` at rollout
        (``explore_factor`` plus e.g. ``reference_action``,
        ``delta_factor``, ``delta``).  Each field is truncated to
        ``(T, ...)``.  Returns ``None`` when the episode recorded no
        ctx for this agent — the buffer then treats every field as
        neutral (explore_factor=0).
        """
        fields = episode.sampling_contexts.get(agent_id)
        if not fields:
            return None
        return {
            name: np.asarray(arr, dtype=np.float32)[:T]
            for name, arr in fields.items()
        }

    @staticmethod
    def _agent_from_rollout_seed(seed: int) -> str:
        rng = np.random.default_rng(int(seed) + 937)
        return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"

    def _build_selfplay_jobs(
        self,
        env_pb: ParameterizedEnvBlueprint,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        stochastic: bool = True,
    ) -> List[Job]:
        rng = np.random.default_rng(base_seed)
        sampling = self._sampling_spec()

        if self.agent_used == "both":
            env_bp = env_pb.materialize(max_steps=self.max_steps)
            jobs: List[Job] = []
            for i in range(n_episodes):
                seed = int(base_seed + i)
                initial_distance = float(
                    rng.uniform(self.init_distance_min, self.init_distance_max)
                )
                jobs.append(Job(
                    policy_a_bp=policy_bp,
                    policy_b_bp=policy_bp,
                    env_bp=env_bp,
                    seed=seed,
                    episode_options={"initial_distance": initial_distance},
                    sampling_a=sampling,
                    sampling_b=sampling,
                    stochastic=stochastic,
                ))
            return jobs

        # Single-agent modes: robot_a, robot_b, or random
        agent_ids: Tuple[str, ...]
        if self.agent_used == "random":
            agent_ids = ("robot_a", "robot_b")
        else:
            agent_ids = (self.agent_used,)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=self.max_steps, agent_id=aid)
            for aid in agent_ids
        }

        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            if self.agent_used == "random":
                agent_id = self._agent_from_rollout_seed(seed)
            else:
                agent_id = self.agent_used
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bps[agent_id],
                seed=seed,
                episode_options={"agent_id": agent_id, "initial_distance": initial_distance},
                sampling_a=sampling,
                sampling_b=sampling,
                stochastic=stochastic,
            ))
        return jobs

    # ------------------------------------------------------------------
    # State persistence (ExperimentPPO interface)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """Persist the trained-version reference chain.

        Subclasses overriding ``state()`` should merge
        ``super().state()`` into their dict so the chain survives
        resume — otherwise a resumed run's reference ensemble restarts
        empty and diverges from a continuous run.
        """
        return {
            "ref_history": [bp.to_dict() for bp in self._ref_history],
        }

    def load_state(self, state: dict) -> None:
        self._ref_history = [
            PolicyBlueprint.from_dict(d)
            for d in state.get("ref_history", [])
        ]
