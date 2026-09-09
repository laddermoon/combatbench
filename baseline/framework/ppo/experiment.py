"""ExperimentPPO — clean PPO-only experiment abstraction.

This module defines the experiment interface for PPO training.  It
provides a single, coherent contract between the experiment author
and the PPO training loop.

Design principles
-----------------

1. **Experiment owns the full data pipeline.**

   ``build_trajectories(episode)`` is the single source of truth.  The
   experiment decides how to slice the episode into trajectories,
   per-channel rewards, per-channel termination, and per-channel
   actor_weight (how much this channel's advantage influences the
   policy gradient).  actor_weight can vary per trajectory, enabling
   curriculum scheduling without framework involvement.

2. **Reward channels are first-class.**

   Each ``RewardChannel`` declares its own ``gamma`` and ``gae_lambda``.
   The framework builds one critic per channel and uses the channel's
   parameters for GAE computation.  Per-channel ``gae_lambda`` allows
   sparse terminal rewards to use high λ (low bias) while dense shaping
   rewards use lower λ (low variance).

3. **PPO only.**

   No SAC support.  If SAC is needed in the future, a separate
   ``ExperimentSAC`` class can be created without polluting the PPO
   interface.

4. **One job builder.**

   ``build_jobs(policy_bp, base_seed, n_episodes, stochastic=...)`` handles
   both training and evaluation.  ``stochastic=True`` wraps policies in
   ``ExploratoryPolicy`` for training rollouts; ``stochastic=False`` uses
   policies directly as ``Policy`` for deterministic evaluation.

5. **Exploration is a split responsibility.**

   The experiment owns exploration *intent* (``exploration()`` returns
   an ``ExplorationSpec`` per update); the policy owns exploration
   *mechanism* (maps ``explore_factor`` to its own internal
   parameters).  The framework only routes.

   See ``DESIGN_unified_exploration_control.md`` for the full design.

Data flow
---------

::

    ActorPolicyBlueprint (from framework)
         │
         ▼
    build_jobs(policy_bp, base_seed, n_episodes, stochastic=...)
         │
         ▼  (ParallelRollouter collects)
    List[Episode]
         │
         ▼  (per episode)
    build_trajectories(episode)
         │
         ▼
    List[Trajectory]
    ┌──────────────────────────────────┐
    │ obs, actions, last_obs           │
    │ channels: {                      │
    │   "r_x": ChannelData(            │──▶ Critic "r_x": GAE(γ_x, λ_x) → adv_x, ret_x
    │     reward, is_terminated,       │      → critic loss: MSE(V, ret_x)
    │     actor_weight                 │
    │   ),                             │
    │   "r_y": ChannelData(            │──▶ Critic "r_y": GAE(γ_y, λ_y) → adv_y, ret_y
    │     reward, is_terminated,       │      → critic loss: MSE(V, ret_y)
    │     actor_weight                 │
    │   ),                             │
    │ }                                │
    │ explore_factor                │
    └──────────────────────────────────┘
         │
         ▼  (PPOBuffer concatenates all trajectories)
    ppo_update(actor, critics, buf, ...)
         │
         ├── Per-channel: normalize advantages (z-score on active frames)
         ├── L1-normalize actor_weights per frame: Σ_c |aw_c| = 1
         ├── Combine: combined_adv = Σ_c  aw_c_normed * confidence_c * norm_adv_c
         ├── Critic update: minimize MSE(V_c(s), ret_c) on active frames
         └── Actor update: PPO clipped surrogate on combined_adv

What the Experiment controls vs what the framework handles
----------------------------------------------------------

| Stage              | Experiment                          | Framework                     |
|--------------------|-------------------------------------|-------------------------------|
| Model building     | build_actor, build_critic           | Creates optimizers            |
| Job construction   | build_jobs                          | ParallelRollouter.collect     |
| Episode→Trajectory | build_trajectories (full control)   | Calls it per episode          |
| GAE computation    | reward_channels (γ, λ config)       | Executes compute_gae          |
| Adv normalization  | —                                   | Default: z-score on active    |
| Adv combination    | —                                   | Default: weighted by aw*conf  |
| Critic update      | —                                   | MSE on returns, masked        |
| Actor update       | —                                   | PPO clipped surrogate         |
| Eval & scheduling  | on_eval (full control)              | Runs eval rollouts, exports   |
| Exploration (train) | exploration() → ExplorationSpec   | Returns uncertainty_floor / uncertainty_coef |
| Exploration (rollout) | build_jobs() → Job.explore_factor | Routes explore_factor to policy.act / evaluate_actions |
| Uncertainty floor  | uncertainty_floor via ExplorationSpec | Computes relu(floor - U) |
| Checkpointing      | state/load_state                    | Save/load model + config.json |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Tuple,
    TYPE_CHECKING,
)

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import Policy, PolicyBlueprint

if TYPE_CHECKING:
    from baseline.framework.ppo.debug.knobs import KnobCheck
    from baseline.framework.ppo.debug.whatif import WhatifParam

from baseline.framework.ppo.stochastic_policy import StochasticPolicy
from baseline.framework.rollout.job import Job


# ---------------------------------------------------------------------------
# Exploration contract
#
# Exploration has two owners, deliberately separated:
#
#   * The **experiment** owns exploration *intent*: how much exploration
#     is wanted at update N, possibly reacting to the previous update's
#     statistics.  It expresses this as an ``ExplorationSpec`` returned
#     from ``ExperimentPPO.exploration()``.
#
#   * The **policy** owns exploration *mechanism*: what each
#     ``explore_factor`` value concretely means for its own
#     distribution.  It receives the value per-frame via
#     ``evaluate_actions`` and per-step via ``act``.  The specific
#     mapping is policy-defined.
#
# Two primary knobs:
#
#   * ``explore_factor`` ∈ [-1, 1] — rollout side: additive exploration
#     strength (0 = neutral, +1 = max explore, -1 = max suppress).
#     The policy maps this to its internal parameters.
#
#   * ``uncertainty_floor`` ∈ [0, 1] — training side: the minimum
#     uncertainty the policy is allowed to have.  The framework computes a
#     one-sided hinge loss ``uncertainty_coef * relu(floor - U)`` that
#     only activates when the policy's uncertainty drops below the floor.
#
# The framework only routes between the two owners.  It never inspects a
# spec field beyond ``uncertainty_floor`` / ``uncertainty_coef`` nor
# interprets a stat key.
#
# See ``DESIGN_unified_exploration_control.md`` for the full design.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplorationSpec:
    """A per-update exploration directive from experiment to policy.

    Two fields, both optional (``None`` = "no opinion, keep current"):

    - ``uncertainty_floor`` ∈ [0, 1]: training-side uncertainty floor.
    - ``uncertainty_coef``: coefficient for the uncertainty floor loss.

    ``explore_factor`` (rollout-time sampling exploration) is **NOT**
    part of this spec — it is decided inside ``build_jobs`` and placed
    into each :class:`Job`'s ``explore_factor_a`` /
    ``explore_factor_b`` fields.  This separation gives ``build_jobs``
    per-job / per-agent / per-frame granularity that a single spec
    field cannot express.

    PPO trust-region knobs (``clip_eps``, ``target_kl``) live in
    :class:`PPOParams` and are not overridable per-update.

    Attributes:
        uncertainty_floor: Training-side uncertainty floor ∈ [0, 1],
            expressed in the policy's uncertainty metric.  The specific
            meaning of 0 and 1 is defined by the policy.  The framework
            computes ``uncertainty_floor_loss = uncertainty_coef *
            relu(floor - U)`` — a one-sided hinge that only activates
            when the policy's uncertainty drops below the floor,
            analogous to PPO clip.  ``None`` = no opinion (policy keeps
            its current floor).
        uncertainty_coef: Coefficient for the uncertainty floor loss.
            ``None`` = use default (0.0).
    """

    uncertainty_floor: Optional[float] = None
    uncertainty_coef: Optional[float] = None


@dataclass
class ActorEval:
    """Result of one :meth:`TrainablePolicy.evaluate_actions` call.

    Attributes:
        log_prob: ``(B,)`` log-probability of the given actions under the
            *current* parameters.  Must be differentiable — this is the
            numerator of the PPO importance ratio.  Action-dependent.
        uncertainty: ``(B,)`` uncertainty ∈ [0, 1], action-independent.
            Must be differentiable — the framework uses it to compute
            the uncertainty floor loss.  The specific meaning of 0 and 1
            is defined by the policy, not the framework.  The framework
            only requires the range [0, 1] and that it is
            action-independent (depends only on observation and policy
            parameters).  This makes it immune to the on-policy
            gradient-zero problem that plagues ``-log_prob.mean()``.
        stats: Diagnostics describing the policy's exploration state,
            populated only when ``want_stats=True``.  Keys are chosen by
            the policy; the framework merges them into its stats dict
            without interpretation.
    """

    log_prob: torch.Tensor
    uncertainty: torch.Tensor
    stats: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Actor protocol
# ---------------------------------------------------------------------------

class TrainablePolicy(StochasticPolicy, Policy, ABC):
    """Interface that the PPO trainer requires from an actor.

    Inherits both :class:`StochasticPolicy` (``sample()`` for rollout
    sampling) and :class:`Policy` (``act()`` for deterministic
    deployment / evaluation), and adds two PPO-specific methods:

    ===================================  ==================  ===================
    when                                 call                yields
    ===================================  ==================  ===================
    training rollout (via ExploratoryPolicy) ``sample``     action + log_prob
    eval / deployment                    ``act``             deterministic action
    once per update, buffer construction ``evaluate_actions``
                                         ``want_stats=True`` batch-wide stats
    ~epochs x minibatches per update     ``evaluate_actions`` log_prob + uncertainty
    ===================================  ==================  ===================

    The exported policy implements both ``Policy`` and
    ``StochasticPolicy``.  Training rollouts wrap it in
    :class:`ExploratoryPolicy` (which calls ``sample()``); evaluation
    and deployment use it directly as a ``Policy`` (which calls
    ``act()``).  This is controlled by the ``Job.stochastic`` flag, not
    by the blueprint.

    Exploration is **not** a mutable state on the policy.  The policy
    receives ``explore_factor`` as a per-frame data field (via
    ``evaluate_actions``) or per-step parameter (via ``sample``), and
    applies its own mapping from it on every call.  This makes the
    rollout→scoring consistency a data guarantee, not a timing
    guarantee.

    Distributional statistics ride on ``ActorEval.stats``, anchored to
    the one call that has a clean definition — the buffer's single
    batched pass over the whole rollout under theta_old.
    """

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        explore_factor: torch.Tensor,
        *, want_stats: bool = False,
    ) -> ActorEval:
        """Recompute log_prob and uncertainty for obs/actions.

        Returns an :class:`ActorEval` with:
        - ``log_prob``: action-dependent, used for PPO importance ratio.
        - ``uncertainty``: action-independent uncertainty ``U(π(·|s))``
          in [0, 1], used by the framework for the uncertainty floor loss.
        - ``stats``: optional diagnostics (only when ``want_stats=True``).

        ``explore_factor`` is a ``(B,)`` tensor recording the per-frame
        exploration intensity used at rollout time.  The policy uses it
        to reproduce the same distribution that produced the actions,
        ensuring the PPO importance ratio is correct.  ``uncertainty``
        uses the policy's own distribution without exploration scaling.

        Args:
            obs: ``(B, obs_dim)`` observations.
            actions: ``(B, action_dim)`` actions taken at rollout time.
            explore_factor: ``(B,)`` per-frame exploration intensity
                recorded at rollout time.  Required — the policy must
                know what distribution produced the actions.
            want_stats: When True, also populate ``ActorEval.stats`` with
                distributional diagnostics over this batch.  The
                framework sets this only for the single whole-batch call
                in ``PPOBuffer``; it is left False inside the
                minibatch loop because building a float dict forces a
                GPU sync on every minibatch.

        Returns:
            An :class:`ActorEval`.  Both ``log_prob`` and ``uncertainty``
            must be differentiable.  Note the buffer's call happens
            under ``torch.no_grad()``, so they are non-differentiable
            there and only the stats are consumed.
        """
        raise NotImplementedError

    def exploration_grad_diagnostics(
        self,
        policy_loss: torch.Tensor,
        floor_loss: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Optional: report gradient diagnostics for exploration parameters.

        P1-7: This hook lets the policy report how much ``policy_loss``
        vs ``floor_loss`` each contribute to the gradient on the
        policy's exploration parameters (e.g. Gaussian ``log_std``,
        mixture temperature, flow scale).  The trainer calls this on
        the first minibatch of each update and prints the result as a
        ``[GradDiag]`` line.

        The policy decides what "exploration parameters" means — the
        framework no longer reaches into ``actor.log_std`` directly.
        Returning ``None`` means "no diagnostics" (the default for
        policies that don't have exploration parameters or don't want
        to report them).

        Args:
            policy_loss: The PPO clipped surrogate loss (scalar tensor,
                still part of the autograd graph).
            floor_loss: The uncertainty floor hinge loss (scalar tensor,
                still part of the autograd graph).

        Returns:
            A dict with keys ``pol_abs``, ``floor_abs``, ``pol_sign``,
            ``floor_sign``, ``floor_active_frac`` — or ``None`` to skip.
            The trainer uses ``pol_abs`` and ``floor_abs`` to compute
            the ratio ``floor_abs / pol_abs`` that indicates whether
            the floor is strong enough to counteract the policy gradient.
        """
        return None

    def action_dim_grad_norms(self) -> Optional[np.ndarray]:
        """Optional: return per-action-dimension gradient norms.

        S0: This hook lets the policy report how much gradient each
        action dimension is receiving from the current loss.  The
        trainer calls this after ``backward()`` and before
        ``step()``/``zero_grad()``, and records the result in
        ``UpdateStats.action_dim_grad_norms``.

        The policy decides how to map its parameters to action
        dimensions — the framework cannot do this generically because
        different policy families have different architectures.  For
        example, a Gaussian MLP with a linear output layer can report
        the per-row L2 norm of the output weight gradient.

        Returning ``None`` (the default) means "not supported" — the
        ``UpdateStats.action_dim_grad_norms`` field will be ``None``
        and downstream tools will report "per-dim gradients not
        available for this policy family".

        Returns:
            ``(action_dim,)`` float32 numpy array of gradient norms,
            or ``None``.
        """
        return None

    @abstractmethod
    def to_blueprint(
        self, dest_path: str,
    ) -> PolicyBlueprint:
        """Export a deployable policy blueprint.

        The exported artifact implements both ``Policy`` (deterministic
        ``act()``) and ``StochasticPolicy`` (sampling ``sample()``).
        Whether it is used stochastically (training rollout) or
        deterministically (eval / deployment) is controlled by the
        ``Job.stochastic`` flag at rollout time, not by the blueprint.

        Args:
            dest_path: Directory path for the exported blueprint.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParams:
    """Training parameters shared across all PPO experiments.

    Per-channel ``gamma`` and ``gae_lambda`` are declared in
    ``RewardChannel`` via ``reward_channels()``, not here.
    """

    name: str
    learning_rate: float
    critic_learning_rate: float
    grad_clip_norm: float
    episodes_per_update: int
    max_updates: int
    eval_interval: int
    eval_episodes: int
    video_eval_interval: int
    rollout_workers: int
    seed: int


@dataclass(frozen=True)
class PPOParams:
    """PPO hyperparameters.

    Strictly the knobs of the PPO *algorithm*.  ``clip_eps`` and
    ``target_kl`` are the trust-region parameters; they are not
    overridable per-update.  Per-channel ``gae_lambda`` is declared in
    ``RewardChannel`` via ``reward_channels()`` — it is NOT a global
    parameter here.

    Policy-specific parameters (e.g. log_std bounds) belong to the
    actor, not here.  Entropy floor coefficient is carried by
    ``ExplorationSpec.uncertainty_coef``.

    ``target_kl`` semantics:
      - ``target_kl > 0.0``: per-minibatch KL early-stop is active.  If
        the running mean KL within an epoch exceeds ``target_kl``, the
        actor stops updating for the rest of this and all subsequent
        epochs (critics continue — see B1 in ``ppo_update``).
      - ``target_kl == 0.0``: KL early-stop is **disabled**, not
        "zero-tolerance".  The actor runs every epoch and minibatch.
        This is the default behavior when you want to run vanilla PPO
        without a trust-region guard.
    """

    clip_eps: float
    target_kl: float
    update_epochs: int
    minibatch_size: int

    def __post_init__(self):
        # Validate at construction so misconfiguration surfaces immediately
        # rather than as a silent behavioral difference mid-training.
        if self.target_kl < 0.0:
            raise ValueError(
                f"target_kl must be >= 0.0, got {self.target_kl}. "
                f"Use 0.0 to disable KL early-stop."
            )


# ---------------------------------------------------------------------------
# UpdateStats — typed summary of one ppo_update call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateStats:
    """Typed summary of one ``ppo_update`` call, passed to ``on_update``.

    The framework guarantees every typed field.  Per-channel dicts are
    keyed by ``RewardChannel.name``.  The ``policy_stats`` sub-mapping
    carries whatever the actor contributed via ``ActorEval.stats`` — its
    keys are the policy's choice and are **not** guaranteed across
    policy families.  Treat ``policy_stats`` as opaque hints, not a
    contract.

    Use :meth:`to_log_dict` to produce the flat dict format expected by
    ``__RAW_STATS__`` logging and ``analyze_training.py``.
    """

    # --- PPO core ---
    approx_kl: float
    max_kl: float
    early_stop_kl: float
    clip_frac: float
    ratio_mean: float
    ratio_max: float
    policy_loss: float
    value_loss: float
    grad_norm_actor: float
    epochs_done: int
    # P0-1: Number of epochs where the actor actually took at least one
    # minibatch step.  Under B1 (critic/actor early-stop decoupling),
    # `epochs_done` always equals `update_epochs` because critics keep
    # running after the actor stops, so it no longer carries information
    # about whether the actor stopped early.  `actor_epochs_done` does.
    # When early stop is disabled or never triggers, actor_epochs_done ==
    # epochs_done == update_epochs.
    actor_epochs_done: int
    n_batches: int
    n_episodes: int
    total_steps: int
    ep_len_mean: float
    ep_len_min: float
    ep_len_max: float
    epoch_kl_stats: List[Dict[str, Any]]

    # --- Per-channel (keyed by channel name) ---
    critic_losses: Dict[str, float]
    explained_variance: Dict[str, float]
    confidence: Dict[str, float]
    adv_mean: Dict[str, float]
    adv_std: Dict[str, float]
    ret_mean: Dict[str, float]
    ret_std: Dict[str, float]
    critic_grad_norms: Dict[str, float]

    # --- Policy-contributed (no contract) ---
    policy_stats: Mapping[str, float]

    # --- S0: Debug aggregates (always-on, no sink required) ---
    # Per-channel L1-normalized actor_weight mean (after per-frame L1
    # normalization, i.e. Σ_c |aw_c| = 1 per frame).  This is the
    # *effective* weight, not the configured value — see DEBUG_GUIDE.md
    # §7.2 "配置值 ≠ 实际影响力".
    actor_weight_normed: Dict[str, float] = field(default_factory=dict)
    # Per-channel influence share: Σ_frames |aw_normed × conf × normed_adv|,
    # normalized across channels to sum=1.  Answers "who is driving the
    # policy this update?" — see DEBUG_GUIDE.md §3.3 `attribute`.
    influence_share: Dict[str, float] = field(default_factory=dict)
    # Fraction of frames where Σ_c |aw_c| <= 1e-12 (no actor gradient
    # contribution at all).  High values mean most frames are dead weight.
    dead_frame_ratio: float = 0.0
    # Per-action-dimension gradient norms (action_dim,), or None if the
    # policy does not implement action_dim_grad_norms().  Answers "which
    # joints are being trained?" — see DEBUG_GUIDE.md §3.3 `attribute
    # --by action-dim`.
    action_dim_grad_norms: Optional[np.ndarray] = None

    # --- Diagnostics (human-readable lines, not for programmatic use) ---
    diagnostics: List[str] = field(default_factory=list)

    # P0-3: Marks an update that was skipped because the buffer was empty
    # (build_trajectories returned []).  Consumers like on_update() can
    # check this to avoid polluting KL history with zeros.
    is_empty: bool = False

    @classmethod
    def empty(cls, reward_keys: Tuple[str, ...]) -> "UpdateStats":
        """Construct a zeroed UpdateStats for a skipped (empty-buffer) update.

        P0-3: When ``build_trajectories`` returns ``[]``, ``ppo_update``
        returns this instead of crashing.  All numeric fields are 0.0,
        per-channel dicts are keyed by ``reward_keys`` with 0.0 values,
        and ``is_empty=True`` so downstream consumers can skip it.
        """
        return cls(
            approx_kl=0.0,
            max_kl=0.0,
            early_stop_kl=0.0,
            clip_frac=0.0,
            ratio_mean=1.0,
            ratio_max=1.0,
            policy_loss=0.0,
            value_loss=0.0,
            grad_norm_actor=0.0,
            epochs_done=0,
            actor_epochs_done=0,
            n_batches=0,
            n_episodes=0,
            total_steps=0,
            ep_len_mean=0.0,
            ep_len_min=0.0,
            ep_len_max=0.0,
            epoch_kl_stats=[],
            critic_losses={k: 0.0 for k in reward_keys},
            explained_variance={k: 0.0 for k in reward_keys},
            confidence={k: 0.0 for k in reward_keys},
            adv_mean={k: 0.0 for k in reward_keys},
            adv_std={k: 0.0 for k in reward_keys},
            ret_mean={k: 0.0 for k in reward_keys},
            ret_std={k: 0.0 for k in reward_keys},
            critic_grad_norms={k: 0.0 for k in reward_keys},
            policy_stats={},
            actor_weight_normed={k: 0.0 for k in reward_keys},
            influence_share={k: 0.0 for k in reward_keys},
            dead_frame_ratio=0.0,
            action_dim_grad_norms=None,
            diagnostics=[],
            is_empty=True,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        """Flatten to the legacy dict format for ``__RAW_STATS__`` logging.

        ``policy_stats`` is spread to top level so that
        ``analyze_training.py`` paths like ``stats.std_min`` keep working.
        The framework's own keys always win collisions (spread first).
        """
        d: Dict[str, Any] = dict(self.policy_stats)
        d.update({
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "approx_kl": self.approx_kl,
            "max_kl": self.max_kl,
            "early_stop_kl": self.early_stop_kl,
            "epochs_done": self.epochs_done,
            "actor_epochs_done": self.actor_epochs_done,
            "ep_len_mean": self.ep_len_mean,
            "ep_len_min": self.ep_len_min,
            "ep_len_max": self.ep_len_max,
            "epoch_kl_stats": self.epoch_kl_stats,
            "n_batches": self.n_batches,
            "n_episodes": self.n_episodes,
            "total_steps": self.total_steps,
            "clip_frac": self.clip_frac,
            "ratio_mean": self.ratio_mean,
            "ratio_max": self.ratio_max,
            "grad_norm_actor": self.grad_norm_actor,
        })
        for key, val in self.critic_losses.items():
            d[f"vloss_{key}"] = val
        for key, val in self.explained_variance.items():
            d[f"ev_{key}"] = val
        for key, val in self.confidence.items():
            d[f"confidence_{key}"] = val
        for key, val in self.adv_mean.items():
            d[f"adv_mean_{key}"] = val
        for key, val in self.adv_std.items():
            d[f"adv_std_{key}"] = val
        for key, val in self.ret_mean.items():
            d[f"ret_mean_{key}"] = val
        for key, val in self.ret_std.items():
            d[f"ret_std_{key}"] = val
        for key, val in self.critic_grad_norms.items():
            d[f"grad_norm_{key}"] = val
        # S0 aggregates — flattened for analyze_training.py auto-discovery.
        d["dead_frame_ratio"] = self.dead_frame_ratio
        for key, val in self.actor_weight_normed.items():
            d[f"aw_normed_{key}"] = val
        for key, val in self.influence_share.items():
            d[f"influence_share_{key}"] = val
        # action_dim_grad_norms: flatten to grad_dim_00, grad_dim_01, ...
        # so analyze_training.py's scalar auto-discovery picks them up.
        if self.action_dim_grad_norms is not None:
            for i, g in enumerate(self.action_dim_grad_norms):
                d[f"grad_dim_{i:02d}"] = float(g)
        return d


# ---------------------------------------------------------------------------
# S5: Behavior probes + metric verifiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehaviorProbe:
    """A single behavior predicate evaluated on an Episode.

    S5: The predicate signature is ``(Episode, agent_id) -> bool``, so
    it can run offline on any :class:`Episode` (from rollout, snapshot,
    or test fixture) and can be unit tested without a live
    environment.  See ``DEBUG_GUIDE.md`` §3.7 ``probe``.

    Probes answer ①环 of the nine-link signal chain: "did the target
    behavior ever physically happen?"  They are the only direct way
    to confirm ①环 — metrics are indirect and can be fooled by
    contact jitter.
    """

    name: str
    predicate: Callable[..., bool]
    """``(Episode, agent_id) -> bool`` — pass/fail for one agent in one
    episode.  ``Episode`` is :class:`baseline.framework.rollout.episode.Episode`.
    """


@dataclass(frozen=True)
class ProbeSuite:
    """A named collection of behavior probes with fixed seeds.

    Fixed seeds ensure the same initial states are used across
    updates, making probe results directly comparable.  Episode
    options (e.g. ``initial_distance``) are the same for all seeds
    in a suite.
    """

    name: str
    probes: Tuple[BehaviorProbe, ...]
    seeds: Tuple[int, ...]
    """One seed per episode.  Each seed produces one deterministic
    episode (``stochastic=False``).  Length = number of episodes
    per probe run."""
    episode_options: Mapping[str, Any] = field(default_factory=dict)
    """Environment-only options forwarded to ``simulator.reset``."""


# ---------------------------------------------------------------------------
# ExperimentPPO ABC
# ---------------------------------------------------------------------------

class ExperimentPPO(ABC):
    """Clean PPO-only experiment abstraction.

    This is the new generation of experiment interface, designed to work
    with V2 ``Trajectory`` data structures.  See the module docstring for
    the full design rationale.

    Subclasses must implement all abstract methods.  Optional override
    methods have default implementations (return None → framework default).

    The experiment is a plain Python object (typically a singleton
    ``EXPERIMENT = MyConfig()`` at module level).  It is not re-instantiated
    per update — the framework calls its methods repeatedly.

    Typical subclass structure::

        class MyExperiment(ExperimentPPO):
            # Class-attribute configuration
            name = "my_experiment"
            obs_dim = 96
            action_dim = 21
            # ...

            def reward_channels(self):
                return (
                    RewardChannel("r_balance", gamma=0.99, gae_lambda=0.95),
                    RewardChannel("r_posture", gamma=0.99, gae_lambda=0.90),
                )

            def common_params(self):
                return CommonParams(
                    name=self.name,
                    reward_keys=tuple(ch.name for ch in self.reward_channels()),
                    obs_dim=self.obs_dim,
                    # ...
                )

            def build_actor(self, device):
                ...

            def build_critic(self, channel_name, device):
                ...

            def build_jobs(self, policy_bp, base_seed, n_episodes):
                ...

            def build_trajectories(self, episode):
                ...

            def compute_episode_metrics(self, episode):
                ...

            def compare_eval(self, esum, best_esum):
                ...

            def scheduler_info(self):
                ...
    """

    # ==================================================================
    # Phase 0: Configuration & Model Building
    # ==================================================================

    @abstractmethod
    def reward_channels(self) -> Tuple["RewardChannel", ...]:
        """Declare all reward channels for this experiment.

        Returns one ``RewardChannel`` per critic.  The framework builds
        one V(s) critic per channel and uses the channel's ``gamma`` and
        ``gae_lambda`` for GAE computation.

        The order of channels defines ``reward_keys`` — the framework
        extracts ``ch.name`` for each channel to form the keys tuple.
        This replaces v1's separate ``reward_keys`` and ``gammas`` dict.

        Returns:
            Tuple of RewardChannel, e.g.::

                (
                    RewardChannel("r_stand", gamma=0.99, gae_lambda=0.95),
                    RewardChannel("r_balance", gamma=0.99, gae_lambda=0.90),
                )
        """
        ...

    @abstractmethod
    def common_params(self) -> CommonParams:
        """Return training parameters (lr, episodes_per_update, etc.).

        ``reward_keys`` must match the names from ``reward_channels()``.
        ``gammas`` is NOT included — gamma lives in ``RewardChannel``.
        """
        ...

    @abstractmethod
    def ppo_params(self) -> PPOParams:
        """Return PPO-specific hyperparameters."""
        ...

    @abstractmethod
    def build_actor(self, device: torch.device) -> TrainablePolicy:
        """Build and return the actor policy on the given device."""
        ...

    @abstractmethod
    def build_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """Build a V(s) critic for one reward channel.

        Called once per channel at training start.  The critic must
        accept an observation tensor of shape ``(B, obs_dim)`` and return
        a value tensor of shape ``(B,)`` or ``(B, 1)``.

        Args:
            channel_name: The ``RewardChannel.name`` for this critic.
            device: Torch device to place the model on.
        """
        ...

    # ==================================================================
    # Update feedback & Exploration scheduling
    # ==================================================================
    #
    # These two hooks form a symmetric pair, mirroring the on_eval /
    # build_trajectories pair for curriculum scheduling:
    #
    #   on_update(stats, update)  →  experiment absorbs training stats
    #                                 into internal state (e.g. KL history)
    #   exploration(update)       →  experiment reads internal state and
    #                                 returns an ExplorationSpec (or None)
    #
    # The framework calls on_update *after* ppo_update and exploration
    # *before* the next rollout.  On the first update, exploration runs
    # before any on_update has been called, so the experiment's initial
    # state (set in __init__ or class attributes) is used.

    def on_update(
        self, stats: "UpdateStats", update: int,
    ) -> None:
        """Absorb training statistics into internal state.

        Called once per update **after** ``ppo_update`` completes, with
        the typed :class:`UpdateStats` for that update.  The experiment
        can accumulate history (e.g. a rolling KL window) into instance
        state, which ``exploration()`` will read on the next update.

        This is the training-stats counterpart of ``on_eval()``:
        ``on_eval`` closes the loop on *reward weighting* using eval
        episodes, while ``on_update`` closes the loop on *exploration
        strength* using training statistics.

        The default implementation does nothing — an experiment that
        does not need closed-loop exploration scheduling can ignore
        this method entirely.

        Args:
            stats: Typed summary of this update's PPO results.  See
                :class:`UpdateStats` for the full field list.  The
                ``policy_stats`` sub-mapping carries policy-contributed
                diagnostics but has **no cross-family contract** —
                treat it as opaque hints.
            update: Current update index (1-based, matches the loop).
        """
        pass

    def exploration(
        self, update: int,
    ) -> Optional["ExplorationSpec"]:
        """Return this update's PPO update parameters, or None to keep defaults.

        Called once per update **before** ``ppo_update``.  Returns
        ``uncertainty_floor`` and ``uncertainty_coef`` for the uncertainty floor
        loss.  Reads whatever internal state ``on_update`` has
        accumulated.

        Note: ``explore_factor`` (rollout-time sampling) is NOT part
        of this spec — it is decided inside ``build_jobs`` and placed
        into each :class:`Job`'s ``explore_factor_a`` /
        ``explore_factor_b`` fields.

        Args:
            update: Current update index (1-based, matches the loop).

        Returns:
            An ``ExplorationSpec``, or ``None`` to use defaults
            (uncertainty_floor=0.0, uncertainty_coef=0.0).
        """
        return None

    # ==================================================================
    # Phase 1: Job Construction
    # ==================================================================

    @abstractmethod
    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        *,
        stochastic: bool = True,
    ) -> List[Job]:
        """Build rollout jobs for training or evaluation.

        This unified method replaces v1's separate ``build_rollout_jobs``
        and ``build_eval_jobs``.

        The experiment decides ``explore_factor_a`` /
        ``explore_factor_b`` internally — it may read class
        attributes, internal state, or any other source.  This is the
        experiment's implementation detail, not a framework parameter.

        Args:
            policy_bp: The actor's exported policy blueprint.
            base_seed: Base random seed for this batch.  Each job should
                use ``base_seed + i`` as its seed.
            n_episodes: Number of episodes to build.
            stochastic: If True (default), jobs are stochastic —
                policies are wrapped in :class:`ExploratoryPolicy` and
                ``sample()`` is called for training rollouts.  If False,
                jobs are deterministic — policies are used directly as
                ``Policy`` and ``act()`` is called for evaluation.

        Returns:
            List of :class:`Job` instances, one per episode.
        """
        ...

    # ==================================================================
    # Phase 2: Episode → Trajectory
    # ==================================================================

    @abstractmethod
    def build_trajectories(self, episodes: List["Episode"]) -> List["Trajectory"]:
        """Convert all episodes into training trajectories.

        Receives the full batch of episodes at once so the experiment can
        compute global statistics (e.g. phase frame-count ratios) and adjust
        per-trajectory ``actor_weight`` accordingly before returning.

        This is the single source of truth for:
        - How each episode is sliced into trajectories (phase-based,
          gating-based, or whole-episode).
        - Per-channel rewards (dense shaping, terminal bonuses, penalties).
        - Per-channel termination (``is_terminated`` → V=0, or
          ``truncated`` → bootstrap from critic).
        - Per-channel ``actor_weight`` (how much this channel's advantage
          influences the policy gradient).  This is where curriculum
          scheduling happens — the experiment decides the weight, not the
          framework.
        - Which channels are active on each trajectory (channels absent
          from ``Trajectory.channels`` are inactive).

        Returns an empty list to skip all episodes entirely.  When ``[]``
        is returned, the framework skips the PPO update for this round
        (``ppo_update`` returns a zeroed ``UpdateStats`` with
        ``is_empty=True``), skips ``on_update`` so the experiment's KL
        history is not polluted, but still runs eval, checkpoint, and
        logging as normal.
        """
        ...

    # ==================================================================
    # Phase 3 & 4: Critic & Actor Update
    # ==================================================================
    #
    # The framework handles GAE computation, advantage normalization
    # (z-score on active frames), advantage combination (weighted by
    # actor_weight * confidence), and PPO clipped surrogate.  These are
    # not customizable — the experiment controls the pipeline through
    # ``reward_channels()`` (γ, λ) and ``ChannelData.actor_weight``
    # (per-channel influence on the actor).

    # ==================================================================
    # Evaluation
    # ==================================================================

    @abstractmethod
    def on_eval(
        self, episodes: List["Episode"], update: int,
    ) -> Dict[str, Any]:
        """Process evaluation results and update internal state.

        Called once per eval cycle with all raw eval episodes.  The
        experiment is responsible for:

        - Computing per-episode and aggregate metrics (replaces v1's
          ``compute_episode_metrics`` + framework aggregation).
        - Updating internal curriculum/scheduler state based on eval
          results (replaces v1's ``next_weights``).
        - Determining whether this eval is a new best (replaces v1's
          ``compare_eval``).
        - Returning logging info (replaces v1's ``scheduler_info``).

        The framework does NOT interpret any metrics — it only uses
        ``is_new_best`` to decide whether to export the policy, and
        passes ``info`` through to the logging line.

        Args:
            episodes: Raw eval episodes from rollout.
            update: Current update index (0-based).

        Returns:
            Dict with at least::

                {
                    "is_new_best": bool,   # export policy if True
                    "info": Dict[str, Any],  # free-form logging info
                    "stop_training": bool,  # optional: request early stop
                }

            The ``info`` dict is printed by the framework as-is (e.g.
            ``{"phase": "stability", "mean_length": 187.5, ...}``).
            If ``stop_training`` is present and ``True``, the framework
            breaks the training loop after the current update.
        """
        ...

    # ==================================================================
    # State Persistence (for checkpoint resume)
    # ==================================================================

    def state(self) -> dict:
        """Serialize all internal state for checkpointing.

        The framework calls this when saving a checkpoint and passes
        the returned dict to ``load_state()`` on resume.

        Suggested keys (experiment decides what's relevant):

        - ``best_eval``: Best eval result so far (for ``on_eval`` to
          compare against and determine ``is_new_best``).
        - ``curriculum``: Current curriculum phase, stage, or any
          scheduling state that affects ``build_trajectories`` behavior
          (e.g. actor_weight schedule, phase transitions).
        - ``update_count``: Number of updates completed (if the
          experiment tracks this internally).
        - Any other mutable state that influences ``build_jobs``,
          ``build_trajectories``, or ``on_eval``.

        Returns:
            JSON-serializable dict.  Empty dict if no state to persist.
        """
        return {}

    def load_state(self, state: dict) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state: The dict previously returned by ``state()``.
        """
        pass

    # ==================================================================
    # S5: Behavior probes + metric verifiers (optional)
    # ==================================================================

    def probe_suites(self) -> Tuple[ProbeSuite, ...]:
        """Declare behavior probe suites for this experiment.

        S5: Returns ``()`` by default — "this experiment has no probes"
        is a complete answer, not a missing one.  Override to declare
        probes that answer ①环 (did the behavior ever happen?).

        Probes use deterministic rollout (``stochastic=False``) with the
        suite's fixed seeds, so results are comparable across updates.
        The CLI ``debug.py probe`` loads a policy export for a given
        update, runs the suite, and reports per-probe pass rates.

        See ``DEBUG_GUIDE.md`` §3.7 ``probe`` and
        ``DESIGN_debug_system.md`` §5.2.

        Returns:
            Tuple of :class:`ProbeSuite`.  Empty by default.
        """
        return ()

    def metric_verifiers(self) -> Dict[str, Callable[..., float]]:
        """Declare strict metric definitions for verification.

        S5: Returns ``{}`` by default.  Override to provide strict
        versions of metrics that can be compared with the production
        definitions via ``debug.py metric --verify``.

        The key is the metric name (e.g. ``"steps"``), and the value is
        a function ``(Episode, agent_id) -> float`` that computes the
        strict metric value for one agent in one episode.

        See ``DEBUG_GUIDE.md`` §3.6 ``metric --verify`` and
        ``DESIGN_debug_system.md`` §5.3.

        Returns:
            Dict mapping metric name to strict computation function.
            Empty by default.
        """
        return {}

    # ==================================================================
    # S2: Experiment intermediate quantities (optional)
    # ==================================================================

    def debug_arrays(
        self,
        episodes: List["Episode"],
        trajectories: List["Trajectory"],
    ) -> Dict[str, "np.ndarray"]:
        """Export per-frame named arrays for snapshot/replay.

        S2: Returns ``{}`` by default — "this experiment has no debug
        arrays" is a complete answer.  Override to export intermediate
        quantities (phase masks, contacts, state-machine branches) that
        align with the trajectory timeline.

        Contract:
            - The 0th dimension of every returned array MUST equal the
              total frame count after trajectory concatenation
              (``sum(len(t.obs) for t in trajectories)``).
            - Array order MUST align with ``trajectories`` (i.e. already
              truncated to trajectory length, not the full episode length
              ``T_full``).  Experiments computing on ``T_full`` must
              slice to the trajectory length before returning.
            - Must call the same helpers as production code (P2 from
              ``DESIGN_debug_system.md`` — no logic duplication).

        The framework passes the full ``episodes`` list (from rollout)
        plus the already-built ``trajectories``.  Use
        ``traj.provenance`` (S1) to map each trajectory back to its
        source episode + agent when reconstructing per-frame quantities.

        See ``DESIGN_debug_system.md`` §5.1.

        Args:
            episodes: All rollout episodes for this update.
            trajectories: Trajectories built from ``episodes`` by
                ``build_trajectories``.

        Returns:
            Dict mapping array name to ``(total_frames, ...)`` numpy
            array.  Empty by default.
        """
        return {}

    # ==================================================================
    # S6: Knob checks (optional)
    # ==================================================================

    def knob_checks(self) -> Tuple["KnobCheck", ...]:
        """Declare experiment-specific knob checks for ``intervene-check``.

        S6: Returns ``()`` by default — "this experiment has no custom
        knobs" is a complete answer.  Override to add experiment-specific
        knobs that ``debug.py intervene-check`` will verify alongside the
        framework's built-in knobs.

        A knob check verifies that a configured parameter actually entered
        the training data pathway.  The framework registers built-in
        knobs (explore_factor, uncertainty_floor, floor_weight, resume,
        observer); experiments add their own here.

        See ``DEBUG_GUIDE.md`` §3.8 ``intervene-check`` and
        ``DESIGN_debug_system.md`` §5.4.

        Returns:
            Tuple of :class:`KnobCheck`.  Empty by default.
        """
        return ()

    # ==================================================================
    # S4: whatif overrides (optional)
    # ==================================================================

    def whatif_params(self) -> Dict[str, "WhatifParam"]:
        """Declare parameters overridable by ``debug.py whatif --set/--sweep``.

        S4: Returns ``{}`` by default — "this experiment declares no
        whatif params" is a complete answer.  Override to declare
        experiment-specific knobs that ``debug.py whatif`` can apply.

        Each param's ``requires_rebuild`` flag tells the framework whether
        applying it requires re-running ``build_trajectories`` + ``PPOBuffer``
        (e.g. reward computation, ``actor_weight`` masks) or only affects
        ``ppo_update``-time inputs (e.g. the exploration spec, which is
        read inside ``replay_snapshot`` via ``experiment.exploration(update)``).

        The framework parses ``--set``/``--sweep``, validates keys against
        this declaration, parses values according to each param's ``type``,
        and calls :meth:`apply_whatif_overrides` on a fresh experiment
        instance before re-running the standard replay path.  Generic code
        never guesses experiment-specific field names (P5).

        See ``DEBUG_GUIDE.md`` §3.5 ``whatif`` and
        ``DESIGN_debug_system.md`` §7 S4.

        Returns:
            Dict mapping param name to :class:`WhatifParam`.  Empty by
            default.
        """
        return ()

    def apply_whatif_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply a set of ``whatif`` overrides to this experiment instance.

        S4: Called by ``whatif.py`` after validation against
        :meth:`whatif_params`.  The default implementation raises
        ``NotImplementedError`` — experiments must opt in by declaring
        params via :meth:`whatif_params` and implementing this hook.

        The experiment mutates its own fields / internal config here.
        The framework then calls the standard replay path
        (``build_trajectories``, ``debug_arrays``, ``PPOBuffer``,
        ``ppo_update``) on the mutated instance.  ``replay_snapshot``
        deep-copies the actor + critics, so the snapshot is not mutated;
        the experiment instance is reconstructed fresh per variant via
        the registry, so override mutations on one variant do not leak
        to the next.

        Args:
            overrides: Dict mapping declared param name to parsed value.
                Keys are guaranteed to be in :meth:`whatif_params`;
                values are parsed to the declared ``type``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement whatif overrides. "
            f"Declare whatif_params() and implement apply_whatif_overrides() "
            f"to enable `debug.py whatif`."
        )

