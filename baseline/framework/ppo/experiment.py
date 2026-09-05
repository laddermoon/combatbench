"""ExperimentPPO — clean PPO-only experiment abstraction.

This module defines the new generation of experiment interface, designed
from scratch to work with the V2 ``Trajectory`` data structures.  It
eliminates the legacy baggage accumulated in ``experiment.py`` (dual
PPO/SAC support, three generations of segment APIs, framework-managed
weight scheduling) and provides a single, coherent contract between the
experiment author and the PPO training loop.

Design principles
-----------------

1. **Experiment owns the full data pipeline.**

   The v1 design split reward extraction (``extract_rewards``), episode
   splitting (``prepare_segments``), and weight scheduling
   (``initial_weights`` / ``next_weights``) across separate methods, with
   the framework normalizing weights and passing them back into the
   trajectory builder.  This created a circular dependency: the experiment
   needed to know the framework's normalization to control actor_weight,
   and the framework needed to know the experiment's weight semantics to
   combine advantages.

   In V2, ``build_trajectories(episode)`` is the single source of truth.
   The experiment decides:
   - How to slice the episode into trajectories (phase-based, gating-based,
     or whole-episode).
   - Per-channel rewards (dense shaping, terminal bonuses, penalties).
   - Per-channel termination (``is_terminated`` → V=0 bootstrap, or
     ``truncated`` → bootstrap from critic).
   - Per-channel actor_weight (how much this channel's advantage influences
     the policy gradient).  This can vary per trajectory, enabling
     curriculum scheduling without framework involvement.

2. **Reward channels are first-class.**

   Each ``RewardChannel`` declares its own ``gamma`` and ``gae_lambda``.
   The framework builds one critic per channel and uses the channel's
   parameters for GAE computation.  This replaces the v1 pattern of a
   separate ``gammas`` dict in ``CommonParams`` and a global
   ``gae_lambda`` in ``PPOParams``.

   Per-channel ``gae_lambda`` is a V2 unique capability: sparse terminal
   rewards benefit from high λ (low bias), while dense shaping rewards
   benefit from lower λ (low variance).  The experiment author can now
   tune this per channel without framework changes.

3. **PPO only.**

   SAC support is removed.  No ``SACParams``, no ``build_q_critic``, no
   ``sac_params()``.  This eliminates the algorithm-dispatch complexity
   that complicated v1's abstract methods.  If SAC is needed in the
   future, a separate ``ExperimentSAC`` class can be created without
   polluting the PPO interface.

4. **One job builder.**

   ``build_jobs(policy_bp, base_seed, n_episodes)`` replaces the separate
   ``build_rollout_jobs`` and ``build_eval_jobs`` methods.  The caller
   (training loop) controls whether the policy is stochastic (training
   rollout) or deterministic (evaluation) by passing the appropriate
   ``PolicyBlueprint``.  The experiment does not need to know whether it
   is building jobs for training or evaluation — it just builds N jobs
   with the given blueprint and seed.

5. **Exploration is a first-class, split responsibility.**

   The experiment owns exploration *intent* (``exploration()`` returns an
   ``ExplorationSpec`` per update, optionally reacting to the previous
   update's stats); the policy owns exploration *mechanism*
   (``set_exploration(explore_intensity)`` interprets the value for its own distribution
   family, ``evaluate_actions`` returns a normalized entropy for the
   framework to use in the entropy floor loss).  The framework only
   routes.

   Two primary knobs, both ∈ [0, 1]: ``explore_intensity`` (symmetric
   temperature-like control, 0.5 = neutral) and ``entropy_floor``
   (training-side entropy floor).
   For the common case where both should move together, set them to the
   same value.  The framework computes the entropy floor loss from
   ``ActorEval.entropy`` — a
   one-sided hinge ``relu(floor - H_norm)`` analogous to PPO clip.

   This replaces a design where the framework hard-coded
   ``loss -= entropy_coef * entropy`` and read ``actor.log_std``
   directly — assumptions that hold only for a diagonal Gaussian with a
   state-independent sigma, and that made the ``TrainablePolicy``
   protocol incorrect for any other family.  It also replaces the
   intermediate ``ActorEval.regularizer`` design where the policy
   computed its own loss term — the coefficient is now a framework
   concern, applied uniformly via the entropy floor.

6. **Coexistence, not replacement.**

   The old ``Experiment`` ABC in ``experiment.py`` and
   ``CombatExperimentBase`` in ``base.py`` remain untouched.  The 40+
   existing v1 experiments continue to work via the legacy training path
   (``train_ppo()``).  The training CLI (``train.py``) dispatches to
   ``train_ppo()`` or ``train_ppo()`` based on whether the experiment
   is an ``ExperimentPPO`` instance.

   New experiments should inherit from ``ExperimentPPO`` directly (or from
   a future ``CombatExperimentPPOBase`` that provides shared combat
   defaults).  Old experiments are not migrated.

Data flow
---------

::

    ActorPolicyBlueprint (from framework)
         │
         ▼
    build_jobs(policy_bp, base_seed, n_episodes)
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
    │ importance, mode                 │
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
| Exploration        | exploration() → ExplorationSpec     | Routes explore_intensity → set_exploration |
| Entropy floor      | entropy_floor via ExplorationSpec   | Computes relu(floor - H_norm) |
| Checkpointing      | state/load_state                    | Save/load model + config.json |

Removed from v1
---------------

- ``initial_weights()`` / ``next_weights()``: The experiment manages
  actor_weight internally in ``build_trajectories``.  No external weight
  scheduling loop.
- ``extract_rewards()``: Folded into ``build_trajectories``.
- ``prepare_segments()`` / ``prepare_training_segments()``: Folded into
  ``build_trajectories``.
- ``build_rollout_jobs()`` / ``build_eval_jobs()``: Merged into
  ``build_jobs()``.
- ``video_env_blueprint()``: Deprecated; video uses eval_jobs[0].
- ``Segment`` dataclass: Replaced by ``Trajectory`` + ``ChannelData``.
- ``SACParams``, ``sac_params()``, ``build_q_critic()``: PPO only.
- ``gammas`` dict in ``CommonParams``: Replaced by ``reward_channels()``.
- ``stage_weights`` parameter in ``ppo_update``: Replaced by per-frame
  ``actor_weight`` from ``ChannelData``.
- ``_current_actor_weights`` hack: Experiment owns weight scheduling.
- ``compute_episode_metrics()`` / ``compare_eval()`` / ``scheduler_info()``:
  Merged into ``on_eval()`` — the experiment receives raw episodes and
  handles metrics, best-of-run judgment, and state updates internally.
- ``normalize_advantages()`` / ``combine_advantages()`` /
  ``normalize_sample_weights()``: Removed — framework defaults are not
  customizable.  The experiment controls the pipeline through
  ``reward_channels()`` and ``ChannelData.actor_weight``.
- ``to_dict()`` / ``save_run_config()``: Removed — serialization is the
  framework's responsibility.  The framework builds ``config.json`` from
  ``reward_channels()``, ``common_params()``, ``ppo_params()``, and
  ``state()``.
- ``PPOParams.log_std_min`` / ``log_std_max``: Moved to the actor — they
  describe a Tanh-Gaussian, not PPO.  Set them in ``build_actor()``.
- ``PPOParams.entropy_coef``: Moved to ``ExplorationSpec.entropy_coef``
  with a default linked to ``explore_intensity``.  The framework
  computes the entropy floor loss from ``ActorEval.entropy`` — the
  policy no longer computes its own loss term via
  ``ActorEval.regularizer`` (field removed).
- ``ExplorationSpec.temperature``, ``.entropy_target``, ``.clip_eps``,
  ``.target_kl``, ``.noise_tau_steps``, ``.noise_scale``,
  ``.policy_extras``: Removed.  ``explore_intensity`` replaces
  ``temperature`` and ``noise_scale``; ``entropy_floor`` replaces
  ``entropy_target``; ``clip_eps``/``target_kl`` are not per-update
  overrides (stay in ``PPOParams``); ``noise_tau_steps`` is an init-time
  policy config; ``policy_extras`` is unnecessary with only three
  fields.
- ``ActorEval.regularizer``: Removed.  Replaced by ``ActorEval.entropy``
  — a per-obs, differentiable, normalized entropy in [0, 1] that the
  framework uses to compute ``relu(entropy_floor - H_norm)``.  This
  decouples the loss coefficient (framework concern) from the entropy
  computation (policy concern).
- ``evaluate_actions() -> (log_prob, entropy)``: Now returns
  ``ActorEval`` with ``log_prob``, ``entropy``, and optional ``stats``,
  so a policy without a closed-form entropy can return a sampled
  estimate without fabricating a regularizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable,
)

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


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
#   * The **policy** owns exploration *mechanism*: what "explore_intensity
#     0.5" concretely means for its own distribution family.  It receives
#     the value via ``set_exploration(explore_intensity)``.  The spec has
#     only three fields: ``explore_intensity``, ``entropy_floor``, and
#     ``entropy_coef``.
#
# Two primary knobs, both ∈ [0, 1]:
#
#   * ``explore_intensity`` — rollout side: how much noise to inject when
#     sampling.  The policy maps this to its internal parameters (σ
#     offset, noise_scale, etc.).
#
#   * ``entropy_floor`` — training side: the minimum normalized entropy
#     the policy is allowed to have.  The framework computes a one-sided
#     hinge loss ``entropy_coef * relu(floor - H_norm)`` that only
#     activates when the policy's entropy drops below the floor —
#     analogous to PPO clip's "only intervene when out of bounds".
#
# For the common case where both should move together, set them to the
# same value.  For scenarios that require independent control (e.g.
# on-policy + anti-collapse, or strong exploration + fast convergence),
# set them separately.
#
# The framework computes the entropy floor loss from ``ActorEval.entropy``
# (a per-obs, differentiable, normalized entropy in [0, 1] that the
# policy returns from ``evaluate_actions``).  This replaces the old
# ``ActorEval.regularizer`` design where the policy computed its own
# loss term — the coefficient is now a framework concern, not a policy
# concern.
#
# The framework only routes between the two owners.  It never inspects a
# spec field beyond ``resolve()`` nor interprets a stat key.  This is
# what allows non-Gaussian actors (mixture, flow, diffusion) to be
# dropped in without touching ppo.trainer / ppo.loop.
#
# See ``DESIGN_unified_exploration_control.md`` for the full design.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExplorationSpec:
    """A per-update exploration directive from experiment to policy.

    Three fields, all optional (``None`` = "no opinion, keep current"):

    - ``explore_intensity`` ∈ [0, 1]: symmetric temperature-like control
      centered at 0.5 (0.5 = neutral, 0 = compress, 1 = expand).
    - ``entropy_floor`` ∈ [0, 1]: training-side entropy floor.
    - ``entropy_coef``: coefficient for the entropy floor loss.

    For the common case where exploration and anti-collapse should move
    together, set ``explore_intensity`` and ``entropy_floor`` to the
    same value.  For independent control (on-policy + anti-collapse,
    strong exploration + fast convergence, async annealing), set them
    separately.

    PPO trust-region knobs (``clip_eps``, ``target_kl``) live in
    :class:`PPOParams` and are not overridable per-update.  OU temporal
    correlation (``noise_tau_steps``, ``noise_scale``) is an init-time
    policy configuration, not a per-update directive.  Policy-family-
    specific scaling is handled by ``explore_intensity`` — the policy
    maps it to its own internal parameters.

    See ``DESIGN_unified_exploration_control.md`` for the full design.

    Attributes:
        explore_intensity: Symmetric temperature-like control ∈ [0, 1].
            ``0.5`` = neutral (policy uses its learned σ as-is),
            ``→ 0`` = compress σ (less noise), ``→ 1`` = expand σ
            (more noise).  The policy maps this to its internal
            parameters (e.g. Gaussian log_std offset).  ``None`` = no
            opinion.
        entropy_floor: Training-side entropy floor ∈ [0, 1], expressed
            in the policy's *normalized* entropy (0 = fully certain,
            1 = policy's maximum entropy).  The framework computes
            ``entropy_floor_loss = entropy_coef * relu(floor - H_norm)``
            — a one-sided hinge that only activates when the policy's
            entropy drops below the floor, analogous to PPO clip.
            ``None`` = no opinion (policy keeps its current floor).
        entropy_coef: Coefficient for the entropy floor loss.  When
            ``None``, the framework uses a default linked to
            ``explore_intensity`` (e.g. ``0.01 * explore_intensity``).
            Override when you want the coefficient independent of
            exploration strength.
    """

    explore_intensity: Optional[float] = None
    entropy_floor: Optional[float] = None
    entropy_coef: Optional[float] = None

    def resolve(self) -> Tuple[float, float]:
        """Return ``(explore_intensity, entropy_floor)`` with fallbacks.

        Returns the explicit values, defaulting to ``0.5`` for
        ``explore_intensity`` (neutral) and ``0.0`` for
        ``entropy_floor`` when ``None``.

        Returns:
            ``(explore_intensity, entropy_floor)`` as floats in [0, 1].
        """
        return (
            self.explore_intensity if self.explore_intensity is not None else 0.5,
            self.entropy_floor if self.entropy_floor is not None else 0.0,
        )


@dataclass
class ActorEval:
    """Result of one :meth:`TrainablePolicy.evaluate_actions` call.

    Attributes:
        log_prob: ``(B,)`` log-probability of the given actions under the
            *current* parameters.  Must be differentiable — this is the
            numerator of the PPO importance ratio.  Action-dependent.
        entropy: ``(B,)`` normalized entropy of the policy's own
            distribution ``H(π(·|s))``, expressed in [0, 1] where 0 =
            fully certain and 1 = the policy's maximum entropy.  Must be
            differentiable — the framework uses it to compute the
            entropy floor loss.  **Action-independent**: it depends
            only on the observation and policy parameters, not on
            which action was taken.  This is what makes it immune to
            the on-policy gradient-zero problem that plagues
            ``-log_prob.mean()``.

            Each policy family is responsible for defining its own
            ``H_max`` and normalizing: a Gaussian uses
            ``H_norm = (H - H_min) / (H_max - H_min)``; a mixture uses
            ``H_norm = H / H_max``; a flow uses a sampled estimate.  The
            framework does not interpret the normalization — it trusts
            the policy's [0, 1] output.
        stats: Diagnostics describing the policy's exploration state,
            populated only when ``want_stats=True``.  Keys are chosen by
            the policy; the framework merges them into its stats dict
            without interpretation.
    """

    log_prob: torch.Tensor
    entropy: torch.Tensor
    stats: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Actor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TrainablePolicy(Protocol):
    """Interface that the PPO trainer requires from an actor.

    Two methods, each with an unambiguous call site in the loop:

    ===================================  ==================  ===================
    when                                 call                yields
    ===================================  ==================  ===================
    once per update, buffer construction ``evaluate_actions``
                                         ``want_stats=True`` batch-wide stats
    ~epochs x minibatches per update     ``evaluate_actions`` log_prob + entropy
    ===================================  ==================  ===================

    Exploration is **not** a mutable state on the policy.  The policy
    receives ``explore_intensity`` as a per-frame data field (via
    ``evaluate_actions``) or per-step parameter (via ``act``), and
    computes its effective σ from it on every call.  This makes the
    rollout→scoring consistency a data guarantee, not a timing
    guarantee.

    Distributional statistics ride on ``ActorEval.stats``, anchored to
    the one call that has a clean definition — the buffer's single
    batched pass over the whole rollout under theta_old.
    """

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
        explore_intensity: torch.Tensor,
        *, want_stats: bool = False,
    ) -> ActorEval:
        """Recompute log_prob and entropy for obs/actions.

        Returns an :class:`ActorEval` with:
        - ``log_prob``: action-dependent, used for PPO importance ratio.
        - ``entropy``: action-independent normalized entropy ``H(π(·|s))``
          in [0, 1], used by the framework for the entropy floor loss.
        - ``stats``: optional diagnostics (only when ``want_stats=True``).

        ``explore_intensity`` is a ``(B,)`` tensor recording the per-frame
        exploration intensity used at rollout time.  The policy uses it
        to compute the effective σ for log_prob evaluation, ensuring the
        PPO importance ratio is computed under the same distribution that
        produced the actions.  ``entropy`` (uncertainty) uses the policy's
        own σ without exploration scaling.

        Args:
            obs: ``(B, obs_dim)`` observations.
            actions: ``(B, action_dim)`` actions taken at rollout time.
            explore_intensity: ``(B,)`` per-frame exploration intensity
                recorded at rollout time.  Required — the policy must
                know what distribution produced the actions.
            want_stats: When True, also populate ``ActorEval.stats`` with
                distributional diagnostics over this batch.  The
                framework sets this only for the single whole-batch call
                in ``PPOBuffer``; it is left False inside the
                minibatch loop because building a float dict forces a
                GPU sync on every minibatch.

        Returns:
            An :class:`ActorEval`.  Both ``log_prob`` and ``entropy``
            must be differentiable.  Note the buffer's call happens
            under ``torch.no_grad()``, so they are non-differentiable
            there and only the stats are consumed.
        """
        ...

    def to_blueprint(
        self, dest_path: str, *, stochastic: bool = False,
    ) -> PolicyBlueprint:
        """Export a rollout-ready policy blueprint.

        The exported artifact is **exploration-neutral**: it does not
        bake in any explore_intensity.  The rollout worker receives
        explore_intensity as a per-step parameter via ``act``, ensuring
        the sampling distribution matches what ``evaluate_actions``
        later scores under.

        Args:
            dest_path: Directory path for the exported blueprint.
            stochastic: If True, the blueprint uses stochastic sampling
                (for training rollouts).  If False (default), it uses
                deterministic mean actions (for evaluation).
        """
        ...


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParams:
    """Training parameters shared across all PPO experiments.

    Unlike v1, this does NOT include ``gammas`` — per-channel gamma and
    gae_lambda are declared in ``RewardChannel`` via ``reward_channels()``.
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

    Strictly the knobs of the PPO *algorithm*.  Two categories used to
    live here and no longer do:

    - ``log_std_min`` / ``log_std_max``: properties of a Tanh-Gaussian
      actor, not of PPO.  They were here only because the trainer used to
      reach into ``actor.log_std`` to clamp it for logging.  They now
      belong to the actor (set in ``build_actor``).
    - ``entropy_coef``: the strength of the entropy floor loss.  It is
      now carried by ``ExplorationSpec.entropy_coef`` with a default
      linked to ``explore_intensity``.  The framework computes
      ``entropy_coef * relu(entropy_floor - H_norm)`` from
      ``ActorEval.entropy`` — a policy-family-agnostic, normalized
      entropy in [0, 1].  This replaces the old
      ``ActorEval.regularizer`` design where the policy computed its own
      loss term.

    ``clip_eps`` and ``target_kl`` stay because they are genuinely PPO's.
    They are not overridable per-update — the trust region is a fixed
    property of the PPO configuration, not part of the exploration story.

    Per-channel ``gae_lambda`` is declared in ``RewardChannel`` via
    ``reward_channels()`` — it is NOT a global parameter here.
    """

    clip_eps: float
    target_kl: float
    update_epochs: int
    minibatch_size: int


# ---------------------------------------------------------------------------
# UpdateStats — typed summary of one ppo_update call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateStats:
    """Typed summary of one ``ppo_update`` call, passed to ``on_update``.

    The framework guarantees every typed field.  Per-channel dicts are
    keyed by ``RewardChannel.name``.  The ``policy_stats`` sub-mapping
    carries whatever the actor contributed via ``ActorEval.stats`` — its
    keys are the policy's choice (e.g. ``entropy_normalized``,
    ``entropy_raw``, ``std_mean`` for a Tanh-Gaussian) and are **not**
    guaranteed across policy families.  Treat ``policy_stats`` as opaque
    hints, not a contract.

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

    # --- Diagnostics (human-readable lines, not for programmatic use) ---
    diagnostics: List[str] = field(default_factory=list)

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
        return d


# ---------------------------------------------------------------------------
# Job type alias
# ---------------------------------------------------------------------------

# A rollout job is a tuple of:
#   (policy_a_blueprint, policy_b_blueprint, env_blueprint, seed, episode_options)
# The experiment builds these in build_jobs(); the framework's
# ParallelRollouter.collect() consumes them.
Job = Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]


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
                diagnostics (e.g. ``entropy_normalized``, ``entropy_raw``,
                ``std_mean`` for a Tanh-Gaussian) but has **no
                cross-family contract** —
                treat it as opaque hints.
            update: Current update index (1-based, matches the loop).
        """
        pass

    def exploration(
        self, update: int,
    ) -> Optional["ExplorationSpec"]:
        """Return this update's exploration directive, or None to keep.

        Called once per update **before** the rollout blueprint is
        exported, so a returned ``temperature`` still affects sampling.
        Reads whatever internal state ``on_update`` has accumulated.

        Args:
            update: Current update index (1-based, matches the loop).

        Returns:
            An ``ExplorationSpec``, or ``None`` to leave the policy's
            current exploration configuration untouched.  The default
            implementation returns ``None``, so an experiment that does
            not care about exploration behaves exactly as before.
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
        explore_intensity: float = 0.5,
    ) -> List[Job]:
        """Build rollout jobs for training or evaluation.

        This unified method replaces v1's separate ``build_rollout_jobs``
        and ``build_eval_jobs``.  The caller controls whether the policy
        is stochastic (training) or deterministic (eval) by passing the
        appropriate ``policy_bp``.

        ``explore_intensity`` is injected into each job's
        ``episode_options["explore_intensity"]`` so the rollout worker
        passes it to ``policy.act`` at every step.  For evaluation,
        the caller passes ``explore_intensity=0.5`` (neutral) — the
        deterministic policy ignores it.

        Args:
            policy_bp: The actor's exported policy blueprint.  For
                training rollouts, this has ``stochastic=True``.  For
                evaluation, it is deterministic (mean action).
            base_seed: Base random seed for this batch.  Each job should
                use ``base_seed + i`` as its seed.
            n_episodes: Number of episodes to build.
            explore_intensity: Exploration intensity for this batch.
                Default 0.5 (neutral).

        Returns:
            List of Job tuples:
            ``(policy_a_bp, policy_b_bp, env_bp, seed, episode_options)``
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

        Returns an empty list to skip all episodes entirely.
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

