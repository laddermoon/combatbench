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
    Any, Dict, List, Mapping, Optional, Tuple,
)

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import Policy, PolicyBlueprint

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


@dataclass(frozen=True)
class LRSpec:
    """A per-update optimizer LR directive from experiment to framework.

    Returned by :meth:`ExperimentPPO.lr_schedule`.  Both fields are
    **absolute** learning rates (not multipliers), matching the resume
    path which force-aligns optimizer LRs to ``CommonParams`` values.
    ``None`` = no opinion (keep the current LR).

    Attributes:
        actor_lr: LR applied to every actor optimizer param group.
        critic_lr: LR applied to every critic optimizer (all channels
            share one value, mirroring ``critic_learning_rate``).
    """

    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None


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

    Naming convention (framework-owned fields only — ``policy_stats``
    keys are the policy's own namespace and exempt):

    - Aggregation suffix is mandatory: ``*_mean`` / ``*_max`` /
      ``*_min`` / ``*_std``.  A bare name therefore always denotes a
      non-aggregated value (counts, spec echoes, flags, per-channel
      dicts, distributions).
    - ``post_*`` marks the post-update endpoint cross-section (final
      actor evaluated once over the whole buffer).  Metrics sampled
      per minibatch *during* the update carry no prefix — unprefixed
      means "proc" by convention.
    - Where a name diverges from conventional PPO vocabulary
      (``approx_kl`` → ``kl_mean``, ``clip_frac`` → ``clip_frac_mean``,
      ``policy_loss`` → ``policy_loss_mean``), the debug catalog maps
      the conventional name for display in chart legends.

    Use :meth:`to_log_dict` to produce the flat dict format expected by
    ``__RAW_STATS__`` logging and the debug viewer.
    """

    # --- Update shape (what entered this update) ---
    # len(buf.traj_lengths) — trajectory segments in the buffer; an
    #   episode can contribute more than one.
    n_trajectories: int
    # sum(buf.traj_lengths) — total frames in the buffer.
    total_steps: int
    # max(1, ceil(total_steps / minibatch_size)) — minibatches per
    #   epoch; the split is even, no tiny remainder batch.
    n_batches: int
    # len(epoch_kl_stats) — under B1 this equals update_epochs because
    #   critics run every epoch even after the actor early-stops; see
    #   actor_epochs_done for actor-side progress.
    epochs_done: int
    # P0-1: Number of epochs where the actor actually took at least one
    # minibatch step.  Under B1 (critic/actor early-stop decoupling),
    # `epochs_done` always equals `update_epochs` because critics keep
    # running after the actor stops, so it no longer carries information
    # about whether the actor stopped early.  `actor_epochs_done` does.
    # When early stop is disabled or never triggers, actor_epochs_done ==
    # epochs_done == update_epochs.
    actor_epochs_done: int

    # --- Process dynamics (per-minibatch samples, aggregated over the
    #     actor minibatches actually run) ---
    # Mean of the per-minibatch k3 KL estimates  mean[(r-1) - log r]
    #   over ALL actor minibatches actually run — r = exp(new_lp-old_lp)
    #   is recomputed per minibatch against the frozen rollout policy.
    #   Aggregated globally, not the last epoch's mean, so it survives
    #   actor early-stop.  (Conventional name: approx_kl.)
    kl_mean: float
    # Max over the same per-minibatch k3 means — the worst single
    #   minibatch displacement seen during this update.
    kl_max: float
    # Running mean of the current epoch's per-minibatch k3 values at the
    #   minibatch where target_kl early-stop fired (running_mean_kl >
    #   target_kl).  0.0 if early stop never triggered.
    early_stop_kl_mean: float
    # Ratio family — r = exp(new_lp - old_lp) per minibatch; the three
    #   aggregations (mean / min / max) describe center and both tails.
    #   ratio_min is the suppression-direction tail — trending toward 0
    #   means sampled actions are being zeroed out (exploration
    #   collapse precursor); ratio_max is the boost-direction tail.
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    # Clip family — fraction of minibatch samples with ratio outside
    #   [1-eps, 1+eps], averaged over actor minibatches.  hi = r>1+eps
    #   boost tail, lo = r<1-eps suppression tail; the masks are
    #   disjoint so clip_frac_mean = hi + lo exactly.  A tail-crossing
    #   sample only loses its surrogate gradient when the advantage
    #   sign matches (hi & A>0, lo & A<0) — these are tail fractions,
    #   not effective clip fractions.  (Conventional name: clip_frac.)
    clip_frac_mean: float
    clip_frac_hi_mean: float
    clip_frac_lo_mean: float
    # Mean over actor minibatches of the clipped surrogate loss
    #   -mean[min(r*A, clip(r,1-eps,1+eps)*A) * w], where A is the
    #   combined advantage and w = per-frame sample_weight renormalized
    #   to mean 1 within the minibatch.  (Conventional name: policy_loss.)
    policy_loss_mean: float
    # Mean over actor minibatches of the PRE-clip total grad L2 norm
    #   (clip_grad_norm_ returns the norm before clipping).
    grad_norm_actor_mean: float
    # One dict per epoch: {kl_mean, kl_max, kl_std, n_minibatches}
    #   computed over that epoch's actor-minibatch k3 values.  Epochs
    #   where the actor was stopped appear with n_minibatches=0 and
    #   zeroed stats.  len(epoch_kl_stats) == epochs_done.
    epoch_kl_stats: List[Dict[str, Any]]

    # --- Buffer descriptors (pre-update, over buf.*) ---
    # mean/min/max over buf.traj_lengths.  Buffer trajectories are the
    #   training unit — an episode can contribute more than one.  Real
    #   episode lengths live in episode_stats (ep.*).
    traj_len_mean: float
    traj_len_min: float
    traj_len_max: float

    # --- Uncertainty floor mechanism (exploration regularizer) ---
    # buf.uncertainty.mean() — mean per-frame uncertainty U over the
    #   whole buffer at theta_old.  Framework-owned: aggregated from the
    #   ActorEval.uncertainty contract field, not from policy_stats.
    uncertainty_mean: float
    # uncertainty_floor / uncertainty_coef: the ExplorationSpec values
    #   actually in force for this update (resolved by the loop, not by
    #   the trainer).  Logged so scheduled floor/coef changes are
    #   visible per-update rather than only as the initial config.
    uncertainty_floor: float
    uncertainty_coef: float
    # floor_loss_mean: the uncertainty-floor hinge term
    #   uncertainty_coef * mean(relu(floor - U)^2 * floor_weight),
    #   averaged over actor minibatches actually run. 0.0 when the
    #   floor mechanism is off (coef=0 or floor=0).
    floor_loss_mean: float
    # action_grad_pol_mean / action_grad_floor_mean: L2 norm of each
    #   loss term's gradient over ALL actor parameters (autograd.grad,
    #   allow_unused → untouched params count as 0), sampled on each
    #   epoch's first minibatch and averaged. Framework-owned — no
    #   policy hook.
    action_grad_pol_mean: float
    action_grad_floor_mean: float

    # --- Post-update cross-section (final actor, whole buffer) ---
    # post_kl_*: k3 KL ((r-1) - log r) between pi_old and the FINAL actor,
    #   computed once over the whole buffer at update end — the actual
    #   trust-region displacement, cleanly comparable across updates
    #   (unlike kl_mean, whose minibatch mean mixes iterates and whose
    #   sample set shrinks when the actor early-stops).  pos/neg split by
    #   advantage sign shows which side the displacement concentrated on.
    post_kl_mean: float
    post_kl_max: float
    post_kl_pos_mean: float
    post_kl_neg_mean: float
    # post_clip_dloss_mean: delta of the double-clipped surrogate vs the
    #   r=1 baseline — -mean[w*A*(clip(r,1-eps,1+eps)-1)] evaluated once
    #   at update end.  Negative = net alignment with the fixed
    #   advantages; unlike the PPO min() surrogate both tails are
    #   clamped, so the value is a bounded per-sample measure of
    #   direction.
    post_clip_dloss_mean: float
    # post_ratio_bins: 9 fractions of ALL samples (sum=1) — final ratio
    #   in 4 bands (r<1-eps, [1-eps,1), [1,1+eps], >1+eps) split by
    #   advantage sign, plus the exact-zero-advantage share.
    post_ratio_bins: Dict[str, float]

    # --- Per-channel (keyed by channel name) ---
    # Mean over that channel's minibatch value losses: masked weighted
    #   MSE  sum((V_c - ret_c)^2 * mask * w) / n_active  on frames where
    #   the channel is active (mask excludes inactive segments).
    critic_loss_mean: Dict[str, float]
    # 1 - Var(ret_c - V_c) / Var(ret_c) on channel-active frames, using
    #   theta_old values (before this update's critic steps); 0.0 when
    #   Var(ret_c) < 1e-8.
    explained_variance: Dict[str, float]
    # clip(EV_c, 0, 1) ** 0.5 — the advantage multiplier actually
    #   applied to channel c this update (1.0 for all channels when
    #   use_confidence=False).
    confidence: Dict[str, float]
    # Mean/std of the RAW per-channel GAE advantages on channel-active
    #   frames — before the z-score normalization and aw*confidence
    #   weighting that produce the combined advantage.
    adv_mean: Dict[str, float]
    adv_std: Dict[str, float]
    # Mean/std of per-channel GAE lambda-returns (ret = adv + V_old,
    #   the critic's regression target) on channel-active frames.
    ret_mean: Dict[str, float]
    ret_std: Dict[str, float]
    # Mean over minibatches of that critic's pre-clip grad L2 norm.
    critic_grad_norm_mean: Dict[str, float]

    # --- Policy-contributed (no contract) ---
    # dict(buf.actor_stats) — whatever the policy contributed via
    #   ActorEval.stats at rollout-eval time.
    policy_stats: Mapping[str, float]

    # --- Diagnostics (human-readable lines, not for programmatic use) ---
    # Warning/info strings collected inside ppo_update (per-channel
    #   return ranges, [kl_stats] per-epoch CV lines, zero-confidence /
    #   zero-variance warnings, floor diagnostics); the loop prints them
    #   to train.log — they are NOT part of __RAW_STATS__ flattening.
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
            kl_mean=0.0,
            kl_max=0.0,
            early_stop_kl_mean=0.0,
            clip_frac_mean=0.0,
            clip_frac_hi_mean=0.0,
            clip_frac_lo_mean=0.0,
            ratio_mean=1.0,
            ratio_min=1.0,
            ratio_max=1.0,
            policy_loss_mean=0.0,
            grad_norm_actor_mean=0.0,
            epochs_done=0,
            actor_epochs_done=0,
            n_batches=0,
            n_trajectories=0,
            total_steps=0,
            uncertainty_mean=0.0,
            traj_len_mean=0.0,
            traj_len_min=0.0,
            traj_len_max=0.0,
            epoch_kl_stats=[],
            uncertainty_floor=0.0,
            uncertainty_coef=0.0,
            floor_loss_mean=0.0,
            action_grad_pol_mean=0.0,
            action_grad_floor_mean=0.0,
            post_kl_mean=0.0,
            post_kl_max=0.0,
            post_kl_pos_mean=0.0,
            post_kl_neg_mean=0.0,
            post_clip_dloss_mean=0.0,
            post_ratio_bins={},
            critic_loss_mean={k: 0.0 for k in reward_keys},
            explained_variance={k: 0.0 for k in reward_keys},
            confidence={k: 0.0 for k in reward_keys},
            adv_mean={k: 0.0 for k in reward_keys},
            adv_std={k: 0.0 for k in reward_keys},
            ret_mean={k: 0.0 for k in reward_keys},
            ret_std={k: 0.0 for k in reward_keys},
            critic_grad_norm_mean={k: 0.0 for k in reward_keys},
            policy_stats={},
            diagnostics=[],
            is_empty=True,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        """Flatten to the dict format for ``__RAW_STATS__`` logging.

        ``policy_stats`` is spread to top level so that flat consumers
        keep working.  The framework's own keys always win collisions
        (spread first).
        """
        d: Dict[str, Any] = dict(self.policy_stats)
        d.update({
            # --- Update shape ---
            "n_trajectories": self.n_trajectories,
            "total_steps": self.total_steps,
            "n_batches": self.n_batches,
            "epochs_done": self.epochs_done,
            "actor_epochs_done": self.actor_epochs_done,
            # --- Process dynamics ---
            "kl_mean": self.kl_mean,
            "kl_max": self.kl_max,
            "early_stop_kl_mean": self.early_stop_kl_mean,
            "ratio_mean": self.ratio_mean,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "clip_frac_mean": self.clip_frac_mean,
            "clip_frac_hi_mean": self.clip_frac_hi_mean,
            "clip_frac_lo_mean": self.clip_frac_lo_mean,
            "policy_loss_mean": self.policy_loss_mean,
            "grad_norm_actor_mean": self.grad_norm_actor_mean,
            "epoch_kl_stats": self.epoch_kl_stats,
            # --- Buffer descriptors ---
            "traj_len_mean": self.traj_len_mean,
            "traj_len_min": self.traj_len_min,
            "traj_len_max": self.traj_len_max,
            # --- Uncertainty floor ---
            "uncertainty_mean": self.uncertainty_mean,
            "uncertainty_floor": self.uncertainty_floor,
            "uncertainty_coef": self.uncertainty_coef,
            "floor_loss_mean": self.floor_loss_mean,
            "action_grad_pol_mean": self.action_grad_pol_mean,
            "action_grad_floor_mean": self.action_grad_floor_mean,
            # --- Post-update cross-section ---
            "post_kl_mean": self.post_kl_mean,
            "post_kl_max": self.post_kl_max,
            "post_kl_pos_mean": self.post_kl_pos_mean,
            "post_kl_neg_mean": self.post_kl_neg_mean,
            "post_clip_dloss_mean": self.post_clip_dloss_mean,
        })
        for key, val in self.post_ratio_bins.items():
            d[f"rbin_{key}"] = val
        for key, val in self.critic_loss_mean.items():
            d[f"vloss_mean_{key}"] = val
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
        for key, val in self.critic_grad_norm_mean.items():
            d[f"grad_norm_mean_{key}"] = val
        return d


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
    ) -> Optional[Mapping[str, float]]:
        """Absorb training statistics into internal state.

        Called once per update **after** ``ppo_update`` completes, with
        the typed :class:`UpdateStats` for that update.  The experiment
        can accumulate history (e.g. a rolling KL window) into instance
        state, which ``exploration()`` will read on the next update.

        This is the training-stats counterpart of ``on_eval()``:
        ``on_eval`` closes the loop on *reward weighting* using eval
        episodes, while ``on_update`` closes the loop on *exploration
        strength* using training statistics.

        Args:
            stats: Typed summary of this update's PPO results.  See
                :class:`UpdateStats` for the full field list.  The
                ``policy_stats`` sub-mapping carries policy-contributed
                diagnostics but has **no cross-family contract** —
                treat it as opaque hints.
            update: Current update index (1-based, matches the loop).

        Returns:
            Optional experiment-defined metrics for this update.  Any
            returned mapping is logged under the ``experiment`` key of
            ``__RAW_STATS__`` and surfaces in the viewer as ``exp.*``
            charts — the update-time mirror of ``on_eval``'s ``info``
            dict.  ``None`` (the default) logs nothing.  Only finite
            scalars with ``[a-z0-9_]`` keys are kept.

            Metrics needing episode/trajectory data are accumulated
            during ``build_trajectories()`` (which sees every episode)
            and reported here — e.g. stash ``self._final_pots`` while
            building, then ``return {"online_success": ...}``.  A
            subclass that overrides ``on_update`` for its own state
            tracking should ``super().on_update(stats, update)`` and
            merge the returned dict so it doesn't drop base metrics.
        """
        return None

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

    def lr_schedule(
        self, update: int,
    ) -> Optional["LRSpec"]:
        """Return this update's optimizer LR, or None to keep current.

        Called once per update before ``ppo_update``, symmetric with
        ``exploration()``.  Returns **absolute** learning rates (not
        multipliers), consistent with the resume path that force-aligns
        optimizer LRs to ``CommonParams`` values.

        May read state accumulated by ``on_update()`` — enabling
        closed-loop schedules (e.g. decay once a rolling KL window
        saturates).  The *actual* applied LRs are logged to
        ``__RAW_STATS__`` under ``stats.actor_lr`` / ``stats.critic_lr``
        so the realized schedule is always observable.

        Args:
            update: Current update index (1-based, matches the loop).

        Returns:
            An ``LRSpec``, or ``None`` to keep the current LR.
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

