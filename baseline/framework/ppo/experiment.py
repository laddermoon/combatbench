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
from dataclasses import dataclass, field, fields as _dc_fields, replace
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
      - ``target_kl > 0.0``: per-minibatch KL early-stop is active.  The
        actor stops updating when the mean KL over the last
        ``early_stop_kl_window`` minibatches exceeds ``target_kl`` — a
        sliding window that crosses epoch boundaries, so the criterion
        is identical at every minibatch position (critics continue —
        see B1 in ``ppo_update``).
      - ``target_kl == 0.0``: KL early-stop is **disabled**, not
        "zero-tolerance".  The actor runs every epoch and minibatch.
        This is the default behavior when you want to run vanilla PPO
        without a trust-region guard.

    ``early_stop_kl_window``: number of most-recent minibatches whose
    KL is averaged for the early-stop decision.  ``1`` = instantaneous
    per-minibatch KL (most responsive, noisiest); larger values smooth
    minibatch sampling noise at the cost of reacting a few minibatches
    later.  The window spans epoch boundaries — it is NOT reset per
    epoch, which keeps the effective threshold position-independent.
    """

    clip_eps: float
    target_kl: float
    update_epochs: int
    minibatch_size: int
    early_stop_kl_window: int

    # Per-channel advantage normalization applied to the full buffer
    # before channels are combined:
    #   "zscore" — (A − mean)/std.  Batch-mean centering; frames near
    #       the batch mean can flip sign, which also flips that frame's
    #       surrogate gradient direction.
    #   "std"    — A/std.  Scale-only normalization; preserves the raw
    #       advantage sign of every frame.
    #   "gauss_rank" — rank → uniform quantile → Φ⁻¹.  Output is exactly
    #       N(0,1)-shaped; preserves ordering, replaces magnitudes with
    #       normal order statistics, implicitly bounds the tail at
    #       ±Φ⁻¹(1−1/2n) with no threshold parameter.
    adv_norm: str = "zscore"

    # Winsorize (缩尾) the normalized COMBINED advantage at ±sigma
    # before it enters the surrogate.  Bounds the per-frame gradient
    # coefficient |adv| so a single outlier trajectory cannot dominate
    # the update direction.  0.0 disables.
    adv_winsorize_sigma: float = 0.0

    # Dual-Clip floor (Ye et al. 2020): for adv<0 the standard
    # min(surr1,surr2) selects the unclipped branch once ratio > 1+eps,
    # and the surrogate diverges to −∞ as ratio grows — gradient
    # coefficient ∝ |adv·ratio|.  dual_clip_c floors the surrogate at
    # c·adv: frames past ratio > c contribute a constant value and zero
    # gradient.  0.0 disables; typical c ≈ 3.
    dual_clip_c: float = 0.0

    # --- ADV gradient-signal diagnostic (theta_old frame sampling) ---
    # DUMP-ONLY diagnostic: it runs only on updates that carry a dump
    # request (loop.py wires it via GradDiagSpec — there is no periodic
    # collection).  At the start of such an update (after combined_adv,
    # before any actor step), the trainer computes the full-buffer
    # aggregate gradient G = mean_i g_i of the actual training loss
    # (surrogate + floor) via chunked backwards, then samples
    # DUMP_GRADSIG_SAMPLE_SIZE buffer frames and computes each frame's
    # improvement-direction gradient g_i.  Per-frame projection
    # p_i = g_i·G_hat and cosine yield the scalars (grad_sig_g_norm /
    # coherence / proj_mean / proj_std / frac_neg / dir_cos) plus the
    # histogram + raw per-frame arrays stored inside the dump's
    # gradsig.npz.
    grad_sig_cos_bins: int = 64
    # Norm bins are equal-mass quantile bins derived from the first
    # computed update (~1/norm_bins of frames per row), then frozen in
    # gradsig/meta.json.  The top bins naturally span the heavy tail.
    grad_sig_norm_bins: int = 40
    # Norm-bin range for the histogram's second axis.  Both > 0 (and
    # hi > lo) pins an explicit log-spaced range; otherwise the range is
    # derived on the first computed update and frozen in
    # gradsig/meta.json for cross-update comparability.
    grad_sig_norm_lo: float = 0.0
    grad_sig_norm_hi: float = 0.0

    def __post_init__(self):
        # Validate at construction so misconfiguration surfaces immediately
        # rather than as a silent behavioral difference mid-training.
        if self.target_kl < 0.0:
            raise ValueError(
                f"target_kl must be >= 0.0, got {self.target_kl}. "
                f"Use 0.0 to disable KL early-stop."
            )
        if self.early_stop_kl_window < 1:
            raise ValueError(
                f"early_stop_kl_window must be >= 1, "
                f"got {self.early_stop_kl_window}."
            )
        _adv_norm_methods = ("zscore", "std", "gauss_rank")
        if self.adv_norm not in _adv_norm_methods:
            raise ValueError(
                f"adv_norm must be one of {_adv_norm_methods}, got "
                f"{self.adv_norm!r}."
            )
        if self.adv_winsorize_sigma < 0.0:
            raise ValueError(
                f"adv_winsorize_sigma must be >= 0.0, got "
                f"{self.adv_winsorize_sigma}."
            )
        if self.dual_clip_c < 0.0 or (0.0 < self.dual_clip_c < 1.0):
            raise ValueError(
                f"dual_clip_c must be 0.0 (disabled) or >= 1.0, got "
                f"{self.dual_clip_c}."
            )
        if self.grad_sig_cos_bins < 4:
            raise ValueError(
                f"grad_sig_cos_bins must be >= 4, "
                f"got {self.grad_sig_cos_bins}."
            )
        if self.grad_sig_norm_bins < 4:
            raise ValueError(
                f"grad_sig_norm_bins must be >= 4, "
                f"got {self.grad_sig_norm_bins}."
            )
        if (self.grad_sig_norm_lo > 0.0) != (self.grad_sig_norm_hi > 0.0):
            raise ValueError(
                f"grad_sig_norm_lo/hi must be set together (or both 0 for "
                f"auto), got lo={self.grad_sig_norm_lo} "
                f"hi={self.grad_sig_norm_hi}."
            )
        if self.grad_sig_norm_lo > 0.0 and self.grad_sig_norm_hi <= self.grad_sig_norm_lo:
            raise ValueError(
                f"grad_sig_norm_hi must be > grad_sig_norm_lo, got "
                f"lo={self.grad_sig_norm_lo} hi={self.grad_sig_norm_hi}."
            )


# ---------------------------------------------------------------------------
# Per-update parameter overrides
# ---------------------------------------------------------------------------
#
# The experiment may return a flat override mapping from
# ``param_overrides(update)`` each update; ``resolve_update_params``
# applies it on top of the base CommonParams/PPOParams for that update
# only.  Keys are routed to whichever dataclass declares the field —
# the two have no colliding names.  Validation happens through
# ``dataclasses.replace`` → ``__post_init__``, so an illegal value fails
# at the update boundary with the normal message.
#
# Blacklisted fields cannot be overridden mid-run:
#   - ``name``            — experiment identity.
#   - ``seed``            — the run's RNG-stream identity; changing it
#       mid-run silently re-derivs every downstream seed.
#   - ``rollout_workers`` — the ParallelRollouter pool is created once
#       before the loop; resizing requires a rebuild the framework
#       does not perform.

_PARAM_OVERRIDE_BLACKLIST = frozenset(
    {"name", "seed", "rollout_workers"}
)


def _param_field_names() -> Tuple[frozenset, frozenset]:
    return (
        frozenset(f.name for f in _dc_fields(CommonParams)),
        frozenset(f.name for f in _dc_fields(PPOParams)),
    )


def resolve_update_params(
    base_cp: CommonParams,
    base_pp: PPOParams,
    override: Optional[Mapping[str, Any]],
) -> Tuple[CommonParams, PPOParams, Dict[str, Any]]:
    """Apply a flat per-update override dict to the owning dataclass.

    Args:
        base_cp, base_pp: The experiment's base parameters.
        override: ``{field_name: value}`` or None.  Applied for this
            update only — repeat the override on every update it should
            stay active.

    Returns:
        ``(cp_effective, pp_effective, applied)`` where ``applied`` is
        the validated override dict actually used (for logging).

    Raises:
        ValueError: Unknown field, blacklisted field, or a value that
            fails dataclass validation — with context naming the field.
    """
    if not override:
        return base_cp, base_pp, {}

    cp_names, pp_names = _param_field_names()
    unknown = sorted(set(override) - cp_names - pp_names)
    if unknown:
        allowed = sorted(cp_names | pp_names)
        raise ValueError(
            f"param_overrides: unknown field(s) {unknown}. "
            f"Allowed fields: {allowed}."
        )
    blocked = sorted(set(override) & _PARAM_OVERRIDE_BLACKLIST)
    if blocked:
        raise ValueError(
            f"param_overrides: field(s) {blocked} cannot be changed "
            f"mid-run (identity or fixed resources). Allowed fields: "
            f"{sorted((cp_names | pp_names) - _PARAM_OVERRIDE_BLACKLIST)}."
        )

    cp_kwargs = {k: v for k, v in override.items() if k in cp_names}
    pp_kwargs = {k: v for k, v in override.items() if k in pp_names}
    try:
        cp_eff = replace(base_cp, **cp_kwargs) if cp_kwargs else base_cp
        pp_eff = replace(base_pp, **pp_kwargs) if pp_kwargs else base_pp
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"param_overrides: invalid value in {override}: {e}"
        ) from e
    return cp_eff, pp_eff, dict(override)


# ---------------------------------------------------------------------------
# GradDiagSpec — per-update request for the ADV gradient-signal diagnostic
# ---------------------------------------------------------------------------

# Frames sampled per dump-triggered diagnostic.  Fixed, not a parameter:
# the diagnostic exists to serve dumps, and a single sampling depth keeps
# gradsig.npz artifacts comparable across runs and updates.
DUMP_GRADSIG_SAMPLE_SIZE = 2000


@dataclass(frozen=True)
class GradDiagSpec:
    """Request for the frame-level gradient-signal diagnostic.

    Built by the training loop for dump-requested updates only — the
    diagnostic is an on-demand investigation tool, not periodic
    telemetry — then passed to ``ppo_update``.  The trainer first
    computes the aggregate
    gradient G = mean_i g_i over the FULL buffer via chunked backwards,
    then samples ``sample_size`` buffer frames with a dedicated RNG
    seeded from ``seed`` (isolated from the training RNG stream) and
    computes each sampled frame's improvement-direction gradient of the
    actual training loss (surrogate + floor hinge) at theta_old.
    Per-frame projections p_i = g_i·G_hat and cosines yield the
    run-level scalars plus a 2D histogram and raw per-frame arrays.

    Attributes:
        sample_size: Frames to sample this update (clamped to buffer
            size by the trainer).
        cos_bins: Number of cosine-similarity bins, fixed linear
            coverage of [-1, 1].
        norm_bins: Number of per-frame-norm bins.  When edges are
            derived (``norm_edges=None``) they are equal-mass quantile
            bins — each row holds ~1/norm_bins of the frames.
        norm_edges: Complete norm-bin edge array (frozen meta.json or
            configured range), or ``None`` to derive quantile edges
            from this update's frame norms (the loop then freezes them
            in ``gradsig/meta.json``).
        seed: Seed for the dedicated sampling RNG — never touches the
            training RNG stream.
        prev_g: The aggregate gradient G from the previous diagnostic
            update, held in loop memory (never persisted), used to
            compute ``grad_sig_dir_cos`` — the direction persistence
            of the aggregate pull across updates.  ``None`` on the
            first diagnostic update.
    """

    sample_size: int
    cos_bins: int
    norm_bins: int
    norm_edges: Optional[np.ndarray]
    seed: int
    prev_g: Optional[np.ndarray] = None


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
    - ``frames`` = environment data (``total_frames``, ``traj_len_*``,
      ``ep_len_*``); ``steps`` = optimizer minibatch steps
      (``actor_steps``).  The two units never share a name.
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
    total_frames: int
    # max(1, ceil(total_frames / minibatch_size)) — minibatches per
    #   epoch; the split is even, no tiny remainder batch.
    n_batches: int
    # Total number of actor minibatch steps actually taken this update,
    #   summed from epoch_kl_stats[*]["n_minibatches"] — the exact count
    #   of minibatches where the actor ran (and usually took a gradient
    #   step, except the triggering minibatch of an early stop).  This is
    #   <= n_batches * actor_epochs_done because the early-stop epoch is
    #   truncated at the KL-threshold minibatch.
    actor_steps: int
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
    #   minibatch where target_kl early-stop fired (sliding-window mean
    #   KL > target_kl).  0.0 if early stop never triggered.
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
    # Additive decomposition of post_clip_dloss_mean into the parts
    #   contributed by advantage-aligned vs anti-aligned per-sample
    #   moves, in the same loss units:
    #     dloss_gain = -mean(relu(contrib))   <= 0  (favorable pull)
    #     dloss_harm = -mean(min(contrib,0))  >= 0  (unfavorable push)
    #   Invariant: dloss_mean = dloss_gain + dloss_harm exactly.
    #   When dloss worsens: gain rising toward 0 = under-optimized
    #   (truncated/weak); harm rising = more samples moved against
    #   their advantage (conflicting/noisy direction).
    post_clip_dloss_gain: float
    post_clip_dloss_harm: float
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
    # Mean/std/min/max of the RAW per-channel GAE advantages on
    #   channel-active frames — before the z-score normalization and
    #   aw*confidence weighting that produce the combined advantage.
    #   min/max are per-frame extrema (not per-trajectory).
    adv_mean: Dict[str, float]
    adv_std: Dict[str, float]
    adv_min: Dict[str, float]
    adv_max: Dict[str, float]
    # Mean/std/min/max of per-channel GAE lambda-returns (ret = adv +
    #   V_old, the critic's regression target) on channel-active frames.
    #   min/max are per-frame extrema (not per-trajectory).
    ret_mean: Dict[str, float]
    ret_std: Dict[str, float]
    ret_min: Dict[str, float]
    ret_max: Dict[str, float]
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

    # --- ADV gradient-signal diagnostic (theta_old, pre-update) ---
    # G = mean_i g_i is the aggregate gradient of the actual training
    #   loss (surrogate + floor hinge) over the FULL buffer, computed
    #   via chunked backwards at theta_old.  g_i is sampled frame i's
    #   improvement-direction gradient; p_i = g_i·G_hat is its signed
    #   projection onto the aggregate direction (positive = helped by
    #   this update's direction, negative = sacrificed).
    # grad_sig_g_norm: ||G|| — net pull strength (intended, pre-optimizer).
    # grad_sig_coherence: ||G|| / mean||g_i|| — fraction of total pull
    #   surviving aggregation (1 = all frames pull the same way; the
    #   denominator is estimated from the sampled frames).
    # grad_sig_proj_mean: mean(p_i) over the sample — an unbiased
    #   estimator of ||G|| (identity mean(p)=||G|| on the full buffer);
    #   large deviation flags a sampling/implementation problem.
    # grad_sig_proj_std: std(p_i) — dispersion of per-frame gains.
    # grad_sig_frac_neg_mean: P(p_i < 0) — fraction of frames this update's
    #   direction sacrifices.
    # grad_sig_dir_cos: cos(G_u, G_{u-1}) — direction persistence of the
    #   aggregate pull across updates; 0 on the first diagnostic update.
    # grad_sig_n_frames: sampled frames with a usable (finite, nonzero-norm)
    #   gradient — coverage indicator; compare with the configured
    #   sample size.
    # grad_sig_norm_mean: mean per-frame gradient L2 norm over the
    #   sample — the coherence denominator: ‖G‖ = mean‖g_i‖ × coherence.
    # grad_sig_time_s: wall time of the diagnostic inside ppo_update.
    # These keys are emitted into __RAW_STATS__ ONLY when the diagnostic
    # actually ran (dump-requested update) — absence = not measured,
    # which must stay distinguishable from a measured zero.
    grad_sig_ran: bool = False
    grad_sig_g_norm: float = 0.0
    grad_sig_coherence: float = 0.0
    grad_sig_proj_mean: float = 0.0
    grad_sig_proj_std: float = 0.0
    grad_sig_frac_neg_mean: float = 0.0
    grad_sig_dir_cos: float = 0.0
    grad_sig_n_frames: int = 0
    grad_sig_norm_mean: float = 0.0
    grad_sig_time_s: float = 0.0
    # Non-logged transport for the per-update histogram artifact — the
    # loop merges it into the dump's ``gradsig.npz`` (the histogram is
    # not a scalar and must not inflate __RAW_STATS__).  None = the
    # diagnostic did not run this update.
    grad_sig_payload: Optional[Dict[str, Any]] = None
    # Non-logged transport of this update's aggregate gradient vector —
    # the loop holds it in memory and passes it back as spec.prev_g on
    # the next diagnostic update to compute grad_sig_dir_cos.  Never
    # persisted (a ~100k-float vector per update would bloat the npz).
    grad_sig_gvec: Optional[np.ndarray] = None

    # Fraction of buffer frames whose normalized combined advantage was
    #   clipped to ±adv_winsorize_sigma this update.  0.0 when winsorize
    #   is disabled, gated off for this update, or nothing exceeded the
    #   bound — so the stat doubles as an activation indicator.
    adv_winsorize_clip_frac: float = 0.0

    # Mean over actor minibatches of the fraction of frames sitting on
    #   the dual-clip floor (adv<0 AND ratio>dual_clip_c) — the blind
    #   quadrant of standard clipping.  0.0 when dual clip is disabled.
    dual_clip_frac_mean: float = 0.0

    # Fraction of actor minibatches whose PRE-clip grad L2 norm exceeded
    #   grad_clip_norm — clip_grad_norm_ returns the pre-clip norm, so
    #   this counts how often the clip actually fired.  grad_norm_actor_
    #   mean alone cannot distinguish "all minibatches at 3×cap" from
    #   "half unclipped at 1×cap, half at 5×cap".  1.0 = every step
    #   clipped (cap binds); 0.0 = clip never active.
    grad_clip_frac: float = 0.0
    # Per-channel equivalent for the critics.
    critic_grad_clip_frac: Dict[str, float] = field(default_factory=dict)

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
            actor_steps=0,
            n_trajectories=0,
            total_frames=0,
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
            post_clip_dloss_gain=0.0,
            post_clip_dloss_harm=0.0,
            post_ratio_bins={},
            critic_loss_mean={k: 0.0 for k in reward_keys},
            explained_variance={k: 0.0 for k in reward_keys},
            confidence={k: 0.0 for k in reward_keys},
            adv_mean={k: 0.0 for k in reward_keys},
            adv_std={k: 0.0 for k in reward_keys},
            adv_min={k: 0.0 for k in reward_keys},
            adv_max={k: 0.0 for k in reward_keys},
            ret_mean={k: 0.0 for k in reward_keys},
            ret_std={k: 0.0 for k in reward_keys},
            ret_min={k: 0.0 for k in reward_keys},
            ret_max={k: 0.0 for k in reward_keys},
            critic_grad_norm_mean={k: 0.0 for k in reward_keys},
            policy_stats={},
            diagnostics=[],
            is_empty=True,
            grad_sig_g_norm=0.0,
            grad_sig_coherence=0.0,
            grad_sig_proj_mean=0.0,
            grad_sig_proj_std=0.0,
            grad_sig_frac_neg_mean=0.0,
            grad_sig_dir_cos=0.0,
            grad_sig_n_frames=0,
            grad_sig_norm_mean=0.0,
            grad_sig_time_s=0.0,
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
            "total_frames": self.total_frames,
            "n_batches": self.n_batches,
            "actor_steps": self.actor_steps,
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
            "post_clip_dloss_gain": self.post_clip_dloss_gain,
            "post_clip_dloss_harm": self.post_clip_dloss_harm,
            "adv_winsorize_clip_frac": self.adv_winsorize_clip_frac,
            "dual_clip_frac_mean": self.dual_clip_frac_mean,
            "grad_clip_frac": self.grad_clip_frac,
        })
        # grad_sig_* scalars only when the dump-triggered diagnostic ran
        # — a measured value is meaningful, an unmeasured zero is not.
        if self.grad_sig_ran:
            d.update({
                "grad_sig_g_norm": self.grad_sig_g_norm,
                "grad_sig_coherence": self.grad_sig_coherence,
                "grad_sig_proj_mean": self.grad_sig_proj_mean,
                "grad_sig_proj_std": self.grad_sig_proj_std,
                "grad_sig_frac_neg_mean": self.grad_sig_frac_neg_mean,
                "grad_sig_dir_cos": self.grad_sig_dir_cos,
                "grad_sig_n_frames": self.grad_sig_n_frames,
                "grad_sig_norm_mean": self.grad_sig_norm_mean,
                "grad_sig_time_s": self.grad_sig_time_s,
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
        for key, val in self.adv_min.items():
            d[f"adv_min_{key}"] = val
        for key, val in self.adv_max.items():
            d[f"adv_max_{key}"] = val
        for key, val in self.ret_mean.items():
            d[f"ret_mean_{key}"] = val
        for key, val in self.ret_std.items():
            d[f"ret_std_{key}"] = val
        for key, val in self.ret_min.items():
            d[f"ret_min_{key}"] = val
        for key, val in self.ret_max.items():
            d[f"ret_max_{key}"] = val
        for key, val in self.critic_grad_norm_mean.items():
            d[f"grad_norm_mean_{key}"] = val
        for key, val in self.critic_grad_clip_frac.items():
            d[f"grad_clip_frac_{key}"] = val
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

    def param_overrides(
        self, update: int,
    ) -> Optional[Mapping[str, Any]]:
        """Return per-update parameter overrides, or None to use base.

        Called once per update, before exploration/lr_schedule.  The
        returned flat ``{field_name: value}`` mapping is routed to
        ``CommonParams`` or ``PPOParams`` by field name, applied via
        ``dataclasses.replace`` for this update only, and recorded in
        ``__RAW_STATS__.param_overrides`` plus the train log.

        Return the override on **every** update it should stay active —
        the framework resolves effective params fresh each update, so a
        resumed run needs no event history::

            def param_overrides(self, update):
                if update >= 282:
                    return {"dual_clip_c": 3.0}
                return None

        May read state accumulated by ``on_update()`` for closed-loop
        schedules.  Blacklisted fields (``name``, ``seed``,
        ``rollout_workers``) and unknown fields raise ``ValueError``.
        CLI ``--param KEY=VALUE[@UPDATE]`` patches apply *after* this
        hook and win on conflicts.
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

