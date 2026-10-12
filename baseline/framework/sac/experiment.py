"""ExperimentSAC — SAC-native experiment abstraction.

Unlike ``ExperimentPPO`` (PPO-only), this interface is designed from the
ground up for off-policy training. The key difference is that the
experiment controls not just reward semantics and trajectory slicing,
but also **data distribution**: what data enters the replay buffer, how
it's tagged, and how it should be sampled.

The framework owns SAC mechanics (Q updates, auto-alpha, target
networks, replay buffer management). The experiment owns semantics
(rewards, termination, actor weights, tags, curriculum, evaluation).

See ``PLAN.md`` §2 for the full design rationale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint

from .actor import SACActor
from .collection import SACFactSpec, SACJob
from .transition import SACTransitionSlice


def _is_intlike(value: Any) -> bool:
    """True only for genuine integers (bool excluded).

    Rejects fractional floats like ``1.5`` *and* integral floats like
    ``256.0`` — config fields typed ``int`` must carry a real integer,
    not a silently truncated float.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_numlike(value: Any) -> bool:
    """True only for real numbers (bool/str excluded)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and (
        not isinstance(value, (bool, np.bool_))
    )


# ---------------------------------------------------------------------------
# Reward channel configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SACRewardChannel:
    """Configuration for one reward channel in SAC.

    Attributes:
        name: Unique channel key (e.g. ``"r_fall"``).
        gamma: Discount factor for this channel's Bellman backup.
        n_step: N-step return horizon (1 = standard TD). Larger values
            propagate sparse rewards faster at the cost of higher
            variance. Per-channel: sparse channels (damage) benefit
            from large n_step, dense channels (posture) from n_step=1.
        n_critics: Number of twin Q critics for this channel (2 =
            standard clipped double-Q). More critics = stronger
            pessimism, useful for sparse/high-variance channels.
        in_target_min: How many of the twin critics to take the min
            over when computing the target. If ``n_critics=5`` and
            ``in_target_min=3``, the target uses the min of 3 randomly
            selected critics (REDQ-style). Default: all.
        trunk_group: Name of the shared trunk group for this channel.
            Channels with the same ``trunk_group`` share a trunk
            network with per-channel heads. If None, auto-groups by
            ``gamma``.
        actor_weight_share: If True (default), this channel's Q
            participates in the actor's weighted-Q objective. If False,
            the channel's critics are still trained but the channel is
            excluded from the actor loss entirely (equivalent to
            actor_weight=0 for this channel on every transition).
    """

    name: str
    gamma: float
    n_step: int = 1
    n_critics: int = 2
    in_target_min: int = 2
    trunk_group: Optional[str] = None
    actor_weight_share: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SACRewardChannel.name must be non-empty")
        if not _is_numlike(self.gamma) or not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                f"channel {self.name!r}: gamma must be a finite number "
                f"in (0,1], got {self.gamma!r}"
            )
        for fname in ("n_step", "n_critics", "in_target_min"):
            if not _is_intlike(getattr(self, fname)):
                raise ValueError(
                    f"channel {self.name!r}: {fname} must be an "
                    f"integer, got {getattr(self, fname)!r}"
                )
        if self.n_step < 1:
            raise ValueError(
                f"channel {self.name!r}: n_step must be >= 1, "
                f"got {self.n_step}"
            )
        if self.n_critics < 1:
            raise ValueError(
                f"channel {self.name!r}: n_critics must be >= 1"
            )
        if not (1 <= self.in_target_min <= self.n_critics):
            raise ValueError(
                f"channel {self.name!r}: in_target_min="
                f"{self.in_target_min} must be in [1, n_critics="
                f"{self.n_critics}]"
            )


# ---------------------------------------------------------------------------
# SAC hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SACParams:
    """SAC algorithm hyperparameters.

    Attributes:
        replay_buffer_size: Max transitions in replay buffer.
        batch_size: Minibatch size for gradient steps.
        warmup_steps: Number of transitions to collect before first
            gradient step.
        utd_ratio: Update-to-data ratio. Number of gradient steps per
            new transition collected. ``n_gradient_steps = utd_ratio *
            n_new_transitions``.
        tau: Soft target update coefficient.
        init_alpha: Initial entropy temperature.
        auto_alpha: If True, tune alpha automatically to maintain
            target_entropy.
        target_entropy: Target entropy for auto-alpha. If None,
            defaults to ``-action_dim``.
        alpha_lr: Learning rate for alpha optimizer.
        log_alpha_min: Lower clamp for log_alpha.
        log_alpha_max: Upper clamp for log_alpha.
        use_grad_norm: RESERVED, currently unsupported — must be False.
            Action-gradient normalization for actor loss is not
            implemented; setting True raises at construction. If ever
            implemented, it would normalize per-channel action-gradient
            contributions; False (required) uses the weighted Q sum.
        grad_norm_est_interval: RESERVED knob for the unimplemented
            gradient-normalization mechanism (re-estimation interval in
            gradient steps). Validated but has no effect while
            ``use_grad_norm`` must be False.
        grad_norm_ema_decay: RESERVED knob — EMA decay for gradient
            scale statistics. No effect while ``use_grad_norm`` is
            unsupported.
        q_hidden_dim: Hidden dimension for Q networks.
        q_layer_norm: If True, use LayerNorm in Q trunk (improves
            stability, recommended for high-DOF).
        reward_scale: Global reward scaling factor applied before
            Bellman target computation.
    """

    replay_buffer_size: int = 500_000
    batch_size: int = 256
    warmup_steps: int = 10_000
    utd_ratio: float = 1.0
    max_grad_steps_per_round: int = 10_000
    tau: float = 0.005
    init_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    alpha_lr: float = 3e-4
    log_alpha_min: float = -10.0
    log_alpha_max: float = 2.0
    use_grad_norm: bool = False
    grad_norm_est_interval: int = 10
    grad_norm_ema_decay: float = 0.99
    q_hidden_dim: int = 256
    q_layer_norm: bool = False
    reward_scale: float = 1.0
    # A4.5: M — uniform samples enumerated per mixture component in
    # target/actor expectations (K components are always enumerated).
    expectation_samples: int = 1
    # A4.6: mutually exclusive regularizer modes.  "shannon" is the
    # baseline; "u_bonus"/"u_floor" are the alternative U route with
    # fixed λ.  u_kind resolves "native" → peak (single) / l2 (mixture).
    regularizer_mode: str = "shannon"
    reg_lambda: float = 0.0
    u_floor: float = 0.0
    u_kind: str = "native"

    def __post_init__(self) -> None:
        import math

        errs = []
        bad_ints = set()
        for fname in (
            "replay_buffer_size",
            "batch_size",
            "warmup_steps",
            "max_grad_steps_per_round",
            "grad_norm_est_interval",
            "q_hidden_dim",
            "expectation_samples",
        ):
            if not _is_intlike(getattr(self, fname)):
                bad_ints.add(fname)
                errs.append(
                    f"{fname} must be an integer, "
                    f"got {getattr(self, fname)!r}"
                )
        if "batch_size" not in bad_ints and self.batch_size <= 0:
            errs.append("batch_size must be > 0")
        if (
            "replay_buffer_size" not in bad_ints
            and "batch_size" not in bad_ints
            and self.replay_buffer_size < self.batch_size
        ):
            errs.append(
                f"replay_buffer_size={self.replay_buffer_size} < "
                f"batch_size={self.batch_size}"
            )
        if "warmup_steps" not in bad_ints and self.warmup_steps < 0:
            errs.append("warmup_steps must be >= 0")
        if not _is_numlike(self.utd_ratio) or not math.isfinite(
            float(self.utd_ratio)
        ):
            errs.append(
                f"utd_ratio must be a finite number, got {self.utd_ratio!r}"
            )
        elif self.utd_ratio <= 0.0:
            errs.append("utd_ratio must be > 0")
        if (
            "max_grad_steps_per_round" not in bad_ints
            and self.max_grad_steps_per_round <= 0
        ):
            errs.append("max_grad_steps_per_round must be > 0")
        if not _is_numlike(self.tau) or not (0.0 <= self.tau <= 1.0):
            errs.append(
                f"tau must be a finite number in [0,1], got {self.tau!r}"
            )
        if not _is_numlike(self.init_alpha) or not math.isfinite(
            float(self.init_alpha)
        ):
            errs.append(
                f"init_alpha must be a finite number, got {self.init_alpha!r}"
            )
        elif self.init_alpha <= 0.0:
            errs.append(f"init_alpha must be > 0, got {self.init_alpha}")
        if not _is_numlike(self.log_alpha_min) or math.isnan(
            float(self.log_alpha_min)
        ):
            errs.append(f"log_alpha_min must not be NaN/non-numeric")
        if not _is_numlike(self.log_alpha_max) or math.isnan(
            float(self.log_alpha_max)
        ):
            errs.append(f"log_alpha_max must not be NaN/non-numeric")
        if _is_numlike(self.log_alpha_min) and _is_numlike(
            self.log_alpha_max
        ) and self.log_alpha_min >= self.log_alpha_max:
            errs.append(
                f"log_alpha_min={self.log_alpha_min} >= "
                f"log_alpha_max={self.log_alpha_max}"
            )
        if not _is_numlike(self.alpha_lr) or not math.isfinite(
            float(self.alpha_lr)
        ):
            errs.append(
                f"alpha_lr must be a finite number, got {self.alpha_lr!r}"
            )
        elif self.alpha_lr <= 0.0:
            errs.append("alpha_lr must be > 0")
        if (
            _is_numlike(self.init_alpha)
            and math.isfinite(float(self.init_alpha))
            and self.init_alpha > 0.0
            and _is_numlike(self.log_alpha_min)
            and _is_numlike(self.log_alpha_max)
            and not math.isnan(float(self.log_alpha_min))
            and not math.isnan(float(self.log_alpha_max))
        ):
            if not (
                self.log_alpha_min <= math.log(self.init_alpha)
                <= self.log_alpha_max
            ):
                errs.append(
                    f"init_alpha={self.init_alpha} outside "
                    f"[exp(log_alpha_min), exp(log_alpha_max)] clamps"
                )
        if self.target_entropy is not None and (
            not _is_numlike(self.target_entropy)
            or not math.isfinite(float(self.target_entropy))
        ):
            errs.append(
                f"target_entropy must be a finite number or None, "
                f"got {self.target_entropy!r}"
            )
        if self.use_grad_norm:
            errs.append(
                "use_grad_norm=True is not supported: action-gradient "
                "normalization is reserved but unimplemented; set "
                "use_grad_norm=False"
            )
        if (
            "grad_norm_est_interval" not in bad_ints
            and self.grad_norm_est_interval < 1
        ):
            errs.append("grad_norm_est_interval must be >= 1")
        if not _is_numlike(self.grad_norm_ema_decay) or not (
            0.0 <= self.grad_norm_ema_decay < 1.0
        ):
            errs.append(
                f"grad_norm_ema_decay must be a finite number in [0,1), "
                f"got {self.grad_norm_ema_decay!r}"
            )
        if "q_hidden_dim" not in bad_ints and self.q_hidden_dim < 1:
            errs.append("q_hidden_dim must be >= 1")
        if (
            "expectation_samples" not in bad_ints
            and self.expectation_samples < 1
        ):
            errs.append("expectation_samples must be >= 1")
        if not _is_numlike(self.reward_scale) or not math.isfinite(
            float(self.reward_scale)
        ):
            errs.append(
                f"reward_scale must be a finite number, "
                f"got {self.reward_scale!r}"
            )
        if (
            not _is_numlike(self.reg_lambda)
            or not math.isfinite(float(self.reg_lambda))
            or self.reg_lambda < 0.0
        ):
            errs.append(
                f"reg_lambda must be a finite number >= 0, "
                f"got {self.reg_lambda!r}"
            )
        if not _is_numlike(self.u_floor) or not math.isfinite(
            float(self.u_floor)
        ):
            errs.append(
                f"u_floor must be a finite number, got {self.u_floor!r}"
            )
        if self.regularizer_mode not in ("shannon", "u_bonus", "u_floor"):
            errs.append(
                f"regularizer_mode must be shannon/u_bonus/u_floor, "
                f"got {self.regularizer_mode!r}"
            )
        if self.u_kind not in ("native", "peak", "l2"):
            errs.append(f"u_kind must be native/peak/l2, got {self.u_kind!r}")
        if errs:
            raise ValueError("invalid SACParams: " + "; ".join(errs))


# ---------------------------------------------------------------------------
# Common parameters (shared with PPO V2 but SAC-specific)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParamsSAC:
    """Training parameters for SAC experiments.

    Unlike PPO's ``CommonParams``, the clock is ``env_step``-based:
    eval_interval and checkpoint_interval are measured in environment
    steps, not updates. This is because SAC's update count depends on
    UTD ratio, making update-based scheduling non-comparable across
    configurations.

    Attributes:
        name: Experiment name.
        learning_rate: Actor learning rate.
        critic_learning_rate: Q critic learning rate.
        grad_clip_norm: Max gradient norm for all networks.
        episodes_per_update: Episodes to collect per rollout round.
        max_env_steps: Total environment steps to train for.
        eval_interval: Evaluate every N env steps.
        eval_episodes: Number of episodes per evaluation.
        video_eval_interval: Record video every N evals (0 = off).
        rollout_workers: Number of parallel rollout workers.
        seed: Random seed.
    """

    name: str
    learning_rate: float
    critic_learning_rate: float
    grad_clip_norm: float
    episodes_per_update: int
    max_env_steps: int
    eval_interval: int
    eval_episodes: int
    video_eval_interval: int
    rollout_workers: int
    seed: int

    def __post_init__(self) -> None:
        import math

        errs = []
        if not self.name:
            errs.append("name must be non-empty")
        for fname in (
            "learning_rate",
            "critic_learning_rate",
            "grad_clip_norm",
        ):
            value = getattr(self, fname)
            if not _is_numlike(value) or not math.isfinite(float(value)):
                errs.append(
                    f"{fname} must be a finite number, got {value!r}"
                )
            elif value <= 0.0:
                errs.append(f"{fname} must be > 0, got {value}")
        bad_ints = set()
        for fname in (
            "episodes_per_update",
            "max_env_steps",
            "eval_interval",
            "eval_episodes",
            "video_eval_interval",
            "rollout_workers",
            "seed",
        ):
            if not _is_intlike(getattr(self, fname)):
                bad_ints.add(fname)
                errs.append(
                    f"{fname} must be an integer, "
                    f"got {getattr(self, fname)!r}"
                )
        if (
            "episodes_per_update" not in bad_ints
            and self.episodes_per_update < 1
        ):
            errs.append("episodes_per_update must be >= 1")
        if "max_env_steps" not in bad_ints and self.max_env_steps < 1:
            errs.append("max_env_steps must be >= 1")
        if "eval_interval" not in bad_ints and self.eval_interval < 1:
            errs.append("eval_interval must be >= 1")
        if "eval_episodes" not in bad_ints and self.eval_episodes < 1:
            errs.append("eval_episodes must be >= 1")
        if (
            "video_eval_interval" not in bad_ints
            and self.video_eval_interval < 0
        ):
            errs.append("video_eval_interval must be >= 0")
        if "rollout_workers" not in bad_ints and self.rollout_workers < 1:
            errs.append("rollout_workers must be >= 1")
        if "seed" not in bad_ints and not (0 <= self.seed <= 2**32 - 1):
            errs.append(
                f"seed must be in [0, 2**32-1] (np.random.seed bound), "
                f"got {self.seed}"
            )
        if errs:
            raise ValueError("invalid CommonParamsSAC: " + "; ".join(errs))


# ---------------------------------------------------------------------------
# Data source declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSource:
    """Declares one data source for the replay buffer.

    The experiment returns a tuple of these from ``data_sources()``.
    The framework uses ``sampling_share`` to allocate collection budget
    and ``agent`` to determine which agent's perspective to collect.

    Attributes:
        kind: One of ``"self"``, ``"opponent"``, ``"pool"``,
            ``"scripted"``, ``"recorded"``.
        agent: Which agent to collect data from (``"robot_a"``,
            ``"robot_b"``, or ``"both"``).
        sampling_share: Target fraction of buffer capacity for this
            source. The framework normalizes shares to sum to 1.0.
        policy_blueprint: Optional path to a policy blueprint for
            scripted/opponent sources. None for ``"self"`` (uses
            current actor).
        config: Free-form config dict (e.g. pool JSON path, scripted
            policy parameters).
    """

    kind: str
    agent: str = "robot_a"
    sampling_share: float = 1.0
    policy_blueprint: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        if self.kind not in (
            "self",
            "opponent",
            "pool",
            "scripted",
            "recorded",
        ):
            raise ValueError(
                f"DataSource.kind must be one of "
                f"self/opponent/pool/scripted/recorded, got {self.kind!r}"
            )
        if not _is_numlike(self.sampling_share) or not math.isfinite(
            float(self.sampling_share)
        ):
            raise ValueError(
                f"DataSource.sampling_share must be a finite number, "
                f"got {self.sampling_share!r}"
            )
        if not (self.sampling_share > 0.0):
            raise ValueError(
                f"DataSource.sampling_share must be > 0, "
                f"got {self.sampling_share}"
            )


# ---------------------------------------------------------------------------
# Replay plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayPlan:
    """Declares how the replay buffer should sample and retain data.

    For the MVP, this is a thin config object. Phase 2 will add
    stratified retention and per-channel sampling.

    Attributes:
        stratify_by: Optional tag name for stratified retention.
            If set, the buffer maintains minimum counts per stratum.
        min_per_stratum: Minimum transitions per stratum (if
            stratify_by is set).
        freshness_weight: If > 0, newer transitions are sampled with
            higher probability (exponential decay with this rate).
            0.0 = uniform sampling.
    """

    stratify_by: Optional[str] = None
    min_per_stratum: int = 1000
    freshness_weight: float = 0.0

    def __post_init__(self) -> None:
        import math

        if not _is_intlike(self.min_per_stratum):
            raise ValueError(
                f"ReplayPlan.min_per_stratum must be an integer, "
                f"got {self.min_per_stratum!r}"
            )
        if self.min_per_stratum < 1:
            raise ValueError(
                f"ReplayPlan.min_per_stratum must be >= 1, "
                f"got {self.min_per_stratum}"
            )
        if not _is_numlike(self.freshness_weight) or not math.isfinite(
            float(self.freshness_weight)
        ):
            raise ValueError(
                f"ReplayPlan.freshness_weight must be a finite number, "
                f"got {self.freshness_weight!r}"
            )
        if self.freshness_weight < 0.0:
            raise ValueError(
                f"ReplayPlan.freshness_weight must be >= 0, "
                f"got {self.freshness_weight}"
            )


# ---------------------------------------------------------------------------
# ExperimentSAC ABC
# ---------------------------------------------------------------------------

class ExperimentSAC(ABC):
    """SAC-native experiment interface.

    Design principles (see PLAN.md):
    1. Experiment owns data distribution (data_sources, build_slices,
       replay_plan).
    2. Framework owns SAC mechanics (Q updates, alpha, target nets,
       replay buffer management).
    3. Reward channels are first-class, but the first-version trainer
       accepts only the audited 1-step twin-Q configuration.
    4. Source identity, task facts, and diagnostics are explicit; replay
       relabeling and stratified retention are unsupported in v1.
    """

    # ==================================================================
    # Phase 0: Configuration & Model Building
    # ==================================================================

    @abstractmethod
    def reward_channels(self) -> Tuple[SACRewardChannel, ...]:
        """Declare all reward channels with per-channel SAC config."""
        ...

    @abstractmethod
    def sac_params(self) -> SACParams:
        """Return SAC hyperparameters."""
        ...

    @abstractmethod
    def common_params(self) -> CommonParamsSAC:
        """Return common training parameters (env_step-based clock)."""
        ...

    @abstractmethod
    def build_actor(self, device: torch.device) -> SACActor:
        """Build and return the actor policy."""
        ...

    def build_critic(
        self,
        obs_dim: int,
        action_dim: int,
        channels: Tuple[SACRewardChannel, ...],
        device: torch.device,
    ) -> nn.Module:
        """Build the multi-channel Q critic used by the training loop.

        Contract-level hook: experiments may override to substitute a
        different critic architecture. The returned module must satisfy
        the ``MultiHeadQCritic`` runtime contract (per-channel twin-Q
        heads, ``channel_parameters``, ``load_state_dict`` with the
        strict six-key layout, ``set_lr``).

        Default: ``MultiHeadQCritic`` grouped by channel ``trunk_group``/
        gamma, sized by ``sac_params.q_hidden_dim``/``q_layer_norm`` and
        ``common_params.critic_learning_rate``.
        """
        from .networks import MultiHeadQCritic

        sp = self.sac_params()
        cp = self.common_params()
        return MultiHeadQCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            channels=channels,
            hidden_dim=sp.q_hidden_dim,
            layer_norm=sp.q_layer_norm,
            critic_lr=cp.critic_learning_rate,
            device=device,
        )

    def build_q_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """RESERVED legacy hook — not consulted by the training loop.

        The per-channel-module shape does not match the actual
        gamma-grouped shared-trunk critic architecture. Experiments
        that need a custom critic override :meth:`build_critic`
        instead. Do not implement this for new experiments.
        """
        return None

    # ==================================================================
    # Phase 1: Data Source Declaration
    # ==================================================================

    @abstractmethod
    def data_sources(self) -> Tuple[DataSource, ...]:
        """Declare all data sources for the replay buffer.

        For the MVP, return a single ``DataSource(kind="self")``.
        Phase 2 will support opponent, pool, and scripted sources.
        """
        ...

    # ==================================================================
    # Phase 2: Job Construction
    # ==================================================================

    @abstractmethod
    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        *,
        collection_round: int = 0,
        run_id: str = "",
        deterministic: bool = False,
    ) -> List[SACJob]:
        """Build SAC collection jobs for training or evaluation.

        ``deterministic=True`` denotes evaluation behavior and must not
        write replay data. Collection metadata is carried by SACJob.
        """
        ...

    def pre_action_fact_specs(self) -> Tuple[SACFactSpec, ...]:
        """Declare facts captured before ``runtime.step()``.

        Default: no task facts. Experiments that require them (e.g.
        ``phi_pre``) must override this method.
        """
        return ()

    # ==================================================================
    # Phase 3: Episode → SACTransitionSlice
    # ==================================================================

    @abstractmethod
    def build_slices(self, episodes: List[Any]) -> List[SACTransitionSlice]:
        """Convert collected episodes into validated ``sac_transition_v2``
        slices for replay admission.

        This is the single source of truth for reward semantics,
        per-channel gates, task facts, boundary flags, and provenance.
        """
        ...

    # ==================================================================
    # Phase 4: Replay Plan (optional override)
    # ==================================================================

    def replay_plan(self) -> ReplayPlan:
        """Declare replay sampling and retention strategy.

        Default: uniform sampling, no stratification.
        """
        return ReplayPlan()

    # ==================================================================
    # Per-round metrics (optional override)
    # ==================================================================

    def param_overrides(self) -> Tuple[Tuple[int, str, Any], ...]:
        """Round-boundary parameter patches for the training loop.

        Returns ``(from_round, field, value)`` triples applied at the
        start of the matching collection round (before policy export
        and job construction). ``field`` must be in the loop's
        patchable allowlist; CLI ``--param`` patches win on conflicts.
        Deterministic by round, so resume equivalence is preserved.
        """
        return ()

    def post_round_metrics(self, episodes: List[Any]) -> Dict[str, float]:
        """Return task-level metrics for the just-finished collection round.

        Called once per round after ``build_slices``. Keys are emitted into
        the round metrics event under the ``task.`` namespace (e.g.
        ``{"online_success": 0.5}`` → ``task.online_success``). Only finite
        numeric values are emitted.
        """
        return {}

    # ==================================================================
    # Evaluation
    # ==================================================================

    @abstractmethod
    def on_eval(
        self, episodes: List[Any], env_step: int,
    ) -> Dict[str, Any]:
        """Process evaluation results and update internal state.

        Same contract as PPO V2's on_eval. Returns dict with at least:
        - ``is_new_best``: bool
        - ``info``: dict (free-form logging)
        - ``stop_training``: bool (optional)
        - ``request_relabel``: unsupported in v1 and fails loudly in the loop

        Args:
            episodes: Raw eval episodes.
            env_step: Current environment step count.
        """
        ...

    def eval_episode_details(self, episodes: List[Any]) -> List[Dict[str, Any]]:
        """Optional: per-episode/per-agent rows backing ``on_eval``.

        Returned rows must sum to the same verdict ``on_eval`` computes —
        they are the auditable denominator of the aggregate (e.g. one row
        per episode×agent with that agent's success/survival flag). The
        held-out evaluator records these verbatim; the default is an
        empty list, which honest callers report as "details unavailable".

        Args:
            episodes: Raw eval episodes (same objects ``on_eval`` got).
        """
        return []

    # ==================================================================
    # State Persistence
    # ==================================================================

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
