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

import torch
import torch.nn as nn

from envs.framework.policy import PolicyBlueprint

from .actor import SACActor
from .collection import SACFactSpec, SACJob
from .transition import SACTransitionSlice


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
        actor_weight_share: If True (default), this channel
            participates in the action-gradient normalization that
            balances per-channel influence on the policy. If False,
            the channel's Q is trained but does not influence the
            actor (equivalent to actor_weight=0 everywhere).
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
        if not (0.0 < float(self.gamma) <= 1.0):
            raise ValueError(
                f"channel {self.name!r}: gamma must be in (0,1], "
                f"got {self.gamma}"
            )
        if int(self.n_step) < 1:
            raise ValueError(
                f"channel {self.name!r}: n_step must be >= 1, "
                f"got {self.n_step}"
            )
        if int(self.n_critics) < 1:
            raise ValueError(
                f"channel {self.name!r}: n_critics must be >= 1"
            )
        if not (1 <= int(self.in_target_min) <= int(self.n_critics)):
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
        use_grad_norm: If True, use action-gradient normalization for
            actor loss (the primary mechanism). If False, use naive
            weighted Q sum (fallback / baseline for comparison).
        grad_norm_est_interval: How often (in gradient steps) to
            re-estimate the per-channel gradient scale statistics.
        grad_norm_ema_decay: EMA decay for running gradient scale
            statistics.
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
        if self.batch_size <= 0:
            errs.append("batch_size must be > 0")
        if self.replay_buffer_size < self.batch_size:
            errs.append(
                f"replay_buffer_size={self.replay_buffer_size} < "
                f"batch_size={self.batch_size}"
            )
        if self.warmup_steps < 0:
            errs.append("warmup_steps must be >= 0")
        if self.utd_ratio <= 0.0:
            errs.append("utd_ratio must be > 0")
        if self.max_grad_steps_per_round <= 0:
            errs.append("max_grad_steps_per_round must be > 0")
        if not (0.0 <= self.tau <= 1.0):
            errs.append(f"tau must be in [0,1], got {self.tau}")
        if self.init_alpha <= 0.0:
            errs.append(f"init_alpha must be > 0, got {self.init_alpha}")
        if self.log_alpha_min >= self.log_alpha_max:
            errs.append(
                f"log_alpha_min={self.log_alpha_min} >= "
                f"log_alpha_max={self.log_alpha_max}"
            )
        if self.alpha_lr <= 0.0:
            errs.append("alpha_lr must be > 0")
        if not (
            self.log_alpha_min <= math.log(self.init_alpha)
            <= self.log_alpha_max
        ):
            errs.append(
                f"init_alpha={self.init_alpha} outside "
                f"[exp(log_alpha_min), exp(log_alpha_max)] clamps"
            )
        if self.target_entropy is not None and not math.isfinite(
            float(self.target_entropy)
        ):
            errs.append(
                f"target_entropy must be finite or None, "
                f"got {self.target_entropy}"
            )
        if self.grad_norm_est_interval < 1:
            errs.append("grad_norm_est_interval must be >= 1")
        if not (0.0 <= self.grad_norm_ema_decay < 1.0):
            errs.append(
                f"grad_norm_ema_decay must be in [0,1), "
                f"got {self.grad_norm_ema_decay}"
            )
        if self.q_hidden_dim < 1:
            errs.append("q_hidden_dim must be >= 1")
        if self.expectation_samples < 1:
            errs.append("expectation_samples must be >= 1")
        if not math.isfinite(float(self.reward_scale)):
            errs.append("reward_scale must be finite")
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
        errs = []
        if not self.name:
            errs.append("name must be non-empty")
        if self.learning_rate <= 0.0:
            errs.append(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.critic_learning_rate <= 0.0:
            errs.append(
                f"critic_learning_rate must be > 0, "
                f"got {self.critic_learning_rate}"
            )
        if self.grad_clip_norm <= 0.0:
            errs.append(
                f"grad_clip_norm must be > 0, got {self.grad_clip_norm}"
            )
        if self.episodes_per_update < 1:
            errs.append("episodes_per_update must be >= 1")
        if self.max_env_steps < 1:
            errs.append("max_env_steps must be >= 1")
        if self.eval_interval < 1:
            errs.append("eval_interval must be >= 1")
        if self.eval_episodes < 1:
            errs.append("eval_episodes must be >= 1")
        if self.video_eval_interval < 0:
            errs.append("video_eval_interval must be >= 0")
        if self.rollout_workers < 1:
            errs.append("rollout_workers must be >= 1")
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

    @abstractmethod
    def build_q_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """Build a Q(s,a) critic for one reward channel.

        The critic must accept (obs, action) tensors and return (B,)
        or (B, 1) Q-values. For multi-head architectures, the framework
        wraps individual channel critics into shared trunks.

        Args:
            channel_name: The SACRewardChannel.name for this critic.
            device: Torch device.
        """
        ...

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

    # ==================================================================
    # State Persistence
    # ==================================================================

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
