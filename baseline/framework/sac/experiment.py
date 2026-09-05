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

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.ppo import TrainablePolicy
from baseline.framework.rollout.job import Job


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
    use_grad_norm: bool = True
    grad_norm_est_interval: int = 10
    grad_norm_ema_decay: float = 0.99
    q_hidden_dim: int = 256
    q_layer_norm: bool = False
    reward_scale: float = 1.0


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
# Trajectory slice — the SAC analog of PPO's Trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectorySlice:
    """A contiguous slice of an episode, prepared for replay insertion.

    This is the SAC analog of PPO V2's ``Trajectory``. The key
    differences are:
    - Per-channel ``done`` is per-step (not per-slice), because SAC
      needs per-transition termination for Bellman targets.
    - ``tags`` and ``reward_features`` are carried for replay
      stratification and relabeling.
    - ``core_states`` is optional, for buffer-based env reset (Phase 2).

    Attributes:
        obs: (T, obs_dim) float32.
        actions: (T, act_dim) float32.
        last_obs: (obs_dim,) float32 — observation after last action.
        rewards: Dict[channel_name, (T,) float32].
        dones: Dict[channel_name, (T,) bool] — per-step per-channel
            termination. True at step t means the episode truly ended
            for that channel at step t (no bootstrap). False means
            truncated (bootstrap from next obs).
        actor_weights: Dict[channel_name, (T,) float32].
        tags: Dict[tag_name, (T,) float32] — per-step tags.
        reward_features: Dict[feature_name, (T,) float32] — raw
            features for relabeling. Optional.
        core_states: Optional (T, state_dim) — for buffer-based reset.
        importance: Sample weight (default 1.0).
    """

    obs: np.ndarray
    actions: np.ndarray
    last_obs: np.ndarray
    rewards: Dict[str, np.ndarray]
    dones: Dict[str, np.ndarray]
    actor_weights: Dict[str, np.ndarray]
    tags: Dict[str, np.ndarray] = field(default_factory=dict)
    reward_features: Dict[str, np.ndarray] = field(default_factory=dict)
    core_states: Optional[np.ndarray] = None
    importance: float = 1.0


# ---------------------------------------------------------------------------
# Job type alias (same structure as PPO V2)
# ---------------------------------------------------------------------------

Job = Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]


# ---------------------------------------------------------------------------
# ExperimentSAC ABC
# ---------------------------------------------------------------------------

class ExperimentSAC(ABC):
    """SAC-native experiment interface.

    Design principles (see PLAN.md):
    1. Experiment owns data distribution (data_sources, build_slices,
       replay_plan, relabel).
    2. Framework owns SAC mechanics (Q updates, alpha, target nets,
       replay buffer management).
    3. Reward channels are first-class with per-channel gamma, n_step,
       and pessimism configuration.
    4. Tags drive stratification, sampling, and diagnostics.
    5. Relabel enables curriculum + off-policy coexistence.
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
    def build_actor(self, device: torch.device) -> TrainablePolicy:
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
    ) -> List[Job]:
        """Build rollout jobs for training or evaluation.

        Same structure as PPO V2's build_jobs. The caller controls
        stochastic vs deterministic by passing the appropriate
        PolicyBlueprint.
        """
        ...

    # ==================================================================
    # Phase 3: Episode → TrajectorySlice
    # ==================================================================

    @abstractmethod
    def build_slices(self, episodes: List[Any]) -> List[TrajectorySlice]:
        """Convert all episodes into trajectory slices for replay.

        Receives the full batch of episodes at once. Each slice is a
        contiguous segment of an episode with per-channel rewards,
        dones, actor_weights, tags, and optional reward_features.

        This is the single source of truth for:
        - How each episode is sliced.
        - Per-channel rewards and termination.
        - Per-channel actor_weights (curriculum control).
        - Tags for stratification and diagnostics.
        - Reward features for relabeling.
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
    # Phase 5: Relabel (optional override)
    # ==================================================================

    def relabel(
        self,
        reward_features: Dict[str, np.ndarray],
        tags: Dict[str, np.ndarray],
        ctx: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Recompute rewards and actor_weights from stored features.

        Called when the experiment signals that curriculum has advanced
        (via ``on_eval`` returning a relabel request). The framework
        scans the entire buffer and calls this per-transition-batch.

        Default: None (no relabeling).
        """
        return None

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
        - ``request_relabel``: bool (optional — triggers buffer relabel)

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
