"""Experiment — unified ABC for curriculum learning experiments (v2).

Supports both PPO and SAC from a single experiment definition.
Algorithm-agnostic methods are abstract; algorithm-specific methods have
defaults that raise NotImplementedError (override in CombatExperimentBase
or your own base class).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


# ---------------------------------------------------------------------------
# Actor protocol — shared by PPO and SAC
# ---------------------------------------------------------------------------

@runtime_checkable
class TrainablePolicy(Protocol):
    """Interface that both PPO and SAC trainers require from an actor.

    For PPO:  evaluate_actions is used for importance-ratio computation.
    For SAC:  sample_action (reparameterized) is used for actor gradient;
              evaluate_actions is used for log-prob recomputation.
    """

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy) for given obs/actions."""
        ...

    def sample_action(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample: return (action, log_prob)."""
        ...

    def to_blueprint(self, dest_path: str) -> PolicyBlueprint:
        """Export a rollout-ready policy blueprint."""
        ...


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonParams:
    """Parameters shared by all algorithms."""
    name: str
    reward_keys: Tuple[str, ...]
    gammas: Dict[str, float]
    obs_dim: int
    action_dim: int
    learning_rate: float
    critic_learning_rate: float
    grad_clip_norm: float
    episodes_per_update: int
    max_updates: int
    eval_interval: int
    eval_episodes: int
    video_eval_interval: int
    rollout_workers: int
    eval_workers: int
    seed: int


@dataclass(frozen=True)
class PPOParams:
    """PPO-specific hyperparameters."""
    log_std_min: float
    log_std_max: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    target_kl: float
    update_epochs: int
    minibatch_size: int


@dataclass(frozen=True)
class SACParams:
    """SAC-specific hyperparameters."""
    log_std_min: float
    log_std_max: float
    tau: float                   # soft target update coefficient
    init_alpha: float            # initial entropy temperature
    auto_alpha: bool             # auto-tune alpha via dual gradient descent
    target_entropy: float        # target entropy for auto-tuning (-action_dim)
    replay_buffer_size: int      # max transitions in replay buffer
    batch_size: int              # minibatch size for SAC updates
    warmup_steps: int            # collect this many transitions before updating
    updates_per_step: int        # gradient steps per collected transition batch
    reward_scale: float = 1.0   # scale rewards to stabilize Q-function


# ---------------------------------------------------------------------------
# Experiment ABC
# ---------------------------------------------------------------------------

class Experiment(ABC):
    """Unified per-experiment configuration for PPO and SAC.

    Algorithm-agnostic methods are abstract.  PPO- and SAC-specific methods
    have default implementations that raise ``NotImplementedError``; the
    concrete base class ``CombatExperimentBase`` provides sensible defaults
    for all of them.
    """

    # ------------------------------------------------------------------
    # Algorithm-agnostic — MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def common_params(self) -> CommonParams:
        """Return algorithm-agnostic parameters."""
        ...

    @abstractmethod
    def build_actor(self, device: torch.device) -> TrainablePolicy:
        """Build and return the actor policy on the given device."""
        ...

    @abstractmethod
    def initial_weights(self) -> Tuple[float, ...]:
        """Return the initial stage-weight tuple (one entry per reward_key)."""
        ...

    @abstractmethod
    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Return the next stage-weight tuple given eval metrics."""
        ...

    @abstractmethod
    def extract_rewards(self, episode: "Episode") -> Dict[str, np.ndarray]:
        """Extract per-step reward arrays from an episode."""
        ...

    @abstractmethod
    def compute_episode_metrics(self, episode: "Episode") -> Dict[str, float]:
        """Compute aggregate metrics for one episode (used for eval & logging)."""
        ...

    @abstractmethod
    def scheduler_info(self) -> Dict[str, Any]:
        """Return extra info dict for logging (phase, consecutive_pass, etc.)."""
        ...

    @abstractmethod
    def compare_eval(self, esum: Dict[str, float], best_esum: Dict[str, float]) -> bool:
        """Return True if esum is better than best_esum."""
        ...

    @abstractmethod
    def build_rollout_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build rollout jobs for one training update."""
        ...

    @abstractmethod
    def build_eval_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build rollout jobs for evaluation."""
        ...

    @abstractmethod
    def video_env_blueprint(self) -> EnvBlueprint:
        """Return the env blueprint to use for video rendering."""
        ...

    # ------------------------------------------------------------------
    # PPO-specific — defaults raise NotImplementedError
    # Override in CombatExperimentBase or your own base class.
    # ------------------------------------------------------------------

    def ppo_params(self) -> PPOParams:
        """Return PPO-specific hyperparameters."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement ppo_params(). "
            "Inherit from CombatExperimentBase or implement manually."
        )

    def build_v_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """Build a V(s) value function for PPO."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement build_v_critic(). "
            "Inherit from CombatExperimentBase or implement manually."
        )

    def prepare_training_segments(
        self, episode: "Episode",
    ) -> List[Tuple[int, int, float]]:
        """Return ``(start, end, weight)`` triples for PPO training.

        Default: full episode as one segment with weight 1.0.
        Return an empty list to skip the episode entirely.
        """
        T = episode.num_frames
        return [(0, T, 1.0)]

    def combine_advantages(
        self,
        advs: Dict[str, np.ndarray],
        stage_weights: Tuple[float, ...],
    ) -> Optional[np.ndarray]:
        """Multi-critic advantage combination.  Return None for framework default."""
        return None

    def normalize_advantages(self, adv: np.ndarray) -> Optional[np.ndarray]:
        """Per-component advantage normalization.  Return None for framework default."""
        return None

    def normalize_sample_weights(self, weights: np.ndarray) -> Optional[np.ndarray]:
        """Sample weight normalization.  Return None for framework default."""
        return None

    # ------------------------------------------------------------------
    # SAC-specific — defaults raise NotImplementedError
    # Override in CombatExperimentBase or your own base class.
    # ------------------------------------------------------------------

    def sac_params(self) -> SACParams:
        """Return SAC-specific hyperparameters."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sac_params(). "
            "Inherit from CombatExperimentBase or implement manually."
        )

    def build_q_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """Build a Q(s,a) critic for SAC."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement build_q_critic(). "
            "Inherit from CombatExperimentBase or implement manually."
        )

    # ------------------------------------------------------------------
    # Optional state persistence — shared by all algorithms
    # ------------------------------------------------------------------

    def scheduler_state(self) -> dict:
        """Serialize mutable scheduler state for checkpointing."""
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        pass

    def training_state(self) -> dict:
        """Serialize training hyperparameters for checkpointing."""
        return {}

    def load_training_state(self, state: dict) -> None:
        """Restore training hyperparameters from a checkpoint."""
        pass
