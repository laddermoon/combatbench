"""Experiment — pure ABC for curriculum learning experiments.

Defines the interface contract between the training framework and experiment
implementations.  No default values, no concrete helpers — those live in
``experiments/base.py`` (CombatExperimentBase).
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


@runtime_checkable
class TrainablePolicy(Protocol):
    """Interface that the framework's PPO trainer requires from an actor."""

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy) for given obs/actions."""
        ...

    def to_blueprint(self, dest_path: str) -> PolicyBlueprint:
        """Export a rollout-ready policy blueprint."""
        ...


@dataclass(frozen=True)
class FrameworkParams:
    """All parameters the framework needs from the experiment."""
    name: str
    reward_keys: Tuple[str, ...]
    gammas: Dict[str, float]
    log_std_min: float
    log_std_max: float
    gae_lambda: float
    learning_rate: float
    critic_learning_rate: float
    clip_eps: float
    entropy_coef: float
    grad_clip_norm: float
    target_kl: float
    update_epochs: int
    minibatch_size: int
    episodes_per_update: int
    max_updates: int
    eval_interval: int
    eval_episodes: int
    video_eval_interval: int
    rollout_workers: int
    eval_workers: int
    seed: int


class Experiment(ABC):
    """Abstract per-experiment configuration.

    The framework calls ``framework_params()`` once to obtain all parameters,
    then calls the abstract methods during training.  Optional hooks allow
    experiments to customise PPO internals; returning ``None`` from a hook
    means "use the framework default".
    """

    # --- Framework parameter access ---

    @abstractmethod
    def framework_params(self) -> FrameworkParams:
        """Return all parameters the framework needs."""
        ...

    # --- Model construction ---

    @abstractmethod
    def build_actor(self, device: torch.device) -> TrainablePolicy:
        """Build and return the actor policy on the given device."""
        ...

    @abstractmethod
    def build_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """Build and return a value function for the given reward key."""
        ...

    # --- Abstract methods ---

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

    # --- Optional hooks (return None to use framework default) ---

    def prepare_training_segments(
        self, episode: "Episode",
    ) -> List[Tuple[int, int, float]]:
        """Return ``(start, end, weight)`` triples for training.

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

    # --- Optional state persistence ---

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
