"""CombatExperimentV2Base — shared base for humanoid21 V2 PPO experiments.

Provides default values for all framework parameters, shared helpers
(self-play job construction, actor/critic building), and state persistence.
PPO-only — no SAC support.

Subclass and override class attributes + abstract methods.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.experiment_v2 import (
    CommonParams,
    ExperimentV2,
    PPOParams,
    TrainablePolicy,
)
from baseline.common.policies import CriticMLP


class CombatExperimentV2Base(ExperimentV2):
    """Class-attribute style base for humanoid21 combat V2 experiments.

    Subclass and override:
    - Class attributes (name, obs_dim, action_dim, etc.)
    - ``reward_channels()`` — declare reward channels
    - ``build_trajectories()`` — episode → trajectories
    - ``on_eval()`` — eval processing + best-of-run
    - ``_env_pb()`` — return the ParameterizedEnvBlueprint for this experiment
    """

    # --- Identity ---
    name: str = ""

    # --- Network shape ---
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    log_std_min: float = -4.0
    log_std_max: float = 0.0

    # --- Shared training ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0

    # --- PPO knobs ---
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 8192

    # --- Rollout schedule ---
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)
    eval_workers: int = max(1, (os.cpu_count() or 1) // 4)

    seed: int = 42

    # --- Free-form experiment-specific parameters ---
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "rollout_distance_min": 1.5,
        "rollout_distance_max": 3.5,
        "max_steps": 200,
        "terminal_fall_penalty": 1.0,
    }

    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    # ------------------------------------------------------------------
    # Parameter access (ExperimentV2 interface)
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
            eval_workers=self.eval_workers,
            seed=self.seed,
        )

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")
        actor = bp.build().to(device)
        actor.log_std_min = float(self.log_std_min)
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
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build self-play rollout jobs.

        Subclass can override for non-self-play scenarios.
        """
        return self._build_selfplay_jobs(
            self._env_pb(), policy_bp, base_seed, n_episodes,
        )

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        """Return the ParameterizedEnvBlueprint for this experiment.

        Subclass must override to specify the env blueprint file.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _env_pb()"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        max_steps = self.custom_config["max_steps"]
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
            for aid in ("robot_a", "robot_b")
        }

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(
                    self.custom_config["rollout_distance_min"],
                    self.custom_config["rollout_distance_max"],
                )
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    # ------------------------------------------------------------------
    # State persistence (ExperimentV2 interface)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
