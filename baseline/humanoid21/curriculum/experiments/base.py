"""CombatExperimentBase v2 — unified PPO/SAC base for combat curriculum experiments.

Provides default values for all framework parameters, shared helpers
(self-play job construction, video blueprint), and serialization.
Supports both PPO and SAC via default implementations of all
algorithm-specific methods.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

from baseline.framework.experiment import (
    CommonParams,
    Experiment,
    PPOParams,
    SACParams,
    TrainablePolicy,
)
from baseline.common.policies import CriticMLP


class CombatExperimentBase(Experiment):
    """Class-attribute style base for humanoid21 combat curriculum experiments.

    Subclass and override class attributes + abstract methods.
    Provides default implementations for both PPO and SAC specific methods.
    """

    # --- Identity ---
    name: str = ""
    reward_keys: Tuple[str, ...] = ()
    gammas: Dict[str, float] = {}

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
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 8192

    # --- SAC knobs ---
    sac_tau: float = 0.005
    sac_init_alpha: float = 0.1
    sac_auto_alpha: bool = False
    sac_replay_buffer_size: int = 1_000_000
    sac_batch_size: int = 256
    sac_warmup_steps: int = 10_000
    sac_updates_per_step: int = 1
    sac_reward_scale: float = 0.1

    # --- Rollout schedule ---
    episodes_per_update: int = 256 * 8
    max_updates: int = 10000
    eval_interval: int = 5
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)

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
    # Algorithm-agnostic parameter access
    # ------------------------------------------------------------------

    def common_params(self) -> CommonParams:
        return CommonParams(
            name=self.name,
            reward_keys=self.reward_keys,
            gammas=self.gammas,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            learning_rate=self.learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            episodes_per_update=self.episodes_per_update,
            max_updates=self.max_updates,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=self.video_eval_interval,
            rollout_workers=self.rollout_workers,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # PPO-specific defaults
    # ------------------------------------------------------------------

    def ppo_params(self) -> PPOParams:
        return PPOParams(
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
        )

    def build_v_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """V(s) critic for PPO. Delegates to build_critic for backward compat."""
        return self.build_critic(reward_key, device)

    def build_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """V(s) critic (legacy name, still supported)."""
        return CriticMLP(
            obs_dim=self.obs_dim, hidden_dim=self.critic_hidden_dim,
        ).to(device)

    # ------------------------------------------------------------------
    # SAC-specific defaults
    # ------------------------------------------------------------------

    def sac_params(self) -> SACParams:
        return SACParams(
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            tau=self.sac_tau,
            init_alpha=self.sac_init_alpha,
            auto_alpha=self.sac_auto_alpha,
            target_entropy=-0.5 * float(self.action_dim),
            replay_buffer_size=self.sac_replay_buffer_size,
            batch_size=self.sac_batch_size,
            warmup_steps=self.sac_warmup_steps,
            updates_per_step=self.sac_updates_per_step,
            reward_scale=self.sac_reward_scale,
        )

    def build_q_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        """Q(s,a) critic for SAC."""
        from baseline.framework.sac_trainer import QCriticMLP
        return QCriticMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.critic_hidden_dim,
        ).to(device)

    # ------------------------------------------------------------------
    # Model construction (shared)
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = Path(__file__).resolve().parent.parent.parent / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")
        actor = bp.build().to(device)
        actor.log_std_min = float(self.log_std_min)
        return actor

    # --- Shared helpers ---

    @staticmethod
    def _agent_from_rollout_seed(seed: int) -> str:
        rng = np.random.default_rng(int(seed) + 937)
        return "robot_a" if int(rng.integers(0, 2)) == 0 else "robot_b"

    def _make_video_blueprint(self, env_pb: ParameterizedEnvBlueprint) -> EnvBlueprint:
        return env_pb.materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id="robot_a",
        )

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

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reward_keys": list(self.reward_keys),
            "gammas": self.gammas,
            "initial_weights": list(self.initial_weights()),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "custom_config": dict(self.custom_config),
            "learning_rate": self.learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "grad_clip_norm": self.grad_clip_norm,
            # PPO
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "target_kl": self.target_kl,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            # SAC
            "sac_tau": self.sac_tau,
            "sac_init_alpha": self.sac_init_alpha,
            "sac_auto_alpha": self.sac_auto_alpha,
            "sac_replay_buffer_size": self.sac_replay_buffer_size,
            "sac_batch_size": self.sac_batch_size,
            "sac_warmup_steps": self.sac_warmup_steps,
            "sac_updates_per_step": self.sac_updates_per_step,
            # Shared
            "episodes_per_update": self.episodes_per_update,
            "max_updates": self.max_updates,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "video_eval_interval": self.video_eval_interval,
            "video_env_blueprint": self.video_env_blueprint(),
            "rollout_workers": self.rollout_workers,
            "seed": self.seed,
        }

    def save_run_config(self, run_dir: Path, *, smoke: bool = False, algo: str = "ppo") -> None:
        payload = {
            "experiment": self.to_dict(),
            "algorithm": algo,
            "smoke": smoke,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    # --- State persistence ---

    def training_state(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
        }
