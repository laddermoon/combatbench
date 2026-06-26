"""CombatExperimentBase — class-attribute style base for combat curriculum experiments.

Provides default values for all framework parameters, shared helpers
(self-play job construction, video blueprint), and serialization.
Experiment-specific parameters (custom_config, weight_target_total, etc.)
are defined here as plain class attributes — the framework does not read them
directly.
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

from baseline.humanoid21.curriculum.framework.experiment import (
    Experiment,
    FrameworkParams,
    TrainablePolicy,
)
from baseline.common.policies import CriticMLP


class CombatExperimentBase(Experiment):
    """Class-attribute style base for humanoid21 combat curriculum experiments.

    Subclass and override class attributes + abstract methods.
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

    # --- GAE ---
    gae_lambda: float = 0.95

    # --- PPO knobs ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    grad_clip_norm: float = 1.0
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

    # --- Framework parameter access ---

    def framework_params(self) -> FrameworkParams:
        return FrameworkParams(
            name=self.name,
            reward_keys=self.reward_keys,
            gammas=self.gammas,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            gae_lambda=self.gae_lambda,
            learning_rate=self.learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            grad_clip_norm=self.grad_clip_norm,
            target_kl=self.target_kl,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            episodes_per_update=self.episodes_per_update,
            max_updates=self.max_updates,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=self.video_eval_interval,
            rollout_workers=self.rollout_workers,
            eval_workers=self.eval_workers,
            seed=self.seed,
        )

    # --- Model construction ---

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = Path(__file__).resolve().parent.parent.parent / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / "init_policy.yaml")
        actor = bp.build().to(device)
        actor.log_std_min = float(self.log_std_min)
        return actor

    def build_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        return CriticMLP(
            obs_dim=self.obs_dim, hidden_dim=self.critic_hidden_dim,
        ).to(device)

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
            "gae_lambda": self.gae_lambda,
            "custom_config": dict(self.custom_config),
            "learning_rate": self.learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "clip_eps": self.clip_eps,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "grad_clip_norm": self.grad_clip_norm,
            "target_kl": self.target_kl,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "episodes_per_update": self.episodes_per_update,
            "max_updates": self.max_updates,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "video_eval_interval": self.video_eval_interval,
            "video_env_blueprint": self.video_env_blueprint(),
            "rollout_workers": self.rollout_workers,
            "eval_workers": self.eval_workers,
            "seed": self.seed,
        }

    def save_run_config(self, run_dir: Path, *, smoke: bool = False) -> None:
        payload = {
            "experiment": self.to_dict(),
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
