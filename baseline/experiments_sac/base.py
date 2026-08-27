"""CombatExperimentSACBase — shared base for humanoid21 SAC experiments.

Provides default values for SAC-specific parameters, shared helpers
(self-play job construction, actor/Q-critic building), and state
persistence defaults. SAC-only — no PPO support.

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

from baseline.framework.ppo import TrainablePolicy
from baseline.framework.sac.experiment import (
    CommonParamsSAC,
    DataSource,
    ExperimentSAC,
    ReplayPlan,
    SACParams,
    SACRewardChannel,
    TrajectorySlice,
)


class CombatExperimentSACBase(ExperimentSAC):
    """Class-attribute style base for humanoid21 combat SAC experiments.

    Subclass and override:
    - Class attributes (name, obs_dim, action_dim, env_blueprint, etc.)
    - ``reward_channels()`` — declare SAC reward channels
    - ``build_slices()`` — episode → trajectory slices
    - ``on_eval()`` — eval processing + best-of-run
    """

    # --- Identity ---
    name: str = ""

    # --- Network shape ---
    obs_dim: int = 96
    action_dim: int = 21
    actor_hidden_dim: int = 256
    q_hidden_dim: int = 256

    # --- Exploration (SAC: wide range, alpha controls exploration) ---
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # --- Shared training ---
    learning_rate: float = 3e-4
    # Lower critic LR for stability (3e-4 caused Q overestimation and
    # divergence with large reward scales).
    critic_learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0

    # --- SAC knobs ---
    replay_buffer_size: int = 500_000
    batch_size: int = 256
    warmup_steps: int = 10_000
    utd_ratio: float = 1.0
    max_grad_steps_per_round: int = 10_000
    tau: float = 0.005
    init_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float = -21.0  # -action_dim for 21-DOF
    alpha_lr: float = 3e-4
    log_alpha_min: float = -10.0
    log_alpha_max: float = 2.0
    use_grad_norm: bool = True
    q_layer_norm: bool = False
    reward_scale: float = 1.0

    # --- Rollout schedule ---
    episodes_per_update: int = 64
    max_env_steps: int = 2_000_000
    eval_interval: int = 20_000
    eval_episodes: int = 16

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Parallelism ---
    rollout_workers: int = max(1, (os.cpu_count() or 1) // 2)
    eval_workers: int = max(1, (os.cpu_count() or 1) // 4)

    seed: int = 42

    # --- Policy blueprint ---
    actor_blueprint: str = "init_policy.yaml"

    # --- Rollout / env configuration ---
    env_blueprint: str = ""
    agent_used: str = "random"
    max_steps: int = 200
    init_distance_min: float = 1.5
    init_distance_max: float = 3.5

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def common_params(self) -> CommonParamsSAC:
        return CommonParamsSAC(
            name=self.name,
            learning_rate=self.learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            grad_clip_norm=self.grad_clip_norm,
            episodes_per_update=self.episodes_per_update,
            max_env_steps=self.max_env_steps,
            eval_interval=self.eval_interval,
            eval_episodes=self.eval_episodes,
            video_eval_interval=self.video_eval_interval,
            rollout_workers=self.rollout_workers,
            eval_workers=self.eval_workers,
            seed=self.seed,
        )

    def sac_params(self) -> SACParams:
        return SACParams(
            replay_buffer_size=self.replay_buffer_size,
            batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            utd_ratio=self.utd_ratio,
            max_grad_steps_per_round=self.max_grad_steps_per_round,
            tau=self.tau,
            init_alpha=self.init_alpha,
            auto_alpha=self.auto_alpha,
            target_entropy=self.target_entropy,
            alpha_lr=self.alpha_lr,
            log_alpha_min=self.log_alpha_min,
            log_alpha_max=self.log_alpha_max,
            use_grad_norm=self.use_grad_norm,
            q_hidden_dim=self.q_hidden_dim,
            q_layer_norm=self.q_layer_norm,
            reward_scale=self.reward_scale,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> TrainablePolicy:
        blueprint_dir = (
            Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        )
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        actor = bp.build().to(device)
        if hasattr(actor, "log_std_min"):
            actor.log_std_min = float(self.log_std_min)
        if hasattr(actor, "log_std_max"):
            actor.log_std_max = float(self.log_std_max)
        return actor

    def build_q_critic(self, channel_name: str, device: torch.device) -> nn.Module:
        """Build a single-channel Q critic.

        The MultiHeadQCritic wrapper will group these into shared trunks.
        For the MVP, we return a simple QTrunkHeads with one channel.
        The wrapper handles the actual grouping.

        Actually, the MultiHeadQCritic builds its own networks based on
        the channel configs. This method is not used by the current
        MultiHeadQCritic — it builds networks internally. We keep it
        for interface compatibility and future per-channel customization.
        """
        from baseline.framework.sac.networks import QTrunkHeads
        return QTrunkHeads(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.q_hidden_dim,
            channel_names=(channel_name,),
            layer_norm=self.q_layer_norm,
        ).to(device)

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def data_sources(self) -> Tuple[DataSource, ...]:
        return (DataSource(kind="self", agent="random"),)

    # ------------------------------------------------------------------
    # Job construction
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        return self._build_selfplay_jobs(
            self._env_pb(), policy_bp, base_seed, n_episodes,
        )

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        if not self.env_blueprint:
            raise ValueError(
                f"{self.__class__.__name__} must set env_blueprint "
                "to a blueprint filename"
            )
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "blueprints" / self.env_blueprint
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
        rng = np.random.default_rng(base_seed)

        if self.agent_used == "both":
            env_bp = env_pb.materialize(max_steps=self.max_steps)
            jobs = []
            for i in range(n_episodes):
                seed = int(base_seed + i)
                initial_distance = float(
                    rng.uniform(self.init_distance_min, self.init_distance_max)
                )
                jobs.append((
                    policy_bp, policy_bp,
                    env_bp, seed,
                    {"initial_distance": initial_distance},
                ))
            return jobs

        agent_ids: Tuple[str, ...]
        if self.agent_used == "random":
            agent_ids = ("robot_a", "robot_b")
        else:
            agent_ids = (self.agent_used,)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=self.max_steps, agent_id=aid)
            for aid in agent_ids
        }

        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            if self.agent_used == "random":
                agent_id = self._agent_from_rollout_seed(seed)
            else:
                agent_id = self.agent_used
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass
