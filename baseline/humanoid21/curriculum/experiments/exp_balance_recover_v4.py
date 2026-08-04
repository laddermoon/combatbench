"""Balance recovery v4: online impulse perturbation training.

Uses ImpulsePerturbationPlugin to generate physically realistic perturbed
states on-the-fly during training. Each episode gets a fresh random impulse
(force/duration/direction sampled from configured ranges), providing infinite
state diversity for better generalization.

The state bank (from generate_state_bank.py) is used externally by
recovery_iter_loop.py to calibrate the force range based on the recovery
boundary — it is NOT used for state injection during training.

Configuration via environment variables (no framework changes needed)::

    IMPULSE_FORCE_MIN=10 IMPULSE_FORCE_MAX=200 \\
    IMPULSE_DURATION_MIN=1 IMPULSE_DURATION_MAX=8 \\
    POLICY_BLUEPRINT_PATH=baseline/runs/.../policy/policy_blueprint.yaml \\
    BASE_POLICY_PATH=baseline/runs/.../policy \\
    python3 baseline/framework/train.py --experiment balance_recover_v4 --algo ppo \\
        --run-name recover_v4_gen0

    # Smoke test
    IMPULSE_FORCE_MIN=50 IMPULSE_FORCE_MAX=150 \\
    IMPULSE_DURATION_MIN=2 IMPULSE_DURATION_MAX=4 \\
    POLICY_BLUEPRINT_PATH=baseline/runs/.../policy/policy_blueprint.yaml \\
    python3 baseline/framework/train.py --experiment balance_recover_v4 --algo ppo --smoke
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import os

import numpy as np
import torch

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverV4Config(CombatExperimentBase):
    """Balance-recovery v4: online impulse perturbation with pure survival reward.

    Each episode: ImpulsePerturbationPlugin applies a random impulse
    (force/duration/direction from configured ranges) via internal sim,
    producing a physically realistic perturbed initial state.
    Reward = per-step survival bonus + terminal fall/survive signal.
    No reward observers — maximizes rollout speed.
    """

    name = "balance_recover_v4"
    reward_keys = ("r_fall",)
    gammas = {"r_fall": 0.99}

    max_steps = 600

    BLUEPRINT = "balance_recover_v4_env.yaml"

    # --- PPO tuning ---
    log_std_min: float = -1.8

    max_updates: int = 20000
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    entropy_coef: float = 1.5e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 2048
    eval_episodes: int = 128

    # --- Reward ---
    per_step_survival_reward: float = 0.01

    # --- Default custom config ---
    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "rollout_distance_min": 1.5,
        "rollout_distance_max": 3.5,
        "max_steps": 600,
        "terminal_fall_penalty": 1.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        rw = os.environ.get("ROLLOUT_WORKERS")
        if rw:
            self.rollout_workers = int(rw)
        tu = os.environ.get("TRAIN_UPDATES")
        if tu:
            self.max_updates = int(tu)

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _impulse_params(self) -> Dict[str, Any]:
        """Read impulse range from environment variables."""
        force_min = float(os.environ.get("IMPULSE_FORCE_MIN", "10"))
        force_max = float(os.environ.get("IMPULSE_FORCE_MAX", "200"))
        dur_min = int(os.environ.get("IMPULSE_DURATION_MIN", "1"))
        dur_max = int(os.environ.get("IMPULSE_DURATION_MAX", "8"))
        policy_bp_path = os.environ.get("POLICY_BLUEPRINT_PATH")

        if not policy_bp_path:
            raise ValueError(
                "balance_recover_v4 requires POLICY_BLUEPRINT_PATH environment variable. "
                "Example: POLICY_BLUEPRINT_PATH=baseline/runs/.../policy/policy_blueprint.yaml"
            )

        return {
            "force_magnitude": [force_min, force_max],
            "duration_action_steps": [dur_min, dur_max],
            "direction_mode": "random_horizontal",
            "policy_blueprint_path": str(Path(policy_bp_path).resolve()),
        }

    def video_env_blueprint(self) -> EnvBlueprint:
        return self._env_pb().materialize(
            max_steps=self.max_steps,
            agent_id="robot_a",
            tolerance=6,
            **self._impulse_params(),
        )

    # --- Warm-start from base policy ---
    def build_actor(self, device: torch.device) -> Any:
        from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy

        base_path = os.environ.get("BASE_POLICY_PATH")
        if not base_path:
            return super().build_actor(device)

        model_pt = Path(base_path) / "model.pt"
        if not model_pt.exists():
            raise FileNotFoundError(f"model.pt not found at {model_pt}")

        payload = torch.load(model_pt, map_location="cpu")
        hidden_dim = int(payload.get("hidden_dim", payload.get("actor_hidden_dim", 256)))

        actor = TanhGaussianMLPPolicy(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=hidden_dim,
            log_std_min=float(self.log_std_min),
            log_std_max=float(self.log_std_max),
            device=device,
        )
        actor.load_state_dict(payload["state_dict"], strict=False)
        actor = actor.to(device)
        actor.log_std_min = float(self.log_std_min)
        return actor

    # --- Rollout job construction ---
    def _build_impulse_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        max_steps = self.max_steps
        env_pb = self._env_pb()
        perturb = self._impulse_params()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(
                max_steps=max_steps,
                agent_id=aid,
                tolerance=6,
                **perturb,
            )
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

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_impulse_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_impulse_jobs(policy_bp, base_seed, self.eval_episodes)

    # --- Evaluation ---
    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        return (1.0,)

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    # --- Reward extraction ---
    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        fell = "imbalance" in episode.termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        return {"r_fall": r_fall}

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        fell = "imbalance" in episode.termination_proposals
        return {
            "survived": 0.0 if fell else 1.0,
            "ep_length": float(episode.num_frames),
            "level": 0.0,
        }

    def eval_summary(self, episodes) -> Dict[str, float]:
        if not episodes:
            return {"survived": 0.0, "ep_length": 0.0}
        metrics = [self.compute_episode_metrics(ep) for ep in episodes]
        survived = np.mean([m["survived"] for m in metrics])
        ep_length = np.mean([m["ep_length"] for m in metrics])
        return {"survived": float(survived), "ep_length": float(ep_length)}

    # --- Scheduler ---
    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "impulse_force_range": [
                os.environ.get("IMPULSE_FORCE_MIN", ""),
                os.environ.get("IMPULSE_FORCE_MAX", ""),
            ],
            "impulse_duration_range": [
                os.environ.get("IMPULSE_DURATION_MIN", ""),
                os.environ.get("IMPULSE_DURATION_MAX", ""),
            ],
            "policy_blueprint_path": os.environ.get("POLICY_BLUEPRINT_PATH", ""),
            "base_policy_path": os.environ.get("BASE_POLICY_PATH", ""),
        }

    def scheduler_state(self) -> dict:
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        pass


# Singleton instance for the registry
EXPERIMENT = BalanceRecoverV4Config()
