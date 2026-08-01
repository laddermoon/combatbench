
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverV2PureConfig(CombatExperimentBase):
    """Balance-recovery with ONLY survival reward (r_fall).

    Simplified version of balance_recover_v2: no posture, cross-support,
    wall-contact, or any other reward. Only per-step survival bonus + terminal
    fall penalty. No reward observers are registered in the env blueprint,
    minimizing per-step computation overhead.

    Goal: observe what balance strategy emerges from pure survival signal.
    """

    name = "balance_recover_v2_pure"
    reward_keys = ("r_fall",)
    gammas = {"r_fall": 0.99}

    max_steps = 100

    BLUEPRINT = "balance_recover_v2_pure_env.yaml"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def video_env_blueprint(self):
        perturb = self._current_perturb_params()
        return self._env_pb().materialize(
            max_steps=self.max_steps,
            agent_id="robot_a",
            tolerance=6,
            **perturb,
        )

    # --- PPO tuning (same as balance_recover_v2) ---
    log_std_min: float = -1.8

    max_updates: int = 20000
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    entropy_coef: float = 1.5e-3

    # --- Rollout schedule (same as balance_recover_v2) ---
    episodes_per_update: int = 2048
    eval_episodes: int = 128

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01

    # --- Progressive perturbation curriculum (same as balance_recover_v2) ---
    PERTURB_FULL: Dict[str, float] = {
        "joint_pos_delta_max": 0.5,
        "joint_vel_delta_max": 2.0,
        "root_tilt_deg_max": 20.0,
        "root_linear_velocity_delta_max": 2.0,
        "root_angular_velocity_delta_max": 1.0,
    }
    LEVEL_SCALES: Tuple[float, ...] = (
        0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.78, 0.85, 0.90, 0.95, 1.0,
    )
    PROMOTE_SURVIVAL: float = 0.92
    PROMOTE_PATIENCE: int = 1

    # --- Stateful scheduler ---
    _level: int = 0
    _consecutive_pass: int = 0
    _survival_rate: float = 0.0

    # --- Perturbation scale helpers ---
    @property
    def current_scale(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_SCALES) - 1))
        return float(self.LEVEL_SCALES[idx])

    def _current_perturb_params(self) -> Dict[str, float]:
        scale = self.current_scale
        return {k: float(v) * scale for k, v in self.PERTURB_FULL.items()}

    # --- Rollout job construction: inject scaled perturbation params ---
    def _build_perturbed_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        max_steps = self.max_steps
        env_pb = self._env_pb()
        perturb = self._current_perturb_params()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid, tolerance=6, **perturb)
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
        return self._build_perturbed_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_perturbed_jobs(policy_bp, base_seed, self.eval_episodes)

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        level = esum.get("level", 0.0)
        best_level = best_esum.get("level", 0.0)
        if level != best_level:
            return level > best_level
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate

        if self._level < len(self.LEVEL_SCALES) - 1:
            if survival_rate >= self.PROMOTE_SURVIVAL:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        return (1.0,)

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """r_fall: per-step survival bonus + terminal signal. That's it."""
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
            "level": float(self._level),
        }

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "perturb_scale": round(self.current_scale, 3),
            "consecutive_pass": self._consecutive_pass,
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._survival_rate = float(state.get("survival_rate", 0.0))


# Singleton instance for the registry
EXPERIMENT = BalanceRecoverV2PureConfig()
