
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import _extract_per_step_scalar
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverConfig(ExperimentConfig):
    """P0 balance-recovery policy (IDEA.md step 2).

    Trained on top of the basic-standing policy. At every episode reset the
    robot's state is randomly perturbed (joint positions/velocities, root
    tilt, root linear/angular velocity); the robot must learn to recover
    balance from any such starting condition — this is the fallback policy.

    The curriculum is **progressive**: perturbations start small and grow
    stronger level by level. A single scalar ``scale`` in ``[0, 1]`` scales
    every full-strength magnitude in :pyattr:`PERTURB_FULL`. When the eval
    survival rate stays at/above :pyattr:`PROMOTE_SURVIVAL` for
    :pyattr:`PROMOTE_PATIENCE` consecutive evaluations, the next (harder)
    level is unlocked. The env blueprint file never changes across levels;
    only the perturbation parameters passed to ``materialize`` do.
    """

    name = "balance_recover"
    reward_keys = ("r_fall", "r_cross")
    gammas = {"r_fall": 0.99, "r_cross": 0.99}

    BLUEPRINT = "balance_recover_env.yaml"

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def video_env_blueprint(self):
        perturb = self._current_perturb_params()
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id="robot_a",
            **perturb,
        )

    # --- PPO tuning (see training analysis) ---
    # Raise the log_std floor so the policy can't collapse to saturated,
    # near-deterministic actions — the main driver of the KL explosions /
    # exploding policy_loss observed in the first run.
    log_std_min: float = -2.0

    # Per-experiment PPO overrides: smaller actor LR + tighter KL/grad
    # clipping + fewer epochs to keep each PPO update from diverging.
    learning_rate: float = 1e-4      # was 3e-4: slow the actor down
    target_kl: float = 0.02          # was 0.05: early-stop sooner
    grad_clip_norm: float = 0.5      # was 1.0: tighter gradient clipping
    update_epochs: int = 3           # was 4: less policy drift per batch

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01

    # --- Progressive perturbation curriculum ---
    # Full-strength perturbation magnitudes, reached at scale == 1.0. These
    # mirror the blueprint defaults; the per-level scale multiplies them.
    PERTURB_FULL: Dict[str, float] = {
        "joint_pos_delta_max": 0.15,
        "joint_vel_delta_max": 0.15,
        "root_tilt_deg_max": 20.0,
        "root_linear_velocity_delta_max": 0.4,
        "root_angular_velocity_delta_max": 1.5,
    }
    # Per-level scale factors (level 0 = mild, last level = full strength).
    LEVEL_SCALES: Tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
    # Promote once survival >= threshold for N consecutive evaluations.
    PROMOTE_SURVIVAL: float = 0.9
    PROMOTE_PATIENCE: int = 2

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
        max_steps = self.custom_config["max_steps"]
        env_pb = self._env_pb()
        perturb = self._current_perturb_params()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid, **perturb)
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

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Advance the perturbation level when the policy reliably recovers.

        Weights stay ``(1.0,)`` throughout; the curriculum knob is the
        perturbation scale, advanced once survival holds at/above
        ``PROMOTE_SURVIVAL`` for ``PROMOTE_PATIENCE`` consecutive evals.
        """
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

        return (3.0, 1.0)

    
    def initial_weights(self) -> Tuple[float, ...]:
        return (3.0, 1.0)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """r_fall: per-step survival bonus + terminal signal.
        r_cross: cross-support balance reward from CrossSupportBalanceRewarder.
        """
        fell = "imbalance" in termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])
        if fell:
            r_fall[-1] = -penalty
        else:
            r_fall[-1] = penalty

        r_cross = _extract_per_step_scalar(observer_outputs, "cross_support", T)

        return {"r_fall": r_fall, "r_cross": r_cross}


    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, float]:
        """``survived`` = 0 only if the robot fell (imbalance termination)."""
        fell = "imbalance" in termination_proposals
        return {"survived": 0.0 if fell else 1.0}

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
EXPERIMENT = BalanceRecoverConfig()
