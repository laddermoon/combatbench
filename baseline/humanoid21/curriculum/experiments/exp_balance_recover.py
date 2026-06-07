
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from envs.framework.blueprint import EnvBlueprint
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
    reward_keys = ("r_fall",)  # Single reward: per-step survival + terminal
    gammas = {"r_fall": 0.99}

    env_blueprint = "balance_recover_env.yaml"

    # --- PPO tuning (see training analysis) ---
    # Raise the log_std floor so the policy can't collapse to saturated,
    # near-deterministic actions — the main driver of the KL explosions /
    # exploding policy_loss observed in the first run.
    log_std_min: float = -2.0

    # Per-experiment TrainConfig overrides: smaller actor LR + tighter KL/grad
    # clipping + fewer epochs to keep each PPO update from diverging.
    train_overrides: Dict[str, Any] = {
        "learning_rate": 1e-4,      # was 3e-4: slow the actor down
        "target_kl": 0.02,          # was 0.05: early-stop sooner
        "grad_clip_norm": 0.5,      # was 1.0: tighter gradient clipping
        "update_epochs": 3,         # was 4: less policy drift per batch
    }

    # Terminal fall penalty (also used as the terminal survival reward).
    terminal_fall_penalty: float = 1.0

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
    def build_rollout_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
        max_steps: int,
        *,
        policy_bp_b: PolicyBlueprint | None = None,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Same self-play scheme as the base, but materialize the env with the
        current curriculum level's perturbation magnitudes."""
        env_pb = self._get_env_pb()
        perturb = self._current_perturb_params()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid, **perturb)
            for aid in ("robot_a", "robot_b")
        }

        bp_b = policy_bp_b if policy_bp_b is not None else policy_bp

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(self.rollout_distance_min, self.rollout_distance_max)
            )
            jobs.append((
                policy_bp, bp_b,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

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

        return (1.0,)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """r_fall: small positive reward every alive step + terminal signal."""
        fell = "imbalance" in termination_proposals
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        if fell:
            r_fall[-1] = -float(self.terminal_fall_penalty)
        else:
            r_fall[-1] = float(self.terminal_fall_penalty)
        return {"r_fall": r_fall}

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
