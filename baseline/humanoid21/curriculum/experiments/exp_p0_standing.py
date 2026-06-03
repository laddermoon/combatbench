"""P0 experiment: Standing balance with fall-only penalty (2-stage curriculum).

Stage 1: Basic standing with fall detection only.
Stage 2: Add random initial state perturbation (tilt, angular velocity, joint offsets).

Transition: Stage 1 -> Stage 2 when eval survival rate reaches 100%.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig


class P0StandingConfig(ExperimentConfig):
    """P0: Single-reward (r_fall) standing balance with 2-stage initialization."""

    name = "p0_standing"
    reward_keys = ("r_fall",)  # Single reward: terminal fall penalty only
    gammas = {"r_fall": 0.99}   # Per-step reward is effectively 0, gamma for terminal

    # Stage blueprints
    STAGE1_BLUEPRINT = "p0_stage1_env.yaml"  # Basic: fall detection only
    STAGE2_BLUEPRINT = "p0_stage2_env.yaml"  # With random initial perturbation

    # Default static blueprint (used by tooling that reads env_blueprint).
    # The ACTIVE blueprint per stage is resolved by current_env_blueprint().
    env_blueprint = STAGE1_BLUEPRINT
    ppo_overrides: Dict[str, Any] = {}

    # Stateful scheduler
    _stage: int = 1
    _survival_rate: float = 0.0

    # --- Blueprint ownership: pick blueprint by current stage ---
    def current_env_blueprint(self) -> str:
        return self.STAGE1_BLUEPRINT if self._stage == 1 else self.STAGE2_BLUEPRINT

    def initial_weights(self) -> Tuple[float, ...]:
        """Single reward weight."""
        return (1.0,)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        """Stage transition: 1 -> 2 when eval survival rate reaches 100%.

        Both stages use identical weights (1.0,); only the blueprint differs
        (stage 2 adds InitialStatePerturbationPlugin). The active blueprint is
        resolved by current_env_blueprint() from self._stage, so simply
        advancing the stage switches the env on the next rollout.
        """
        # survived = fraction of eval episodes that did NOT terminate (= did not
        # fall) within the horizon, aggregated by compute_episode_metrics.
        survival_rate = float(eval_metrics.get("survived", 0.0))
        self._survival_rate = survival_rate

        if self._stage == 1 and survival_rate >= 1.0:
            self._stage = 2

        # Weights always (1.0,) regardless of stage.
        return (1.0,)

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
    ) -> Dict[str, np.ndarray]:
        """P0 uses only r_fall, which the framework injects automatically.

        Returning an empty dict lets PPOBuffer fill r_fall with the terminal
        fall penalty (-terminal_fall_penalty on the last step of terminated
        episodes). No per-step reward signal is used.
        """
        return {}

    def compute_episode_metrics(
        self,
        observer_outputs: dict,
        T: int,
        terminated: bool,
    ) -> Dict[str, float]:
        """Per-episode metrics. ``survived`` averages to the survival rate."""
        return {"survived": 0.0 if terminated else 1.0}

    def scheduler_info(self) -> Dict[str, Any]:
        """Return current scheduler state for logging."""
        return {
            "stage": self._stage,
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        """Serialize scheduler state for checkpoint."""
        return {
            "stage": self._stage,
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._stage = int(state.get("stage", 1))
        self._survival_rate = float(state.get("survival_rate", 0.0))


# Singleton instance for the registry
EXPERIMENT = P0StandingConfig()
