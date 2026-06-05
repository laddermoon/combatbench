
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig


class BasicBalanceConfig(ExperimentConfig):

    name = "basic_balance"
    reward_keys = ("r_fall",)  # Single reward: terminal fall penalty only
    gammas = {"r_fall": 0.99}   # Per-step reward is effectively 0, gamma for terminal

    # Stage blueprints
    BLUEPRINT = "basic_balance_env.yaml"  # Basic: fall detection only

    # Default static blueprint (used by tooling that reads env_blueprint).
    # The ACTIVE blueprint per stage is resolved by current_env_blueprint().
    env_blueprint = BLUEPRINT

    # Terminal fall penalty (set by training loop before buffer construction).
    terminal_fall_penalty: float = 1.0

    # Stateful scheduler
    _survival_rate: float = 0.0

    # --- Blueprint ownership: pick blueprint by current stage ---
    def current_env_blueprint(self) -> str:
        return self.BLUEPRINT

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

        # Weights always (1.0,) regardless of stage.
        return (1.0,)

    # Small per-step survival bonus (each alive step is worth this much).
    per_step_survival_reward: float = 0.01

    def extract_rewards(
        self,
        observer_outputs: dict,
        T: int,
        termination_proposals: Tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        """r_fall: small positive reward every step + terminal signal."""
        fell = "imbalance" in termination_proposals
        # Every step starts with a small positive reward for still being alive.
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        # Terminal override on the last step.
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
        """Per-episode metrics. ``survived`` = 0 only if the robot fell.

        ``"imbalance"`` in termination_proposals means ImbalanceTerminationPlugin
        triggered (robot fell). ``"timeout"`` means the robot stood the full
        horizon — that counts as survived.
        """
        fell = "imbalance" in termination_proposals
        return {"survived": 0.0 if fell else 1.0}

    def scheduler_info(self) -> Dict[str, Any]:
        """Return current scheduler state for logging."""
        return {
            "survival_rate": round(self._survival_rate, 3),
        }

    def scheduler_state(self) -> dict:
        """Serialize scheduler state for checkpoint."""
        return {
            "survival_rate": self._survival_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._survival_rate = float(state.get("survival_rate", 0.0))


# Singleton instance for the registry
EXPERIMENT = BasicBalanceConfig()
